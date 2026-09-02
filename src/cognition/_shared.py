# -*- coding: utf-8 -*-
"""M1 认知核心内部共享工具（模块私有，不对外导出）。

边界约定（architecture.md §2）：M1 不 import M2/M3 —— 嵌入/分词在本模块内
自实现确定性哈希词袋（算法与 M2 桩 embed_text 逐字一致，当下两模块嵌入空间
数值兼容；演进位：后续统一为共享嵌入服务，接口不变）。
"""
from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timedelta, timezone

EMBED_DIM = 64  # 统一嵌入维度（architecture §3 I1）
_CST = timezone(timedelta(hours=8))  # Asia/Shanghai
_WORD_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")


class CognitionError(Exception):
    """认知层错误（2xxx，algorithm-design §5.5）：不抛裸 msg，携带 code。"""

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code, self.msg = int(code), str(msg)


def now_ts() -> float:
    return datetime.now(_CST).timestamp()


def now_iso() -> str:
    return datetime.now(_CST).isoformat(timespec="seconds")


def tokenize(text) -> list:
    """分词：英文/数字整词（小写），中文按字符 bigram（与 M2 桩约定一致）。"""
    tokens = []
    for m in _WORD_RE.findall(str(text or "")):
        if len(m) > 1 and re.match(r"[\u4e00-\u9fff]", m):
            tokens.extend(m[i:i + 2] for i in range(len(m) - 1))
        else:
            tokens.append(m.lower() if m.isascii() else m)
    return tokens


def embed(text) -> list:
    """64 维确定性嵌入：MD5 双槽哈希词袋 + L2 归一化（零参数、可复现）。"""
    vec = [0.0] * EMBED_DIM
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % EMBED_DIM] += 1.0 if (h >> 8) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


def cosine(a, b) -> float:
    """余弦相似度（输入须为数值 list；空/非法 → 0.0）。"""
    if not a or not b:
        return 0.0
    dims = min(len(a), len(b))
    s = 0.0
    for x, y in zip(a[:dims], b[:dims]):
        s += x * y
    na = math.sqrt(sum(x * x for x in a[:dims])) or 1.0
    nb = math.sqrt(sum(y * y for y in b[:dims])) or 1.0
    return s / (na * nb)


def softmax(xs, temperature: float = 0.5) -> list:
    """数值稳定 softmax（T=0.5，对齐 algorithm-design §1.2）。"""
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp((x - m) / temperature) for x in xs]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def l2_normalize(vec) -> list:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def mean_vector(vectors) -> list:
    """向量均值并 L2 归一化（巩固簇质心用；空输入 → 零向量）。"""
    if not vectors:
        return [0.0] * EMBED_DIM
    n = len(vectors)
    return l2_normalize([sum(col) / n for col in zip(*vectors)])


def obs_text(o) -> str:
    """观测 → 规则池文本：meta.query 优先，退化到 tokens 拼接。"""
    if not isinstance(o, dict):
        return ""
    meta = o.get("meta") if isinstance(o.get("meta"), dict) else {}
    q = meta.get("query")
    if isinstance(q, str) and q.strip():
        return q
    toks = o.get("tokens")
    if isinstance(toks, list) and toks:
        return " ".join(str(t) for t in toks)
    return ""


# ---- Phase 4 共享语义空间桥接（目标①）----
# M1/M2 禁止互相 import（architecture §2 / api-spec §0.2）：由 M4 编排，把 M2
# 的 embed_text(query) 真实语义嵌入包装成一条「查询观测」（meta.role="query"，
# source="m4-inject"）随 I1 obs 一并传入；M1 侧仅按观测字段消费，不触碰 M2。
QUERY_ROLE = "query"


def is_query_obs(o) -> bool:
    """是否 M4 注入的查询观测（meta.role == 'query'）。"""
    if not isinstance(o, dict):
        return False
    meta = o.get("meta") if isinstance(o.get("meta"), dict) else {}
    return meta.get("role") == QUERY_ROLE


def query_obs_from(obs) -> dict:
    """取首条查询观测（无则 None）。"""
    for o in (obs or []):
        if is_query_obs(o):
            return o
    return None


def query_ctx(query, obs):
    """查询语义上下文 → (qemb, qset, qo)。

    优先复用 M4 注入查询观测的 M2 真实 64 维语义嵌入与 tokens（共享空间，
    attention α 通道 / 记忆检索 / 情景回写共用）；查询观测缺席时退化为 M1
    自建哈希嵌入（self-test / 直连调用无感知，行为同 Phase 3）。
    """
    qo = query_obs_from(obs)
    if qo is not None:
        emb = qo.get("embedding")
        if isinstance(emb, list) and emb:
            toks = qo.get("tokens")
            if isinstance(toks, list) and toks:
                return list(emb), set(map(str, toks)), qo
            return list(emb), set(tokenize(query)), qo
    return embed(query), set(tokenize(query)), None
