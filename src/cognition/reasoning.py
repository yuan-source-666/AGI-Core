# -*- coding: utf-8 -*-
"""M1 混合推理引擎（C1 reason）—— 规则前向链接 + 记忆证据投票 + 模板回退三通道仲裁。

对齐 docs/algorithm-design.md v1.0 §3（设计）/§5.3（接口）；
Thought 严格符合 I2（steps: recall→attend→infer 三步，op 枚举固定，api-spec §3.3）。

置信度仲裁（§3.2）：
    conf = (w_r·C_rule + w_m·C_mem + w_t·C_tpl) / (w_r + w_m + w_t)
    默认 w_r=0.5, w_m=0.35, w_t=0.15；C_tpl=0.3（保守常数）。
- C_rule = rule.weight × pattern 覆盖率（命中关键词数/总关键词数）；
- C_mem = 1 − exp(−Σ w_m·sim)（证据充分性饱和归一，单调不减且 ∈ [0,1)）；
- 模板通道：E 为空或最高检索分 < θ=0.2 时启用（澄清追问，低幻觉优先）；
- 通道仲裁：规则结论优先 → 记忆证据次之 → 模板追问兜底；
- 前向链接链深 ≤ 3（规则结论回流文本池，可触发链式规则，访问集防环）。
门控（conf < 0.6 → clarify）由 M4 plan() 执行（I4），本模块只产出 Thought。
"""
from __future__ import annotations

import hashlib
import json
import math
import os

from ._shared import (CognitionError, is_query_obs, obs_text, query_ctx,
                      tokenize)
from . import attention, memory

W_RULE, W_MEM, W_TPL = 0.5, 0.35, 0.15
C_TPL = 0.3           # 模板通道保守置信度
SIM_THETA = 0.2       # 模板通道启用阈值（E 空或最高分 < θ）
K_RECALL = 5          # 证据召回条数
ATTEND_K = 8          # 注意力 Top-K
CHAIN_DEPTH = 3       # 规则前向链接最大链深
RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules.json")

_rules_cache = None


def load_rules(path=None, refresh=False) -> list:
    """载入 IF-THEN 规则库（rules.json）；不可读 → 降级为空规则集（不阻塞）。"""
    global _rules_cache
    if path is None and _rules_cache is not None and not refresh:
        return _rules_cache
    try:
        with open(path or RULES_PATH, "r", encoding="utf-8") as f:
            rules = json.load(f)
    except Exception:
        rules = []
    rules = [r for r in rules if isinstance(r, dict) and r.get("id")
             and (r.get("all") or r.get("any"))]
    if path is None:
        _rules_cache = rules
    return rules


# ------------------------------------------------------------ 通道实现 --
def _kw_hit(kw: str, text: str, tset: set) -> bool:
    return kw in tset or kw in text


def _rule_coverage(rule: dict, text: str, tset: set):
    """返回覆盖率 [0,1]；不满足 all/any 语义 → None（未命中）。"""
    allk = [str(k) for k in (rule.get("all") or [])]
    anyk = [str(k) for k in (rule.get("any") or [])]
    if not allk and not anyk:
        return None
    n_all = sum(1 for k in allk if _kw_hit(k, text, tset))
    if n_all < len(allk):          # all 组：须全部在场
        return None
    n_any = sum(1 for k in anyk if _kw_hit(k, text, tset))
    if anyk and n_any == 0:        # any 组：须至少一个在场
        return None
    total = len(allk) + len(anyk)
    return (n_all + n_any) / total if total else 1.0


def _forward_chain(text: str, tset: set, rules: list, depth: int = CHAIN_DEPTH) -> list:
    """前向链接（链深≤3）：规则结论回流文本池，可触发链式规则。"""
    triggered, visited = [], set()
    for _ in range(max(1, depth)):
        fired = []
        for r in rules:
            rid = str(r.get("id"))
            if rid in visited:
                continue
            cov = _rule_coverage(r, text, tset)
            if cov is not None:
                fired.append((r, cov))
                visited.add(rid)
        if not fired:
            break
        triggered.extend(fired)
        add = " ".join(str(r.get("then", "")) for r, _ in fired)
        text = text + " " + add
        tset |= set(tokenize(add))
    return triggered


def _evidence_cm(E: list):
    """证据投票：C_mem = 1 − exp(−Σ top-3 检索分)（饱和归一）。"""
    if not E:
        return 0.0, None
    ssum = 0.0
    for m in E[:3]:
        s = float(m.get("score", 0.0) or 0.0)
        if s > 0:
            ssum += s
    return 1.0 - math.exp(-ssum), E[0]


# ------------------------------------------------------------ C1 入口 --
def reason(obs, query, session_id, rules: list = None, k_recall: int = K_RECALL) -> dict:
    """C1 reason(obs, query, session_id) -> Thought（I2 对齐，api-spec §3.3）。"""
    if not isinstance(obs, list):
        raise CognitionError(2001, "obs 非法：期望 list[Observation]")
    query = str(query or "")
    session_id = str(session_id)
    rules = load_rules() if rules is None else rules

    # 查询语义上下文（Phase 4 共享空间桥接）：qemb/qset 优先取 M4 注入查询
    # 观测的 M2 真实语义嵌入；query_obs 缺席时退化为 M1 自建哈希嵌入。
    qemb, qset, _ = query_ctx(query, obs)

    # 通道0：记忆检索（检索失败就地降级为空证据，不阻塞推理）
    try:
        E = memory.mem_recall(query, k=k_recall, qemb=qemb, qset=qset)
    except Exception:
        E = []

    # 候选观测 = 当前 obs（剔除查询观测）∪ 会话工作记忆（algorithm-design §4.1 第 3 步）
    try:
        working = memory.get_working(session_id)
    except Exception:
        working = []
    cands = [o for o in obs if isinstance(o, dict) and not is_query_obs(o)]
    for m in working:
        cands.append({"obs_id": str(m.get("mem_id") or "wobs-?"), "modality": "text",
                      "embedding": m.get("embedding") or [],
                      "tokens": m.get("tokens") or tokenize(m.get("content", "")),
                      "meta": {"salience_prior": 1.0, "source": "working",
                               "ts": m.get("ts")}})

    # 注意力聚焦（A1；空候选 → 空焦点，不抛错；共享空间 α 通道）
    if cands:
        try:
            F = attention.attend(query, cands, k=ATTEND_K, qemb=qemb, qset=qset)
        except CognitionError:
            F = {"items": [], "weights": [], "scores": [], "k": 0}
    else:
        F = {"items": [], "weights": [], "scores": [], "k": 0}
    focus_ids = set(F.get("items") or [])
    focused = [o for o in cands if str(o.get("obs_id")) in focus_ids]

    # 通道1：规则前向链接（文本池 = 查询 + 证据 + 聚焦观测）
    pool = query + " " + " ".join(str(m.get("content", "")) for m in E) + " " + \
           " ".join(obs_text(o) for o in focused)
    tset = set(tokenize(pool))
    trig = _forward_chain(pool, tset, rules)
    C_rule, best_rule, chain = 0.0, None, []
    if trig:
        scored = [(min(1.0, float(r.get("weight", 0.5)) * cov), r) for r, cov in trig]
        best = max(scored, key=lambda t: t[0])
        C_rule, best_rule = round(best[0], 4), best[1]
        chain = [str(r.get("id")) for r, _ in trig]

    # 通道2：记忆证据投票
    C_mem, best_mem = _evidence_cm(E)
    best_sim = float(E[0].get("score", 0.0) or 0.0) if E else 0.0

    # 通道3：模板回退（E 为空或最高相似度 < θ）
    tpl_active = (not E) or (best_sim < SIM_THETA)
    C_tpl = C_TPL if tpl_active else 0.0

    # 仲裁：conf = Σ w·C / Σ w（§3.2）
    conf = (W_RULE * C_rule + W_MEM * C_mem + W_TPL * C_tpl) / (W_RULE + W_MEM + W_TPL)

    # 答案选择：规则优先 → 证据次之 → 模板兜底
    if best_rule is not None and C_rule >= 0.05:
        answer = str(best_rule.get("then", "")).strip()
        if best_mem is not None and best_sim >= SIM_THETA:
            answer += f"（记忆佐证：{str(best_mem.get('content', ''))[:32]}…）"
        tag = str(best_rule.get("id"))
    elif E and best_sim >= SIM_THETA:
        answer = f"基于记忆证据：{str(best_mem.get('content', ''))[:80]}"
        tag = "evidence-vote"
    else:
        answer = ("（暂无足够证据支撑结论）能否补充更多信息，"
                  "例如具体的颜色/形状或完整的问题描述？")
        tag = "template"
    if not answer.strip():
        raise CognitionError(2003, "推理无可用证据且模板不可用")

    thought = {
        "thought_id": "th-" + hashlib.md5(
            f"{session_id}|{query}|{len(obs)}|{tag}|{len(E)}"
            .encode("utf-8")).hexdigest()[:8],
        "steps": [
            {"step": 1, "op": "recall",
             "used_mem": [str(m.get("mem_id")) for m in E]},
            {"step": 2, "op": "attend",
             "focus_obs": [str(i) for i in F.get("items") or []]},
            {"step": 3, "op": "infer", "rule": tag, "chain": chain,
             "channels": {"rule": C_rule, "memory": round(C_mem, 4),
                          "template": round(C_tpl, 4)},
             "focus_k": F.get("k", 0)},
        ],
        "answer": answer,
        "confidence": round(max(0.0, min(1.0, conf)), 4),
    }
    json.dumps(thought, ensure_ascii=False)  # I2 契约：可 JSON 序列化自检
    return thought
