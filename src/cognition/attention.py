# -*- coding: utf-8 -*-
"""M1 注意力机制（A1 attend）—— 零参数打分 + Top-K 硬筛选 + softmax 加权。

对齐 docs/algorithm-design.md v1.0 §1（设计）/§5.1（接口）：

    score(q, o) = α·cos(e_q, e_o) + β·overlap(T_q, T_o) + γ·exp(−Δt/τ) + δ·salience(o)

- overlap：IDF 加权 Jaccard（词频/稀有度驱动的相关性，idf=log((N+1)/(df+1))+1）；
- 新鲜度：Δt = now − ts(o)，τ=3600s（注意瞬态衰减；无 ts 视为刚到达）；
- salience：模态先验（text 1.0 / image 0.9 / audio 0.85）× 实体密度（数字/大写词占比）；
- 默认权重 α=0.45, β=0.30, γ=0.15, δ=0.10（cfg 可覆盖，演示可配置）；
- Top-K 硬筛选（K=8，控制推理上下文长度）+ softmax(scores/T=0.5) 加权。

出参 FocusSet（§5.1）：{"items": [obs_id...], "weights": [...], "scores": [...], "k"}；
items 引用 obs_id，供 I2 Thought.steps[].focus_obs 直接引用。
错误：obs 空/非法 → CognitionError(2001)。
"""
from __future__ import annotations

import math
from collections import Counter

from ._shared import (CognitionError, cosine, embed, is_query_obs, now_ts,
                      softmax, tokenize)

ALPHA, BETA, GAMMA, DELTA = 0.45, 0.30, 0.15, 0.10
TAU_FRESHNESS = 3600.0
K_DEFAULT = 8
SOFTMAX_T = 0.5
MODALITY_PRIOR = {"text": 1.0, "image": 0.9, "audio": 0.85}
_CFG_KEYS = ("alpha", "beta", "gamma", "delta")


def _obs_tokens(o: dict) -> set:
    toks = o.get("tokens")
    if isinstance(toks, list) and toks:
        return set(map(str, toks))
    if o.get("content"):
        return set(tokenize(o["content"]))
    return set()


def _entity_density(tokens) -> float:
    """实体密度：数字词 / ASCII 大写词 占比（专名、数字、大写词，§1.1）。"""
    toks = [str(t) for t in (tokens or [])]
    if not toks:
        return 0.0
    n = 0
    for t in toks:
        if any(c.isdigit() for c in t):
            n += 1
        elif t.isascii() and t.isalpha() and not t.islower():
            n += 1
    return n / len(toks)


def _salience(o: dict) -> float:
    meta = o.get("meta") if isinstance(o.get("meta"), dict) else {}
    prior = meta.get("salience_prior")
    if not isinstance(prior, (int, float)) or prior <= 0:
        prior = MODALITY_PRIOR.get(o.get("modality"), 0.5)
    return float(prior) * _entity_density(o.get("tokens"))


def _freshness(o: dict, now: float) -> float:
    ts = o.get("ts")
    if ts is None and isinstance(o.get("meta"), dict):
        ts = o["meta"].get("ts")
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        ts = now
    return math.exp(-max(0.0, now - ts) / TAU_FRESHNESS)


def attend(query, obs, k: int = K_DEFAULT, cfg: dict = None,
           qemb=None, qset=None) -> dict:
    """A1 attend(query, obs, k=8, cfg, qemb=None, qset=None) -> FocusSet。

    对候选观测逐项打分 → Top-K 硬筛选 → softmax 温度归一化权重。
    qemb/qset：共享空间查询嵌入/分词（可选，Phase 4 M4 桥接 M2 语义嵌入）；
    不传时优先取 obs 内注入的查询观测，否则退化 M1 自建哈希嵌入。
    """
    if not isinstance(obs, list) or not obs:
        raise CognitionError(2001, "注意力输入为空/非法：obs 期望非空 list[Observation]")
    valid = [o for o in obs if isinstance(o, dict)]
    if not valid:
        raise CognitionError(2001, "注意力输入非法：无有效 Observation")

    weights = {k2: v for k2, v in zip(_CFG_KEYS, (ALPHA, BETA, GAMMA, DELTA))}
    if isinstance(cfg, dict):
        for key in _CFG_KEYS:
            if key in cfg:
                weights[key] = float(cfg[key])

    query = str(query or "")
    # Phase 4 共享语义空间：查询嵌入优先取显式 qemb/qset 入参，其次注入查询
    # 观测的 M2 真实语义嵌入（α 通道 cos 对齐 M2 64 维空间）；缺席退化
    # M1 自建哈希嵌入。
    qemb_out = None
    if qemb is not None:
        qemb_out = list(qemb)
    else:
        for o in valid:
            if is_query_obs(o) and isinstance(o.get("embedding"), list) \
                    and o["embedding"]:
                qemb_out = list(o["embedding"])
                break
    if qemb_out is None:
        qemb_out = embed(query)
    # 分词集合：qset 入参 → 注入查询观测 tokens → 自建分词
    qset_out = set(map(str, qset)) if qset is not None else set(tokenize(query))
    if qset is None:
        for o in valid:
            if is_query_obs(o) and isinstance(o.get("tokens"), list) \
                    and o["tokens"]:
                qset_out = set(map(str, o["tokens"]))
                break
    # 查询观测本身不参与评分（避免与 query 自相关占据焦点）
    cands = [o for o in valid if not is_query_obs(o)]
    if not cands:
        return {"items": [], "weights": [], "scores": [], "k": 0}
    try:
        k = max(1, min(int(k), len(cands)))
    except (TypeError, ValueError):
        k = min(K_DEFAULT, len(cands))

    now = now_ts()
    N = len(cands)

    df = Counter()
    tsets = []
    for o in cands:
        tset = _obs_tokens(o)
        tsets.append(tset)
        df.update(tset)
    idf_cache = {}

    def idf(w):
        v = idf_cache.get(w)
        if v is None:
            v = math.log((N + 1) / (df.get(w, 0) + 1)) + 1.0
            idf_cache[w] = v
        return v

    scored = []
    for o, tset in zip(cands, tsets):
        inter = qset_out & tset
        union = qset_out | tset
        if union:
            num = sum(idf(w) for w in inter)
            den = sum(idf(w) for w in union)
            overlap = num / den if den > 0 else 0.0
        else:
            overlap = 0.0
        s = (weights["alpha"] * cosine(qemb_out, o.get("embedding") or [])
             + weights["beta"] * overlap
             + weights["gamma"] * _freshness(o, now)
             + weights["delta"] * _salience(o))
        scored.append((round(s, 6), str(o.get("obs_id") or "obs-?")))
    scored.sort(key=lambda t: (-t[0], t[1]))  # 分数降序，obs_id 升序保证确定性

    top = scored[:k]
    raw = [s for s, _ in top]
    ws = softmax(raw, SOFTMAX_T)
    return {"items": [oid for _, oid in top],
            "weights": [round(w, 6) for w in ws],
            "scores": [round(s, 4) for s in raw],
            "k": k}
