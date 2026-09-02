# -*- coding: utf-8 -*-
"""A3 语音编码器（multimodal-design.md §1.4）：语气原型匹配 + 意图/语气锚 + 私有区投影。

- 输入：8 维合成声学特征 dict {"values":[8×float], "tone":...}（或直接 list）；
- 语气软匹配：对 3 个语气中心（T1 原型，缺席→T0 内置表）算余弦取 top-1；
- 嵌入：B 区 = tone 词 + intent 词哈希锚（权重 = 匹配置信，与文本触发词同槽）；
  C 区 = P_aud·f8；A 区置零（语音不含视觉概念）；
- intent 解码：question→qa，statement→describe（默认），command→command；
  retrieve 依赖文本/上下文，audio-only 场景由 M1/M4 融合判定。
"""
from __future__ import annotations

import math

from . import concepts as C
from .image_encoder import parse_feat
from .space import B_END, EMBED_DIM, P_AUD, add_term, l2_normalize, project

__all__ = ["encode_audio"]


def _centers(protos):
    t1 = (protos or {}).get("audio_protos") or {}
    return {**C.T0_AUDIO_CENTERS, **{k: v for k, v in t1.items() if len(v) == 8}}


def encode_audio(feat, protos=None):
    """语音特征 → {"embedding","tokens","tone","intent","confidence"}。非法输入 → None。"""
    vals = parse_feat(feat)
    if vals is None:
        return None
    v = l2_normalize(vals)
    centers = _centers(protos)
    nv = math.sqrt(sum(x * x for x in v)) or 1e-9
    scored = []
    for tone in sorted(centers):
        cc = centers[tone]
        nc = math.sqrt(sum(x * x for x in cc)) or 1e-9
        scored.append((tone, sum(a * b for a, b in zip(v, cc)) / (nv * nc)))
    scored.sort(key=lambda t: (-t[1], t[0]))
    tone, conf = scored[0]
    intent = C.INTENT_OF_TONE.get(tone)

    vec = [0.0] * EMBED_DIM
    add_term(vec, tone, conf, "B")     # 语气词锚（B 区）
    if intent:
        add_term(vec, intent, conf, "B")  # 意图词锚（与文本触发词共享槽位）
    vec[B_END:EMBED_DIM] = project(P_AUD, v)

    emb = l2_normalize(vec)
    if not any(emb):
        return None  # 1003
    label_cn = C.TONES.get(tone, (tone,))[0]
    return {"embedding": [round(x, 6) for x in emb],
            "tokens": [label_cn, tone, intent] if intent else [label_cn, tone],
            "tone": tone,
            "intent": intent,
            "concept_hits": [],
            "confidence": round(conf, 4)}
