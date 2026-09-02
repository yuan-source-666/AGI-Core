# -*- coding: utf-8 -*-
"""A2 图像编码器（multimodal-design.md §1.3）：原型 top-2 概念软匹配 + 私有区投影。

- 输入：8 维合成特征 dict {"values":[8×float],...}（或直接 list）；
- 概念软匹配：对 30 个概念中心（T1 原型，缺席→T0 内置表）算余弦取 top-2，
  权重 w_k=(cos_k+1)/2 归一；
- MATCH_DIMS=slice(0,5)：概念锁定维（RGB+圆度+边缘密度）。后 3 维为数据管道声明的
  样本级自由风格维（data-pipeline.md §8），不参与概念匹配，仅入私有区——
  该选择使概念解码对风格噪声免疫（实现口径，README 有记录）；
- 嵌入：A 区 = top-2 概念的组合词+颜色词+形状词哈希锚（与文本同槽）；若 T2 校准
  已启用则 A 区 = W·[f8,1]；C 区 = P_img·f8；B 区置零；L2 归一。
"""
from __future__ import annotations

import math

from . import concepts as C
from .space import A_END, B_END, EMBED_DIM, P_IMG, add_term, l2_normalize, project

__all__ = ["encode_image", "MATCH_DIMS", "parse_feat"]

MATCH_DIMS = slice(0, 5)  # 概念锁定维（后 3 维为自由风格维）


def parse_feat(feat):
    """ModalInput.raw → values[8×float]；非法 → None（1002 由 perceive 处置）。"""
    if isinstance(feat, dict):
        vals = feat.get("values")
    elif isinstance(feat, (list, tuple)):
        vals = list(feat)
    else:
        return None
    if not isinstance(vals, (list, tuple)) or len(vals) != 8:
        return None
    try:
        out = [float(x) for x in vals]
    except (TypeError, ValueError):
        return None
    if any(math.isnan(x) or math.isinf(x) for x in out):
        return None
    return out


def _centers(protos):
    """T1 原型（缺席概念回填 T0）→ 概念中心表 {label: 8 维}。"""
    t1 = (protos or {}).get("image_protos") or {}
    return {**C.T0_IMAGE_CENTERS, **{k: v for k, v in t1.items() if len(v) == 8}}


def _match_top2(vals, centers):
    """对概念中心算余弦（MATCH_DIMS 子空间）→ top-2 [(label, cos)]（确定性排序）。"""
    v = vals[MATCH_DIMS]
    nv = math.sqrt(sum(x * x for x in v)) or 1e-9
    scored = []
    for label in sorted(centers):
        cc = centers[label][MATCH_DIMS]
        nc = math.sqrt(sum(x * x for x in cc)) or 1e-9
        scored.append((label, sum(a * b for a, b in zip(v, cc)) / (nv * nc)))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored[:2]


def encode_image(feat, protos=None, W=None):
    """图像特征 → {"embedding","tokens","concept_hits","concept_weights","confidence"}。

    W：T2 校准系数（48×9）；None → top-2 原型软匹配路径。非法输入 → None。
    """
    vals = parse_feat(feat)
    if vals is None:
        return None
    v = l2_normalize(vals)
    top = _match_top2(vals, _centers(protos))
    raw = [(cos + 1.0) / 2.0 for _, cos in top]
    s = sum(raw) or 1.0
    ws = [w / s for w in raw]

    vec = [0.0] * EMBED_DIM
    if W is None:
        for (label, _), w in zip(top, ws):
            word = C.LABEL_TO_WORD.get(label, label.replace("-", ""))
            for term in (word, word[0], word[1:]):
                add_term(vec, term, w, "A")
    else:  # T2：A 区 = W·[f8 归一, 1]
        x = v + [1.0]
        vec[:A_END] = [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(A_END)]
    vec[B_END:EMBED_DIM] = project(P_IMG, v)  # 私有区保留细节区分度

    emb = l2_normalize(vec)
    if not any(emb):
        return None  # 1003：范数为 0
    tokens = []
    for (label, _), _w in zip(top, ws):
        word = C.LABEL_TO_WORD.get(label, label.replace("-", ""))
        for t in (word, word[0], word[1:]):
            if t not in tokens:
                tokens.append(t)
    return {"embedding": [round(x, 6) for x in emb],
            "tokens": tokens,
            "concept_hits": [label for label, _ in top],
            "concept_weights": [round(w, 4) for w in ws],
            "confidence": round(top[0][1], 4)}
