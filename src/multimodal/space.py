# -*- coding: utf-8 -*-
"""M2 64 维共享语义空间（multimodal-design.md §1.5 / §1.2 / §2.2）。

分区：A 概念锚区 [0,48) ｜ B 意图锚区 [48,56) ｜ C 模态私有区 [56,64)。
锚定：带符号双槽哈希（md5 + 双盐，同一概念名三模态同槽 → 余弦自然高）；
度量：掩码余弦 sim_cross = cos(a[0:56], b[0:56])，同模态 sim_full = cos(a, b)。
投影：P_img / P_aud 为 seed=42 确定性 8×8 高斯投影（保留模态私有细节区分度）。
"""
from __future__ import annotations

import hashlib
import math
import random

EMBED_DIM = 64
A_END = 48         # A 概念锚区上界（开区间）
B_END = 56         # B 意图锚区上界（开区间）
SHARED_DIM = 56    # A∪B 共享区上限（掩码余弦用）
PROJ_SEED = 42     # 固定投影种子（确定性可复现）


def _md5i(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def term_slots(term: str, region: str = "A"):
    """带符号双槽哈希：term → ((dim1, sign1), (dim2, sign2))，dim 落在指定锚区。

    双槽 + 独立符号盐降低碰撞偏置（§1.2）；region="A" → [0,48)，"B" → [48,56)。
    """
    base, span = (0, A_END) if region == "A" else (B_END, EMBED_DIM - B_END)
    d1 = base + _md5i("slot1|" + term) % span
    s1 = 1.0 if _md5i("sign1|" + term) & 1 else -1.0
    d2 = base + _md5i("slot2|" + term) % span
    s2 = 1.0 if _md5i("sign2|" + term) & 1 else -1.0
    return (d1, s1), (d2, s2)


def add_term(vec: list, term: str, weight: float, region: str = "A") -> None:
    """把 term 的双槽锚叠加进 vec（region="A"/"B"）。"""
    for d, s in term_slots(term, region):
        vec[d] += s * weight


def l2_normalize(vec) -> list:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def cosine(a, b, lo: int = 0, hi: int = EMBED_DIM) -> float:
    """区间余弦（掩码余弦基础；空区间或零向量 → 0.0，不抛错）。"""
    va, vb = a[lo:hi], b[lo:hi]
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return sum(x * y for x, y in zip(va, vb)) / (na * nb)


def masked_cosine(a, b, cross: bool = True) -> float:
    """§2.2 掩码余弦：cross=True → 前 56 维共享区；False → 全 64 维。"""
    return cosine(a, b, 0, SHARED_DIM if cross else EMBED_DIM)


def _proj_matrix(seed: int, size: int = 8):
    """seed 确定性高斯投影矩阵（scale=1/√cols 保证单位向量投影范数≈1）。"""
    rng = random.Random(seed)
    scale = 1.0 / math.sqrt(size)
    return tuple(tuple(rng.gauss(0.0, scale) for _ in range(size)) for _ in range(size))


P_IMG = _proj_matrix(PROJ_SEED)      # 图像私有区投影（seed=42）
P_AUD = _proj_matrix(PROJ_SEED + 1)  # 语音私有区投影（独立 seed）


def project(P, v) -> list:
    """P·v（固定投影；v 建议 L2 归一后传入）。"""
    return [sum(P[i][j] * v[j] for j in range(len(v))) for i in range(len(P))]
