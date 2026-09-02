# -*- coding: utf-8 -*-
"""M2 多模态感知模块（真实实现，替换骨架期 STUB；I1/I1'/§2.2 签名冻结不变）。

文件结构（multimodal-design §4.4 + 任务书命名兼容层）：
    concepts.py        概念词表 + T0 确定性映射表（30 组合概念 / 4 intent / 3 tone）
    space.py           64 维共享空间：分区 / 带符号双槽哈希 / 固定投影 / 掩码余弦
    text_encoder.py    A1 文本编码（概念词典最大匹配 + idf 词袋哈希）
    image_encoder.py   A2 图像编码（原型 top-2 概念软匹配 + 私有区投影）
    vision_encoder.py  image_encoder 的任务命名兼容层（≡ 同一实现）
    audio_encoder.py   A3 语音编码（语气原型匹配 + 意图/语气锚）
    aligner.py         A4 跨模态对齐（T1 原型 / T2 校准 / 掩码余弦检索 / 评测）
    align.py           aligner 的设计文档命名兼容层（≡ 同一实现）
    perceive.py        I1 感知入口 + 容错降级（1001-1004 / μ 矩阵 / meta.missing）
    selftest.py        自测脚本：python src/multimodal/selftest.py
    prototypes.json    T1 原型统计缓存（selftest 首跑生成；缺席 → T0 兜底）
"""
from __future__ import annotations

from . import aligner, concepts, space
from .aligner import align_eval, fit, fit_t2, load_protos, load_t2, retrieve
from .audio_encoder import encode_audio
from .image_encoder import encode_image
from .perceive import MU_MATRIX, SALIENCE_PRIOR, PerceptionError, perceive
from .space import EMBED_DIM, masked_cosine
from .text_encoder import embed_text, encode_text, tokenize

__all__ = ["perceive", "embed_text", "similarity", "PerceptionError", "EMBED_DIM",
           "encode_text", "encode_image", "encode_audio", "tokenize", "fit",
           "fit_t2", "align_eval", "retrieve", "load_protos", "load_t2",
           "MU_MATRIX", "SALIENCE_PRIOR", "concepts", "space", "aligner"]


def similarity(a, b, cross: bool = True) -> float:
    """§2.2 相似度：cross=True → 掩码余弦（前 56 维共享区）；False → 全 64 维。"""
    return round(masked_cosine(a, b, cross), 6)
