# -*- coding: utf-8 -*-
"""M2 概念词表与 T0 确定性映射表（multimodal-design.md §1.1 / §2.3 T0）。

概念槽与 data/build_dataset.py 生成规则镜像一致：
- 颜色 6 × 形状 5 = 30 组合概念；图像 8 维特征 = [R,G,B,圆度,边缘密度,亮度,对比度,纹理熵]，
  前 5 维概念锁定，后 3 维为样本级自由风格维（data-pipeline.md §8）——
  故概念匹配只取前 5 维（见 image_encoder.MATCH_DIMS），风格维噪声天然被剔除；
- T0 内置中心：数据集缺席时兜底（风格维取生成均值 0.5，与 build_dataset.CENTERS 一致）。
"""
from __future__ import annotations

COLORS = {"红": (0.90, 0.10, 0.10), "橙": (0.95, 0.50, 0.15), "黄": (0.90, 0.85, 0.20),
          "绿": (0.15, 0.80, 0.25), "蓝": (0.10, 0.30, 0.90), "紫": (0.55, 0.15, 0.80)}
SHAPES = {"圆": (0.95, 0.30), "方": (0.75, 0.65), "三角": (0.45, 0.80),
          "星": (0.20, 0.45), "六边": (0.55, 0.20)}
STYLE_MEAN = 0.5  # 自由风格维（亮度/对比度/纹理熵）生成均值

# 别名 → 规范概念词（提升文本侧概念命中："蓝色三角形" → 蓝 / 三角）
COLOR_ALIASES = {f"{c}色": c for c in COLORS}
SHAPE_ALIASES = {"圆形": "圆", "方形": "方", "正方": "方", "三角形": "三角",
                 "星形": "星", "六边形": "六边"}

# 30 组合概念：词形 "蓝三角"（分词/哈希锚定用）；标签 "蓝-三角"（数据集 label 口径）
COMBO_WORDS = [f"{c}{s}" for c in COLORS for s in SHAPES]
CONCEPT_LABELS = {w: w[0] + "-" + w[1:] for w in COMBO_WORDS}
LABEL_TO_WORD = {v: k for k, v in CONCEPT_LABELS.items()}
CONCEPT_TERMS = set(COMBO_WORDS) | set(COLORS) | set(SHAPES)

# 语音语气原型（T0）：8 维 = [基频,音强,语速,过零率,频谱质心,频谱带宽,音长,静音比]
TONES = {"question": ("疑问-升调", [0.75, 0.55, 0.45, 0.40, 0.70, 0.55, 0.40, 0.30]),
         "statement": ("陈述-平调", [0.50, 0.50, 0.50, 0.35, 0.50, 0.45, 0.50, 0.35]),
         "command": ("指令-短促", [0.62, 0.80, 0.75, 0.50, 0.60, 0.65, 0.25, 0.20])}
INTENT_OF_TONE = {"question": "qa", "statement": "describe", "command": "command"}
TONE_OF_INTENT = {"qa": "question", "describe": "statement",
                  "retrieve": "statement", "command": "command"}
INTENTS = ("qa", "describe", "retrieve", "command")

# 意图触发词（§1.1 基础集 + 12 句式模板覆盖扩展）；冲突按 INTENT_PRIORITY 解析
TRIGGERS = [("什么", "qa"), ("吗", "qa"), ("哪", "qa"), ("怎么", "qa"), ("如何", "qa"),
            ("描述", "describe"), ("看看", "describe"), ("说说", "describe"), ("形容", "describe"),
            ("哪张", "retrieve"), ("检索", "retrieve"), ("找", "retrieve"), ("有没有", "retrieve"),
            ("请", "command"), ("帮我", "command"), ("播放", "command"),
            ("朗读", "command"), ("停止", "command")]
INTENT_PRIORITY = ("qa", "describe", "retrieve", "command")

# T0 内置概念中心（镜像 build_dataset.CENTERS：RGB+形状锁定，风格维取均值）
T0_IMAGE_CENTERS = {}
for _w in COMBO_WORDS:
    _r, _g, _b = COLORS[_w[0]]
    _circ, _edge = SHAPES[_w[1:]]
    T0_IMAGE_CENTERS[CONCEPT_LABELS[_w]] = [_r, _g, _b, _circ, _edge, 0.5, 0.5, 0.5]
T0_AUDIO_CENTERS = {k: list(v[1]) for k, v in TONES.items()}
