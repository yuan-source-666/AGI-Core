# -*- coding: utf-8 -*-
"""任务侧命名兼容层：vision_encoder ≡ image_encoder（multimodal-design §4.4 为
image_encoder.py；任务书口径为 vision_encoder.py，两者同实现，均可用）。"""
from .image_encoder import encode_image, parse_feat, MATCH_DIMS, _match_top2  # noqa: F401

__all__ = ["encode_image", "parse_feat", "MATCH_DIMS"]
