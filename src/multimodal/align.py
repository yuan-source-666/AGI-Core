# -*- coding: utf-8 -*-
"""设计文档命名兼容层：align ≡ aligner（multimodal-design §4.4 为 align.py；
任务书口径为 aligner.py，两者同实现，均可用）。"""
from .aligner import (align_eval, fit, fit_t2, load_idf, load_protos,  # noqa: F401
                      load_t2, retrieve)

__all__ = ["fit", "fit_t2", "align_eval", "retrieve", "load_protos", "load_idf", "load_t2"]
