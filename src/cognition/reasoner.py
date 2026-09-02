# -*- coding: utf-8 -*-
"""C1 兼容别名：algorithm-design §5.6 以 reasoner.py 命名本模块，任务实现采用
reasoning.py；此文件保持两条命名路径等价（re-export，避免文档/实现命名分歧）。"""
from .reasoning import load_rules, reason  # noqa: F401

__all__ = ["reason", "load_rules"]
