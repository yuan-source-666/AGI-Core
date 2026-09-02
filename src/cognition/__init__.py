# -*- coding: utf-8 -*-
"""M1 认知核心模块（Phase 3 阿瑟交付，替换骨架期 STUB）。

对外契约冻结不变（api-spec §3.3 / architecture §3 I2）：
- run_cognition(obs, query, session_id) -> Thought  # M4 经 router 调度
- CognitionError(code, msg)                        # 2xxx 错误码（§5.5）

文件结构（algorithm-design §5.6）：
    attention.py / memory.py / reasoning.py（=设计文档 reasoner.py，别名见
    reasoner.py）/ cognition.py（编排）/ rules.json（种子规则库）/ selftest.py
    selftest 运行：python src/cognition/selftest.py
"""
from __future__ import annotations

from ._shared import CognitionError
from .cognition import run_cognition

__all__ = ["run_cognition", "CognitionError"]
__version__ = "1.0"
