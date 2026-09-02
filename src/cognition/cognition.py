# -*- coding: utf-8 -*-
"""M1 认知循环编排（algorithm-design §4.1 / §5.4）—— run_cognition 为 M4 唯一依赖入口（I2）。

认知循环（M1 负责段）：
  ② obs → 工作记忆写入（会话级，滑动淘汰）
  ③④ reason：recall（记忆检索）→ attend（注意力聚焦）→ 三通道 infer（仲裁）
  ⑦ 情景回写：{query, answer, conf} 落盘（对话即学习）
  ⑧ 巩固（consolidate）+ 衰减（decay）触发
⑤⑥（plan/render）归 M4，此处不越界（对齐 api-spec §0 模块边界）。
记忆子模块故障就地降级（R4：单点故障不阻塞整体演示），仅 obs 非法抛 2001。
"""
from __future__ import annotations

from ._shared import (CognitionError, embed, is_query_obs, now_ts, obs_text,
                      query_obs_from)
from . import memory as _memory
from . import reasoning as _reasoning

__all__ = ["run_cognition"]


def run_cognition(obs, query, session_id) -> dict:
    """I2 契约入口（签名冻结，api-spec §3.3）→ Thought。

    obs: list[Observation]（I1，来自 M2 perceive）
    query: str（会话当前查询）
    session_id: str（会话标识，工作记忆/情景回写的会话隔离键）
    """
    if not isinstance(obs, list):
        raise CognitionError(2001, "obs 非法：期望 list[Observation]")
    query = str(query or "")
    session_id = str(session_id)
    valid = [o for o in obs if isinstance(o, dict)]

    # ② 工作记忆写入（失败降级：跳过；查询观测为 M4 桥接元数据，不入记忆池）
    for o in valid:
        if is_query_obs(o):
            continue
        try:
            _memory.mem_save({"type": "working",
                              "content": obs_text(o) or f"<{o.get('modality', 'unknown')}>",
                              "embedding": o.get("embedding"),
                              "session": session_id, "ts": now_ts()})
        except Exception:
            pass  # 2002 就地降级（R4）

    # ③④ 推理（reason 内部：recall → attend → infer 三步 + 仲裁）
    thought = _reasoning.reason(valid, query, session_id)

    # ⑦ 情景回写（对话即学习：以「Q ⇒ A」整句落盘）
    #    嵌入优先取共享空间查询嵌入（M2 真实语义，M4 注入；语义空间一致），
    #    缺席退化为 M1 自建哈希嵌入。
    answer = str(thought.get("answer") or "")
    if query.strip() and answer.strip():
        qo = query_obs_from(valid)
        qemb = qo.get("embedding") if (qo is not None
                                       and isinstance(qo.get("embedding"), list)
                                       and qo["embedding"]) else None
        try:
            _memory.mem_save({"type": "episodic",
                              "content": f"{query} ⇒ {answer}",
                              "embedding": qemb or embed(f"{query} ⇒ {answer}"),
                              "session": session_id, "ts": now_ts()})
        except Exception:
            pass  # 2002 就地降级（R4）

    # ⑧ 巩固 + 衰减（同步执行，量小；失败降级不阻塞）
    try:
        _memory.consolidate()
    except Exception:
        pass
    try:
        _memory.decay()
    except Exception:
        pass
    return thought
