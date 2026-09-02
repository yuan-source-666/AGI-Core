# -*- coding: utf-8 -*-
"""AGI-Core M4 统一调度入口（骨架实现）。

对齐文档：
- docs/api-spec.md v1.0（本模块为其规范实现）
- docs/architecture.md v1.0 §3（I1-I6 契约 / 信封 / Response）

职责：
1. dispatch(request) -> response：四层顺序编排（感知→认知→决策→输出），永不抛异常；
2. Registry：路由注册 / 懒导入 / 桩降级（R4：模块可缺席，缺席走 __init__.py 桩）；
3. plan/render：I4 决策（置信度门控 0.6）/ I5 输出封装 的骨架实现。

边界：M4 是唯一允许 import M1/M2/M3 的模块（架构分层裁决），且仅经
ROUTE_TABLE 懒导入顶层函数符号，不触碰各模块内部实现。

演进位（Phase 4，见 api-spec §6）：会话状态持久化、认知循环情景回写、
HTTP POST 适配层（dispatch 签名不变）。
"""
from __future__ import annotations

import importlib
import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone

__all__ = ["dispatch", "plan", "render", "register", "registry",
           "make_envelope", "ROUTE_TABLE", "VERSION", "GATE_THRESHOLD",
           "session_store", "SessionStore", "SESSION_FILE"]

VERSION = "1.0"
GATE_THRESHOLD = 0.6          # I4 置信度门控（algorithm-design §3.2）
LATENCY_BUDGET_MS = 5000.0    # 单轮演示基线（architecture §6）
_CST = timezone(timedelta(hours=8))  # Asia/Shanghai

# ---- 4xxx 错误码（M4 自身，api-spec §4）----
ERR_BAD_REQUEST = 4001   # request 格式非法
ERR_UNKNOWN_ROUTE = 4002 # 未知路由/模块未注册
ERR_MODULE_CRASH = 4003  # 模块调用未捕获异常（兜底）
ERR_RENDER = 4004        # 信封/Response 封装失败

# ---- 路由表（api-spec §2.1）----
ROUTE_TABLE = {
    "perceive":     {"module": "src.multimodal",  "attr": "perceive",     "stage": 1, "desc": "感知 I1"},
    "cognition":    {"module": "src.cognition",   "attr": "run_cognition", "stage": 2, "desc": "认知 I2"},
    "load_dataset": {"module": "data.build_dataset", "attr": "load_dataset", "stage": 0, "desc": "数据 I6"},
    # plan / render 为 M4 内部函数，不走懒导入
}


def _now_iso() -> str:
    return datetime.now(_CST).isoformat(timespec="seconds")


def make_envelope(src: str, dst: str, msg_type: str,
                  payload=None, error=None) -> dict:
    """统一 JSON 信封（api-spec §1 / architecture §3.1）。"""
    return {
        "msg_id": str(uuid.uuid4()),
        "ts": _now_iso(),
        "src": src,
        "dst": dst,
        "type": msg_type,
        "version": VERSION,
        "payload": payload,
        "error": error,
    }


# ---------------------------------------------------------------- Registry --
class Registry:
    """路由注册器（api-spec §2.2）：同名默认拒绝，overwrite=True 方可覆盖。"""

    def __init__(self):
        self._routes = {}

    def register(self, name: str, fn, stage: int = 0, overwrite: bool = False) -> bool:
        if not callable(fn):
            return False
        if name in self._routes and not overwrite:
            return False
        self._routes[name] = {"fn": fn, "stage": stage}
        return True

    def get(self, name: str):
        entry = self._routes.get(name)
        return entry["fn"] if entry else None

    def list_modules(self) -> list:
        return [(n, ROUTE_TABLE.get(n, {}).get("module", "<external>"),
                 ROUTE_TABLE.get(n, {}).get("desc", "user")) for n in self._routes]


registry = Registry()


def register(name: str, fn, stage: int = 0, overwrite: bool = False) -> bool:
    """注册路由（测试替身/模块升级入口，api-spec §2.2）。"""
    return registry.register(name, fn, stage=stage, overwrite=overwrite)


# ------------------------------------------------- 会话状态持久化（§6 演进位）--
_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         os.pardir, os.pardir, "data"))
SESSION_FILE = os.path.join(_DATA_DIR, "session_state.jsonl")


class SessionStore:
    """会话状态持久化（session_id → {rounds, first_seen, last_seen, history[]}）。

    JSONL 追加式：进程内索引 + 批量回写（R4：读写失败降级为内存态，不阻塞演示）。
    """

    def __init__(self, path: str):
        self.path = str(path)
        self._sessions = {}   # sid -> dict
        self._dirty = set()
        self._load()

    def _load(self):
        try:
            if os.path.isfile(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(row, dict) and row.get("session_id"):
                            self._sessions[str(row["session_id"])] = row
        except OSError:
            pass  # 读失败 → 空态（R4 降级）

    def touch(self, session_id, meta: dict = None) -> int:
        """记一轮交互 → 返回累计轮数。meta 为可选的轮次快照（query/confidence）。"""
        sid = str(session_id)
        now = _now_iso()
        if sid not in self._sessions:
            self._sessions[sid] = {"session_id": sid, "rounds": 0,
                                   "first_seen": now, "last_seen": now,
                                   "last_confidence": None, "history": []}
        s = self._sessions[sid]
        s["rounds"] = int(s.get("rounds") or 0) + 1
        s["last_seen"] = now
        if isinstance(meta, dict):
            if meta.get("query") is not None:
                s["last_query"] = str(meta["query"])
            if meta.get("confidence") is not None:
                s["last_confidence"] = float(meta["confidence"])
            hist = s.setdefault("history", [])
            hist.append({"round": s["rounds"], "ts": now, **meta})
            if len(hist) > 50:      # 轮次历史有界，防止无限膨胀
                s["history"] = hist[-50:]
        self._dirty.add(sid)
        self._flush()
        return s["rounds"]

    def get(self, session_id):
        return self._sessions.get(str(session_id))

    def _flush(self):
        if not self._dirty:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            rows = [self._sessions[sid] for sid in sorted(self._sessions)]
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(tmp, self.path)
            self._dirty.clear()
        except OSError:
            pass  # 写失败 → 内存态继续（R4 降级）


session_store = SessionStore(SESSION_FILE)


# ---- 共享语义空间桥接（Phase 4 目标①）：M4 -> M2 懒取 embed_text/tokenize ----
_M2_EMBED = None


def _m2_query_embed(query: str):
    """经 M4 懒取 M2.embed_text/tokenize，产出共享空间查询嵌入 (qemb, qhash)。

    M1/M2 禁止互相 import（api-spec §0.2）：此处由 M4 编排，仅把 M2 查询嵌入
    包装成 I1 观测（meta.role="query"）注入 M1，不触碰 M1 内部实现。
    M2 缺席/调用失败 → 返回 (None, None)，M1 走自建哈希嵌入（向后兼容）。
    """
    global _M2_EMBED
    if _M2_EMBED is None:
        try:
            mod = importlib.import_module("src.multimodal")
            _M2_EMBED = (getattr(mod, "embed_text", None),
                         getattr(mod, "tokenize", None))
        except Exception:
            _M2_EMBED = (None, None)
    embed_text, tokenize = _M2_EMBED
    if not callable(embed_text):
        return None, None
    try:
        emb = embed_text(str(query))
        if not isinstance(emb, list) or not emb or not any(emb):
            return None, None
        toks = tokenize(str(query)) if callable(tokenize) else []
        return [round(float(x), 6) for x in emb], [str(t) for t in toks]
    except Exception:
        return None, None


def _bind(name: str, trace: list, envelopes: list):
    """绑定路由函数：显式注册优先 → ROUTE_TABLE 懒导入 → M4 内置最小桩。

    返回 (fn, is_stub)。缺席模块不阻塞（R4），trace 标记 stub。
    """
    fn = registry.get(name)
    if callable(fn):
        return fn, False

    route = ROUTE_TABLE.get(name)
    if route is None:
        raise RouteError(ERR_UNKNOWN_ROUTE, f"未知路由: {name}")
    try:
        mod = importlib.import_module(route["module"])
        fn = getattr(mod, route["attr"], None)
        if callable(fn):
            return fn, False
    except Exception as exc:  # 模块缺席/损坏 → 桩降级
        trace.append(f"{name}:模块导入失败({type(exc).__name__})→桩降级")

    fn = _FALLBACKS.get(name)
    if fn is None:
        raise RouteError(ERR_UNKNOWN_ROUTE, f"路由 {name} 无可用绑定")
    return fn, True


class RouteError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code, self.msg = code, msg


# M4 内置最小桩（双保险：模块 __init__.py 桩也失效时的最后降级，对齐 R4）
def _fallback_perceive(inputs):
    out = []
    for i, it in enumerate(inputs or []):
        if isinstance(it, dict) and it.get("modality") in ("text", "image", "audio"):
            text = it.get("raw") if isinstance(it.get("raw"), str) else ""
            out.append({"obs_id": f"obs-{i:03d}", "modality": it["modality"],
                        "embedding": [0.125] * 64, "tokens": [],
                        "meta": {"dim": 64, "source": "fallback", "missing": [],
                                 "confidence_factor": 0.5, "salience_prior": 0.5}})
            if text:
                out[-1]["tokens"] = [text[i:i + 2] for i in range(max(len(text) - 1, 0))] or [text]
    return out


def _fallback_cognition(obs, query, session_id):
    return {"thought_id": "th-fallback", "steps": [
        {"step": 1, "op": "recall", "used_mem": []},
        {"step": 2, "op": "attend", "focus_obs": [o.get("obs_id", "?") for o in obs[:2]]},
        {"step": 3, "op": "infer", "rule": "fallback"}],
        "answer": "（M4 兜底桩）认知模块不可用。", "confidence": 0.3}


_FALLBACKS = {"perceive": _fallback_perceive, "cognition": _fallback_cognition}


# ------------------------------------------------------------ I4 / I5 骨架 --
def plan(thought: dict) -> dict:
    """I4 决策骨架：置信度门控（>=0.6 → reply，<0.6 → clarify 追问）。"""
    conf = float(thought.get("confidence", 0.0) or 0.0)
    if conf >= GATE_THRESHOLD:
        return {"action_type": "reply",
                "payload": {"text": thought.get("answer", "")},
                "gate": {"confidence": conf, "threshold": GATE_THRESHOLD}}
    return {"action_type": "clarify",
            "payload": {"text": "（置信度不足，追问）能否补充更多信息，"
                                "例如具体颜色/形状或完整的问题描述？"},
            "gate": {"confidence": conf, "threshold": GATE_THRESHOLD}}


def render(action: dict, ctx: dict) -> dict:
    """I5 输出骨架：封装标准 Response（architecture §3.3 / api-spec §3.1）。"""
    return {
        "output": action.get("payload", {}).get("text", ""),
        "confidence": action.get("gate", {}).get("confidence", 0.0),
        "trace": ctx.get("trace", []),
        "latency_ms": round((time.perf_counter() - ctx["t0"]) * 1000, 1),
        "version": VERSION,
        "error": ctx.get("error"),
    }


def _error_response(code: int, msg: str, trace: list, t0: float) -> dict:
    """错误响应（dispatch 永不抛异常，api-spec §4 错误传播规则 3）。"""
    return {"output": "", "confidence": 0.0, "trace": trace,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "version": VERSION, "error": {"code": code, "msg": msg}}


# --------------------------------------------------------------- dispatch --
def dispatch(request) -> dict:
    """统一调度入口（api-spec §3.1/§3.2）：request -> response。

    编排：①校验 ②感知(I1) ③认知(I2) ④决策(I4) ⑤输出(I5)。
    每步产出经信封封装（收集于内部流水，供调试扩展）；
    模块异常按信封错误码降级，M4 自身异常兜底为 4003。
    """
    t0 = time.perf_counter()
    trace, envelopes = [], []

    try:
        # ① request 校验（4001）
        if not isinstance(request, dict):
            raise RouteError(ERR_BAD_REQUEST, "request 必须为 dict")
        session_id = request.get("session_id")
        inputs = request.get("inputs")
        if not session_id or not isinstance(session_id, (str, int)):
            raise RouteError(ERR_BAD_REQUEST, "缺少合法 session_id")
        if not isinstance(inputs, list) or not inputs:
            # 空 inputs 按感知层 1004 语义整体失败（api-spec §4 传播规则 2）
            return _error_response(1004, "inputs 为空（全模态缺失）", trace, t0)
        mode = request.get("mode", "standard")
        if mode not in ("standard", "fast"):
            mode = "standard"
        trace.append(f"请求(session={session_id}, mode={mode}, inputs={len(inputs)})")

        # ② 感知（I1）
        perceive_fn, stub_p = _bind("perceive", trace, envelopes)
        obs = perceive_fn(inputs)
        obs = obs if isinstance(obs, list) else []
        envelopes.append(make_envelope("perception", "cognition", "observation",
                                       payload={"count": len(obs)}))
        trace.append(f"感知({len(obs)} obs{'|stub' if stub_p else ''})")
        # query 提取规则（api-spec §3.2）：首个 text 模态的 raw
        query = ""
        for it in inputs:
            if isinstance(it, dict) and it.get("modality") == "text":
                raw = it.get("raw")
                if isinstance(raw, str) and raw.strip():
                    query = raw.strip()
                break

        # 共享语义空间桥接（目标①）：M4 懒取 M2.embed_text(query) 真实语义嵌入，
        # 包装成 meta.role="query" 的查询观测注入 M1（不违反 M1/M2 边界）。
        # M2 缺席 → qemb=None，M1 走自建哈希嵌入（向后兼容）。
        qemb, qhash = _m2_query_embed(query)
        if qemb is not None:
            obs = list(obs) + [{
                "obs_id": "q-query",
                "modality": "text",
                "embedding": qemb,
                "tokens": qhash,
                "meta": {"dim": 64, "role": "query", "source": "m4-inject",
                         "confidence_factor": 1.0, "missing": [],
                         "salience_prior": 1.0},
            }]
            trace.append("共享空间(M4→M2 embed_text 注入 M1)")

        # ③ 认知（I2）
        cognition_fn, stub_c = _bind("cognition", trace, envelopes)
        thought = cognition_fn(obs, query, str(session_id))
        thought = thought if isinstance(thought, dict) else {}
        envelopes.append(make_envelope("cognition", "decision", "thought",
                                       payload={"thought_id": thought.get("thought_id")}))
        trace.append(f"认知({len(thought.get('steps', []))} steps{'|stub' if stub_c else ''})")

        # ④ 决策（I4）
        action = plan(thought)
        trace.append(f"决策({action['action_type']}, conf={action['gate']['confidence']})")

        # ⑤ 输出（I5）
        response = render(action, {"trace": trace, "t0": t0})
        # 会话状态持久化（§6 演进位：rounds/history 落盘 data/session_state.jsonl）
        try:
            rounds = session_store.touch(
                session_id, {"query": query,
                             "confidence": action.get("gate", {}).get("confidence"),
                             "action": action.get("action_type"),
                             "shared_space": qemb is not None})
            trace.append(f"会话(session_state 保存，第 {rounds} 轮)")
        except Exception:
            trace.append("会话(session_state 保存失败→跳过)")
        response["trace"] = trace  # trace 含 render 步
        return response

    except Exception as exc:  # 兜底（api-spec §4 传播规则 3）
        code = getattr(exc, "code", None)
        if isinstance(code, int) and 1000 <= code <= 4999:
            # 模块信封式错误（1xxx/2xxx/3xxx）透传语义（api-spec §4）
            msg = str(getattr(exc, "msg", None) or exc)
            return _error_response(code, msg, trace, t0)
        return _error_response(
            ERR_MODULE_CRASH, f"{type(exc).__name__}: {exc}", trace, t0)


if __name__ == "__main__":  # 最小冒烟：python src/api/router.py
    import json
    import os
    import sys
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         os.pardir, os.pardir))
    if _root not in sys.path:  # 脚本直跑时补包上下文（包导入用法无需此处理）
        sys.path.insert(0, _root)
    print(json.dumps(dispatch({
        "session_id": "smoke-001", "mode": "standard",
        "inputs": [{"modality": "text", "raw": "哪张图是蓝色三角形？"}],
    }), ensure_ascii=False, indent=2))
