# -*- coding: utf-8 -*-
"""M1 认知核心自测（algorithm-design §6 验收口径）。

运行：cd /home/z/my-project && python src/cognition/selftest.py
覆盖：
  1) 单元：①注意力排序 ②衰减单调性+归档 ③巩固触发(hits≥3/簇≥5) ④门控行为
  2) 冒烟：3 轮同主题会话 → 召回首轮情景记忆且 conf 单调提升（记忆增强可量化）
  3) 性能：1000 条记忆暴力 kNN < 50ms/次（热缓存口径）
  4) 契约：Thought I2 结构 / 2001·2002 错误码 / M4 dispatch 集成 / JSONL 持久化
所有测试使用临时目录（tempfile），不污染 data/ 正式存储。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.cognition import CognitionError, run_cognition  # noqa: E402
from src.cognition import attention, memory, reasoning  # noqa: E402
from src.cognition._shared import embed, now_ts, tokenize  # noqa: E402

_TESTS = []
_TMP_ROOT = tempfile.mkdtemp(prefix="agi-cog-selftest-")


def test(fn):
    _TESTS.append(fn)
    return fn


def _fresh() -> str:
    """为当前测试切换一个干净的临时存储目录。"""
    d = tempfile.mkdtemp(dir=_TMP_ROOT)
    memory.configure(d)
    return d


def _mk_obs(text: str, idx: int) -> dict:
    """构造符合 I1 契约的文本观测（等价 M2 桩 perceive 输出）。"""
    return {"obs_id": f"obs-{idx:03d}", "modality": "text",
            "embedding": embed(text), "tokens": tokenize(text),
            "meta": {"dim": 64, "confidence_factor": 0.8,
                     "missing": ["image", "audio"], "salience_prior": 1.0,
                     "source": "selftest", "query": text}}


# ---------------------------------------------------------------- 1) 单元 --
@test
def attention_ranking_and_contract():
    """① 注意力排序：相关观测须排前；FocusSet 契约与 2001 错误码。"""
    _fresh()
    query = "蓝色三角形在哪里"
    obs = [
        {"obs_id": "obs-irr", "modality": "text",
         "tokens": ["今天", "天气", "晴朗"], "embedding": embed("今天天气晴朗"),
         "meta": {"salience_prior": 1.0}},
        {"obs_id": "obs-rel", "modality": "text",
         "tokens": ["蓝色", "色三", "三角", "角形"], "embedding": embed("蓝色三角形在左边"),
         "meta": {"salience_prior": 1.0}},
    ]
    F = attention.attend(query, obs, k=2)
    assert set(F) >= {"items", "weights", "scores", "k"}, F
    assert F["items"][0] == "obs-rel", f"相关观测应排首位: {F}"
    assert F["scores"] == sorted(F["scores"], reverse=True), F
    assert abs(sum(F["weights"]) - 1.0) < 1e-6, F["weights"]
    assert F["k"] == 2 and len(F["items"]) == 2
    # Top-K 截断
    F3 = attention.attend(query, obs + [obs[0]], k=1)
    assert F3["items"] == ["obs-rel"] and F3["k"] == 1, F3
    # 空输入 → 2001
    try:
        attention.attend(query, [])
        raise AssertionError("空 obs 应抛 2001")
    except CognitionError as e:
        assert e.code == 2001, e.code
    return f"top={F['items'][0]} scores={F['scores']}"


@test
def memory_save_recall_dedup():
    """B1/B2 写入、关键词检索排序与相似合并（合并相似记忆）。"""
    _fresh()
    r = memory.mem_save({"type": "episodic", "content": "蓝色三角形在左上角"})
    assert r["ok"] and r["mem_id"].startswith("mem-") and not r.get("merged"), r
    r2 = memory.mem_save({"type": "episodic", "content": "红色圆形在右下角"})
    hits = memory.mem_recall("红色圆形", k=2)
    assert hits and hits[0]["mem_id"] == r2["mem_id"], hits
    got = memory.mem_recall("蓝色三角形", k=2)
    assert any(m["mem_id"] == r["mem_id"] for m in got), got
    # 相似合并：同内容再写 → 合并到既有条目（hits+1）
    r3 = memory.mem_save({"type": "episodic", "content": "blue triangle test"})
    r4 = memory.mem_save({"type": "episodic", "content": "蓝色三角形在左上角"})
    assert r4.get("merged") is True and r4["mem_id"] == r["mem_id"], r4
    assert memory.stats()["episodic"] == 3, memory.stats()
    # 记忆巩固合并后 hits 递增（合并即强化）
    after = memory.mem_recall("蓝色三角形在左上角", k=1)
    assert int(after[0]["hits"]) >= 2, after[0]
    return "dedup merged hits ok"


@test
def decay_monotonic_and_archive():
    """② 衰减单调性：strength 随 t 单调不增；旧情景 <0.2 归档退出召回池。"""
    _fresh()
    r_old = memory.mem_save({"type": "episodic", "content": "很久以前的旧记忆",
                             "ts": now_ts() - 40 * 86400})   # 40 天前（τ=7d）
    memory.mem_save({"type": "episodic", "content": "刚刚的新记忆"})
    items = memory.mem_recall("旧记忆", k=5)
    m = next(m for m in items if m["mem_id"] == r_old["mem_id"])
    t = now_ts()
    s1 = memory.strength_at(m, t)
    s2 = memory.strength_at(m, t + 86400)
    s3 = memory.strength_at(m, t + 30 * 86400)
    assert s1 >= s2 >= s3 > 0, (s1, s2, s3)
    res = memory.decay()
    assert res["archived"] >= 1, res
    st = memory.stats()
    assert st["archived_rows"] >= 1 and r_old["mem_id"] not in memory.mem_ids("episodic"), st
    got = memory.mem_recall("很久以前的旧记忆", k=10)
    assert all(m["mem_id"] != r_old["mem_id"] for m in got), "归档后不应再被召回"
    assert isinstance(res["boosted"], int)
    return f"strength {s1:.4f}→{s2:.4f}→{s3:.4f}, archived={res['archived']}"


@test
def consolidate_hits_and_cluster():
    """③ 巩固触发：hits≥3 单条蒸馏；簇≥5（cos>0.75）合并为语义记忆。"""
    # ① hits≥3 路径
    _fresh()
    v = embed("巩固测试基准向量")
    r1 = memory.mem_save({"type": "episodic", "content": "高频情景记忆甲",
                          "embedding": v, "hits": 3})
    res = memory.consolidate()
    assert res["created"] >= 1 and res["archived"] >= 1, res
    sem = memory.mem_recall("高频情景", k=5, mem_type="semantic")
    assert sem and "巩固" in sem[0]["content"] and "高频情景记忆甲" in sem[0]["content"], sem
    assert r1["mem_id"] not in memory.mem_ids("episodic"), "源情景应退出召回池"
    # ② 簇≥5 路径（合成嵌入 pairwise cos=0.875：>0.75 可聚类、<0.95 不触发写入去重）
    _fresh()
    base = [1.0] * 7 + [0.0] * 57          # 7 个公共维 + 各自独占维
    for i in range(5):
        v_i = list(base)
        v_i[7 + i] = 1.0                   # 两两 cos = 7/8 = 0.875
        memory.mem_save({"type": "episodic", "content": f"聚类样本编号{i}",
                         "embedding": v_i, "hits": 0})
    res2 = memory.consolidate()
    assert res2["created"] >= 1 and res2["archived"] >= 5, res2
    sem2 = memory.mem_recall("聚类样本", k=5, mem_type="semantic")
    assert sem2 and "聚类样本" in sem2[0]["content"] and "n=5" in sem2[0]["content"], sem2
    assert len(memory.mem_ids("episodic")) == 0
    return f"hits路径 {res} / 簇路径 {res2}"


@test
def reason_thought_contract():
    """C1 reason → Thought 严格符合 I2（三步/字段/置信度域/可序列化）。"""
    _fresh()
    obs = [_mk_obs("蓝色三角形在左边", 1)]
    th = reasoning.reason(obs, "哪张图是蓝色三角形？", "sess-contract")
    for k in ("thought_id", "steps", "answer", "confidence"):
        assert k in th, k
    assert th["thought_id"].startswith("th-")
    assert [s["op"] for s in th["steps"]] == ["recall", "attend", "infer"], th["steps"]
    assert [s["step"] for s in th["steps"]] == [1, 2, 3]
    assert isinstance(th["answer"], str) and th["answer"].strip()
    assert isinstance(th["confidence"], float) and 0.0 <= th["confidence"] <= 1.0
    assert th["steps"][2]["rule"] and th["steps"][1]["focus_obs"]
    json.dumps(th, ensure_ascii=False)   # 可 JSON 序列化
    # 空观测 + 空记忆 → 模板兜底，不抛异常（低置信）
    th2 = reasoning.reason([], "随便问点什么", "sess-contract")
    assert th2["steps"][1]["focus_obs"] == [] and th2["confidence"] < 0.6, th2
    # 前向链接可触发链式规则（R001 结论含「定位」→ R011）
    chain = reasoning.reason(obs, "哪张图是蓝色三角形？", "sess-contract")["steps"][2]["chain"]
    assert "R001" in chain and "R011" in chain, chain
    return f"conf={th['confidence']} chain={chain}"


@test
def gate_behavior_conf():
    """④ 门控行为：conf<0.6 → clarify；≥0.6 → reply（M4 plan，I4 对齐）。"""
    from src.api.router import GATE_THRESHOLD, plan
    low = plan({"confidence": 0.3, "answer": "低置信结论"})
    assert low["action_type"] == "clarify" and low["gate"]["threshold"] == GATE_THRESHOLD, low
    high = plan({"confidence": 0.82, "answer": "高置信结论"})
    assert high["action_type"] == "reply" and high["payload"]["text"] == "高置信结论", high
    return f"threshold={GATE_THRESHOLD} clarify/reply ok"


# ---------------------------------------------------------------- 2) 冒烟 --
@test
def smoke_memory_augmented_conversation():
    """3 轮同主题会话：召回首轮情景记忆且 conf 单调提升（§6.2）。"""
    _fresh()
    q = "什么是蓝色三角形？"
    th1 = run_cognition([_mk_obs(q, 1)], q, "sess-smoke")
    ep_round1 = memory.mem_ids("episodic")      # 第 1 轮回写快照
    assert ep_round1, "第 1 轮应回写情景记忆"
    th2 = run_cognition([_mk_obs(q, 2)], q, "sess-smoke")
    th3 = run_cognition([_mk_obs(q, 3)], q, "sess-smoke")
    thoughts = [th1, th2, th3]
    c1, c2, c3 = (t["confidence"] for t in thoughts)
    assert c1 < c2 < c3, f"conf 应单调提升: {c1} {c2} {c3}"
    used2 = set(th2["steps"][0]["used_mem"])
    used3 = set(th3["steps"][0]["used_mem"])
    assert used2 & set(ep_round1) and used3 & set(ep_round1), (used2, used3, ep_round1)
    assert c3 >= 0.6, f"记忆增强后应过门控: {c3}"
    for t in thoughts:
        assert [s["op"] for s in t["steps"]] == ["recall", "attend", "infer"]
    return f"conf {c1}→{c2}→{c3}，第2/3轮召回首轮记忆 {ep_round1}"


# ---------------------------------------------------------------- 3) 性能 --
@test
def perf_brute_knn_1000():
    """性能基线：1000 条记忆暴力 kNN < 50ms/次（热缓存口径，§6.3）。"""
    _fresh()
    items = [{"type": "episodic",
              "content": f"样本编号{i} 颜色{('红', '蓝', '绿')[i % 3]}"
                         f"形状{('三角', '圆', '方')[i % 3]} 位置{i % 10}",
              "ts": now_ts()} for i in range(1000)]
    memory.bulk_save(items)
    assert memory.stats()["rows"] == 1000, memory.stats()
    memory.mem_recall("预热 颜色 形状", k=5)     # 热缓存
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        memory.mem_recall("蓝色 三角 形状 位置", k=5)
        times.append((time.perf_counter() - t0) * 1000.0)
    avg = sum(times) / len(times)
    assert avg < 50.0, f"千条 kNN 均值 {avg:.1f}ms 超基线"
    return f"1000条 kNN avg={avg:.1f}ms min={min(times):.1f}ms"


# ---------------------------------------------------------------- 4) 集成 --
@test
def dispatch_integration():
    """M4 全链路集成：dispatch → 感知(M2 桩)→ 认知(M1 真)→ 决策 → 输出。"""
    _fresh()
    from src.api.router import dispatch
    r = dispatch({"session_id": "selftest-s1", "mode": "standard",
                  "inputs": [{"modality": "text", "raw": "你好，请介绍一下你自己"}]})
    for k in ("output", "confidence", "trace", "latency_ms", "version", "error"):
        assert k in r, k
    assert r["error"] is None, r["error"]
    assert isinstance(r["output"], str) and r["output"].strip()
    assert isinstance(r["confidence"], (int, float))
    assert r["version"] == "1.0" and r["latency_ms"] < 5000 and len(r["trace"]) >= 5
    return f"conf={r['confidence']} latency={r['latency_ms']}ms"


@test
def persistence_roundtrip():
    """JSONL 持久化：落盘后新 store 实例同目录可召回（跨进程语义）。"""
    d = _fresh()
    memory.mem_save({"type": "semantic", "content": "巴黎是法国的首都"})
    memory.flush()
    store2 = memory.MemoryStore(d)
    got = store2.recall("法国的首都", k=3)
    assert got and "巴黎" in got[0]["content"], got
    return "roundtrip ok"


@test
def error_codes_2001_2002():
    """错误码：2001 注意力空输入；2002 记忆文件读写失败。"""
    _fresh()
    try:
        attention.attend("q", [])
        raise AssertionError("应抛 2001")
    except CognitionError as e:
        assert e.code == 2001
    blocker = os.path.join(_TMP_ROOT, "blocker.file")
    with open(blocker, "w", encoding="utf-8") as f:
        f.write("not-a-dir")
    memory.configure(blocker)   # 以普通文件充当目录 → 读写必败
    try:
        memory.mem_save({"type": "episodic", "content": "x"})
        raise AssertionError("应抛 2002")
    except CognitionError as e:
        assert e.code == 2002, e.code
    _fresh()                    # 恢复可用存储
    assert memory.mem_save({"type": "episodic", "content": "恢复测试"})["ok"]
    return "2001/2002 ok"


def main() -> int:
    print("=" * 64)
    print("M1 认知核心自测（src/cognition/selftest.py）")
    print("=" * 64)
    failed = []
    t0 = time.perf_counter()
    for fn in _TESTS:
        try:
            detail = fn() or ""
            print(f"  [PASS] {fn.__name__}{(' — ' + detail) if detail else ''}")
        except Exception as exc:  # noqa: BLE001
            failed.append((fn.__name__, f"{type(exc).__name__}: {exc}"))
            print(f"  [FAIL] {fn.__name__} — {type(exc).__name__}: {exc}")
    elapsed = (time.perf_counter() - t0) * 1000.0
    print("-" * 64)
    total, ok = len(_TESTS), len(_TESTS) - len(failed)
    print(f"结果：{ok}/{total} PASS，总耗时 {elapsed:.0f}ms"
          + ("" if not failed else f"；FAIL: {[n for n, _ in failed]}"))
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
