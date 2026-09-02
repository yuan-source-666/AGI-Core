# -*- coding: utf-8 -*-
"""M1 记忆系统（B1~B4）—— 三层记忆 / JSONL 存储 / 遗忘曲线衰减 / 情景→语义巩固。

对齐：docs/algorithm-design.md v1.0 §2（设计）/§5.2（接口）；
     docs/api-spec.md §3.3 I3（MemoryItem：mem_id/type/content/embedding/ts/hits，
     附加字段 strength/consolidated/links/tokens/session 不破坏契约）。

实现要点（CPU 简化路线，纯标准库，单线程）：
- working：会话级内存 deque（容量 16，滑动淘汰；同会话高相似写入即合并）；
- episodic/semantic：data/memory.jsonl 追加写 + flush 批量回写（compaction）；
- 检索：暴力 kNN 四因子分 s = λ1·cos + λ2·IDF 重叠 + λ3·新近度 + λ4·命中数；
- 衰减：strength(m,t) = m.strength·exp(−(t−m.ts)/τ_type)；情景 <0.2 归档（不删除）；
- 巩固：hits≥3 或 簇内 cos>0.75 且簇≥5 → 蒸馏语义记忆（合并相似记忆），
  源情景标记 consolidated=true 退出召回池（保留在文件中）；
- 相似合并（写入路径）：与既有同类型记忆 cos≥0.95 → 合并（hits+1、strength+0.3）；
- 容量：主文件行数 >1000 按 strength 升序淘汰入归档（对齐 R5）。
"""
from __future__ import annotations

import atexit
import json
import math
import os
from collections import Counter, deque
from hashlib import md5

from ._shared import (CognitionError, cosine, embed, mean_vector, now_ts,
                      tokenize)

# ---- 常量（algorithm-design §2）----
WORKING_CAP = 16          # 工作记忆容量 W
TOTAL_CAP = 1000          # 主文件总容量（R5）
TAU = {"working": 600.0, "episodic": 7 * 86400.0, "semantic": 365 * 86400.0}
S0 = 1.0                  # 初始强度
BOOST_STEP = 0.3          # 召回强化步长
STRENGTH_CAP = 2.0        # 强度上限
DECAY_THETA = 0.2         # 情景归档阈值
CONSOLIDATE_HITS = 3      # 巩固触发①：hits ≥ 3
CONSOLIDATE_CLUSTER = 5   # 巩固触发②：同簇情景数 ≥ 5
CONSOLIDATE_COS = 0.75    # 簇内相似度阈值
DEDUP_COS = 0.95          # 写入合并阈值
L1, L2, L3, L4 = 0.4, 0.3, 0.2, 0.1  # 检索四因子权重

_TYPES = ("working", "episodic", "semantic")


def strength_at(item: dict, now: float = None) -> float:
    """记忆强度：strength·exp(−Δt/τ_type)（类 Ebbinghaus，Δt 非负）。"""
    now = now_ts() if now is None else float(now)
    ts = float(item.get("ts") or now)
    tau = TAU.get(item.get("type"), TAU["episodic"])
    s0 = float(item.get("strength") or S0)
    return s0 * math.exp(-max(0.0, now - ts) / tau)


class MemoryStore:
    """JSONL 记忆存储（一个 data_dir 一个实例；working 为进程内存态）。"""

    def __init__(self, data_dir: str):
        self.data_dir = str(data_dir)
        self.main_path = os.path.join(self.data_dir, "memory.jsonl")
        self.archive_path = os.path.join(self.data_dir, "memory_archive.jsonl")
        self._working = {}      # session_id -> deque[dict]
        self._cache = None      # list[{"item", "tokset"}]（文件行缓存）
        self._df = Counter()    # token -> 文档频（文件层）
        self._dirty = set()     # 待回写 mem_id（boost/consolidate 标记）
        self._boosted = 0       # 自上次 decay 以来的召回强化次数
        self._seq = 0           # id 序号（防同内容同刻碰撞）

    # ------------------------------------------------------------- 底层 IO --
    def _load(self):
        if self._cache is not None:
            return
        self._cache = []
        try:
            if os.path.isfile(self.main_path):
                with open(self.main_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue  # 容错：坏行跳过
                        if isinstance(row, dict) and row.get("mem_id"):
                            self._index(row)
        except OSError as exc:
            raise CognitionError(2002, f"记忆文件读失败: {exc}") from exc

    def _index(self, row: dict):
        toks = row.get("tokens")
        if not isinstance(toks, list) or not toks:
            toks = tokenize(row.get("content", ""))
        self._cache.append({"item": row, "tokset": set(map(str, toks))})
        self._df.update(map(str, toks))

    def _append(self, row: dict):
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(self.main_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise CognitionError(2002, f"记忆文件写失败: {exc}") from exc

    def _append_archive(self, item: dict):
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(self.archive_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise CognitionError(2002, f"归档文件写失败: {exc}") from exc

    def flush(self):
        """compaction：缓存全量回写（含 boost/consolidated 标记），原子替换。"""
        self._load()
        rows = [rec["item"] for rec in self._cache]
        tmp = self.main_path + ".tmp"
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(tmp, self.main_path)
            self._dirty.clear()
        except OSError as exc:
            raise CognitionError(2002, f"记忆文件回写失败: {exc}") from exc

    # ------------------------------------------------------------- B1 写入 --
    def _new_id(self, kind: str, content: str, ts: float) -> str:
        self._seq += 1
        raw = f"{kind}|{content}|{ts:.6f}|{self._seq}"
        return "mem-" + md5(raw.encode("utf-8")).hexdigest()[:8]

    @staticmethod
    def _build_row(item: dict) -> dict:
        mtype = item.get("type")
        if mtype not in _TYPES:
            raise CognitionError(
                2002, f"mem_save type 非法: {mtype!r}（须 working|episodic|semantic）")
        content = str(item.get("content") or "")
        if not content.strip():
            raise CognitionError(2002, "mem_save content 为空")
        ts = float(item.get("ts") or now_ts())
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            embedding = embed(content)
        tokens = item.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            tokens = tokenize(content)
        return {"mem_id": "", "type": mtype, "content": content,
                "embedding": [round(float(x), 6) for x in embedding],
                "ts": ts, "hits": int(item.get("hits") or 0),
                "strength": float(item.get("strength") or S0),
                "consolidated": bool(item.get("consolidated")),
                "links": [str(x) for x in (item.get("links") or [])],
                "tokens": [str(t) for t in tokens],
                "session": item.get("session")}

    def _save_working(self, row: dict) -> dict:
        session = str(row.get("session") or "*")
        dq = self._working.setdefault(session, deque(maxlen=WORKING_CAP))
        for m in dq:  # 同会话高相似 → 合并（滑动淘汰由 deque maxlen 保证）
            if cosine(row["embedding"], m.get("embedding") or []) >= DEDUP_COS:
                m["hits"] = int(m.get("hits") or 0) + 1
                m["strength"] = min(float(m.get("strength") or S0) + BOOST_STEP,
                                    STRENGTH_CAP)
                m["ts"] = max(float(m.get("ts") or 0.0), float(row["ts"]))
                return {"mem_id": m["mem_id"], "ok": True, "merged": True}
        row["mem_id"] = self._new_id("working", row["content"], row["ts"])
        dq.append(row)
        return {"mem_id": row["mem_id"], "ok": True, "merged": False}

    def save(self, item: dict, dedup: bool = True) -> dict:
        """B1 mem_save：写入（type ∈ working|episodic|semantic）。"""
        if not isinstance(item, dict):
            raise CognitionError(2002, "mem_save item 非法：期望 dict(MemoryItem)")
        row = self._build_row(item)
        if row["type"] == "working":
            return self._save_working(row)
        self._load()
        if dedup:  # 相似记忆合并：cos ≥ 0.95 → 强化既有条目
            for rec in self._cache:
                m = rec["item"]
                if (m.get("type") == row["type"]
                        and cosine(row["embedding"], m.get("embedding") or []) >= DEDUP_COS):
                    m["hits"] = int(m.get("hits") or 0) + 1
                    m["strength"] = min(float(m.get("strength") or S0) + BOOST_STEP,
                                        STRENGTH_CAP)
                    m["ts"] = max(float(m.get("ts") or 0.0), float(row["ts"]))
                    self._dirty.add(str(m.get("mem_id")))
                    return {"mem_id": str(m.get("mem_id")), "ok": True, "merged": True}
        row["mem_id"] = self._new_id(row["type"], row["content"], row["ts"])
        self._append(row)
        self._index(row)
        return {"mem_id": row["mem_id"], "ok": True, "merged": False}

    def bulk_save(self, items) -> int:
        """批量写入（跳过相似合并；自测性能基线/数据灌入用）。"""
        self._load()
        rows = [self._build_row(it) for it in (items or []) if isinstance(it, dict)]
        if not rows:
            return 0
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(self.main_path, "a", encoding="utf-8") as f:
                for row in rows:
                    row["mem_id"] = self._new_id(row["type"], row["content"], row["ts"])
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise CognitionError(2002, f"记忆文件批量写失败: {exc}") from exc
        for row in rows:
            self._index(row)
        return len(rows)

    # ------------------------------------------------------------- B2 检索 --
    def _pool(self, mem_type=None):
        """召回池：文件层（排除已巩固条目）+ working（内存层）。"""
        self._load()
        for rec in self._cache:
            m = rec["item"]
            if m.get("consolidated"):
                continue  # 已巩固情景退出召回池（保留在文件）
            if mem_type is not None and m.get("type") != mem_type:
                continue
            yield rec
        if mem_type is None or mem_type == "working":
            for dq in self._working.values():
                for m in dq:
                    if not m.get("consolidated"):
                        yield {"item": m,
                               "tokset": set(m.get("tokens") or
                                             tokenize(m.get("content", "")))}

    def _overlap(self, qset, tset, idf) -> float:
        if not qset and not tset:
            return 0.0
        inter = qset & tset
        union = qset | tset
        if not union:
            return 0.0
        num = sum(idf(w) for w in inter)
        den = sum(idf(w) for w in union)
        return num / den if den > 0 else 0.0

    def recall(self, query, k: int = 5, mem_type=None,
               qemb=None, qset=None) -> list:
        """B2 mem_recall：暴力 kNN 四因子混合分（返回 MemoryItem[]，附 score）。

        qemb/qset 可显式注入共享空间查询嵌入（Phase 4：M4 桥接 M2 语义嵌入），
        缺席时内部退化 M1 自建哈希嵌入（自测/直连调用无感知）。
        """
        try:
            k = max(1, int(k))
        except (TypeError, ValueError):
            k = 5
        self._load()
        query = str(query or "")
        if qemb is None:
            qemb = embed(query)
        if qset is None:
            qset = set(tokenize(query))
        now = now_ts()
        N = max(1, len(self._cache))
        idf_cache = {}

        def idf(w):
            v = idf_cache.get(w)
            if v is None:
                v = math.log((N + 1) / (self._df.get(w, 0) + 1)) + 1.0
                idf_cache[w] = v
            return v

        scored = []
        for rec in self._pool(mem_type):
            m = rec["item"]
            recency = math.exp(
                -max(0.0, now - float(m.get("ts") or now))
                / TAU.get(m.get("type"), TAU["episodic"]))
            s = (L1 * cosine(qemb, m.get("embedding") or [])
                 + L2 * self._overlap(qset, rec["tokset"], idf)
                 + L3 * recency
                 + L4 * min(int(m.get("hits") or 0), 5) / 5.0)
            scored.append((round(s, 6), str(m.get("mem_id")), m))
        scored.sort(key=lambda t: (-t[0], t[1]))
        top = scored[:k]
        out = []
        for s, mid, m in top:  # 召回计数反哺重要性（越用越牢）
            m["hits"] = int(m.get("hits") or 0) + 1
            m["strength"] = min(float(m.get("strength") or S0) + BOOST_STEP,
                                STRENGTH_CAP)
            if m.get("type") in ("episodic", "semantic"):
                self._dirty.add(mid)
            self._boosted += 1
            out.append({**m, "score": round(s, 4)})
        return out

    # ------------------------------------------------------------- B3 巩固 --
    def consolidate(self, now=None) -> dict:
        """B3 consolidate：情景→语义蒸馏（hits≥3 或 簇≥5 且簇内 cos>0.75）。"""
        now = now_ts() if now is None else float(now)
        self._load()
        epis = [rec for rec in self._cache
                if rec["item"].get("type") == "episodic"
                and not rec["item"].get("consolidated")]
        if not epis:
            return {"created": 0, "archived": 0}
        epis.sort(key=lambda rec: (-strength_at(rec["item"], now),
                                   str(rec["item"].get("mem_id"))))
        clusters = []  # 贪心种子聚类（按强度排序，与簇首相似即入簇）
        for rec in epis:
            emb = rec["item"].get("embedding") or []
            placed = False
            for cl in clusters:
                if cosine(emb, cl[0]["item"].get("embedding") or []) >= CONSOLIDATE_COS:
                    cl.append(rec)
                    placed = True
                    break
            if not placed:
                clusters.append([rec])
        created = archived = 0
        for cl in clusters:
            max_hits = max(int(r["item"].get("hits") or 0) for r in cl)
            if len(cl) < CONSOLIDATE_CLUSTER and max_hits < CONSOLIDATE_HITS:
                continue  # 未达触发条件
            center = cl[0]["item"]
            freq = Counter()
            for r in cl:
                freq.update(r["tokset"])
            kws = [w for w, _ in freq.most_common(5)]
            content = (f"[巩固|n={len(cl)}|hits≥{max_hits}] "
                       f"{str(center.get('content', ''))[:60]}"
                       f"（关键词：{'、'.join(kws)}）")
            emb = mean_vector([r["item"].get("embedding") or [] for r in cl])
            dup = any(cosine(emb, rec["item"].get("embedding") or []) >= 0.9
                      for rec in self._cache
                      if rec["item"].get("type") == "semantic")
            if not dup:
                row = {"mem_id": self._new_id("semantic", content, now),
                       "type": "semantic", "content": content,
                       "embedding": [round(x, 6) for x in emb], "ts": now,
                       "hits": sum(int(r["item"].get("hits") or 0) for r in cl),
                       "strength": 1.5, "consolidated": False,
                       "links": [str(r["item"].get("mem_id")) for r in cl],
                       "tokens": sorted(freq.keys()), "session": None}
                self._append(row)
                self._index(row)
                created += 1
            for r in cl:  # 源情景退出召回池（保留文件，可追溯）
                r["item"]["consolidated"] = True
                self._dirty.add(str(r["item"].get("mem_id")))
                archived += 1
        self.flush()
        return {"created": created, "archived": archived}

    # ------------------------------------------------------------- B4 衰减 --
    def decay(self, now=None) -> dict:
        """B4 decay：强度衰减 + 情景归档 + 容量淘汰 + 脏数据回写。"""
        now = now_ts() if now is None else float(now)
        self._load()
        archived = 0
        for rec in list(self._cache):
            m = rec["item"]
            if (m.get("type") == "episodic" and not m.get("consolidated")
                    and strength_at(m, now) < DECAY_THETA):
                self._archive(rec)
                archived += 1
        if len(self._cache) > TOTAL_CAP:  # 容量控制（working 不占文件额度）
            excess = len(self._cache) - TOTAL_CAP
            ordered = sorted(self._cache,
                             key=lambda rec: (strength_at(rec["item"], now),
                                              str(rec["item"].get("mem_id"))))
            for rec in ordered[:excess]:
                self._archive(rec)
                archived += 1
        boosted = self._boosted
        self._boosted = 0
        self.flush()
        return {"archived": archived, "boosted": boosted}

    def _archive(self, rec):
        self._append_archive(rec["item"])
        if rec in self._cache:
            self._cache.remove(rec)
            self._df.subtract(rec["tokset"])

    # ------------------------------------------------------------ 观测/统计 --
    def get_working(self, session="*") -> list:
        dq = self._working.get(str(session))
        return [dict(m) for m in dq] if dq else []

    def mem_ids(self, mem_type=None) -> list:
        return [str(rec["item"].get("mem_id")) for rec in self._pool(mem_type)]

    def stats(self) -> dict:
        self._load()
        c = {"working": 0, "episodic": 0, "semantic": 0, "consolidated": 0,
             "rows": len(self._cache), "archived_rows": 0}
        for dq in self._working.values():
            c["working"] += len(dq)
        for rec in self._cache:
            t = rec["item"].get("type")
            if t in ("episodic", "semantic"):
                c[t] += 1
            if rec["item"].get("consolidated"):
                c["consolidated"] += 1
        try:
            if os.path.isfile(self.archive_path):
                with open(self.archive_path, "r", encoding="utf-8") as f:
                    c["archived_rows"] = sum(1 for line in f if line.strip())
        except OSError:
            pass
        return c


# ------------------------------------------------------- 模块级默认接口 B1~B4 --
_default_store = None


def _default_data_dir() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        os.pardir, os.pardir))
    return os.path.join(root, "data")


def _store() -> MemoryStore:
    global _default_store
    if _default_store is None:
        _default_store = MemoryStore(_default_data_dir())
    return _default_store


def configure(data_dir=None) -> MemoryStore:
    """切换默认存储目录（自测隔离/多数据源用）；旧 store 尽力落盘。"""
    global _default_store
    if _default_store is not None:
        try:
            _default_store.flush()
        except Exception:
            pass
    _default_store = MemoryStore(str(data_dir) if data_dir else _default_data_dir())
    return _default_store


def mem_save(item: dict) -> dict:
    """B1：{mem_id, ok: true[, merged]}。"""
    return _store().save(item)


def mem_recall(query, k: int = 5, mem_type=None, qemb=None, qset=None) -> list:
    """B2：MemoryItem[]（附检索分 score），top-k 降序。

    qemb/qset：共享空间查询嵌入（可选，Phase 4 M4 桥接 M2 语义嵌入）。
    """
    return _store().recall(query, k, mem_type, qemb=qemb, qset=qset)


def bulk_save(items) -> int:
    """批量写入（跳过去重，性能压测/初始化用）→ 写入条数。"""
    return _store().bulk_save(items)


def consolidate(now=None) -> dict:
    """B3：{"created": n, "archived": n}。"""
    return _store().consolidate(now)


def decay(now=None) -> dict:
    """B4：{"archived": n, "boosted": n}。"""
    return _store().decay(now)


def get_working(session="*") -> list:
    return _store().get_working(session)


def mem_ids(mem_type=None) -> list:
    return _store().mem_ids(mem_type)


def stats() -> dict:
    return _store().stats()


def flush():
    _store().flush()


@atexit.register
def _atexit_flush():
    try:
        _store().flush()
    except Exception:
        pass
