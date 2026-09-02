# -*- coding: utf-8 -*-
"""A4 跨模态对齐（multimodal-design.md §2）：T1 原型统计 / T2 闭式校准 / 掩码余弦检索。

- T1（§2.3）：读 data/dataset/train.jsonl 统计 30 概念图像原型 + 3 语气原型 + 文本
  idf（df 口径与 M1 同式：idf = log((N+1)/(df+1))+1），落盘 prototypes.json；
  数据集缺席 → 返回 None，编码器走 T0 内置表（1005 语义，警告不阻塞）；
- T2：T1 评测 top-1<0.70 时启用的闭式岭回归校准（X=图像 8 维特征+偏置 →
  Y=配对 caption 的 A 区嵌入，λ=0.1，纯 Python 正规方程+高斯消元，无 numpy 依赖）；
- 检索（§2.2）：retrieve(query_text, gallery_obs, k) → 掩码余弦 top-k；
- 评测（§2.4）：align_eval() 输出概念解码一致率 / text→image top-1·top-3 /
  图文 5 选 1 / 语气解码一致率。
"""
from __future__ import annotations

import json
import math
import os

from .space import A_END, EMBED_DIM, l2_normalize, masked_cosine

__all__ = ["fit", "load_protos", "load_idf", "load_t2", "fit_t2",
           "retrieve", "align_eval", "TRAIN_PATH", "EVAL_PATH", "PROTO_PATH"]

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, os.pardir, os.pardir))
TRAIN_PATH = os.path.join(_ROOT, "data", "dataset", "train.jsonl")
EVAL_PATH = os.path.join(_ROOT, "data", "dataset", "eval.jsonl")
PROTO_PATH = os.path.join(_DIR, "prototypes.json")
T2_PATH = os.path.join(_DIR, "align_W.json")

_PROTOS_CACHE = None
_W_CACHE = None


def _read_jsonl(path):
    rows = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except (ValueError, TypeError):
                        pass
    return rows


def fit(save: bool = True, train_path: str = None):
    """T1 原型统计（§2.3）→ protos dict；数据集缺席 → None（T0 兜底）。"""
    from .text_encoder import tokenize
    rows = _read_jsonl(train_path or TRAIN_PATH)
    if not rows:
        return None
    img_acc, aud_acc, df, n_docs = {}, {}, {}, 0
    for r in rows:
        inp = r.get("input") or {}
        imf = inp.get("image_feat") or {}
        vals, label = imf.get("values"), imf.get("label")
        if isinstance(vals, list) and len(vals) == 8 and label:
            try:
                img_acc.setdefault(str(label), []).append([float(x) for x in vals])
            except (TypeError, ValueError):
                pass
        auf = inp.get("audio_feat") or {}
        avals, tone = auf.get("values"), auf.get("tone")
        if isinstance(avals, list) and len(avals) == 8 and tone:
            try:
                aud_acc.setdefault(str(tone), []).append([float(x) for x in avals])
            except (TypeError, ValueError):
                pass
        text = inp.get("text")
        if isinstance(text, str) and text.strip():
            n_docs += 1
            terms, _, _ = tokenize(text)
            for t in set(terms):
                df[t] = df.get(t, 0) + 1

    def _mean(acc):
        return {k: [round(sum(col) / len(vecs), 6) for col in zip(*vecs)]
                for k, vecs in acc.items() if vecs}

    protos = {"version": "1.0", "kind": "t1-prototypes", "seed": 42,
              "n_train_samples": len(rows), "n_train_docs": n_docs,
              "image_protos": _mean(img_acc), "audio_protos": _mean(aud_acc),
              "idf": {k: round(math.log((n_docs + 1) / (v + 1)) + 1.0, 6)
                      for k, v in sorted(df.items())}}
    if save:
        with open(PROTO_PATH, "w", encoding="utf-8") as f:
            json.dump(protos, f, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return protos


def load_protos(refresh: bool = False):
    """读 prototypes.json（T1）；缺席/损坏 → None（T0 兜底，1005 警告语义）。"""
    global _PROTOS_CACHE
    if _PROTOS_CACHE is not None and not refresh:
        return _PROTOS_CACHE
    p = None
    if os.path.isfile(PROTO_PATH):
        try:
            with open(PROTO_PATH, "r", encoding="utf-8") as f:
                p = json.load(f)
            if not (isinstance(p, dict) and "image_protos" in p and "idf" in p):
                p = None
        except (ValueError, OSError):
            p = None
    _PROTOS_CACHE = p
    return p


def load_idf():
    """文本 idf 表（T1）；缺席 → None（均匀权重）。"""
    p = load_protos()
    return (p or {}).get("idf")


def load_t2(refresh: bool = False):
    """读 align_W.json（T2 校准系数）；缺席 → None。"""
    global _W_CACHE
    if _W_CACHE is not None and not refresh:
        return _W_CACHE
    w = None
    if os.path.isfile(T2_PATH):
        try:
            with open(T2_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
            w = d.get("W") if isinstance(d, dict) else None
            if not (isinstance(w, list) and len(w) == A_END):
                w = None
        except (ValueError, OSError):
            w = None
    _W_CACHE = w
    return w


def retrieve(query_text, gallery_obs, k: int = 3):
    """§2.2 跨模态检索：query 文本 × gallery(Observation[]) → 掩码余弦 top-k。"""
    from .text_encoder import embed_text
    q = embed_text(query_text)
    if not q or not any(q):
        return []
    items = []
    for g in gallery_obs or []:
        if not isinstance(g, dict):
            continue
        emb = g.get("embedding")
        if isinstance(emb, list) and len(emb) == EMBED_DIM and any(emb):
            items.append((masked_cosine(q, emb, cross=True),
                          str(g.get("obs_id") or "obs-?"), g.get("modality")))
    items.sort(key=lambda t: (-t[0], t[1]))  # 分数降序，obs_id 升序保证确定性
    return [{"obs_id": oid, "modality": m, "score": round(s, 6)}
            for s, oid, m in items[:max(0, int(k))]]


def _gauss_solve(A, rhs_list):
    """解 A·x = b_k（同一 A，多右端）→ [x_k]；纯 Python 高斯消元（列主元）。"""
    n = len(A)
    m = len(rhs_list)
    aug = [A[i][:] + [rhs_list[k][i] for k in range(m)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            continue
        aug[col], aug[piv] = aug[piv], aug[col]
        p = aug[col][col]
        aug[col] = [x / p for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[col])]
    return [[aug[i][n + k] for i in range(n)] for k in range(m)]


def fit_t2(protos=None, lam: float = 0.1, save: bool = True):
    """T2 闭式校准（§2.3）：岭回归 X=[f8 归一,1] → Y=caption A 区嵌入（48 维）。

    λ 不施加于偏置位；输出 W（48×9）落盘 align_W.json，供 encode_image 启用。
    """
    from .text_encoder import encode_text
    rows = _read_jsonl(TRAIN_PATH)
    if not rows:
        return None
    idf = (protos or load_protos() or {}).get("idf")
    X, Y = [], []
    for r in rows:
        inp = r.get("input") or {}
        imf = inp.get("image_feat") or {}
        vals = imf.get("values")
        if not (isinstance(vals, list) and len(vals) == 8):
            continue
        try:
            v = l2_normalize([float(x) for x in vals])
        except (TypeError, ValueError):
            continue
        emb, _, _ = encode_text(inp.get("text") or "", idf)
        if not emb:
            continue
        X.append(v + [1.0])
        Y.append(emb[:A_END])
    if len(X) < 10:
        return None
    d = len(X[0])
    XtX = [[sum(X[k][i] * X[k][j] for k in range(len(X))) for j in range(d)]
           for i in range(d)]
    for i in range(1, d):  # 偏置位不正则
        XtX[i][i] += lam
    XtY = [[sum(X[k][i] * Y[k][o] for k in range(len(X))) for i in range(d)]
           for o in range(A_END)]  # 48×d
    W = _gauss_solve(XtX, XtY)  # 48 行 × 9 列：A[o] = Σ_j W[o][j]·x[j]
    if save:
        with open(T2_PATH, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0", "kind": "t2-ridge", "lambda": lam,
                       "n": len(X), "W": [[round(x, 8) for x in row] for row in W]},
                      f, ensure_ascii=False, separators=(",", ":"))
    return W


def align_eval(protos=None, W=None):
    """§2.4 对齐质量评测（train 概念解码一致率 / eval text→image 检索 / 图文 5 选 1
    / 语气解码一致率）。W 非 None → 图像 A 区走 T2 校准路径。"""
    from .text_encoder import encode_text
    from .image_encoder import encode_image
    from .audio_encoder import encode_audio
    metrics = {"n_train": 0, "n_eval": 0}

    train = _read_jsonl(TRAIN_PATH)
    if train:
        ok = n = 0
        for r in train:
            imf = (r.get("input") or {}).get("image_feat") or {}
            res = encode_image(imf, protos, W)
            if res:
                n += 1
                ok += (res["concept_hits"][0] == imf.get("label"))
        metrics["n_train"] = n
        metrics["concept_consistency"] = round(ok / n, 4) if n else 0.0

    ev = _read_jsonl(EVAL_PATH)
    if not ev:
        return metrics
    idf = (protos or {}).get("idf")
    gallery = []  # (id, embedding, label)
    texts = []    # (id, embedding, label)
    for r in ev:
        inp = r.get("input") or {}
        label = (inp.get("image_feat") or {}).get("label")
        res = encode_image(inp.get("image_feat") or {}, protos, W)
        if res:
            gallery.append((r.get("id"), res["embedding"], label))
        emb, _, _ = encode_text(inp.get("text") or "", idf)
        if emb:
            texts.append((r.get("id"), emb, label))

    # 双口径：retrieval_top1 = 配对样本 id 排第 1；retrieval_top1_label = top-1 命中
    # 同概念（语义正确——eval 含同概念多图，共享空间私有区被掩码后本就无法区分）。
    hit1 = hit1l = hit3 = n = 0
    for rid, q, qlabel in texts:
        sims = sorted(((masked_cosine(q, g_emb), gid, glabel) for gid, g_emb, glabel in gallery),
                      key=lambda t: (-t[0], str(t[1])))
        rank = [gid for _, gid, _ in sims].index(rid) + 1
        n += 1
        hit1 += (rank == 1)
        hit1l += (sims[0][2] == qlabel)
        hit3 += (rank <= 3)
    metrics["n_eval"] = n
    metrics["retrieval_top1"] = round(hit1 / n, 4) if n else 0.0
    metrics["retrieval_top1_label"] = round(hit1l / n, 4) if n else 0.0
    metrics["retrieval_top3"] = round(hit3 / n, 4) if n else 0.0

    ok5 = n5 = 0
    tmap = {rid: emb for rid, emb, _ in texts}
    for idx, r in enumerate(ev):
        inp = r.get("input") or {}
        res = encode_image(inp.get("image_feat") or {}, protos, W)
        rid = r.get("id")
        if not res or rid not in tmap:
            continue
        # 自身 + 4 个确定性轮转负例（§2.4 图文 5 选 1）
        cands = [rid] + [ev[(idx + j) % len(ev)]["id"] for j in range(4)]
        cands = [c for c in dict.fromkeys(cands) if c in tmap]
        if len(cands) < 2:
            continue
        best = sorted(cands, key=lambda c: (-masked_cosine(res["embedding"], tmap[c],
                                                           cross=True), str(c)))[0]
        n5 += 1
        ok5 += (best == rid)
    metrics["match5"] = round(ok5 / n5, 4) if n5 else 0.0

    okT = nT = 0
    for r in ev:
        auf = (r.get("input") or {}).get("audio_feat") or {}
        res = encode_audio(auf, protos)
        if res:
            nT += 1
            okT += (res["tone"] == auf.get("tone"))
    metrics["tone_decode"] = round(okT / nT, 4) if nT else 0.0
    return metrics
