# -*- coding: utf-8 -*-
"""M2 自测脚本（multimodal-design.md §6.3）。

运行：cd /home/z/my-project && python src/multimodal/selftest.py

断言（§6.3 口径）：
①I1 契约（字段/64 维/L2=1/μ/missing/salience）②分离度（"蓝色三角" vs 蓝三角图 >0.5，
vs 红圆图 <0.2）③T1 概念解码一致率 ≥0.95 ④eval text→image 检索 top-1 ≥0.70
（不达标 → 自动 T2 校准复测；仍不达标 → 降级 top-3 口径并在 README 记录）⑤图文 5 选 1
≥0.70 ⑥语气解码 ≥0.90 ⑦缺失容错（audio-only/非法模态/坏特征/1004）⑧确定性（同输入
两次运行逐位一致 + 两次 fit 落盘逐字节一致）⑨性能（三模态 perceive <100ms）⑩M4
dispatch 集成冒烟。exit code 0=全过 / 1=有失败。
"""
from __future__ import annotations

import json
import os
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.multimodal import (EMBED_DIM, PerceptionError, align_eval, encode_image,  # noqa: E402
                            embed_text, fit, fit_t2, load_t2, perceive, retrieve,
                            similarity)
from src.multimodal import concepts as C
from src.multimodal.aligner import PROTO_PATH, T2_PATH
from src.multimodal.space import masked_cosine

_CHECKS = []


def _check(name, cond, detail=""):
    ok = bool(cond)
    _CHECKS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def _img_feat(label):
    return {"values": list(C.T0_IMAGE_CENTERS[label]), "label": label}


def _audio_feat(tone):
    return {"values": list(C.T0_AUDIO_CENTERS[tone]), "label": C.TONES[tone][0], "tone": tone}


_INPUTS3 = [
    {"modality": "text", "raw": "哪张图是蓝色三角形？"},
    {"modality": "image", "raw": _img_feat("蓝-三角")},
    {"modality": "audio", "raw": _audio_feat("question")},
]


def main() -> int:
    print("=" * 62)
    print("M2 多模态模块自测（multimodal-design.md §6.3）")
    print("=" * 62)

    # T2 决策须基于本次评测 → 清理旧校准文件（行为可复现）
    if os.path.isfile(T2_PATH):
        os.remove(T2_PATH)

    # ---- ① T1 原型统计 ----
    protos = fit(save=True)
    _check("T1 原型统计：30 概念 + 3 语气 + idf 落盘",
           bool(protos) and len(protos.get("image_protos") or {}) == 30
           and len(protos.get("audio_protos") or {}) == 3 and bool(protos.get("idf")),
           f"train 样本={protos.get('n_train_samples')}，文本 doc={protos.get('n_train_docs')}")

    # ---- ② I1 契约 ----
    obs = perceive(_INPUTS3)
    _check("I1：三模态 → 3 条 Observation", len(obs) == 3, f"n={len(obs)}")
    ok_contract = all(
        isinstance(o.get("obs_id"), str) and o.get("modality") in ("text", "image", "audio")
        and isinstance(o.get("embedding"), list) and len(o["embedding"]) == EMBED_DIM
        and abs(sum(x * x for x in o["embedding"]) - 1.0) < 1e-4
        and isinstance(o.get("tokens"), list) and all(isinstance(t, str) for t in o["tokens"])
        and isinstance(o.get("meta"), dict) and o["meta"].get("dim") == 64
        and o["meta"].get("confidence_factor") == 1.0 and o["meta"].get("missing") == []
        and o["meta"].get("salience_prior") in (1.0, 0.9, 0.85)
        for o in obs)
    _check("I1：字段/dim=64/L2=1/μ=1.0/missing=[]", ok_contract)
    o_text = next((o for o in obs if o["modality"] == "text"), {})
    o_img = next((o for o in obs if o["modality"] == "image"), {})
    o_aud = next((o for o in obs if o["modality"] == "audio"), {})
    _check("I1：meta 扩展（query/concept_hits/intent/tone）",
           o_text.get("meta", {}).get("query") == "哪张图是蓝色三角形？"
           and o_text["meta"].get("intent") == "retrieve"  # 触发词"哪张"
           and (o_img.get("meta", {}).get("concept_hits") or [""])[0] == "蓝-三角"
           and o_aud.get("meta", {}).get("tone") == "question"
           and o_aud.get("meta", {}).get("intent") == "qa",
           f"text.intent={o_text.get('meta', {}).get('intent')}，"
           f"img.hits={o_img.get('meta', {}).get('concept_hits')}")

    # ---- ③ 分离度（§6.3-②）----
    q = embed_text("蓝色三角")
    e_bt = encode_image(_img_feat("蓝-三角"), protos)
    e_hy = encode_image(_img_feat("红-圆"), protos)
    s_bt = similarity(q, e_bt["embedding"])
    s_hy = similarity(q, e_hy["embedding"])
    _check("分离度：sim_cross('蓝色三角', 蓝三角图) > 0.5", s_bt > 0.5, f"sim={s_bt}")
    _check("分离度：sim_cross('蓝色三角', 红圆图) < 0.2", s_hy < 0.2, f"sim={s_hy}")

    # ---- ④ 对齐评测（T1 → 语义 top-1<0.70 时自动 T2 → 仍不达标降级 top-3 口径）----
    # 检索主口径 = retrieval_top1_label（top-1 命中同概念；eval 含同概念多图，
    # 掩码余弦下样本级 id 配对在同概念内本质并列，作参考指标）。
    metrics = align_eval(protos=protos)
    route = "T1"
    if metrics.get("retrieval_top1_label", 0) < 0.70:
        W = fit_t2(protos=protos)
        if W:
            route = "T2"
            metrics = align_eval(protos=protos, W=load_t2(refresh=True))
    _check(f"概念解码一致率 ≥ 0.95（{route} 路径）",
           metrics.get("concept_consistency", 0) >= 0.95,
           f"consistency={metrics.get('concept_consistency')}")
    top1l = metrics.get("retrieval_top1_label", 0)
    top3 = metrics.get("retrieval_top3", 0)
    if top1l >= 0.70:
        _check(f"eval text→image 语义检索 top-1 ≥ 0.70（{route}）", True,
               f"top1_label={top1l}，top3={top3}，"
               f"top1_id={metrics.get('retrieval_top1')}（参考）")
    else:
        _check(f"eval 检索 top-1<0.70（{route}）→ 降级口径 top-3 ≥ 0.85",
               top3 >= 0.85,
               f"top1_label={top1l}，top3={top3}（降级，README 记录）")
    _check("图文 5 选 1 匹配 ≥ 0.70", metrics.get("match5", 0) >= 0.70,
           f"match5={metrics.get('match5')}")
    _check("语气解码一致率 ≥ 0.90", metrics.get("tone_decode", 0) >= 0.90,
           f"tone_decode={metrics.get('tone_decode')}")

    # ---- ⑤ 缺失容错（§6.3-⑤）----
    obs_a = perceive([{"modality": "audio", "raw": _audio_feat("command")}])
    _check("容错：audio-only 不崩溃 + missing/μ 正确",
           len(obs_a) == 1 and obs_a[0]["meta"]["missing"] == ["text", "image"]
           and abs(obs_a[0]["meta"]["confidence_factor"] - 0.50) < 1e-9,
           f"missing={obs_a[0]['meta']['missing'] if obs_a else '?'}")
    obs_bad = perceive([{"modality": "video", "raw": "x"},
                        {"modality": "text", "raw": "红圆"},
                        {"modality": "image", "raw": {"values": [0.1, 0.2, 0.3]}}])
    _check("容错：非法模态/坏特征丢弃继续（1001/1002）",
           len(obs_bad) == 1 and obs_bad[0]["modality"] == "text",
           f"存活 obs={[o['modality'] for o in obs_bad]}")
    raised = False
    try:
        perceive([])
    except PerceptionError as e:
        raised = (e.code == 1004)
    _check("容错：空 inputs → PerceptionError(1004)", raised)

    # ---- ⑥ 确定性（§6.3-⑥）----
    a1 = perceive(_INPUTS3)
    b1 = perceive(_INPUTS3)
    same_emb = all(a["embedding"] == b["embedding"] and a["obs_id"] == b["obs_id"]
                   for a, b in zip(a1, b1))
    before = open(PROTO_PATH, "r", encoding="utf-8").read() if os.path.isfile(PROTO_PATH) else ""
    fit(save=True)
    after = open(PROTO_PATH, "r", encoding="utf-8").read() if os.path.isfile(PROTO_PATH) else ""
    _check("确定性：同输入两次运行嵌入逐位一致", len(a1) == len(b1) == 3 and same_emb)
    _check("确定性：两次 fit 落盘逐字节一致", before != "" and before == after)

    # ---- ⑦ 检索接口 + T0 兜底 ----
    gal = perceive([{"modality": "image", "raw": _img_feat("蓝-三角")},
                    {"modality": "image", "raw": _img_feat("红-圆")}])
    topk = retrieve("蓝色三角形", gal, k=2)
    ok_ret = (isinstance(topk, list) and len(topk) == 2
              and all({"obs_id", "modality", "score"} <= set(t) for t in topk)
              and topk[0]["obs_id"] == gal[0]["obs_id"]
              and topk[0]["score"] > topk[1]["score"])
    _check("retrieve：掩码余弦 top-k 语义正确", ok_ret,
           f"top1={topk[0]['obs_id'] if topk else '?'} score={topk[0]['score'] if topk else '?'}")
    res_t0 = encode_image(_img_feat("绿-星"), None)
    _check("T0 兜底：protos=None 走内置概念表",
           bool(res_t0) and res_t0["concept_hits"][0] == "绿-星",
           f"hits={res_t0['concept_hits'] if res_t0 else None}")

    # ---- ⑧ 性能基线（§6.4）----
    t0 = time.perf_counter()
    for _ in range(10):
        perceive(_INPUTS3)
    lat = (time.perf_counter() - t0) / 10 * 1000.0
    _check("性能：三模态 perceive 平均 < 100ms", lat < 100, f"avg={lat:.1f}ms")

    # ---- ⑨ M4 集成冒烟 ----
    from src.api.router import dispatch  # noqa: E402
    resp = dispatch({"session_id": "mm-selftest", "mode": "standard",
                     "inputs": [_INPUTS3[0], _INPUTS3[1]]})
    _check("M4 集成：dispatch(text+image) 全链路无 error",
           not resp.get("error") and isinstance(resp.get("confidence"), (int, float)),
           f"confidence={resp.get('confidence')}")

    print("-" * 62)
    n_ok = sum(1 for _, ok in _CHECKS if ok)
    total = len(_CHECKS)
    print(f"selftest {n_ok}/{total} PASS" + ("  [OK] 全部通过" if n_ok == total else "  [FAIL] 存在失败"))
    print(f"metrics({route})={json.dumps(metrics, ensure_ascii=False)}")
    print("=" * 62)
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
