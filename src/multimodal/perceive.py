# -*- coding: utf-8 -*-
"""I1 感知入口 + 模态缺失容错降级（multimodal-design.md §3 / §4.1 / §4.3）。

perceive(inputs) → Observation[]，字段严格对齐 I1：
    {"obs_id", "modality", "embedding"(64 维 L2 归一), "tokens",
     "meta": {dim, concept_hits, intent, tone, confidence_factor(μ),
              missing, salience_prior, source, align, query(text)}}

容错降级（§4.3）：非法模态丢弃继续（1001）｜单模态解码失败丢弃继续（1002）｜
嵌入范数为 0 丢弃继续（1003）｜inputs 空/非列表整体失败（1004）｜
原型文件缺席 → T0 内置表兜底（1005 语义，警告不阻塞，meta.align="T0"）。
μ 矩阵：三模态 1.0 ｜ 缺 audio 0.90 ｜ 缺 text 0.55 ｜ 缺 image 0.50 ｜ 单模态 0.80/0.50。
"""
from __future__ import annotations

import hashlib
import json
import os

from . import aligner
from . import concepts as C
from .audio_encoder import encode_audio
from .image_encoder import encode_image
from .text_encoder import encode_text

__all__ = ["perceive", "PerceptionError", "SALIENCE_PRIOR", "MU_MATRIX"]

SALIENCE_PRIOR = {"text": 1.0, "image": 0.9, "audio": 0.85}
_MODALITIES = ("text", "image", "audio")
# 模态缺失降级矩阵（§4.2，对齐 I4 置信度门控 0.6 的演示余量）；
# 键 = sorted(present)（模态在场组合，字母序保证确定性）
MU_MATRIX = {
    ("audio", "image", "text"): 1.00,   # 三模态齐备
    ("image", "text"): 0.90,            # 缺 audio
    ("audio", "text"): 0.55,            # 缺 image
    ("audio", "image"): 0.50,           # 缺 text
    ("text",): 0.80, ("image",): 0.50, ("audio",): 0.50,
}


class PerceptionError(Exception):
    """感知层错误（1xxx，multimodal-design §4.3）：携带 code，不抛裸 msg。"""

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code, self.msg = int(code), str(msg)


def _obs_id(item, seq: int) -> str:
    """确定性观测 id：内容指纹 + 序号（同输入 → 同 id）。"""
    digest = hashlib.md5(json.dumps(item, ensure_ascii=False,
                                    sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return f"obs-{seq:03d}-{digest}"


def _read_uri(uri):
    if not isinstance(uri, str) or not os.path.isfile(uri):
        raise PerceptionError(1002, f"uri 不可读或不存在: {uri!r}")
    with open(uri, "r", encoding="utf-8") as f:
        return f.read()


def _decode(item, modality):
    """ModalInput → 模态内容（raw 优先，uri 兜底）；失败抛 1002。"""
    raw = item.get("raw")
    if modality == "text":
        if raw is not None and str(raw).strip():
            return str(raw)
        content = _read_uri(item.get("uri"))
        if not content.strip():
            raise PerceptionError(1002, "文本内容为空")
        return content
    # image / audio：raw 为特征 dict（或 list / JSON 字符串），否则 uri 指向 JSON 文件
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            raise PerceptionError(1002, "raw 非法 JSON 特征")
        if isinstance(parsed, (dict, list)):
            return parsed
        raise PerceptionError(1002, "raw 非特征对象")
    content = _read_uri(item.get("uri"))
    try:
        parsed = json.loads(content)
    except ValueError:
        raise PerceptionError(1002, "uri 文件非 JSON 特征")
    if not isinstance(parsed, (dict, list)):
        raise PerceptionError(1002, "uri 文件非特征对象")
    return parsed


def _encode_text_obs(content, protos):
    idf = (protos or {}).get("idf")
    emb, tokens, intent = encode_text(content, idf)
    if emb is None:
        raise PerceptionError(1002, "文本无有效词项")
    hits = [t for t in tokens if t in C.CONCEPT_TERMS]
    return {"embedding": emb, "tokens": tokens, "concept_hits": hits,
            "intent": intent, "tone": None}


def perceive(inputs):
    """I1 契约入口（签名冻结，api-spec §3.2）。"""
    if not isinstance(inputs, list) or not inputs:
        raise PerceptionError(1004, "inputs 为空（全模态缺失）")
    present = sorted({it.get("modality") for it in inputs
                      if isinstance(it, dict) and it.get("modality") in _MODALITIES})
    missing = [m for m in _MODALITIES if m not in present]
    mu = MU_MATRIX.get(tuple(present), 0.50)
    protos = aligner.load_protos()
    W = aligner.load_t2()
    align_src = "T2" if W else ("T1" if protos else "T0")

    observations, seq = [], 0
    for item in inputs:
        if not isinstance(item, dict) or item.get("modality") not in _MODALITIES:
            continue  # 1001：非法模态，丢弃继续
        modality = item["modality"]
        try:
            content = _decode(item, modality)
            if modality == "text":
                res = _encode_text_obs(content, protos)
            elif modality == "image":
                res = encode_image(content, protos, W)
            else:
                res = encode_audio(content, protos)
        except PerceptionError:
            continue  # 1002：该模态解码失败，丢弃继续
        if not res or not res.get("embedding") or not any(res["embedding"]):
            continue  # 1002/1003：非法特征或零范数嵌入
        seq += 1
        meta = {"dim": 64, "concept_hits": res.get("concept_hits", []),
                "intent": res.get("intent"), "tone": res.get("tone"),
                "confidence_factor": mu, "missing": missing,
                "salience_prior": SALIENCE_PRIOR[modality],
                "source": "m2", "align": align_src,
                "query": content if modality == "text" else None}
        observations.append({"obs_id": _obs_id(item, seq), "modality": modality,
                             "embedding": res["embedding"],
                             "tokens": res.get("tokens", []), "meta": meta})
    return observations
