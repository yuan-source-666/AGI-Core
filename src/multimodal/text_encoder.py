# -*- coding: utf-8 -*-
"""A1 文本编码器（multimodal-design.md §1.2）：概念词典最大匹配 + idf 加权词袋哈希。

- 分词：中文概念词典最大匹配（组合词 > 别名 > 单概念 > 触发词），剩余 CJK 切字符
  bigram，英文/数字整词保留；
- 编码：A 区词袋（w = (1+log tf)·idf，概念词 ×2 锚定加权，idf 由 T1 统计，缺席→均匀 1）；
  触发词→B 区意图词锚（与语音侧共享锚）；C 区置零；整向量 L2 归一；
- tokens 同时输出规范概念词与原始词形（兼容 M1 的 bigram 口径，保证 IDF-Jaccard 通道）。
"""
from __future__ import annotations

import math
import re

from . import concepts as C
from .space import EMBED_DIM, add_term, l2_normalize

__all__ = ["tokenize", "encode_text", "embed_text"]

_ASCII_RE = re.compile(r"[A-Za-z0-9]+")


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


# 最大匹配词典：surface → (kind, canonical/intent)
_DICT = {}
for _w in C.COMBO_WORDS:
    _DICT[_w] = ("combo", _w)
for _a, _c in C.COLOR_ALIASES.items():
    _DICT[_a] = ("color", _c)
for _c in C.COLORS:
    _DICT[_c] = ("color", _c)
for _a, _s in C.SHAPE_ALIASES.items():
    _DICT[_a] = ("shape", _s)
for _s in C.SHAPES:
    _DICT[_s] = ("shape", _s)
for _w, _i in C.TRIGGERS:
    _DICT.setdefault(_w, ("trigger", _i))
_MAXLEN = max(len(k) for k in _DICT)


def _dedup(seq) -> list:
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def tokenize(text):
    """概念词典最大匹配 → (terms, surface, intent)。

    terms：A 区锚定词（规范概念词[组合词自动展开颜色+形状] + 未匹配 CJK bigram + ASCII 词）；
    surface：tokens 输出（原始词形 + 规范词形，保证 M1 词项重叠通道可用）；
    intent：触发词按 INTENT_PRIORITY 解析的最高优先级意图（无触发 → None）。
    """
    text = str(text or "")
    terms, surface, intents = [], [], []
    pending = []  # 未匹配 CJK 缓冲（≥2 字切 bigram）

    def _flush():
        if len(pending) >= 2:
            bg = ["".join(pending[i:i + 2]) for i in range(len(pending) - 1)]
            terms.extend(bg)
            surface.extend(bg)
        elif pending:
            surface.append(pending[0])
        pending.clear()

    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isascii() and ch.isalnum():
            _flush()
            m = _ASCII_RE.match(text, i)
            w = m.group().lower()
            terms.append(w)
            surface.append(w)
            i = m.end()
            continue
        if not _is_cjk(ch):
            _flush()
            i += 1
            continue
        hit = None
        for ln in range(min(_MAXLEN, n - i), 0, -1):
            cand = text[i:i + ln]
            if cand in _DICT:
                hit = (cand, _DICT[cand])
                break
        if hit is None:
            pending.append(ch)
            i += 1
            continue
        _flush()
        word, (kind, canon) = hit
        if kind == "trigger":
            intents.append(canon)
            surface.append(word)          # 触发词不进 A 区（意图词锚入 B 区）
        else:
            terms.append(canon)
            surface.append(word)
            if kind == "combo":           # 组合词展开：颜色词 + 形状词同槽锚定
                terms.extend([canon[0], canon[1:]])
                surface.extend([canon[0], canon[1:]])
            elif canon != word:
                surface.append(canon)     # 别名场景补规范词形
        i += len(word)
    _flush()

    intent = None
    if intents:
        intent = min(intents, key=C.INTENT_PRIORITY.index)
    return terms, _dedup(surface), intent


def encode_text(text, idf=None):
    """文本 → (embedding, tokens, intent)。idf=None → 均匀权重（T0 兜底）。

    无有效词项且无触发词 → (None, tokens, None)（由 perceive 按 1002 处置）。
    """
    terms, surface, intent = tokenize(text)
    if not terms and intent is None:
        return None, surface, None
    tf = {}
    for t in terms:
        tf[t] = tf.get(t, 0) + 1
    vec = [0.0] * EMBED_DIM
    for t, f in tf.items():
        w = (1.0 + math.log(f)) * ((idf or {}).get(t, 1.0) if idf else 1.0)
        if t in C.CONCEPT_TERMS:
            w *= 2.0                      # 概念词锚定加权（§1.2）
        add_term(vec, t, w, "A")
    if intent is not None:
        add_term(vec, intent, 1.0, "B")   # 意图词锚（B 区，与语音意图锚共享）
    emb = l2_normalize(vec)
    if not any(emb):
        return None, surface, intent
    return [round(x, 6) for x in emb], surface, intent


def embed_text(text) -> list:
    """I1'：文本 → 64 维嵌入（idf 取 T1 统计，缺席 → T0 均匀权重；空文本 → 零向量）。"""
    from .aligner import load_idf  # 延迟导入避免环（aligner → text_encoder 单向）
    emb, _, _ = encode_text(text, idf=load_idf())
    return emb if emb else [0.0] * EMBED_DIM
