# AGI-Core 示例数据集质量评估报告

> 生成器：`data/build_dataset.py`（seed=42，确定性复现）｜ 数据集版本：1.0 ｜ 报告日期：2026-09-02 ｜ 口径：`docs/data-pipeline.md` §4/§7.2

> 合规声明：全量本地合成数据（`meta.source=synthetic`），零真实个人信息、零版权素材、零外网依赖，天然满足脱敏与伦理要求（§1.5）。


## 1. 规模与管道流程

| 阶段 | 输入 | 剔除 | 产出 | 说明 |
|---|---|---|---|---|
| raw 原始 | — | — | 300 | 噪声注入 dup24/近重复12/格式12/异常12（共 60，占 20%） |
| ① validate 格式校验 | 300 | 12 | 288 | 错误码 3001，隔离 invalid.jsonl |
| ② dedup 去重（精确+近似） | 288 | 36（精确24+近似12） | 252 | 去重率 12.5% |
| ③ outlier 异常剔除 | 252 | 12 | 240 | 异常率 4.8%（{'O1': 6, 'O2': 6, 'O3': 0}） |
| ④ normalize + ⑤ score | 240 | 0 | 240 | NFKC/clip[0,1]/L2 + quality∈[0,1] |
| ⑥ split 分层划分 | 240 | 0 | train 192 / eval 48 | intent×modality 分层 80/20 |

- 最终有效样本（train+eval）**240 条 ≥200**（GOAL 验收 3 达标）；
- 去重率 12.5%（目标 10~14%，与注入量吻合 → 管道正确性证据）；异常率 4.8%（目标 3~6%）。
- eval 含 robust 难子集 9 条（quality≤P25=0.96，测认知层鲁棒性与 clarify 触发）。

## 2. 模态分布（modality = 任务主模态场景，每条 input 均含三模态字段）

| modality | cleaned | train | eval |
|---|---|---|---|
| text | 34 | 27 | 7 |
| image | 77 | 62 | 15 |
| audio | 34 | 27 | 7 |
| multi | 95 | 76 | 19 |

各模态均 ≥30 条（cleaned 口径，目标 ⑦ 达标）。


## 3. 意图分布与均衡度

| intent | cleaned | train | eval |
|---|---|---|---|
| qa | 55 | 44 | 11 |
| describe | 61 | 49 | 12 |
| retrieve | 68 | 54 | 14 |
| command | 56 | 45 | 11 |

意图占比极差：train 5.2pp / eval 6.2pp（目标 ≤15pp，达标 ⑥）。


## 4. 完整性与标签覆盖

| 指标 | 目标 | 实测 | 结论 |
|---|---|---|---|
| 完整性 completeness | ≥98% | 100.0% | ✅ |
| 标签覆盖率 label_coverage | 100% | 100.0% | ✅ |
| 概念槽覆盖（6色×5形状） | 30/30 | 30/30（每槽 4~12 条） | ✅ |

## 5. 质量分布（quality = 0.35完整+0.25标签+0.20特征+0.20文本）

| 统计量 | 均值 | 中位数 | P10 | P90 | 最小 | 最大 |
|---|---|---|---|---|---|---|
| quality | 0.977 | 0.987 | 0.938 | 1.000 | 0.888 | 1.000 |

- 均值 0.977 ≥0.85（达标 ⑨）；quality<0.6 占比 0.0% <5%（达标 ⑩，
I6 门控下 train 全部 ≥0.6）；train 均值 0.976 / eval 均值 0.979。

## 6. 抽检样例（确定性选取：各主模态场景首条 cleaned 样本）


### text 主模态（ds-0001，qa）

```json
{
  "id": "ds-0001",
  "modality": "text",
  "input": {
    "text": "你看到黄圆了吗",
    "image_feat": {
      "values": [
        0.472112,
        0.45749,
        0.111239,
        0.489347,
        0.158763,
        0.276792,
        0.405787,
        0.222478
      ],
      "label": "黄-圆",
      "caption": "一张黄圆图案"
    },
    "audio_feat": {
      "values": [
        0.489253,
        0.379205,
        0.279763,
        0.240649,
        0.487928,
        0.34208,
        0.287718,
        0.206839
      ],
      "label": "疑问-升调",
      "tone": "question"
    }
  },
  "expected": {
    "answer": "这是一张黄圆图案，边缘清晰、对比度中等。",
    "intent": "qa",
    "entities": [
      "黄",
      "圆"
    ]
  },
  "quality": 1.0,
  "meta": {
    "source": "synthetic",
    "lang": "zh",
    "template_id": "T03",
    "noise": null,
    "created_at": "2026-09-02T16:10:00+08:00",
    "text_len": 7,
    "orig_norm": {
      "image": 1.9148,
      "audio": 1.5084
    }
  }
}
```

### image 主模态（ds-0003，retrieve）

```json
{
  "id": "ds-0003",
  "modality": "image",
  "input": {
    "text": "帮我找绿六边的图",
    "image_feat": {
      "values": [
        0.125578,
        0.614024,
        0.170263,
        0.414486,
        0.149461,
        0.246534,
        0.473808,
        0.314331
      ],
      "label": "绿-六边",
      "caption": "一张绿六边图案"
    },
    "audio_feat": {
      "values": [
        0.387813,
        0.375502,
        0.415514,
        0.272393,
        0.375502,
        0.343184,
        0.373193,
        0.252386
      ],
      "label": "陈述-平调",
      "tone": "statement"
    }
  },
  "expected": {
    "answer": "已找到绿六边的匹配图片1张，相关度较高。",
    "intent": "retrieve",
    "entities": [
      "绿",
      "六边"
    ]
  },
  "quality": 1.0,
  "meta": {
    "source": "synthetic",
    "lang": "zh",
    "template_id": "T07",
    "noise": null,
    "created_at": "2026-09-02T16:10:34+08:00",
    "text_len": 8,
    "orig_norm": {
      "image": 1.298,
      "audio": 1.2996
    }
  }
}
```

### audio 主模态（ds-0295，command）

```json
{
  "id": "ds-0295",
  "modality": "audio",
  "input": {
    "text": "播放绿星的语音描述",
    "image_feat": {
      "values": [
        0.110129,
        0.551316,
        0.149077,
        0.151092,
        0.298825,
        0.494237,
        0.5399,
        0.117516
      ],
      "label": "绿-星",
      "caption": "一张绿星图案"
    },
    "audio_feat": {
      "values": [
        0.38883,
        0.494602,
        0.435706,
        0.312507,
        0.378614,
        0.380417,
        0.128008,
        0.110579
      ],
      "label": "指令-短促",
      "tone": "command"
    }
  },
  "expected": {
    "answer": "好的，正在为你朗读绿星的描述内容。",
    "intent": "command",
    "entities": [
      "绿",
      "星"
    ]
  },
  "quality": 0.8999,
  "meta": {
    "source": "synthetic",
    "lang": "zh",
    "template_id": "T10",
    "noise": null,
    "created_at": "2026-09-02T17:33:18+08:00",
    "text_len": 9,
    "orig_norm": {
      "image": 1.4892,
      "audio": 1.664
    }
  }
}
```

### multi 主模态（ds-0009，describe）

```json
{
  "id": "ds-0009",
  "modality": "multi",
  "input": {
    "text": "请描述这张橙三角",
    "image_feat": {
      "values": [
        0.577939,
        0.281575,
        0.089323,
        0.276251,
        0.483883,
        0.343096,
        0.372081,
        0.110027
      ],
      "label": "橙-三角",
      "caption": "一张橙三角图案"
    },
    "audio_feat": {
      "values": [
        0.376185,
        0.391508,
        0.401468,
        0.288076,
        0.381548,
        0.350902,
        0.357797,
        0.253599
      ],
      "label": "陈述-平调",
      "tone": "statement"
    }
  },
  "expected": {
    "answer": "这张橙三角图：颜色饱和度较高，形状规整，整体视觉平衡。",
    "intent": "describe",
    "entities": [
      "橙",
      "三角"
    ]
  },
  "quality": 0.9779,
  "meta": {
    "source": "synthetic",
    "lang": "zh",
    "template_id": "T04",
    "noise": null,
    "created_at": "2026-09-02T16:12:16+08:00",
    "text_len": 8,
    "orig_norm": {
      "image": 1.6905,
      "audio": 1.3052
    }
  }
}
```

## 7. 复现性与接口自测

- 同 seed 二次全管道构建：数据文件**逐字节一致**（✅）；
- `load_dataset`（I6）：train=192 / eval=48，I6 五字段完整，train 双保险过滤生效；
错误路径 3002/3003 按信封语义返回（✅）。

## 8. 结论

**12/12 项硬断言通过**：全部通过；数据集满足 GOAL 验收 3（≥200 条 + 质量报告 + 管道方案）。train/eval 可经 `load_dataset` 供 M2 对齐训练与 M1 评测使用。

## 9. 文件清单（data/dataset/）

| 文件 | 行数 | sha256(前16位) |
|---|---|---|
| raw.jsonl | 300 | d9b5c3a039cc7e8d |
| cleaned.jsonl | 240 | 78ce18bbed8b697b |
| train.jsonl | 192 | d614395006c30cd8 |
| eval.jsonl | 48 | fa56c61b6f4c01c3 |
| invalid.jsonl | 24 | 43ccce90532d2138 |
