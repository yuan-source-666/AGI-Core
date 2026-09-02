# AGI-Core 数据工程管道设计方案

> 版本：v1.1 ｜ 作者：凯文（数据工程专家）｜ 日期：2026-09-02
> 状态：Phase 3 实现已交付（`data/build_dataset.py` 运行 12/12 断言全过，见 §7/§8）｜ 上游依据：`GOAL.md` 验收3、`docs/architecture.md` v1.0（§2 M3、§3 I1/I6、§6 约束）
> 下游读者：莎拉（M2 多模态，特征格式对齐）、诺亚（M4 集成，load_dataset 调用）、阿瑟（M1 认知，评测集口径）
> 实现目录：`data/` ｜ 约束：纯本地 CPU、无外网、标准库+numpy（不可用则纯 Python）

## 0. 设计定位与对齐声明
- M3 数据管道 = **合成生成 → 清洗归一 → 构建划分 → 质量评估** 四阶段，产出可复现、可审计的示例数据集；
- 对齐 I6 契约：`load_dataset(split: "train|eval") -> Sample[]`，`Sample = {"id", "modality", "input", "expected", "quality"}`，quality<0.6 不入训练集（见 §6）；
- 对齐 R2 降级路线：无外网 → 全部数据由本地规则模板合成（确定性随机种子，可复现）；
- 64 维统一嵌入由 M2 在运行时计算，**数据集只存原始简化特征**，不预存嵌入（避免双源不一致）；
- 合规红线：不采集任何真实个人数据，全量合成，天然满足脱敏与伦理要求（见 §1.5）。

## 1. 数据来源规划（合成数据为主）
### 1.1 来源策略
| 来源 | 占比 | 说明 |
|---|---|---|
| 本地模板合成（主） | 100% | 规则+槽位模板生成，`random.seed(42)` 确定性复现 |
| 外部数据 | 0% | 纯本地无外网红线（R2），不预留爬取通道 |
| 运行时沉淀（辅） | 不计入 | M1「对话即学习」产出的情景记忆写 `data/memory.jsonl`，属 M1 运行时数据，不入本数据集 |

### 1.2 模态特征的简化表示（对齐 I1）
多模态样本 = 文本（真实现）+ 图像特征（8 维简化向量）+ 语音特征（8 维简化向量），三者共享**概念标签空间**（颜色/形状/意图词），支撑 M2 跨模态对齐演示：

| 模态 | 表示 | 生成规则 |
|---|---|---|
| text | 中文字符串（query/描述） | 槽位模板：`{颜色}{形状}的图片是什么样子`、`请描述这张{颜色}{形状}` 等 12 类句式 |
| image_feat | `{values: 8×float∈[0,1], label, caption}` | 8 维 = [R, G, B, 圆形度, 边缘密度, 亮度, 对比度, 纹理熵]，由颜色/形状概念确定性映射+微抖动（±0.05） |
| audio_feat | `{values: 8×float, label, tone}` | 8 维 = [基频, 音强, 语速, 过零率, 频谱质心, 频谱带宽, 音长, 静音比]，由意图（疑问/陈述/指令）+音调模板映射 |

### 1.3 内容域（保证分布可设计）
- 概念槽：颜色 6 种（红/橙/黄/绿/蓝/紫）× 形状 5 种（圆/方/三角/星/六边）= 30 个图像概念组合；
- 意图标签 4 类：`qa`（问答）、`describe`（描述）、`retrieve`（图文检索）、`command`（语音指令）；
- 主题域：①视觉问答 ②图文检索 ③语音指令理解 ④多模态组合场景（text+image+audio 同时出现）。

### 1.4 噪声注入设计（为清洗管道提供可验证靶标）
合成数据本身纯净，故**主动注入可控噪声**，使清洗/去重/异常剔除各环节有真实效果可度量：
| 噪声类型 | 注入比例 | 形态 |
|---|---|---|
| 完全重复 | 8% | 整样本复制（校验精确去重） |
| 近重复 | 4% | 文本改 1 词 / 特征向量微扰 ±0.02（校验近似去重） |
| 格式破坏 | 4% | 字段缺失、类型错误、维度≠8（校验格式校验环节） |
| 异常值 | 4% | 特征分量越界 [-1,2]、空文本、超长文本>500 字（校验异常剔除） |

### 1.5 合规与伦理
- 全量合成数据，零真实个人信息、零版权素材、零隐私风险，无需脱敏流程（管道保留 `sanitize` 检查位以便未来接入真实数据）；
- `meta.source` 字段强制标注 `synthetic`，下游可识别数据来源，防止合成数据被误当真实数据使用。

## 2. 数据清洗与预处理流程
### 2.1 管道总览（build_dataset.py 内五步串行）
```
raw(300条,含噪声) → ①validate 格式校验 → ②dedup 去重(精确+近似)
→ ③outlier 异常剔除(阈值作用于归一化前分量) → ④normalize 归一化
→ ⑤score 质量打分 → cleaned(≥240) → 分层抽样划分 → train/eval + stats + quality_report
```
每步记录处理量与剔除原因到 `stats.json`，保证全流程可审计（对齐架构「全程可追溯」原则）。
> 实现差异 D1（见 §8）：outlier 前置于 normalize——§2.5 越界阈值作用于**归一化前**分量，若先 clip[0,1]+L2 会掩盖越界特征、异常剔除率恒为 0，故步骤序与 v1.0 设计对调。

### 2.2 ① 格式校验（validate）
- 依据 `dataset/schema.json`（JSON Schema 草案，见 §3.1）逐条校验：必填字段存在、类型正确、特征维度=8、id 唯一格式 `ds-XXXX`；
- 校验失败样本 → 隔离到 `dataset/invalid.jsonl`（不删除，可追溯），计 3xxx 错误码 3001。

### 2.3 ② 去重（dedup）
- **精确去重**：对样本内容指纹（`modality+input+expected` 规范化序列化，不含 id/meta）取 SHA-256，哈希碰撞即删；同指纹时净样本（`meta.noise=null`）优先保留，剔除注入副本；
- **近似去重**：文本 bigram-Jaccard>0.85 **且** 特征余弦>0.95 联合判定近重复（实现差异 D2：同概念合法样本特征余弦天然>0.95，单特征判据会误杀，故由 v1.0 的 OR 收敛为 AND，以文本显著性为主、特征为佐证），保留 quality 较高者、同分时净样本优先；
- 去重率 = 剔除数/输入数，预期 10%~14%（与注入量吻合，作为管道正确性证据；实测 12.5%）。

### 2.4 ④ 归一化（normalize）
- 文本：NFKC 规范化 → 去首尾空白 → 全角转半角（保留中文）；长度记录入 meta；
- 特征向量：clip 到 [0,1] 后 L2 归一化（另存原始幅值于 `meta.orig_norm` 供异常审计）；
- 时间戳：统一 ISO 8601 带时区（Asia/Shanghai）。

### 2.5 ③ 异常剔除（outlier，阈值作用于归一化前分量）
- 规则阈值（可配置于 build_dataset.py 顶部 CFG）：文本长度 ∉ [2,200]；任一特征分量 ∉ [-0.2, 1.2]（归一化前）；`expected.answer` 为空或纯标点；图像/语音 label 不在 §1.3 概念表内；
- 剔除样本同样隔离到 invalid.jsonl 并记录原因码（O1 文本异常/O2 特征异常/O3 标签异常）。

### 2.6 ⑤ 质量打分（score → quality ∈ [0,1]）
```
quality = 0.35·completeness + 0.25·label_conf + 0.20·feat_norm + 0.20·text_clarity
```
- completeness：必填字段齐备度；label_conf：概念标签与特征映射的一致性（回查模板映射表）；feat_norm：特征向量距概念中心向量的偏差（偏差越小分越高）；text_clarity：文本长度适中、无乱码、模板匹配成功。
- 阈值：quality<0.6 按 I6 不入训练集（评测集保留以测鲁棒性，见 §3.2）。

## 3. 数据集构建规范
### 3.1 样本 JSON Schema（顶层 = I6 核心 5 字段 + meta 扩展，向后兼容只增不删）
> 实现差异 D4：**每条样本 `input` 物理携带全部三模态字段**（text+image_feat+audio_feat，对齐任务书「每条含文本/图像特征/语音特征」），`modality` 字段语义调整为**任务主模态场景标记**（四值保留：qa→text、describe/retrieve→image、command→audio、跨模态组合→multi），便于 M2 跨模态对齐训练与缺失容错矩阵演练（运行时按 modality 掩码非主模态输入即可）。
> 真实样例（摘自 train.jsonl，特征已 clip[0,1]+L2 归一化，原始幅值见 meta.orig_norm）：
```json
{
  "id": "ds-0171",
  "modality": "multi",
  "input": {
    "text": "帮我形容一下紫圆",
    "image_feat": {"values": [0.340369, 0.086154, 0.482948, 0.561821, 0.179589, 0.260889, 0.124377, 0.460499],
                    "label": "紫-圆", "caption": "一张紫圆图案"},
    "audio_feat": {"values": [0.398633, 0.366436, 0.400166, 0.242246, 0.391734, 0.346504, 0.364903, 0.285176],
                    "label": "陈述-平调", "tone": "statement"}
  },
  "expected": {"answer": "这张紫圆图：颜色饱和度较高，形状规整，整体视觉平衡。",
                "intent": "describe", "entities": ["紫", "圆"]},
  "quality": 0.9557,
  "meta": {"source": "synthetic", "lang": "zh", "template_id": "T06",
            "noise": null, "created_at": "2026-09-02T16:58:10+08:00",
            "text_len": 8, "orig_norm": {"image": 1.6482, "audio": 1.3045}}
}
```
- `modality`：`text|image|audio|multi`（任务主模态场景标记，见上方 D4 说明）；
- `meta.noise`：注入噪声标记（`dup|near-dup|format|outlier|null`），供清洗管道回验与质量报告统计；清洗后净样本恒为 null；
- `meta.orig_norm`/`meta.text_len`：归一化步骤回写（原始幅值/文本长度，审计用）；eval 样本另含 `meta.robust` 布尔标记；
- 存储格式：JSONL（每行一个样本，追加友好，与 M1 记忆存储格式统一）；`schema.json` 为机器可读校验依据。

### 3.2 规模与划分（括号内为 Phase 3 实测值）
| 集合 | 规模（目标） | 占比 | 说明 |
|---|---|---|---|
| raw 原始 | 300（300） | — | 含 §1.4 注入噪声 |
| cleaned 清洗后 | ≥240（240） | — | 四类清洗后的存活量 |
| train 训练集 | ≥192（192） | 80% | **强制过滤 quality<0.6**（I6 契约；实测 min=0.888，过滤 0 条=双保险生效证明） |
| eval 评测集 | ≥48（48，含 robust 9） | 20% | 保留相对低置信难子集（robust=true）以评测认知层鲁棒性与 clarify 触发 |
- 实现差异 D3：清洗后全量 quality≥0.6（<0.6 恒为 0，I6 门控双保险下该类样本不存在），robust 子集改取 **quality≤P25 分位**（实测阈值 0.9603）作为「相对低置信」难样本；
- 最终有效样本 ≥200 条（实测 240），满足 GOAL 验收 3 的下限并留 20% 冗余；
- 划分策略：**按 intent×modality 分层抽样**（每层内 80/20，最大余数法凑整），保证两集合标签分布一致，避免评测偏斜（实测 intent 占比极差 train 5.2pp / eval 6.2pp）。

### 3.3 命名与版本规范
- id：`ds-` + 4 位零填充序号，全局唯一，清洗剔除后不复用；
- 文件级版本：`stats.json` 内 `dataset_version: "1.0"`，与信封 `version` 字段语义一致；数据格式升级走小版本（1.1…），只增字段不删字段（向后兼容）。

## 4. 数据质量评估指标（输出至 quality_report.md）
| 指标 | 定义 | 目标值 | 度量时机 |
|---|---|---|---|
| 完整性 completeness | 必填字段齐备样本 / 总样本 | ≥98% | 清洗后全量 |
| 分布均衡度 balance | 各 intent 类占比的极差（max−min）；各概念槽覆盖数 | intent 极差 ≤15pp；30 概念槽全覆盖 | 划分后（train/eval 分别统计） |
| 去重率 dedup_rate | 去重剔除数 / 去重输入数 | 10%~14%（与注入量吻合即管道正确） | 去重步骤 |
| 标签覆盖率 label_coverage | `expected.intent`+`answer` 非空率；图像/语音 label 合法率 | 100% | 清洗后 |
| 异常率 outlier_rate | 异常剔除数 / 该步输入数 | 3%~6% | 异常剔除步骤 |
| 质量分布 quality_dist | quality 均值/中位数/P10；quality<0.6 占比 | 均值 ≥0.85；<0.6 占比 <5% | 打分步骤 |
| 模态覆盖 modality_coverage | 四种 modality 样本量 | 每类 ≥30 条 | 划分后 |
- 评估脚本内嵌于 build_dataset.py（`assess()` 函数），指标落盘 `stats.json`，人读摘要渲染 `quality_report.md`；
- 任一指标不达标 → 脚本 exit code=1 并打印缺口，作为 Phase 3 数据线自测的硬断言。

## 5. 数据目录结构定义（对齐现有 data/ 规划）
```
data/
├── build_dataset.py        # 一体化脚本：生成→清洗→划分→评估（Phase 3 交付）
├── dataset/
│   ├── raw.jsonl           # 合成原始样本（300 条，含注入噪声）
│   ├── cleaned.jsonl       # 清洗+归一化+打分后样本（≥240 条）
│   ├── train.jsonl         # 训练集（≥192，quality≥0.6）
│   ├── eval.jsonl          # 评测集（≥48，含 robust 子集）
│   ├── invalid.jsonl       # 校验/异常剔除隔离区（可追溯）
│   ├── schema.json         # 样本 JSON Schema（校验依据）
│   └── stats.json          # 管道各步统计+质量指标+dataset_version
├── quality_report.md       # 质量评估报告（人读，验收物）
├── memory.jsonl            # M1 运行时记忆存储（规划位，阿瑟产出）
└── memory_archive.jsonl    # M1 记忆归档（规划位，阿瑟产出）
```
- 本设计不改动 README 文件索引中 data/ 的既有三项规划（build_dataset.py/dataset//quality_report.md），仅细化 `dataset/` 内部结构；**Phase 3 已按此结构实际产出**（实测行数：raw 300 / cleaned 240 / train 192 / eval 48 / invalid 24）；
- `memory*.jsonl` 为 M1 运行时文件，与静态数据集物理分离，互不读写，避免污染训练/评测数据；
- rules.json（M1 种子规则库）位于 `src/cognition/`，不属本目录。

## 6. 数据供给接口与错误码（I6 对齐）
- **唯一对外入口**：`load_dataset(split: "train|eval", data_dir="data/dataset") -> Sample[]`
  - 读对应 jsonl → 逐行 JSON 反序列化 → list[dict]（顶层 I6 5 字段 + meta 扩展，见 §3.1）；
  - train 分支在加载时二次过滤 quality<0.6（双保险，即使文件被人工改动也守住 I6 契约）；
  - 纯函数、零全局状态，M4/演示层可直接调用；延迟要求：<200ms（全量 ≤300 条， trivial）。
- **错误码（3xxx，对齐 §3.1）**：3001 样本格式校验失败（构建期触发，隔离 invalid.jsonl）；3002 split 参数非法；3003 数据文件缺失或行解析失败/缺 I6 必填字段；3004 加载期聚合校验不达标（train 过滤 quality<0.6 后为空；构建期等价校验以 exit code=1 + stats.json 断言承担）。
- 错误以信封 `error={"code": 3xxx, "msg": ...}` 形式返回给 M4，不抛裸异常（对齐 R4 模块级隔离）。

## 7. Phase 3 实施结果与验收口径（已交付 ✅）
1. `build_dataset.py` 单文件实现（实测 917 行，超出 400 行预算——增量来自 SCHEMA 常量、质量报告渲染、复现性/降级自测与 I6 冒烟代码，核心九函数结构符合设计）：`generate / validate / dedup / outlier / normalize / score / split / assess / main` + `load_dataset`，`python data/build_dataset.py` 一键产出全套文件，失败重试 3 次后自动降级（140 条规模，GOAL 停止条件 1）；
2. 自测断言（12/12 PASS，实测值）：①cleaned=240≥200/240；②train=192/eval=48；③train min quality=0.888 无 <0.6；④去重率 12.5%∈[10%,14%]；⑤异常率 4.8%∈[3%,6%]；⑥intent 极差 train 5.2pp/eval 6.2pp≤15pp；⑦模态 text34/image77/audio34/multi95 各≥30；⑧30 概念槽 30/30 全覆盖；⑨quality 均值 0.9766≥0.85；⑩<0.6 占比 0%<5%；⑪完整性 100%≥98%；⑫标签覆盖 100%；
3. 复现性：同 seed 二次全管道构建，数据文件**逐字节一致**（✅，含 sha256 清单落盘 stats.json）；I6 冒烟：load_dataset train=192/eval=48、五字段完整、3002/3003 错误路径验证通过；
4. 性能：全管道 CPU 运行实测 **0.3s**（300 条规模，目标 <10s）；
5. 演进位：数据量 >10⁴ 或接入真实多模态数据时，仅替换 generate/load 实现，接口与目录结构不变（对齐架构「version+适配层」演进原则）。

## 8. Phase 3 实现对齐说明（v1.0 → v1.1 变更记录）
> 编码实现与 v1.0 设计稿的 4 处偏差，均以实现为准并回写本节（依任务约束「文档与实际有出入以实际为准」）：
- **D1 步骤序**：outlier 前置于 normalize（§2.1）——越界阈值作用于归一化前分量，先 clip 会掩盖越界；
- **D2 近似去重判据**：bigram-Jaccard>0.85 **且** 特征余弦>0.95（联合判定，§2.3）——单特征判据会误杀同概念合法样本（其特征余弦天然>0.95）；文本分词口径=字符 bigram；
- **D3 robust 难子集**：quality≤P25（§3.2）——清洗后 <0.6 恒为 0，原「保留低质量样本」目标改由相对低置信子集承担；
- **D4 三模态字段全量携带**（§3.1）——每条样本 input 均含 text+image_feat+audio_feat，modality 转为任务主模态场景标记；I6 顶层字段不变（id/modality/input/expected/quality + meta 扩展，只增不删向后兼容，对齐 api-spec §0.5）。
- 其他微调：精确去重指纹=内容指纹（不含 id/meta，§2.3）；图像特征后 3 维（亮度/对比度/纹理熵）为样本级自由风格维、前 5 维概念锁定（保证概念可分性）；S3 keep「净样本优先」规则；3004 语义收敛为加载期聚合校验（§6）。
- 契约影响评估：I1/I2/I6 契约签名与字段集**零变更**，无需架构师评审；本节为设计↔实现一致性回写。

---
*本文档 v1.1 与 architecture.md v1.0 §3 契约（I1/I6）及 GOAL 验收 3 逐条对齐；Phase 3 编码已交付，实现与设计的 4 处偏差以 §8 回写记录为准（I1/I2/I6 契约零变更）。后续契约级变更仍须经首席架构师评审并同步版本号。*
