# AGI-Core 多模态模块设计方案

> 版本：v1.0 ｜ 作者：莎拉（多模态专家）｜ 日期：2026-09-02
> 状态：Phase 2 交付物（多模态线）｜ 上游依据：`docs/architecture.md` v1.0（§2 M2、§3 I1、R1-R4 降级）
> 特征格式对齐：`docs/data-pipeline.md` v1.0 §1.2/§1.3（8 维图像/语音特征、概念槽、intent 标签）
> 下游读者：阿瑟（M1，tokens/salience/嵌入空间耦合）、诺亚（M4，perceive 调用与 demo 编排）
> 实现目录：`src/multimodal/` ｜ 约束：纯本地 CPU、无外网、标准库 + numpy（不可用则纯 Python）

## 0. 设计定位与对齐声明
- M2 多模态 = **三模态编码 + 跨模态语义对齐 + 模态缺失容错**，产出 64 维统一嵌入（I1 Observation）；
- 对齐架构 R3 降级：以「**概念锚定共享哈希语义空间 + 掩码余弦**」替代可学习对比学习（CLIP 式），验收口径从对齐精度改为**可演示的语义检索正确率**；
- 对齐 R1：编码零训练参数（哈希/原型/固定随机投影，seed=42 确定性可复现）；「训练」仅指统计量估计与可选闭式校准（§2.3 T1/T2）；
- 边界：不 import M1/M3，仅经 M4 调度或文件传递（读 `data/dataset/*.jsonl` 估计原型，属架构允许的文件传递）；
- 数据集只存 8 维原始特征，64 维嵌入由 M2 运行时计算（对齐 data-pipeline §0「避免双源不一致」）。

## 1. 三模态输入表示方案（CPU 简化路线）
### 1.1 概念锚词表（三模态共享的「语义货币」）
跨模态对齐的前提是三模态存在公共语义锚点。沿用 data-pipeline §1.3 概念槽，构建内置词表 `concepts.py`：
- 视觉概念：颜色 6（红/橙/黄/绿/蓝/紫）× 形状 5（圆/方/三角/星/六边）= 30 个组合概念，同时收录 11 个单概念词；
- 意图/语气词：4 intent（qa/describe/retrieve/command）+ 3 tone（question/statement/command）及中文触发词（"什么/吗/哪"→qa，"描述/看看"→describe，"找/哪张/检索"→retrieve，"请/帮我"→command）；
- 词表→8 维特征的**确定性映射表**与 data-pipeline 生成规则镜像一致（T0 内置基线，数据集缺席时兜底）。

### 1.2 文本 → 词袋哈希向量（text_encoder.py）
- 分词：中文**概念词典最大匹配**（优先命中概念词，剩余串切字符 bigram），英文/数字整词保留；
- 编码：词袋 + **带符号双槽哈希**入 64 维空间：`hash(term,salt) → dim∈[0,48)`，符号 ±1 由第三盐奇偶决定（降低碰撞偏置）；
- 权重：`w = (1+log tf)·idf`，idf 公式与 M1 注意力**同式**（`log((N+1)/(df+1))+1`，N/df 由 train.jsonl 文本统计，T1 产出）；概念词额外 ×2 锚定加权；
- 意图触发词哈希入意图区 [48,56)；私有区 [56,64) 置零；整向量 L2 归一化。

### 1.3 图像 → 像素统计/合成特征（image_encoder.py）
- 输入：8 维合成特征 `image_feat.values = [R,G,B,圆度,边缘密度,亮度,对比度,纹理熵]`（I1 的 raw 内联或 uri 指向 JSON 文件）；
- 概念软匹配：对 30 个概念中心向量（T1 原型，见 §2.3）算余弦，取 **top-2**，权重 `w_k=(cos_k+1)/2` 归一；
- 嵌入构造：①概念区 [0,48)：按 w_k 激活 top-2 概念的**组合词+颜色词+形状词**哈希锚（与文本同槽，天然对齐）；②私有区 [56,64)：`P_img·f8`（P_img 为 seed=42 的 8×8 固定高斯投影），保留图像细节区分度；③意图区置零；L2 归一化。

### 1.4 语音 → 频谱统计特征（audio_encoder.py）
- 输入：8 维特征 `audio_feat.values = [基频,音强,语速,过零率,频谱质心,频谱带宽,音长,静音比]`；
- 意图软匹配：对 tone/intent 原型（T1 估计）余弦 top-1，映射 tone→intent（question→qa，statement→describe，command→command，retrieve 由文本/上下文决定）；
- 嵌入构造：①意图区 [48,56)：激活 tone+intent 词哈希锚（权重=匹配置信）；②私有区 [56,64)：`P_aud·f8`（独立 seed 投影）；③概念区置零（语音不携带视觉概念，靠与文本/图像共现补齐）；L2 归一化。

### 1.5 64 维共享语义空间分区
| 分区 | 维度 | 内容 | text | image | audio |
|---|---|---|---|---|---|
| A 概念锚区 | [0,48) | 颜色/形状/组合/通用词哈希 | ✔ | ✔（原型软匹配） | — |
| B 意图锚区 | [48,56) | intent/tone 词哈希 | ✔（触发词） | — | ✔（原型匹配） |
| C 模态私有区 | [56,64) | 原始特征固定投影 | 0 | ✔ | ✔ |

## 2. 跨模态语义对齐方案
### 2.1 共享语义空间设计
- **锚定原理**：对齐不靠训练，靠「同一概念名在三模态编码时哈希到同一维度组」。文本写"蓝色三角"、图像特征解码出概念"蓝色-三角"，两者在 A 区激活相同锚 → 余弦自然高（R3 降级路线的工程化落地）；
- 演进一致性：`version` 字段 + 分区表存 `space.py` 常量；未来接 VLM/CLIP 时仅替换三个 encoder，分区与契约不变。

### 2.2 相似度度量（掩码余弦）
```
sim_cross(a, b) = cos(a[0:56], b[0:56])   # 跨模态：A∪B 共享区，剔除私有区干扰
sim_full(a, b)  = cos(a, b)               # 同模态：全 64 维
retrieve(query_text, gallery_obs, k)      # sim_cross 排序取 top-k（暴力扫描，N≤300 无压力）
```
- 跨模态检索一律 `sim_cross`；同模态去重/匹配用 `sim_full`；
- 阈值：sim<0.2 视为无语义关联（对齐 M1 模板通道兜底阈值 θ=0.2）。

### 2.3 对齐「训练」（简化版，三级递进）
| 级 | 名称 | 方法 | 触发 | 开销 |
|---|---|---|---|---|
| T0 | 内置基线 | 概念→8 维确定性映射表（concepts.py 内置，镜像数据生成规则） | 数据集缺席时兜底 | 0 |
| T1 | 原型统计（默认） | 从 `train.jsonl` 按概念标签求 image_feat 均值 = 30 概念中心；按 tone 求 audio_feat 均值 = 意图原型；同步统计文本 df 得 idf | 初始化一次 | <1s |
| T2 | 闭式校准（可选） | 岭回归 `W*=argmin‖XW−Y‖²+λ‖W‖²`（λ=0.1，numpy lstsq）：X=样本 8 维特征，Y=配对 caption 的 A 区嵌入；图像嵌入 A 区改为 `W·f8`，保留私有区 | T1 评测 top-1<70% 时启用 | <1s |
| T3 | 演进位 | 本地小模型对比学习（CLIP-lite） | Phase 5 后 | — |
- T1 产物缓存 `src/multimodal/prototypes.json`（seed=42 可复现）；T2 系数缓存 `align_W.json`；
- 一致性自检：train 集上「图像 top-1 概念 == label」比率 ≥95%（T1 硬断言）。

### 2.4 对齐质量评测口径（对齐 R3「可演示正确率」）
| 指标 | 定义 | 目标 |
|---|---|---|
| 检索 top-1 / top-3 | text→image 跨模态检索首位/前三命中率（eval.jsonl） | ≥0.70 / ≥0.85 |
| 图文匹配准确率 | image→候选文本（5 选 1）配对正确率 | ≥0.70 |
| 概念解码一致率 | 图像 top-1 概念==label 比率（train） | ≥0.95 |
| 编码延迟 | 单样本 encode | <5ms（numpy）/ <20ms（纯 Python） |

## 3. 模态缺失容错策略
### 3.1 降级矩阵（8 种输入组合）
| 输入组合 | 产出 | μ（置信度因子） | 路径说明 |
|---|---|---|---|
| text+image+audio | 3 条 Observation | 1.00 | 全模态，场景 A 主路径 |
| text+image | 2 条 | 0.90 | 图文问答主形态，缺语音仅少意图佐证 |
| text+audio | 2 条 | 0.85 | 语音问答无图：意图区+概念区均可用 |
| image+audio | 2 条 | 0.65 | 无文本查询词：概念+意图齐但无问题锚，M1 多走模板/caption |
| text | 1 条 | 0.80 | 纯文本问答（记忆增强场景主形态） |
| image | 1 条 | 0.55 | 概念解码→caption 回退，conf<0.6 大概率 clarify 追问 |
| audio | 1 条 | 0.50 | 仅意图可判：command 走规则路由，其余 clarify |
| 空 | 0 条 | — | error 1004（见 §4.3） |
### 3.2 降级规则
1. **不造假观测**：缺失模态不产生零向量 Observation，避免污染 M1 注意力池；
2. 每条 Observation 的 `meta.missing` 记录本次输入缺失的模态列表；
3. `meta.confidence_factor = μ`（**可选字段**，不破坏 I1 契约兼容）；建议 M1 仲裁采用 `conf ← conf × mean(μ of focus_obs)`，默认忽略亦不影响现有行为（是否采纳由阿瑟评审，本模块不强加）；
4. 单模态非文本（image/audio only）时 tokens 仍输出概念/意图词，保证 M1 词项重叠通道可用；
5. 解码失败≠全局失败：某模态特征非法 → 该模态信封携带 1xxx 错误码并丢弃，其余模态继续（对齐 R4 模块级隔离）。

## 4. 与认知核心（src/cognition/）的接口契约
### 4.1 I1 逐字对齐：`perceive(inputs: ModalInput[]) -> Observation[]`
- ModalInput 解析规则：

| modality | raw（内联，优先） | uri（次选/兜底） |
|---|---|---|
| text | 字符串正文 | 指向 .txt 文件 |
| image | 特征 dict `{"values":[8×float], "label", "caption"}`（即 Sample.input.image_feat） | 指向同构 .json 文件 |
| audio | 特征 dict `{"values":[8×float], "label", "tone"}` | 指向同构 .json 文件 |

- Observation 出参（对齐 I1 全部必填字段）：
```json
{"obs_id": "obs-001", "modality": "image",
 "embedding": [64×float, L2归一化],
 "tokens": ["蓝色", "三角"],
 "meta": {"dim": 64, "concept_hits": ["蓝色-三角"],
          "intent": null, "tone": null,
          "confidence_factor": 0.9, "missing": ["audio"],
          "salience_prior": 0.9, "source": "synthetic"}}
```
- 信封封装：`src="perception", dst="cognition", type="observation", version="1.0"`；
- `salience_prior` 即 M1 注意力 δ·salience(o) 的模态先验（文本 1.0 / 图像 0.9 / 语音 0.85，与 algorithm-design §1.1 数值一致，避免 M1 重复硬编码）。

### 4.2 与 M1 的三个耦合点（不 import，经 M4 注入）
1. **tokens 口径**：M2 词典切词输出即 M1 IDF-Jaccard overlap 的分词单元（概念词优先），保证跨模态词项重叠可比（文本查询"蓝色三角" ↔ 图像 tokens 命中）；
2. **共享嵌入空间**：M2 额外导出 `embed_text(text) -> embedding`（供 M1 `mem_recall` 的 e_q 使用，确保记忆检索与感知在同一 64 维空间）；M4 初始化时注入 M1；
3. **注意力双通道**：embedding 余弦通道 + tokens 重叠通道在跨模态对上同时生效（图像观测因 tokens 含概念词而可被文本查询聚焦）——这是 M1 attend 对多模态观测生效的关键保障。
### 4.3 感知层错误码（1xxx，对齐 §3.1）
| 码 | 含义 | 处置 |
|---|---|---|
| 1001 | modality 非法（非 text/image/audio） | 丢弃该项，继续 |
| 1002 | 特征解码失败（文件缺失/维度≠8/类型错） | 丢弃该模态，继续（R4） |
| 1003 | 嵌入生成异常（范数为 0 等） | 丢弃该观测，继续 |
| 1004 | inputs 为空（全模态缺失） | 整体失败，返回错误信封 |
| 1005 | 原型/idf 资源缺失 | 降级 T0 内置表，警告不阻塞 |
### 4.4 模块文件结构
```
src/multimodal/
├── __init__.py      # 导出 perceive / embed_text / similarity
├── concepts.py      # 概念词表 + T0 确定性映射表（内置）
├── space.py         # 哈希/分区常量/掩码余弦/归一化
├── text_encoder.py  # A1 encode_text
├── image_encoder.py # A2 encode_image（T1/T2 原型软匹配）
├── audio_encoder.py # A3 encode_audio（意图原型）
├── align.py         # T1 原型统计 / T2 闭式校准 / align_eval 评测
├── perceive.py      # I1 入口 + 容错降级 + 信封封装
├── prototypes.json  # T1 运行时产物（seed=42）
└── align_W.json     # T2 运行时产物（可选）
```
依赖：hashlib/json/math/random（标准库）；numpy 可选（缺失则纯 Python 路径，延迟上限放宽至 §2.4 口径）。

## 5. Demo 场景设计（≥2，供诺亚编排 run_demo.py）
### 场景 A：多模态检索问答（text→image 跨模态检索 + M1 问答）
1. 数据准备：从 `eval.jsonl` 取 20 条含 image_feat 样本建检索库（gallery）；
2. 输入：用户文本查询，如"哪张图是蓝色三角形？"（若样本含 audio_feat 则三模态齐发，走 μ=1.00 主路径）；
3. 感知：`perceive` 产出 Observation，`sim_cross` 对 gallery 排序取 top-3（附相似度分）；
4. 认知：top-1 图像观测 + 文本查询交 `run_cognition`，M1 聚焦该观测，基于 caption 记忆/规则作答；
5. 输出：答案 + top-3 排名 + trace（感知→检索→注意力→推理→决策）。
- 验收：eval 集 top-1 ≥0.70；单轮延迟 <5s。

### 场景 B：图文匹配（image→text 配对打分）
1. 输入：单条 image_feat（或文本指定 id）；
2. 概念解码：top-2 概念 + 置信度，模板生成 caption（"一张蓝色三角形图案"）；
3. 配对：将该图与 5 条候选文本（1 正 4 负，取自 eval）逐一 `sim_cross` 打分排序；
4. 输出：排名表 + 正配对得分 vs 最高负例得分 + 结论。
- 验收：匹配准确率 ≥0.70；展示 1 个典型错例及原因（概念碰撞/近邻概念混淆）。

### 场景 C（备用加分）：语音意图与缺失容错演示
1. audio-only 输入 → 意图解码（如 question）→ μ=0.50，M1 conf<0.6 触发 clarify 追问（演示容错而非硬答）；
2. 同一问题补充 text 后（双模态 μ=0.85）conf 提升、正常作答——量化展示降级矩阵效果；
3. 验收：两次运行 conf 差值 ≥0.15 且行为符合 I4 门控预期。

### run_demo 编排约定（移交诺亚）
- 场景 A/B/C 各封装为函数，M4 仅调用 `perceive / embed_text / run_cognition`，不触碰 M2 内部符号；
- 每场景输出统一 Response（I5）+ 场景级 PASS/FAIL 判定，全场景 <30s。

## 6. Phase 3 实施计划与自测
1. 编码顺序：concepts → space → 三 encoder → perceive → align（纵切先通场景 A，再补 B/C，对齐 R6）；
2. 总代码预算 ≤600 行（含注释与 docstring），全部纯函数、零全局状态；
3. 自测断言：①perceive 输出严格符合 I1 字段与维度=64、L2 范数=1；②"蓝色三角"文本 vs 蓝三角图像 `sim_cross`>0.5，vs 红圆图像 <0.2（对齐分离度）；③T1 概念解码一致率 ≥95%；④eval 检索 top-1≥0.70；⑤缺失容错：audio-only 不崩溃且 meta.missing 正确；⑥同 seed 两次运行嵌入逐位一致；
4. 性能基线：batch(3 模态) perceive <100ms；场景 A 单轮端到端 <5s；
5. 风险与降级：无 numpy → 纯 Python 路径；原型文件缺失 → T0 兜底（1005 警告）；检索不达标 → 启用 T2 校准，仍不达标则验收口径降为 top-3 并在 README 记录（对齐 GOAL 停止条件 1）。

---
*本文档与 architecture.md v1.0 §3 契约（I1）及 data-pipeline.md §1.2/§1.3 特征格式逐条对齐；对 M1 的 μ 建议为可选扩展，采纳与否由算法线评审。契约变更须经首席架构师评审并同步版本号。Phase 3 编码实现以本文 §4 规格为准。*
