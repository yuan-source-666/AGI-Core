# README.md — AGI-Core 项目状态与索引

## 项目简介

构建全量 AGI 原型系统（AGI-Core）：多模态感知（视觉/语音/文本）+ 认知核心（记忆/注意力/推理）+ 系统集成，本地可运行、可演示、可演进。目标与红线详见 `GOAL.md`（所有 Agent 必读）。

## 技术栈

- Python 3（沙箱本地运行，默认 CPU，不假设外网/GPU/外部 API）
- 数据格式：JSON 为主；存储：本地文件
- 模块协作：统一 API 规范（`docs/api-spec.md`）+ 文件传递

## 团队分工

| 成员  | 角色      | 职责范围                   |
| --- | ------- | ---------------------- |
| 皮皮  | 项目总编排   | 任务调度、进度同步、用户对接（不执行子任务） |
| 林图灵 | 首席架构师   | 顶层架构、技术路线图、最终验收        |
| 阿瑟  | 算法研究员   | 认知核心算法（注意力/记忆/推理）      |
| 凯文  | 数据工程专家  | 数据管道、数据集构建、质量评估        |
| 莎拉  | 多模态专家   | 视觉/语音模块、跨模态对齐          |
| 诺亚  | 系统集成工程师 | 代码骨架、API、集成落地          |

## 工作流（方法论）

「设计先行 → 逐模块实现 → 持续集成 → 联调演示 → 验收交付」。  
规则：每个任务完成后必须①更新本 README「当前进度」②在群里发 ≤100 字简报。失败重试上限 3 次后降级。

## 当前进度（更新于 2026-09-02 19:09）

- [x] Phase 0：项目初始化（GOAL/README/目录/任务管道）✅
- [x] Phase 1：总体架构设计（林图灵）✅ → `docs/architecture.md`（分层架构/模块职责/JSON 接口契约/路线图/风险降级）
- [x] Phase 2：算法/数据/多模态三线设计（阿瑟/凯文/莎拉）✅ 全部完成
  - [x] 算法线：认知核心算法设计（阿瑟）✅ → `docs/algorithm-design.md`（注意力打分/三层记忆+衰减巩固/三通道混合推理/认知循环/接口对齐 I2-I3）
  - [x] 数据线：数据管道方案（凯文）✅ → `docs/data-pipeline.md`（合成多模态数据/清洗五步/JSON Schema+分层划分/质量七指标/I6 对齐）
  - [x] 多模态线：多模态方案（莎拉）✅ → `docs/multimodal-design.md`（概念锚定 64 维共享空间/三模态哈希编码/掩码余弦对齐 T0-T2/缺失容错矩阵/I1 契约/3 演示场景）
- [x] Phase 3：工程骨架 + 各模块代码实现 ✅（骨架/认知/数据/多模态四线全部完成，自测与回归全绿）
  - [x] 工程骨架线：统一 API 规范 + 代码骨架（诺亚）✅ → `docs/api-spec.md` + `src/api/router.py` + M1/M2 接口桩 + `demo/run_demo.py`（14/14 自测通过，输出「骨架就绪」）
  - [x] 认知核心实现（阿瑟）✅ → `src/cognition/`（memory/attention/reasoning/cognition 编排 + rules.json + selftest，**自测 11/11 PASS**；`run_cognition` I2 签名不变，M4 零改动；demo 回归 14/14 PASS）
  - [x] 多模态实现（莎拉）✅ → `src/multimodal/`（三模态编码器 + aligner 对齐/检索 + perceive I1 容错入口，纯标准库；**自测 19/19 PASS**；`perceive/embed_text/similarity` 签名不变，M4 零改动；demo 回归 14/14 PASS，M1 自测回归 11/11 PASS）
  - [x] 数据管道实现（凯文）✅ → `data/build_dataset.py` + `data/dataset/`（7 文件）+ `data/quality_report.md`（**自测 12/12 断言 PASS + 复现性字节级一致 + I6 冒烟 PASS**；`docs/data-pipeline.md` 升 v1.1 回写实现差异；demo 回归 14/14 PASS）
- [x] Phase 4：系统集成与联调（诺亚）✅ → `src/api/router.py` 升级（共享语义空间桥接 M4→M2→M1 + 会话状态持久化 `data/session_state.jsonl`）+ `demo/run_demo.py` 三场景完整版（**21/21 PASS，3/3 场景**，端到端 avg 62.8ms）
- [x] Phase 5：验收测试、总结报告与最终交付（林图灵 / 大米复核）✅ 三项齐备
  - [x] 总结报告 ✅ → `docs/final-report.md` v1.0（342 行：**GOAL 六条 6/6 核销** + 量化指标 + §6 八项局限披露 + §7 P0-P2 演进路线 + §1.3 交付清单）
  - [x] 全量测试 ✅ → `tests/acceptance.py`（一键核销入口：**6/6 PASS，exit 0**；独立判定 + subprocess 隔离 + 单项失败不阻塞 + 支持 `--json`）
  - [x] 最终打包 ✅ → 交付清单落 `final-report.md` §1.3：**44 个文件 / 858.3 KB / 逐文件 sha256 审计**，已排除 `__pycache__`、运行时产物与平台预置目录

> **项目状态：已交付（2026-09-02）**。GOAL 六条验收标准全部达成，无降级触发、无停止条件触发。局限性、架构债与演进路线见 `docs/final-report.md` §6–§7。


## 重大事件 / 经验教训

- 2026-09-02 15:04 项目启动，完成初始化；用户待确认：资源边界/交付侧重/是否需要定时进度汇报
- 2026-09-02 15:11 Phase 1 完成：`docs/architecture.md` 发布（v1.0，189 行）。确定感知/认知/决策/输出四层架构与统一 JSON 信封契约（I1-I6）；确立 CPU 简化降级路线（R1-R6），Phase 2 各线设计须对齐 §3 接口契约
- 2026-09-02 15:15 Phase 2 算法线完成：`docs/algorithm-design.md` 发布（v1.0，177 行）。核心决策：①注意力=零参数打分（余弦+IDF-Jaccard+新鲜度+显著性）替代 QK 投影；②三层记忆（working/episodic/semantic，JSONL 存储，遗忘曲线衰减+巩固机制）；③推理=规则+证据投票+模板三通道仲裁，conf<0.6 转 clarify；④认知循环含「对话即学习」回写闭环。Phase 3 实现以 §5 接口为准
- 2026-09-02 15:32 Phase 2 数据线完成：`docs/data-pipeline.md` 发布（v1.0，170 行）。核心决策：①全量本地合成多模态数据（文本+8维图像/语音特征，seed=42 可复现，合规零风险）；②主动注入 20% 四类噪声使清洗管道可度量；③清洗五步（校验/精确+近似去重/归一化/异常剔除/质量打分）；④规模 300 原始→≥240 清洁→train 80%/eval 20% 分层划分（≥200 验收达标）；⑤七项质量指标硬断言；⑥load_dataset 严格对齐 I6（train 双保险过滤 quality<0.6，3xxx 错误码）
- 2026-09-02 15:24 Phase 2 多模态线完成：`docs/multimodal-design.md` 发布（v1.0，181 行），Phase 2 三线设计全部收官。核心决策：①64 维共享语义空间按 A 概念锚区/B 意图区/C 私有区结构化分区，跨模态对齐=「概念名同槽哈希+掩码余弦」（零训练参数，对齐 R3 降级）；②对齐训练三级 T0 内置表→T1 原型统计→T2 岭回归闭式校准（默认 T1，检索 top-1<70% 才启 T2）；③模态缺失 8 组合降级矩阵，μ 置信度因子为可选 meta 字段不破坏 I1；④额外导出 embed_text 供 M1 记忆检索共用嵌入空间；⑤3 个演示场景（检索问答/图文匹配/语音容错），评测口径 top-1≥0.70
- 2026-09-02 15:36 Phase 3 工程骨架线完成（诺亚）：`docs/api-spec.md` 发布（v1.0，191 行）+ 代码骨架落地。核心决策：①统一信封/路由表/懒导入+桩降级（R4 双保险：模块桩→M4 内置兜底桩）；②错误码 1xxx/2xxx/3xxx 汇总 + 新增 4xxx（4001-4005）；③dispatch 永不抛异常、错误按信封码传播；④M1/M2 留 STUB 桩（签名冻结 I1/I2，阿瑟/莎拉 Phase 3 直接替换）；⑤query 提取规则=首个 text 模态 raw。自测：demo 14/14 PASS、注册替身/桩降级/确定性/64 维 L2 契约全过、全链路 <10ms；numpy 2.1.3 在环（可选依赖，requirements.txt 已标注）
- 2026-09-02 16:02 Phase 3 认知核心实现完成（阿瑟）：`src/cognition/` 桩全部替换为真实实现（纯标准库）。核心落地：①memory.py 三层记忆（working 内存 deque/episodic+semantic JSONL），四因子 kNN 检索（cos+IDF-Jaccard+新近度+命中数）、遗忘曲线衰减归档（<0.2 不删除）、巩固（hits≥3 或簇 cos>0.75 且≥5 → 蒸馏语义记忆）、写入相似合并（cos≥0.95）；②attention.py 零参数打分 α/β/γ/δ=0.45/0.30/0.15/0.10 + Top-K(8) + softmax(T=0.5)；③reasoning.py 规则前向链接（链深≤3，rules.json 11 条种子规则）+ 记忆证据投票 + 模板兜底三通道仲裁（w=0.5/0.35/0.15）；④cognition.py 认知循环（工作记忆写入→推理→情景回写→巩固+衰减），记忆子模块故障就地降级（R4）。自测 11/11 PASS（`python src/cognition/selftest.py`：注意力排序/衰减单调/双路巩固/门控/3 轮记忆增强冒烟 conf 0.6577→0.7285 单调升/千条 kNN avg 17.5ms<50ms/契约+2001·2002/持久化）；demo 回归 14/14 PASS，M4 零改动；跨进程复跑 demo conf 0.6577→0.7505 验证记忆持久生效。备注：M1 按边界约定未 import M2，`_shared.py` 内置与 M2 桩一致的哈希词袋嵌入（演进位：统一共享嵌入服务）；reasoning.py 为任务命名，reasoner.py 为设计文档命名别名（等价 re-export）
- 2026-09-02 16:25 Phase 3 数据管道实现完成（凯文）：`data/build_dataset.py`（917 行，纯标准库，seed=42）+ `data/dataset/`（raw 300/cleaned 240/train 192/eval 48/invalid 24/schema/stats，含 sha256 审计）+ `data/quality_report.md`。核心落地：①合成多模态样本 300 条（每条含文本+8 维图像特征+8 维语音特征+标签，30 概念槽×12 句式模板，四模态场景标记）；②注入 20% 四类噪声（重复 8%/近重复 4%/格式 4%/异常 4%）→ 五步清洗（校验/精确+近似去重/异常剔除/归一化/打分）；③12/12 硬断言 PASS：去重率 12.5%、异常率 4.8%、quality 均值 0.9766、<0.6 占比 0%、30 概念槽全覆盖、intent 极差≤6.2pp、全模态≥30 条；④同 seed 二次构建逐字节一致；⑤`load_dataset` I6 落地（train 双保险过滤 + 3xxx 错误码，router 懒导入路径验证通过）。`docs/data-pipeline.md` 升 v1.1：回写 4 处实现差异（D1 outlier 前置/D2 近似去重联合判据/D3 robust=P25/D4 三模态字段全量携带，I1/I2/I6 契约零变更）；demo 回归 14/14 PASS。耗时 0.3s（目标<10s）。经验：同概念样本特征余弦天然>0.95，近似去重须文本+特征联合判定防误杀；异常阈值须作用于归一化前分量
- 2026-09-02 16:28 Phase3c 验收（皮皮）：PASS——独立复验 I6（train 192/eval 48、五字段+三模态字段完整、错误路径 3003 正常），样本 240≥200，GOAL 验收 3 达成；数据线收官，管道无阻塞，自动进入 Phase3d（多模态实现）
- 2026-09-02 16:58 Phase 3 多模态实现完成（莎拉）：`src/multimodal/` 桩全部替换为真实实现（纯标准库，11 文件）。核心落地：①三模态编码器——text 概念词典最大匹配+idf 词袋哈希入 A 区（概念词×2 锚定），image 原型 top-2 概念软匹配（w=(cos+1)/2 归一）+组合/颜色/形状词锚+私有区投影，audio 语气原型匹配+意图/语气锚入 B 区；②64 维共享空间结构化分区（A[0,48)/B[48,56)/C[56,64)）+带符号双槽 md5 哈希（同一概念名三模态同槽）+掩码余弦 sim_cross；③aligner——T1 原型统计（30 概念+3 语气+idf，落盘 prototypes.json）、T2 纯 Python 岭回归校准（备用路径，T1 达标未启用）、retrieve top-k 检索、align_eval 评测；④perceive I1 容错入口（1001-1004 错误码+μ 降级矩阵+meta.missing）。  
  自测 **19/19 PASS**（`python src/multimodal/selftest.py`）：概念解码一致率 1.0（≥0.95✓）、eval 语义检索 top-1=1.0 / 样本级 top-3=0.9167（≥0.85✓）、图文 5 选 1=1.0、语气解码 1.0、分离度 0.731/0.0（>0.5✓/<0.2✓）、同输入逐位一致、三模态 perceive 0.3ms（<100ms✓）、T0 兜底、M4 dispatch 集成冒烟。demo 回归 14/14 PASS、M1 自测回归 11/11 PASS，M4 零改动。  
  实现口径记录（对齐 multimodal-design §6.5 降级约定）：**D1** 概念匹配取图像特征前 5 维（RGB+圆度+边缘密度）——后 3 维为数据管道声明的样本级自由风格维（data-pipeline §8），剔除后概念解码对风格噪声免疫（一致率 1.0）；**D2** 检索 top-1 双口径——eval 含同概念多图（48 样本/30 概念），掩码余弦下同概念样本本质并列，主口径=语义命中（top-1 概念正确，1.0），样本级 id 配对（0.5417）作参考，top-3（0.9167）含正确样本达标，T2 未触发；**D3** 任务书命名兼容层——vision_encoder≡image_encoder、aligner≡align（双命名均可 import，同 M1 的 reasoner 先例）；**D4** 代码量 914 核心行（预算 600，超支因 T2 求解器+双口径评测+兼容层并入单模块交付）。M1 演进位提示：M1 `_shared.py` 复刻的是 M2 旧桩哈希词袋，M2 换概念锚定嵌入后 M1 注意力 α 通道（cos）为跨空间弱信号、β 通道（token Jaccard）与规则通道仍语义有效，统一共享嵌入服务为 Phase 4 待办（阿瑟/莎拉联合）
- 2026-09-02 17:05 Phase3d 验收（皮皮）：PASS——独立复验自测 19/19、demo 回归 14/14、M1 回归 11/11 全绿；指标核验：概念一致率 1.0、语义 top-1=1.0/top-3=0.9167、图文 5 选 1=1.0、语气 1.0、性能 0.3ms。GOAL 验收 4（多模态模块+自测）达成。**Phase 3 收官**，M1/M2 嵌入统一作为可选项已注入 Phase 4 任务书；管道无阻塞，自动进入 Phase 4 系统集成（诺亚）
- 2026-09-02（时间未回写）Phase 4 系统集成完成（诺亚）：`src/api/router.py` 落地两项演进位——①**共享语义空间桥接**：M4 懒取 M2 `embed_text(query)` 包装为 `meta.role="query"` 观测注入 M1，在不违反「M1/M2 禁止互相 import」边界的前提下让注意力 α 通道使用真实语义嵌入，M2 缺席时 qemb=None 自动回退 M1 自建哈希（向后兼容）；②**会话状态持久化**：`SessionStore` JSONL 追加式存储 rounds/history（上限 50 轮），读写失败降级内存态（R4）。`demo/run_demo.py` 由骨架版 14 项结构自检升级为三场景真实度量版（A 多模态检索问答 / B 图文匹配 5 选 1 / C 语音意图与容错），**21/21 PASS、3/3 场景**，超 GOAL「≥2 场景」要求。**教训：Phase 4 完成后未回写 README 进度，导致文档状态落后实现（2026-09-02 19:20 由大米复核时发现并修复）**
- 2026-09-02 18:59 Phase 5 验收交付（林图灵 / 大米独立复核）：`docs/final-report.md` 发布（v1.0）。**验收结论：GOAL 六条 6/6 达成**。复核方式为不采信各模块自我声明，全部指标在目标环境（Python 3.13 / 纯 CPU / 无外网）独立复跑：demo 21/21、M1 自测 11/11、M2 自测 19/19、数据管道 12/12 断言；数据集 5 文件 sha256 前 16 位与 `stats.json`/`quality_report.md` §9 声明逐项匹配（文件未篡改）；端到端 20 轮 avg 62.8ms（p95 72.7ms，预算 5000ms）；记忆增强效应同问 4 轮 conf 0.7176→0.7462 单调不减；错误码 1004/4001/1001-1002 降级/3002 路径全通；模态缺失 4 组合均不崩溃。报告 §6 主动披露 8 项局限与架构债（L1 封闭世界假设 / L2 零学习参数 / L3 感知层非真实信号 / L4 推理仅 11 条规则 / **L5 M1-M2 嵌入空间未统一** / L6 demo 汇总硬编码 / L7 三对冗余别名 / L8 长时程未验证），§7 给出 P0-P2 演进路线。**核心结论：系统价值在于验证了契约先行、降级内建、分层可替换三项架构主张，而非封闭合成数据上的高指标**
- 2026-09-02 19:09 Phase 5 全量测试与最终打包（大米）：补齐全量测试统一入口与交付清单。①新建 `tests/acceptance.py`：逐条核销 GOAL 六条，每条自带客观判据（文件存在性+结构核验+实测断言），子测试经 subprocess 隔离调用并**解析实际输出取证**（不采信模块自我声明），单项超时/崩溃记为 FAIL 且不中断其余项，退出码 0/1 可直接接 CI 门禁；实测 **6/6 PASS，exit 0**。②交付清单落 `final-report.md` §1.3：**44 文件 / 858.3 KB / 逐文件 sha256 前 16 位**，明确排除 `__pycache__`、运行时产物（memory/session_state）与平台预置目录三类非交付物；其中 `final-report.md` 自身哈希标注为**自指不可校验**（写入该行即改变自身哈希），其余 43 个可随时复算比对。  
  过程中修掉两个真实缺陷：**D1** 验收脚本正则 `运行成功场景数：(\d+)/(\d+)` 含两个捕获组，而解析器只取 `group(1)` 得到孤值 `"3"`，`split("/")` 解包失败导致验收 5 误判 FAIL（改为单捕获组 `(\d+/\d+)`）；**D2** manifest 分组逻辑用 if/elif 链且无兜底，`tests/` 目录不匹配任何分支会被**静默丢弃**（补 `tests/` 分支 + `setdefault("other")` 兜底）。经验：验收脚本自身也需被验收——它第一次运行时报出的 FAIL，一半是它自己的 bug
- 2026-09-02 19:09 时间戳更正：上一轮 README 记录的「19:20」为超前时间戳（实际 18:59），已修正为真实时刻。交付记录的时间戳必须真实，超前记录等同伪造进度
- 2026-09-02 19:31 编码缺陷修复与环境无关性加固（大米）：交付后在中文 Windows 环境复跑验收得 4/6（验收 4、5 FAIL），根因为**验收脚本未固定子进程 I/O 编码**——父进程 stdout 为 UTF-8，而 `subprocess` 管道取 locale 编码（GBK），子进程 `print("…✅")` 抛 `UnicodeEncodeError` 崩溃，被误判为被测模块失败。实测确认共 **4 个脚本**在 GBK 下崩溃（`src/multimodal/selftest.py`、`demo/run_demo.py`、`data/build_dataset.py`、`tests/acceptance.py` 自身）。修复两层：①验收脚本为子进程注入 `PYTHONIOENCODING=utf-8` 并以 UTF-8 解码输出，**父子两侧同时钉死**；②删除 4 个脚本 `print` 路径上的 GBK 不可编码字符（✅❌→`[OK]`/`[FAIL]`，⑪⑫→`11.`/`12.`），使项目在默认 GBK 控制台可直接运行。修法选择上**移除装饰字符而非给每个脚本加装 UTF-8 适配层**——emoji 对功能零贡献，为其新增 4 处环境兜底属为冗余实体续命。  
  **过程中大米犯了两个自身错误并已更正**：**E1** 对同一文件并发发起两个 Edit 调用产生「丢失更新」竞态，`❌→[FAIL]` 生效而 `✅→[OK]` 被覆盖丢失，导致修复看似完成实则残留（改为串行编辑后修复）；**E2** 中途依据一次临时哈希采集误判「5 个数据集文件被重跑改动」，经以报告 §1.3 记录值做字节数+哈希双重比对后确认**数据集从未变动**（raw/cleaned/train/eval/invalid 五项字节数与哈希全部与报告一致），虚报已更正——唯一真实变化是 `stats.json`（3,775 字节不变、哈希变更），源于断言名 `⑪`→`11.`（二者 UTF-8 均 3 字节，故长度巧合不变）。  
  连带修正报告三处失准数字：多模态模块 **1,113 → 1,133 行**（数字换位手误，2 处）、「项目目录 67 文件」→「44 个交付文件」（口径错误）、§1.3 清单 6 条过期哈希与 4 处分组小计重采集（858.3→865.4→**866.4 KB**）。修复后在 GBK 与 UTF-8 两种环境各跑一次验收，**均 6/6 PASS、exit 0**。经验：①验收工具的结论不得依赖运行它的终端，否则同一份代码会给出不同判定；②`stats.json` 的 sha256 审计是自指的（它声明自己刚写入文件的哈希），只能发现构建后被篡改，不能发现构建本身漂移，因此报告 §1.3 的独立清单才是真正的基线

## 文件结构索引

```
/home/z/my-project/
├── GOAL.md                 # 目标与红线（核心文件，勿加长文）
├── README.md               # 本文件：状态与索引
├── requirements.txt        # 依赖声明（诺亚，✅标准库优先/numpy 可选已装 2.1.3）
├── docs/                   # 设计文档
│   ├── architecture.md     # 总体架构（林图灵，✅已产出 v1.0）
│   ├── algorithm-design.md # 认知算法方案（阿瑟，✅已产出 v1.0）
│   ├── data-pipeline.md    # 数据管道方案（凯文，✅v1.1：Phase 3 实现交付 + §8 差异回写）
│   ├── multimodal-design.md# 多模态方案（莎拉，✅已产出 v1.0）
│   ├── api-spec.md         # 统一 API 规范（诺亚，✅已产出 v1.0：信封/注册/路由/错误码）
│   └── final-report.md     # 最终总结报告（林图灵，✅已产出 v1.0：6/6 验收核销 + 8 项局限披露 + P0-P2 演进路线）
├── src/
│   ├── __init__.py         # 源码包声明（诺亚，✅）
│   ├── cognition/          # 认知核心实现（阿瑟，✅Phase 3 完成，自测 11/11 PASS）
│   │   ├── __init__.py     # 导出 run_cognition/CognitionError（✅真实实现，替换桩，I2 签名冻结）
│   │   ├── memory.py       # B1~B4 三层记忆：JSONL 存储/关键词 kNN 检索/遗忘曲线衰减/巩固合并（✅）
│   │   ├── attention.py    # A1 注意力：词频(IDF)×相关性打分 + Top-K(8) + softmax（✅）
│   │   ├── reasoning.py    # C1 混合推理：规则前向链接+记忆证据+模板，三通道置信度仲裁（✅）
│   │   ├── reasoner.py     # C1 命名别名（对齐 algorithm-design §5.6，re-export reasoning）（✅）
│   │   ├── cognition.py    # 认知循环编排：run_cognition（工作记忆→推理→情景回写→巩固/衰减）（✅）
│   │   ├── rules.json      # 种子 IF-THEN 规则库（11 条，演示用）（✅）
│   │   ├── selftest.py     # M1 自测脚本：python src/cognition/selftest.py（✅11/11 PASS）
│   │   └── _shared.py      # 模块私有：分词/哈希嵌入/softmax/CognitionError（✅）
│   ├── multimodal/         # 多模态实现（莎拉，✅Phase 3 完成，自测 19/19 PASS）
│   │   ├── __init__.py     # 导出 perceive/embed_text/similarity（✅真实实现，替换桩，I1 签名冻结）
│   │   ├── concepts.py     # 概念词表 + T0 确定性映射表（30 组合概念/4 intent/3 tone）（✅）
│   │   ├── space.py        # 64 维共享空间：分区/带符号双槽哈希/固定投影/掩码余弦（✅）
│   │   ├── text_encoder.py # A1 文本编码：概念词典最大匹配 + idf 词袋哈希（✅）
│   │   ├── image_encoder.py# A2 图像编码：原型 top-2 概念软匹配 + 私有区投影（✅）
│   │   ├── vision_encoder.py # A2 命名别名（任务书口径，re-export image_encoder）（✅）
│   │   ├── audio_encoder.py# A3 语音编码：语气原型匹配 + 意图/语气锚（✅）
│   │   ├── aligner.py      # A4 跨模态对齐：T1 原型/T2 校准/掩码余弦检索/评测（✅）
│   │   ├── align.py        # A4 命名别名（设计文档口径，re-export aligner）（✅）
│   │   ├── perceive.py     # I1 感知入口 + 容错降级（1001-1004/μ 矩阵/meta.missing）（✅）
│   │   ├── prototypes.json # T1 原型统计缓存（selftest 生成；缺席 → T0 兜底）
│   │   └── selftest.py     # M2 自测脚本：python src/multimodal/selftest.py（✅19/19 PASS）
│   └── api/                # 统一调度与集成（诺亚）
│       ├── __init__.py     # M4 包声明（✅）
│       └── router.py       # dispatch 统一调度入口（✅Phase 4：路由表/懒导入+桩降级/I4/I5 + 共享空间桥接 + 会话持久化）
├── data/
│   ├── memory.jsonl        # M1 情景/语义记忆存储（阿瑟，✅运行时自动生成；另有 memory_archive.jsonl 归档）
│   ├── build_dataset.py    # 数据集一体化构建脚本（凯文，✅Phase 3 完成：生成→清洗→划分→评估+load_dataset I6，12/12 断言 PASS）
│   ├── dataset/            # 示例数据集（凯文，✅已产出，seed=42 可复现）
│   │   ├── raw.jsonl       # 合成原始样本（300 条，含 20% 注入噪声）
│   │   ├── cleaned.jsonl   # 清洗+归一化+打分样本（240 条）
│   │   ├── train.jsonl     # 训练集（192 条，quality≥0.6 双保险）
│   │   ├── eval.jsonl      # 评测集（48 条，含 robust 难子集 9 条）
│   │   ├── invalid.jsonl   # 校验/异常剔除隔离区（24 条，可追溯）
│   │   ├── schema.json     # 样本 JSON Schema（机器可读校验依据）
│   │   └── stats.json      # 管道统计+质量指标+文件 sha256 审计清单
│   └── quality_report.md   # 数据质量评估报告（凯文，✅已产出：规模/模态/去重/完整性/抽检样例）
├── demo/
│   └── run_demo.py         # 一键演示入口（诺亚，✅Phase 4 三场景版：A 检索问答/B 图文匹配/C 语音容错，21/21 PASS）
├── tests/
│   └── acceptance.py       # Phase 5 一键验收：GOAL 六条逐条核销（✅6/6 PASS，exit 0；--json 输出交付清单+sha256）
└── agents/, skills/, download/  # 平台预置目录（勿动）
```
