# AGI-Core 认知核心算法设计方案

> 版本：v1.0 ｜ 作者：阿瑟（算法研究员）｜ 日期：2026-09-02
> 状态：Phase 2 交付物（算法线）｜ 上游依据：`docs/architecture.md` v1.0 §3 接口契约
> 实现目录：`src/cognition/` ｜ 约束：本地 CPU、无外网、无外部 API、标准库+numpy（不可用则纯 Python）

## 0. 设计定位
M1 认知核心 = **注意力（筛选）+ 记忆（存取/巩固）+ 推理（混合）** 三个子模块 + 认知循环编排。
- 默认实现走 CPU 简化路线（对齐架构 R1/R5 降级：零参数打分、暴力 kNN、容量上限 1000）；
- 全模块零训练参数、纯确定性推理，保证结果可复现、决策可解释（trace 全程保留）；
- 保留演进位：接口不变，嵌入与推理通道未来可替换为可学习实现或本地小模型。

## 1. 注意力机制设计（attention.py）
### 1.1 简化打分模型（无训练参数）
对查询 q 与候选观测 o（均来自 I1 Observation，含 tokens 与 64 维 embedding）：

score(q, o) = α·cos(e_q, e_o) + β·overlap(T_q, T_o) + γ·exp(-Δt/τ) + δ·salience(o)

- 词项重叠（IDF 加权 Jaccard）：
  overlap(q,o) = Σ_{w∈Tq∩To} idf(w) / Σ_{w∈Tq∪To} idf(w)，其中 idf(w)=log((N+1)/(df(w)+1))+1；
- 时间新鲜度：Δt = now − ts(o)，τ=3600s（近期输入加权，模拟注意瞬态衰减）；
- salience(o)：模态先验（文本 1.0/图像 0.9/语音 0.85）× 实体密度（专名、数字、大写词占比）；
- 默认权重 α=0.45, β=0.30, γ=0.15, δ=0.10（演示场景可配置，存 cfg）。

### 1.2 Top-K 硬筛选 + softmax 加权（伪代码）
```
attend(query, obs_list, K=8):
    for o in obs_list: s[o] = score(query, o)      # 逐项打分
    focus = topK(obs_list, s, K)                    # 硬筛选（控制推理上下文长度）
    w = softmax(s[focus] / T),  T=0.5               # 温度归一化权重（聚焦 vs 均衡）
    return FocusSet{items: focus, weights: w, scores: s[focus]}
```

### 1.3 与 Transformer 注意力的关系（取舍说明）
- 以**确定性打分函数**替代可学习 QK 投影：QKᵀ/√d → 余弦+词项重叠，零参数零训练；
- 保留 Top-K + softmax 结构形态，行为可类比单头注意力（对齐架构 R1 降级路线）；
- 演进位：嵌入升级为可学习后，score 的 cos 项可直接替换为 scaled dot-product（W_Q=W_K=I 起步即等价）。

## 2. 记忆系统设计（memory.py）
### 2.1 三层记忆（对齐 I3 MemoryItem.type）
| 层 | type | 存储 | 容量 | 生命周期 |
|---|---|---|---|---|
| 工作记忆 | working | 内存 deque（会话级） | W=16 | 会话内轮次间滑动淘汰 |
| 情景记忆 | episodic | `data/memory.jsonl`（追加写） | ≤800 | 衰减式，弱记忆归档 |
| 语义记忆 | semantic | `data/memory.jsonl`（同文件按 type 区分） | ≤200 | 长期，衰减极慢 |

MemoryItem 完全对齐 I3：`{mem_id, type, content, embedding, ts, hits}`；另加可选字段 `strength, consolidated, links[]`（均不影响契约兼容）。

### 2.2 检索策略（暴力 kNN，对齐 R5）
```
mem_recall(query, k=5, mem_type=None):
    e_q = embed(query)                        # 与 M2 共用 64 维嵌入空间
    for m in load(mem_type):                  # JSONL 全量加载（总量 ≤1000）
        s[m] = λ1·cos(e_q, e_m) + λ2·overlap(q, m.content)
              + λ3·recency(m) + λ4·norm_hits(m)     # 四因子混合分
    top = topK(s, k);  hits[top] += 1         # 召回计数反哺重要性
    return top
```
复杂度 O(N·d)：N≤1000、d=64，numpy 下单次 <5ms，纯 Python <50ms，满足演示基线。

### 2.3 记忆衰减（类 Ebbinghaus 遗忘曲线）
强度随时间指数衰减：**strength(m, t) = S₀ · exp(−(t − m.ts)/τ_type)**
- τ_working=600s，τ_episodic=7d，τ_semantic=365d（对应不同巩固水平）；
- 召回命中即强化：strength ← min(strength + 0.3, 2.0)（「越用越牢」）；
- 情景记忆 strength < θ_decay=0.2 → 移入 `data/memory_archive.jsonl`（归档不删除，可追溯）。

### 2.4 巩固机制（episodic → semantic，类睡眠系统巩固）
- 触发条件（任一）：①某情景 hits ≥ 3；②同关键词簇（簇内 embedding 余弦 >0.75）情景数 ≥ 5；
- 操作：新建语义记忆，content = 模板化摘要（簇高频词+中心句），embedding = 簇均值归一化，ts=now；源情景标记 `consolidated=true` 退出召回池（保留在文件中）；
- 运行时机：每轮认知循环结束后异步执行，不阻塞主链路。

### 2.5 存储与容量控制
- 主存储 JSONL 追加写 + 定期 compaction（去重、分离已巩固条目，批量写回）；
- SQLite 为演进选项（数据量 >10⁴ 条时切换，仅替换 load/save 实现，接口不变）；
- 总量 >1000（对齐 R5）：按 strength 升序淘汰至达标（working 优先淘汰）。

## 3. 推理引擎设计（reasoner.py）
### 3.1 混合推理三通道（伪代码）
```
reason(obs[], query, session_id) -> Thought:      # 签名对齐 I2
    E = mem_recall(query, k=5)                    # 通道0：检索增强（RAG-lite）
    F = attend(query, obs)                        # 注意力聚焦集
    ch_rule = match_rules(E ∪ F, rules.json)      # 通道1：规则推理（前向链接）
    ch_mem  = vote(E, F)                          # 通道2：证据加权投票
    ch_tpl  = template(query, E)                  # 通道3：模板回退（含澄清追问）
    answer, conf, steps = arbitrate(ch_rule, ch_mem, ch_tpl)  # 仲裁融合
    return Thought{steps, answer, confidence}
```
- 通道1 规则推理：语义记忆及种子 `rules.json` 中 IF-THEN 规则（pattern→conclusion, weight）；pattern 为关键词/正则，前向链接一次匹配即触发，链深 ≤3（防循环）；
- 通道2 证据投票：answer = argmax_c Σ_{m∈E} w_m·sim(q,m)·1[m 支持 c]，近似 kNN 分类；
- 通道3 模板回退：E 为空或最高相似度 <θ=0.2 时启用，产出澄清追问或「基于记忆的保守回复」。

### 3.2 仲裁与置信度
conf = (w_r·C_rule + w_m·C_mem + w_t·C_tpl) / (w_r + w_m + w_t)
- C_rule = rule.weight × pattern 覆盖率；C_mem = Σw_m·sim 归一化至 [0,1]；C_tpl = 0.3（保守常数）；
- 默认 w_r=0.5, w_m=0.35, w_t=0.15（规则优先，检索次之，模板兜底）；
- 门控对齐 I4：conf < 0.6 → Action=clarify（追问用户）而非强行作答。

### 3.3 与端到端大模型的差异与取舍
| 维度 | 端到端 LLM | 本方案（规则+检索混合） | 取舍结论 |
|---|---|---|---|
| 生成能力 | 强（开放域流畅生成） | 弱（受规则/模板/证据约束） | 放弃流畅性换取低幻觉：答案必有证据 trace |
| 知识更新 | 需重训/微调权重 | 记忆即时写入，分钟级生效 | 取即时性：认知在线学习、零训练 |
| 可解释性 | 弱（隐式表征） | 强（steps 全程显式） | 满足「全程可追溯」架构原则 |
| 资源开销 | GPU、GB 级 | CPU、MB 级、<5s/轮 | 契合本地无 GPU 红线 |
| 泛化能力 | 强 | 依赖规则覆盖与记忆积累 | 靠巩固机制持续蒸馏情景→语义，渐进提升 |
| 演进路径 | — | infer 通道可替换为本地小模型 | I2 接口不变，替换不动架构 |

## 4. 认知循环数据流（cognition.py 编排）
### 4.1 单轮循环（一次 dispatch 的完整流转）
```
loop(request):
    1 感知     obs[] = perceive(inputs)                # I1（M2 产出 Observation）
    2 记忆写入 for o in obs: mem_save(working, o)      # 写工作记忆（含 64 维嵌入）
    3 注意力   F = attend(query, obs[] ∪ working)      # Top-K 聚焦集
    4 推理     E = mem_recall(query); thought = infer(E, F, rules)  # 产出 Thought（I2）
    5 决策     action = plan(thought)                  # I4：置信度门控 0.6
    6 输出     resp = render(action)                   # I5
    7 回写     mem_save(episodic, {query, answer, conf})  # 本轮 Q/A 沉淀为情景记忆
    8 巩固     consolidate(); decay(); compaction()    # 异步：巩固+衰减+压实
    return resp
```
- 闭环关键在第 7 步：系统具备「对话即学习」能力，多轮后同类问题命中率与 conf 上升——这是演示场景②（记忆增强对话）的算法基础；
- 失败路径：检索为空 → 模板通道兜底；conf<0.6 → clarify 追问；子模块异常 → 信封携带 2xxx 错误码返回，不阻塞整体（对齐 R4）。

### 4.2 循环数据流图（对齐 architecture.md §1.2）
```
输入 ─→ 感知(I1) ─→ [写工作记忆] ─→ 注意力(Top-K) ─→ 推理(检索+规则+模板)
   ↑                                              │ Thought(I2)
   └── [情景回写+巩固/衰减] ←── 输出(I5) ←── 决策(I4 门控) ←┘
```

## 5. 子模块接口定义（对齐 architecture.md §3）
统一信封（§3.1）与 I2/I3 完全复用不再重复；以下为认知层内部子接口（本地函数调用，入/出参均可 JSON 序列化）。

### 5.1 attention.py — A1 attend
`attend(query: str, obs: Observation[], k=8, cfg) -> FocusSet`
出参 FocusSet：`{"items": ["obs-001", ...], "weights": [0.31, ...], "scores": [0.83, ...], "k": 8}`
items 引用 obs_id，供 I2 Thought.steps[].focus_obs 直接引用。

### 5.2 memory.py — B1~B4
- `B1 mem_save(item: MemoryItem) -> {"mem_id", "ok": true}`（type ∈ working|episodic|semantic）
- `B2 mem_recall(query: str, k=5, mem_type=None) -> MemoryItem[]`（对齐 I3）
- `B3 consolidate(now=None) -> {"created": n, "archived": n}`（情景→语义巩固）
- `B4 decay(now=None) -> {"archived": n, "boosted": n}`（衰减与归档）
存储文件：`data/memory.jsonl`（主）、`data/memory_archive.jsonl`（归档）。

### 5.3 reasoner.py — C1 reason
`reason(obs: Observation[], query: str, session_id: str) -> Thought`（签名与 I2 逐字对齐）
Thought.steps 严格使用 I2 的 op 枚举 `recall → attend → infer`（对应 §4.1 第 3~4 步）；`rule` 字段填触发规则 id 或 "evidence-vote" / "template"。

### 5.4 cognition.py — 编排入口（供 M4 调用）
`run_cognition(obs: Observation[], query: str, session_id: str) -> Thought`
即 I2 的实现函数。M4 只依赖本入口与 B1（回写），禁止 import 子模块内部符号。

### 5.5 认知层错误码（2xxx，对齐 §3.1）
2001 注意力输入为空/非法；2002 记忆文件读写失败；2003 推理无可用证据且模板不可用；2004 置信度门控未过转 clarify（正常流，非故障）。

### 5.6 模块文件结构
```
src/cognition/
├── __init__.py     # 导出 run_cognition
├── attention.py    # A1（打分/Top-K/softmax）
├── memory.py       # B1~B4（JSONL 存储、衰减、巩固）
├── reasoner.py     # C1（规则+检索+模板三通道）
├── cognition.py    # 认知循环编排（§4）
└── rules.json      # 种子 IF-THEN 规则库（演示用，≤20 条）
```

## 6. 自测与实验设计（Phase 3 验收口径）
1. 单元测试：①注意力排序（构造已知相关/无关观测验证 Top-K 序）；②衰减单调性（strength 随 t 单调不增）；③巩固触发（hits≥3 或簇≥5 生成语义记忆）；④门控行为（conf<0.6 → clarify）；
2. 冒烟测试：模拟 3 轮同主题会话，第 2/3 轮应召回第 1 轮情景记忆且 conf 提升（记忆增强可量化验证）；
3. 性能基线：1000 条记忆暴力 kNN <50ms/次；单轮认知循环总延迟 <5s（对齐 architecture.md 演示基线）；
4. 消融建议：分别关闭注意力/记忆/巩固通道跑对照，由 trace 统计各通道贡献率。

---
*本文档与 architecture.md v1.0 §3 契约逐条对齐；契约变更须经首席架构师评审并同步版本号。Phase 3 编码实现以本文 §5 接口为准。*
