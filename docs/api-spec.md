# AGI-Core 统一 API 接口规范

> 版本：v1.0 ｜ 作者：诺亚（系统集成工程师）｜ 日期：2026-09-02
> 状态：Phase 3 交付物（工程骨架线）｜ 上游依据：`docs/architecture.md` v1.0 §3、`docs/algorithm-design.md` §5、`docs/multimodal-design.md` §4、`docs/data-pipeline.md` §6
> 下游读者：阿瑟（M1 实现）、莎拉（M2 实现）、凯文（M3 实现）、皮皮（编排）
> 实现位置：`src/api/router.py`（统一调度入口）｜ 本规范对四线设计文档的接口契约（I1-I6）做**工程落地版汇总**，冲突时以 architecture.md 为准

## 0. 总则

1. **本地函数即 API**：所有跨模块调用为本地 Python 函数调用，入参/出参均为**可 JSON 序列化对象**（dict/list/str/num/bool/None）；预留 HTTP POST 适配位，签名不变（对齐 architecture §3.1）。
2. **模块边界**：M1/M2/M3 之间**禁止互相 import**，一律经 M4（`src/api/router.py`）调度或文件传递。
3. **模块可缺席**：M4 对每个模块做懒导入 + 桩函数降级（对齐 R4：单模块故障不阻塞整体演示）；模块未实现时自动回退到 `__init__.py` 中的接口桩。
4. **纯 CPU / 标准库优先**：依赖限定标准库 + numpy（环境已装 2.1.3，可选使用；缺失则纯 Python 路径，不阻塞）。
5. **版本化**：信封与 Response 均携带 `version: "1.0"`；格式升级只增字段不删字段（向后兼容）。

## 1. 消息格式（统一 JSON 信封）

所有**层间/模块间**消息一律采用如下信封（对齐 architecture.md §3.1，逐字段复用）：

```json
{
  "msg_id": "uuid4 字符串",
  "ts": "ISO8601 带时区，如 2026-09-02T15:30:00+08:00",
  "src": "发送方模块名",
  "dst": "接收方模块名",
  "type": "消息类型（observation / thought / action / response / request / dataset）",
  "version": "1.0",
  "payload": {},
  "error": null
}
```

字段规则：
- `src/dst` 取值：`perception`（M2）/ `cognition`（M1）/ `decision`（M4 决策）/ `output`（M4 渲染）/ `api`（M4 入口）/ `data`（M3）；
- `error` 仅在失败时非空：`{"code": <int>, "msg": <str>}`，code 取值见 §4；
- `payload` 为各契约的结构化数据（I1 Observation / I2 Thought / I4 Action 等，见 §3）。

**I1 Observation（M2 产出，M1 消费）**：
```json
{"obs_id": "obs-001", "modality": "text|image|audio",
 "embedding": [64×float, L2 归一化],
 "tokens": ["蓝色", "三角"],
 "meta": {"dim": 64, "confidence_factor": 0.9, "missing": ["audio"],
          "salience_prior": 1.0, "source": "synthetic", "...": "可选扩展字段"}}
```

**I2 Thought（M1 产出）**：
```json
{"thought_id": "th-001",
 "steps": [{"step": 1, "op": "recall", "used_mem": ["mem-007"]},
           {"step": 2, "op": "attend", "focus_obs": ["obs-001"]},
           {"step": 3, "op": "infer", "rule": "chain-of-thought"}],
 "answer": "初步结论", "confidence": 0.82}
```
`op` 枚举固定为 `recall → attend → infer` 三步（对齐 algorithm-design §5.3）。

**I4 Action（M4 决策层内部）**：
```json
{"action_type": "reply|tool|clarify|none",
 "payload": {"text": "..."}, "gate": {"confidence": 0.82, "threshold": 0.6}}
```

**I6 Sample（M3 产出）**：`{"id", "modality", "input", "expected", "quality"}`，5 字段顶层（对齐 data-pipeline §3.1）。

## 2. 模块注册方式

M4 内置**路由表 + 注册器**（`src/api/router.py`），各模块按「路由名 → 函数」注册，由 M4 懒加载：

### 2.1 路由表（ROUTE_TABLE，M4 骨架已内置）

| 路由名 | 阶段 stage | 目标模块 | 绑定函数（懒导入） | 契约 |
|---|---|---|---|---|
| `perceive` | 1 感知 | M2 `src/multimodal` | `perceive(inputs: ModalInput[]) -> Observation[]` | I1 |
| `cognition` | 2 认知 | M1 `src/cognition` | `run_cognition(obs, query, session_id) -> Thought` | I2 |
| `plan` | 3 决策 | M4 内部 | `plan(thought: dict) -> Action` | I4 |
| `render` | 4 输出 | M4 内部 | `render(action, ctx) -> Response` | I5 |
| `load_dataset` | 数据 | M3 `data/build_dataset.py` | `load_dataset(split, data_dir) -> Sample[]` | I6 |

### 2.2 注册 API（供后续模块接入/测试替身使用）

```python
from src.api.router import registry, register, dispatch

register(name="perceive", fn=my_perceive_fn, stage=1, overwrite=True)  # 注册/替换
registry.get("perceive")   # 取绑定（缺席返回 None → 走桩降级）
registry.list_modules()    # [("perceive", "src.multimodal.perceive", "1 感知"), ...]
```

注册规则：
1. `register()` 的 `fn` 必须是**纯函数**（入/出参可 JSON 序列化，无全局可变状态）；
2. 同名重复注册默认拒绝，`overwrite=True` 方可覆盖（用于测试替身/模块升级）；
3. 模块实现迁移进 `src/` 后**无需显式注册**：M4 首次 `dispatch` 时按 ROUTE_TABLE 懒导入真实实现，导入失败或函数缺失 → 自动回退该模块 `__init__.py` 接口桩（骨架期行为），并在 Response 的 `trace` 中标记 `stub`；
4. 模块实现成熟后应自测通过再合入（对齐 Phase 3 验收），M4 不感知内部细节，只认函数签名。

## 3. 调用协议（函数签名 / 路由规则）

### 3.1 对外唯一入口（M4 → 用户/调用方）

```python
from src.api.router import dispatch
response = dispatch(request)
```

**request（入参）**：
```json
{"session_id": "sess-001",
 "mode": "standard|fast",
 "inputs": [{"modality": "text|image|audio", "uri": "可选本地路径", "raw": "可选内联内容"}]}
```

**response（出参，对齐 architecture §3.3）**：
```json
{"output": "最终回复文本",
 "confidence": 0.82,
 "trace": ["感知(1 obs)", "认知(3 steps)", "决策(reply)", "..."],
 "latency_ms": 12.3,
 "version": "1.0",
 "error": null}
```

### 3.2 dispatch 内部路由规则（四层顺序编排）

```
dispatch(request):
  ① 校验      request 格式（session_id/inputs 必填，mode 默认 standard）→ 失败 4001
  ② 感知      obs[] = perceive(inputs)                 # 路由名 perceive，I1
              query := inputs 中首个 modality=text 的 raw（无文本则 ""）
  ③ 认知      thought = run_cognition(obs, query, session_id)   # I2
  ④ 决策      action = plan(thought)                   # I4：confidence<0.6 → clarify
  ⑤ 输出      response = render(action, ctx)           # I5：封装 output/confidence/trace/latency_ms
```

- 每步产出均经信封封装（`src/dst/type` 按表填写），模块异常时信封带对应错误码，**不中断整体**（降级为带错误信息的标准 Response 返回）；
- `mode`：`standard` 完整链路；`fast` 骨架期行为相同，预留跳过非关键步骤的开关（trace 中记录 mode）；
- 延迟预算：单次 dispatch 全链路 < 5s（对齐演示基线），`latency_ms` 由 M4 统一计时。

### 3.3 各模块函数签名汇总（I1-I6 工程落地版）

| # | 函数签名 | 归属 | 状态 |
|---|---|---|---|
| I1 | `perceive(inputs: list[dict]) -> list[dict]` | M2 `src/multimodal/__init__.py` | 桩已立（莎拉 Phase 3 实现） |
| I1' | `embed_text(text: str) -> list[float]`（64 维，供 M1 记忆检索共用） | M2 同上 | 桩已立 |
| I2 | `run_cognition(obs: list[dict], query: str, session_id: str) -> dict` | M1 `src/cognition/__init__.py` | 桩已立（阿瑟 Phase 3 实现） |
| I3 | `mem_save(item) / mem_recall(query, k, mem_type)` | M1 内部 `memory.py` | M1 内部接口，M4 不直接调用 |
| I4 | `plan(thought: dict) -> dict` | M4 `src/api/router.py` | 骨架已实现（门控阈值 0.6） |
| I5 | `render(action: dict, ctx: dict) -> dict` | M4 `src/api/router.py` | 骨架已实现 |
| I6 | `load_dataset(split: str, data_dir: str) -> list[dict]` | M3 `data/build_dataset.py` | 桩已立（凯文 Phase 3 实现） |

> 桩函数约定：签名与真实实现**逐字一致**；返回值符合契约字段结构（值可为占位）；docstring 首行标注 `STUB`，待实现方替换。这保证 M4 集成联调在模块完成前即可全链路跑通（对齐 R4）。

## 4. 错误码约定

统一格式：`error = {"code": <int>, "msg": <str>}`；`code=0` 或 `error=null` 表示成功。

| 段 | 归属 | 明细 |
|---|---|---|
| **0** | 成功 | `0 = OK` |
| **1xxx** | 感知层 M2 | 1001 modality 非法；1002 特征解码失败；1003 嵌入生成异常；1004 inputs 为空（整体失败）；1005 原型/idf 资源缺失（降级 T0 警告）｜来源：multimodal-design §4.3 |
| **2xxx** | 认知层 M1 | 2001 注意力输入为空/非法；2002 记忆文件读写失败；2003 推理无可用证据且模板不可用；2004 置信度门控未过转 clarify（正常流，非故障）｜来源：algorithm-design §5.5 |
| **3xxx** | 数据管道 M3 | 3001 样本格式校验失败；3002 split 参数非法；3003 数据文件缺失/解析失败；3004 quality 聚合校验不达标｜来源：data-pipeline §6 |
| **4xxx** | API/集成 M4 | **4001** request 格式非法（缺 session_id/inputs 结构错）；**4002** 未知路由/模块未注册；**4003** 模块调用异常（未捕获异常兜底）；**4004** 信封/Response 封装失败；**4005** 声明的外部依赖缺失（如 numpy 强依赖路径）｜本文档新增 |

错误传播规则：
1. 模块内部错误**不抛裸异常**出模块边界（对齐 R4），一律封装进信封 `error` 字段返回 M4；
2. M4 收到带错误的模块信封：`1xxx/2xxx` → 记入 trace 并降级继续（或走 clarify 路径）；`3xxx`（数据供给）→ 直接透传给调用方；`4xxx` → M4 自身错误，直接返回；
3. M4 对未捕获异常兜底捕获（4003），保证 `dispatch` **永不向调用方抛异常**，始终返回结构化 Response。

## 5. 目录结构与运行方式

```
src/api/router.py          # 本规范实现：dispatch/registry/register/plan/render
src/cognition/__init__.py  # M1 桩：run_cognition（I2）
src/multimodal/__init__.py # M2 桩：perceive/embed_text（I1）
demo/run_demo.py           # 一键演示：调 dispatch 跑通全链路 → 输出「骨架就绪」
requirements.txt           # 依赖声明：标准库优先，numpy 可选
```

运行：
```bash
cd /home/z/my-project && python demo/run_demo.py     # 骨架自测（输出「骨架就绪」+ Response 结构校验）
python -c "from src.api.router import dispatch; print(dispatch({'session_id':'s1','mode':'standard','inputs':[{'modality':'text','raw':'你好'}]}))"
```

## 6. Phase 3→4 演进约定

1. 阿瑟/莎拉/凯文实现各自模块时，**替换桩为真实实现、签名不变**；自测通过后在 README 记录，M4 侧零改动；
2. M4 将在 Phase 4 补充：会话管理（session 状态持久化）、认知循环第 7/8 步（情景回写+巩固触发，对齐 algorithm-design §4.1）、`HTTP POST` 适配层（演进位）；
3. 契约变更须经首席架构师（林图灵）评审并同步 architecture.md 与本文档版本号。

---
*本文档为 M4 工程基线，与 architecture.md v1.0 §3 契约逐条对齐；I1/I2/I6 字段定义以其上游设计文档为准，此处不重复展开全部字段语义。*
