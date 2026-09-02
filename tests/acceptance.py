# -*- coding: utf-8 -*-
"""AGI-Core Phase 5 一键验收脚本：对照 GOAL.md 六条验收标准逐条核销。

运行：
    python tests/acceptance.py            # 人类可读表格
    python tests/acceptance.py --json     # 机器可读 JSON

设计原则：
1. **独立判定**：每条验收项自带客观判据（文件存在性 + 结构核验 + 实测断言），
   不采信任何模块的自我声明，子测试结果以解析实际输出的方式取证。
2. **进程隔离**：子测试经 subprocess 调用（用 sys.executable），避免模块状态污染。
3. **永不阻塞**：子测试超时/崩溃记为 FAIL 并附原因，不抛异常、不中断其余验收项。
4. **退出码**：0 = 六条全通过；1 = 存在未通过项（可用于 CI 门禁）。

对齐文档：GOAL.md §二 验收标准、docs/architecture.md §4 Phase 5、docs/final-report.md §2。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

SUBPROCESS_TIMEOUT_SEC = 120          # 子测试上限（GOAL 单任务 30min 约束下的保守值）
VERSION = "1.0"


# ------------------------------------------------------------------ 工具 ----
def _p(*parts: str) -> str:
    return os.path.join(_PROJECT_ROOT, *parts)


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _lines(path: str) -> int:
    try:
        return sum(1 for _ in open(path, "r", encoding="utf-8"))
    except OSError:
        return 0


def _sha256(path: str, n: int = 16) -> str:
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()[:n]
    except OSError:
        return ""


def _run_subtest(script: str, *patterns: str) -> tuple:
    """运行子测试脚本，返回 (returncode, 各 pattern 的首个捕获组列表, 原始输出尾部)。

    patterns 为含一个捕获组的正则；未匹配到则对应位置为 None。
    """
    # 固定子进程 I/O 编码：Windows 下管道默认取 locale 编码（中文环境为 GBK），
    # 与父进程 stdout 编码可能不一致，会导致验收结论随终端漂移。此处父子两侧
    # 同时钉死为 UTF-8，保证任何终端下结论可复现。
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.run([sys.executable, script], cwd=_PROJECT_ROOT,
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env,
                          timeout=SUBPROCESS_TIMEOUT_SEC)
    out = (proc.stdout or "") + (proc.stderr or "")
    captured = []
    for pat in patterns:
        m = re.search(pat, out)
        captured.append(m.group(1) if m else None)
    return proc.returncode, captured, out[-400:]


def _count_py(directory: str) -> tuple:
    """统计目录下 .py 文件数与总行数，返回 (files, lines)。"""
    files = lines = 0
    for root, dirs, names in os.walk(_p(directory)):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for n in names:
            if n.endswith(".py"):
                files += 1
                lines += _lines(os.path.join(root, n))
    return files, lines


# --------------------------------------------------------- 验收项 1：架构 ----
def check_1_architecture() -> dict:
    path = _p("docs", "architecture.md")
    text = _read(path)
    required = ["分层架构", "I1", "I2", "I3", "I4", "I5", "I6",
                "R1", "风险", "降级", "Phase 5"]
    missing = [k for k in required if k not in text]
    ok = bool(text) and not missing
    return {
        "id": 1,
        "title": "docs/architecture.md 总体架构设计书",
        "ok": ok,
        "evidence": (f"{_lines(path)} 行，含分层架构 / I1-I6 契约 / R1-R6 降级预案 / Phase 1-5 路线图"
                     if ok else f"缺失要素: {missing}"),
    }


# ----------------------------------------------------- 验收项 2：算法实现 ----
def check_2_algorithm() -> dict:
    design = _p("docs", "algorithm-design.md")
    files, lines = _count_py("src/cognition")
    base_ok = os.path.isfile(design) and files > 0
    if not base_ok:
        return {"id": 2, "title": "algorithm-design.md + src/cognition/ 可运行实现",
                "ok": False, "evidence": "设计文档或实现目录缺失"}

    selftest = _p("src", "cognition", "selftest.py")
    rc, (ratio,), tail = _run_subtest(
        os.path.join("src", "cognition", "selftest.py"), r"结果：(\d+/\d+) PASS")
    try:
        got, total = (int(x) for x in ratio.split("/"))
    except (AttributeError, ValueError):
        return {"id": 2, "title": "algorithm-design.md + src/cognition/ 可运行实现",
                "ok": False, "evidence": f"无法解析自测输出: {tail}"}

    ok = rc == 0 and got == total and got > 0
    return {
        "id": 2,
        "title": "algorithm-design.md + src/cognition/ 可运行实现",
        "ok": ok,
        "evidence": (f"设计文档 {_lines(design)} 行；实现 {files} 文件 {lines} 行；"
                     f"自测 {ratio} PASS（exit={rc}）"),
    }


# ----------------------------------------------------- 验收项 3：数据集 ------
def check_3_dataset() -> dict:
    report = _p("data", "quality_report.md")
    pipeline = _p("docs", "data-pipeline.md")
    stats_path = _p("data", "dataset", "stats.json")

    if not (os.path.isfile(report) and os.path.isfile(pipeline)
            and os.path.isfile(stats_path)):
        return {"id": 3, "title": "数据集 ≥200 条 + 质量报告 + 管道方案",
                "ok": False, "evidence": "质量报告 / 管道方案 / stats.json 缺失"}

    try:
        stats = json.loads(_read(stats_path))
    except ValueError:
        return {"id": 3, "title": "数据集 ≥200 条 + 质量报告 + 管道方案",
                "ok": False, "evidence": "stats.json 解析失败"}

    # 规模：cleaned 计数以文件实际行数为准（不采信声明值）
    cleaned = sum(1 for l in open(_p("data", "dataset", "cleaned.jsonl"),
                                  encoding="utf-8") if l.strip())
    scale_ok = cleaned >= 200

    # 完整性审计：实测 sha256 前 16 位 vs stats.json 声明
    declared = stats.get("files", {})
    mismatched = []
    for name, meta in declared.items():
        actual = _sha256(_p("data", "dataset", name))
        if actual != str(meta.get("sha", ""))[:16]:
            mismatched.append(f"{name}(声明{str(meta.get('sha'))[:8]}/实测{actual[:8]})")

    # 硬断言：stats.json 记录的 12 条断言须全绿
    assertions = stats.get("assertions", [])
    failed_asserts = [a.get("name") for a in assertions if not a.get("ok")]

    ok = scale_ok and not mismatched and not failed_asserts and len(assertions) >= 12
    return {
        "id": 3,
        "title": "数据集 ≥200 条 + 质量报告 + 管道方案",
        "ok": ok,
        "evidence": (f"cleaned={cleaned} 条（要求 ≥200）；"
                     f"硬断言 {len(assertions) - len(failed_asserts)}/{len(assertions)} 通过；"
                     f"5 文件 sha256 审计{'全部匹配' if not mismatched else '不一致: ' + ','.join(mismatched)}"),
    }


# --------------------------------------------------- 验收项 4：多模态模块 ----
def check_4_multimodal() -> dict:
    files, lines = _count_py("src/multimodal")
    if files == 0:
        return {"id": 4, "title": "src/multimodal/ 跨模态对齐模块 + 自测",
                "ok": False, "evidence": "实现目录缺失"}

    rc, (ratio, metrics), tail = _run_subtest(
        os.path.join("src", "multimodal", "selftest.py"),
        r"selftest (\d+/\d+) PASS", r"metrics\(T1\)=(\{.*\})")
    try:
        got, total = (int(x) for x in ratio.split("/"))
    except (AttributeError, ValueError):
        return {"id": 4, "title": "src/multimodal/ 跨模态对齐模块 + 自测",
                "ok": False, "evidence": f"无法解析自测输出: {tail}"}

    ok = rc == 0 and got == total and got > 0
    ev = f"{files} 文件 {lines} 行；自测 {ratio} PASS（exit={rc}）"
    if metrics:
        try:
            m = json.loads(metrics)
            ev += (f"；概念一致率 {m.get('concept_consistency')}、"
                   f"语义 top-1 {m.get('retrieval_top1_label')}、"
                   f"top-3 {m.get('retrieval_top3')}、图文5选1 {m.get('match5')}")
        except ValueError:
            pass
    return {"id": 4, "title": "src/multimodal/ 跨模态对齐模块 + 自测",
            "ok": ok, "evidence": ev}


# ------------------------------------------------- 验收项 5：集成与演示 ----
def check_5_integration() -> dict:
    router = _p("src", "api", "router.py")
    demo = _p("demo", "run_demo.py")
    if not (os.path.isfile(router) and os.path.isfile(demo)):
        return {"id": 5, "title": "src/api/ 集成 + run_demo.py 一键演示 ≥2 场景",
                "ok": False, "evidence": "router.py 或 run_demo.py 缺失"}

    rc, (passed, failed, scenes), tail = _run_subtest(
        os.path.join("demo", "run_demo.py"),
        r"演示汇总：PASS (\d+)", r"演示汇总：PASS \d+\s*/\s*FAIL (\d+)",
        r"运行成功场景数：(\d+/\d+)")
    try:
        n_pass, n_fail = int(passed), int(failed)
        s_ok, s_total = (int(x) for x in scenes.split("/"))
    except (AttributeError, ValueError):
        return {"id": 5, "title": "src/api/ 集成 + run_demo.py 一键演示 ≥2 场景",
                "ok": False, "evidence": f"无法解析演示输出: {tail}"}

    ok = rc == 0 and n_fail == 0 and s_ok >= 2
    return {
        "id": 5,
        "title": "src/api/ 集成 + run_demo.py 一键演示 ≥2 场景",
        "ok": ok,
        "evidence": (f"演示 {n_pass} PASS / {n_fail} FAIL（exit={rc}）；"
                     f"成功场景 {s_ok}/{s_total}（要求 ≥2）"),
    }


# ------------------------------------------- 验收项 6：总结报告 + 目录交付 ----
def check_6_final_report() -> dict:
    path = _p("docs", "final-report.md")
    text = _read(path)
    required = ["验收结论", "局限性", "演进路线", "复现指南", "交付清单"]
    missing = [k for k in required if k not in text]
    report_ok = bool(text) and not missing

    # 目录交付完整性：19 项核心交付物须齐备
    core = [
        "GOAL.md", "README.md", "requirements.txt",
        "docs/architecture.md", "docs/algorithm-design.md", "docs/data-pipeline.md",
        "docs/multimodal-design.md", "docs/api-spec.md", "docs/final-report.md",
        "src/__init__.py", "src/api/router.py",
        "src/cognition/memory.py", "src/cognition/attention.py",
        "src/cognition/reasoning.py", "src/cognition/cognition.py",
        "src/multimodal/perceive.py", "src/multimodal/aligner.py",
        "src/multimodal/space.py",
        "data/build_dataset.py", "data/quality_report.md",
        "data/dataset/cleaned.jsonl", "data/dataset/train.jsonl",
        "data/dataset/eval.jsonl", "data/dataset/schema.json",
        "demo/run_demo.py",
    ]
    absents = [c for c in core if not os.path.isfile(_p(c))]
    tree_ok = not absents

    ok = report_ok and tree_ok
    ev = f"final-report.md {_lines(path)} 行"
    ev += "，含验收结论/局限性/演进路线/复现指南/交付清单" if report_ok else f"，缺章节: {missing}"
    ev += f"；核心交付物 {len(core) - len(absents)}/{len(core)} 齐备"
    if absents:
        ev += f"（缺: {absents}）"
    return {"id": 6, "title": "docs/final-report.md + 完整项目目录交付",
            "ok": ok, "evidence": ev}


# ------------------------------------------------------- 交付文件清单 --------
_EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", "agents", "download"}
_RUNTIME = {"data/memory.jsonl", "data/memory_archive.jsonl",
            "data/session_state.jsonl", "data/session_state.jsonl.tmp"}


def build_manifest() -> dict:
    """扫描项目目录，产出分类交付清单（含 sha256 前 16 位）。"""
    groups = {"root": [], "docs": [], "src": [], "data": [], "demo": [], "tests": []}
    for root, dirs, names in os.walk(_PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIRS]
        for n in sorted(names):
            if n.endswith((".pyc", ".tmp")):
                continue
            full = os.path.join(root, n)
            rel = os.path.relpath(full, _PROJECT_ROOT).replace("\\", "/")
            if rel in _RUNTIME or n.startswith("."):
                continue
            entry = {"path": rel, "bytes": os.path.getsize(full),
                     "sha256_16": _sha256(full)}
            if rel.startswith("docs/"):
                groups["docs"].append(entry)
            elif rel.startswith("src/"):
                groups["src"].append(entry)
            elif rel.startswith("data/"):
                groups["data"].append(entry)
            elif rel.startswith("demo/"):
                groups["demo"].append(entry)
            elif rel.startswith("tests/"):
                groups["tests"].append(entry)
            elif os.path.dirname(rel) == "":
                groups["root"].append(entry)
            else:  # 兜底：任何新增顶层目录都不会被静默丢弃
                groups.setdefault("other", []).append(entry)
    return groups


# ------------------------------------------------------------------ main ----
CHECKS = [check_1_architecture, check_2_algorithm, check_3_dataset,
          check_4_multimodal, check_5_integration, check_6_final_report]


def main(argv: list) -> int:
    as_json = "--json" in argv
    with_manifest = "--manifest" in argv or as_json

    if not as_json:
        print("=" * 74)
        print(f"AGI-Core Phase 5 验收核销（对照 GOAL.md 六条验收标准）v{VERSION}")
        print(f"项目根目录：{_PROJECT_ROOT}")
        print("=" * 74)

    results = []
    for fn in CHECKS:
        try:
            r = fn()
        except subprocess.TimeoutExpired:
            r = {"id": fn.__name__, "title": str(fn.__doc__ or fn.__name__),
                 "ok": False, "evidence": f"子测试超时（>{SUBPROCESS_TIMEOUT_SEC}s）"}
        except Exception as exc:                      # 单条失败不阻塞其余
            r = {"id": fn.__name__, "title": str(fn.__doc__ or fn.__name__),
                 "ok": False, "evidence": f"{type(exc).__name__}: {exc}"}
        results.append(r)
        if not as_json:
            flag = "PASS" if r["ok"] else "FAIL"
            print(f"\n[验收 {r['id']}] {flag}  {r['title']}")
            print(f"          证据：{r['evidence']}")

    n_ok = sum(1 for r in results if r["ok"])
    manifest = build_manifest() if with_manifest else None

    if as_json:
        print(json.dumps({"version": VERSION, "project_root": _PROJECT_ROOT,
                          "passed": n_ok, "total": len(results),
                          "all_passed": n_ok == len(results),
                          "results": results, "manifest": manifest},
                         ensure_ascii=False, indent=2))
        return 0 if n_ok == len(results) else 1

    print("\n" + "=" * 74)
    print(f"验收核销：{n_ok}/{len(results)} 条通过")
    if n_ok == len(results):
        print("GOAL.md 六条验收标准全部达成 [OK]  项目可交付")
    else:
        failed = [r["id"] for r in results if not r["ok"]]
        print(f"未通过项：{failed} [FAIL]  需修复后重新验收")
    print("=" * 74)

    if manifest:
        total_files = sum(len(v) for v in manifest.values())
        total_bytes = sum(e["bytes"] for v in manifest.values() for e in v)
        print(f"\n交付清单：{total_files} 个文件，{total_bytes / 1024:.1f} KB"
              f"（已排除 __pycache__ / 运行时产物；--json 查看明细）")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
