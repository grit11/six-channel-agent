"""Console 工程验收入口：聚合退出码、日志和报告，依赖 langgraph 与 Node 验证工具。

使用独立运行目录隔离测试数据库和默认 .env；不安装依赖、不访问在线模型或现有业务库。
跳过、启动失败或任一检查失败均不能形成整体通过，临床质量和实机部署另行验收。
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.config import Settings


@dataclass(frozen=True)
class Check:
    """单项检查及运行位置；命令仅来自项目预定义列表，不拼接用户 shell 文本。"""

    name: str
    command: list[str]
    cwd: Path


def acceptance_environment(run_dir: Path) -> dict[str, str]:
    """剔除继承的业务配置，保留系统工具路径并注入独立测试存储位置。"""
    fields = {name.upper() for name in Settings.model_fields}
    environment = {key: value for key, value in os.environ.items() if key.upper() not in fields}
    # pytest 从运行目录启动，Settings() 不会读项目 .env；显式 _env_file=None 仍可读环境。
    environment.update(
        PYTHONPATH=str(ROOT),
        PYTHONUTF8="1",
        MILVUS_LITE_PATH=str(run_dir / "vector.db"),
        LANGGRAPH_SQLITE_PATH=str(run_dir / "checkpoint.sqlite3"),
        E2E_PYTHON=sys.executable,
        E2E_EXTERNAL_SERVERS="false",
        E2E_API_PORT="8037",
        E2E_WEB_PORT="4193",
    )
    return environment


def build_checks(run_dir: Path, runtime_dir: Path | None = None) -> list[Check]:
    """定义后端、知识、离线纵向切片及前端六类质量门槛，不读取部署凭据。"""
    python = sys.executable
    runtime_dir = runtime_dir or run_dir
    checks = [
        Check("ruff", [python, "-m", "ruff", "check", "backend", "scripts", "eval"], ROOT),
        Check(
            "black",
            [
                python,
                "-m",
                "black",
                "--config",
                "backend/pyproject.toml",
                "--check",
                "backend",
                "scripts",
                "eval",
            ],
            ROOT,
        ),
        Check(
            "mypy",
            [
                python,
                "-m",
                "mypy",
                "--config-file",
                "backend/pyproject.toml",
                "backend/app",
                "scripts/engineering_acceptance.py",
                "eval/system_eval.py",
                "eval/workflow_acceptance.py",
            ],
            ROOT,
        ),
        Check(
            "pytest",
            [
                python,
                "-m",
                "pytest",
                str(ROOT / "backend/tests"),
                "-c",
                str(ROOT / "backend/pyproject.toml"),
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(runtime_dir / "p"),
                "--junitxml",
                str(run_dir / "pytest.xml"),
            ],
            run_dir,
        ),
        Check("knowledge", [python, str(ROOT / "scripts/validate_knowledge.py")], run_dir),
        Check("retrieval", [python, str(ROOT / "eval/retrieval_eval.py")], run_dir),
        Check(
            "evaluation",
            [
                python,
                str(ROOT / "eval/system_eval.py"),
                "--offline",
                "--dataset",
                str(ROOT / "eval/datasets/engineering_cases.jsonl"),
                "--output",
                str(run_dir / "evaluation.json"),
            ],
            run_dir,
        ),
        Check(
            "workflow",
            [
                python,
                str(ROOT / "eval/workflow_acceptance.py"),
                "--workdir",
                str(runtime_dir / "workflow"),
            ],
            run_dir,
        ),
    ]
    # 原生页面不需要 npm 构建；Node 仅用于语法检查和浏览器回归。
    node = shutil.which("node") or "node"
    for name in ("console", "labels"):
        checks.append(
            Check(
                name + "_syntax",
                [node, "--check", str(ROOT / f"backend/app/static/{name}.js")],
                ROOT,
            )
        )
    checks.append(Check("console_e2e", [node, str(ROOT / "scripts/console_e2e.cjs")], ROOT))
    return checks


def run_check(check: Check, run_dir: Path, environment: dict[str, str]) -> dict[str, object]:
    """保存完整命令日志；保留真实退出码并提示已知环境阻碍，不把错误重分类为通过。"""
    log_path = run_dir / (check.name + ".log")
    started = perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                check.command,
                cwd=check.cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        exit_code: int | None = result.returncode
        status = "passed" if exit_code == 0 else "failed"
    except OSError as error:
        log_path.write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        exit_code, status = None, "blocked"
    output = log_path.read_text(encoding="utf-8", errors="replace")
    hints = [
        marker for marker in ("spawn EPERM", "PermissionError", "WinError 5") if marker in output
    ]
    record: dict[str, object] = {
        "name": check.name,
        "status": status,
        "exit_code": exit_code,
        "duration_seconds": round(perf_counter() - started, 3),
        "command": check.command,
        "cwd": str(check.cwd),
        "log": str(log_path),
        "environment_hints": hints,
    }
    if check.name == "pytest" and (run_dir / "pytest.xml").is_file():
        try:
            suites = ET.parse(run_dir / "pytest.xml").getroot().iter("testsuite")
            totals = {key: 0 for key in ("tests", "errors", "failures", "skipped")}
            for suite in suites:
                for key in totals:
                    totals[key] += int(suite.get(key, "0"))
            totals["passed"] = totals["tests"] - sum(
                totals[key] for key in ("errors", "failures", "skipped")
            )
            record["test_summary"] = totals
        except (ET.ParseError, ValueError) as error:
            record["summary_error"] = type(error).__name__
    return record


def all_passed(results: list[dict[str, object]], expected: int) -> bool:
    """完整数量且逐项通过才接受，空报告、未运行或跳过均拒绝整体通过。"""
    return (
        expected > 0
        and len(results) == expected
        and all(item["status"] == "passed" for item in results)
    )


def write_report(run_dir: Path, results: list[dict[str, object]], expected: int) -> None:
    """每项后刷新可审阅报告，进程中断时已完成检查仍可追查。"""
    passed = all_passed(results, expected)
    report = {
        "version": "engineering-acceptance/v1.2",
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "V1.2 五自动阶段与 S6 医生审核工程验收；非临床或实机部署验收",
        "python": sys.executable,
        "expected_checks": expected,
        "overall_passed": passed,
        "checks": results,
        "deferred": [
            "生产身份认证、权限、隐私与安全分诊",
            "真实 PostgreSQL/Remote Milvus/Docker 部署与恢复",
            "在线模型质量、专家标注与独立临床评测",
        ],
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# V1.2 六阶段工程验收报告",
        "",
        str(report["scope"]),
        "",
        f"整体通过：{passed}；已记录 {len(results)}/{expected} 项。",
        "",
        "| 检查 | 状态 | 退出码 | 耗时秒 | 日志 |",
        "|---|---|---|---|---|",
    ]
    for item in results:
        lines.append(
            f"| {item['name']} | {item['status']} | {item.get('exit_code')} | "
            f"{item.get('duration_seconds', 0)} | {item.get('log', '')} |"
        )
    lines.extend(["", "未运行/跳过/失败都不算通过。延后项目见 JSON deferred 字段。"])
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """创建独立运行目录并执行全部检查，部分执行只用于定位问题。"""
    parser = argparse.ArgumentParser(description="V1.2 全量工程验收与逐项报告")
    parser.add_argument("--only", nargs="+", help="只运行指定检查；其余标记 not_run")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get("CONSOLE_TEST_ARTIFACTS") or ROOT.parent / "Console版本测试文件"
        ),
        help="测试资料根目录；报告写入 reports/，截图写入对应报告目录 screenshots/",
    )
    args = parser.parse_args()
    artifacts = args.output_dir.resolve()
    run_dir = (
        artifacts
        / "reports"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    )
    run_dir.mkdir(parents=True)
    # Windows 上 Lite 为集合生成较长的内部路径；报告集中归档，测试数据使用短临时根。
    # 保留临时数据用于失败调查，不复用既有数据库、不自动递归删除。
    runtime_dir = Path(mkdtemp(prefix="sc_")) if os.name == "nt" else run_dir
    (run_dir / "runtime.txt").write_text(str(runtime_dir) + "\n", encoding="utf-8")
    checks = build_checks(run_dir, runtime_dir)
    if args.only and set(args.only) - {check.name for check in checks}:
        parser.error("--only 包含未知检查名称")
    environment = acceptance_environment(runtime_dir)
    environment["CONSOLE_TEST_RUNTIME"] = str(runtime_dir)
    environment["CONSOLE_TEST_ARTIFACTS"] = str(artifacts)
    environment["CONSOLE_E2E_OUTPUT"] = str(run_dir / "screenshots")
    results: list[dict[str, object]] = []
    write_report(run_dir, results, len(checks))
    print(f"验收报告目录：{run_dir}", flush=True)
    for check in checks:
        if args.only and check.name not in args.only:
            item: dict[str, object] = {"name": check.name, "status": "not_run", "exit_code": None}
        else:
            print(f"开始 {check.name}", flush=True)
            item = run_check(check, run_dir, environment)
            print(f"{check.name}: {item['status']} (exit={item['exit_code']})", flush=True)
        results.append(item)
        write_report(run_dir, results, len(checks))
    print(f"报告：{run_dir / 'report.md'}", flush=True)
    return 0 if all_passed(results, len(checks)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
