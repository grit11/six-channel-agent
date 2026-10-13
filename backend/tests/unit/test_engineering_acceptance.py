"""验收工具统计回归：失败、部分执行和配置隔离不能冒充全部通过。"""

import sys
from pathlib import Path

import pytest
from scripts.engineering_acceptance import (
    Check,
    acceptance_environment,
    all_passed,
    build_checks,
    run_check,
)


def test_incomplete_or_failed_acceptance_never_passes() -> None:
    """检查未执行、缺失和失败均不满足完成定义。"""
    assert not all_passed([], 0)
    assert not all_passed([{"status": "passed"}], 2)
    assert not all_passed([{"status": "not_run"}], 1)
    assert not all_passed([{"status": "failed"}], 1)
    assert all_passed([{"status": "passed"}], 1)


def test_command_failure_records_exit_and_log(tmp_path: Path) -> None:
    """子进程失败保留原始退出码。"""
    result = run_check(
        Check("failure", [sys.executable, "-c", "print('marker'); raise SystemExit(7)"], tmp_path),
        tmp_path,
        acceptance_environment(tmp_path),
    )
    assert result["status"] == "failed" and result["exit_code"] == 7
    assert "marker" in (tmp_path / "failure.log").read_text()


def test_environment_removes_deployment_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """工程回归不继承部署密钥，不更改父进程。"""
    monkeypatch.setenv("LLM_API_KEY", "fixture-secret")
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    values = acceptance_environment(Path("isolated"))
    assert "LLM_API_KEY" not in values and "RETRIEVAL_MODE" not in values
    assert values["LANGGRAPH_SQLITE_PATH"].endswith("checkpoint.sqlite3")


def test_short_runtime_keeps_reports_separate() -> None:
    """测试数据短路径独立于报告，所有入口必须是真实存在的 V2 文件。"""
    report, runtime = Path("reports"), Path("runtime")
    checks = build_checks(report, runtime)
    command = next(c.command for c in checks if c.name == "pytest")
    assert command[command.index("--basetemp") + 1] == str(runtime / "p")
    assert command[command.index("--junitxml") + 1] == str(report / "pytest.xml")
    assert not any("system_eval" in str(c.command) for c in checks)
