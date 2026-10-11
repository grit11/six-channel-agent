"""S7 验收工具回归：验证失败不被吞掉、部分运行不冒充通过及评测指标边界。

仅测试工具统计和失败报告，外部病例质量与数据库部署不在此测试范围内。
"""

import math
import sys
from pathlib import Path

import pytest
from eval.system_eval import (
    EvaluationCase,
    classification_metrics,
    load_cases,
    path_label_metrics,
    ranking_metrics,
)
from scripts.engineering_acceptance import (
    Check,
    acceptance_environment,
    all_passed,
    build_checks,
    run_check,
)


def test_incomplete_or_failed_acceptance_never_passes() -> None:
    """没有检查、少检查、跳过及失败都不满足完整验收。"""
    assert not all_passed([], 0)
    assert not all_passed([{"status": "passed"}], 2)
    assert not all_passed([{"status": "not_run"}], 1)
    assert not all_passed([{"status": "failed"}], 1)
    assert all_passed([{"status": "passed"}], 1)


def test_command_failure_records_exit_and_log(tmp_path: Path) -> None:
    """子进程错误保留真实退出码与可读日志，不能在报告中转为成功。"""
    result = run_check(
        Check("failure", [sys.executable, "-c", "print('marker'); raise SystemExit(7)"], tmp_path),
        tmp_path,
        acceptance_environment(tmp_path),
    )
    assert result["status"] == "failed" and result["exit_code"] == 7
    assert "marker" in (tmp_path / "failure.log").read_text(encoding="utf-8")


def test_environment_removes_deployment_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """不把部署密钥和在线模式传给离线子进程，也不更改父进程配置。"""
    monkeypatch.setenv("LLM_API_KEY", "fixture-secret")
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    values = acceptance_environment(Path("isolated"))
    assert "LLM_API_KEY" not in values and "RETRIEVAL_MODE" not in values
    assert values["LANGGRAPH_SQLITE_PATH"].endswith("checkpoint.sqlite3")


def test_short_runtime_keeps_reports_separate() -> None:
    """短路径只用于测试数据，JUnit 日志仍留在用户可见的验收报告目录。"""
    report = Path("reports")
    runtime = Path("runtime")
    checks = build_checks(report, runtime)
    command = next(check.command for check in checks if check.name == "pytest")
    assert command[command.index("--basetemp") + 1] == str(runtime / "p")
    assert command[command.index("--junitxml") + 1] == str(report / "pytest.xml")


def test_environment_preserves_browser_module_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试工具模块位置不是模型凭据，应传给浏览器回归子进程。"""
    monkeypatch.setenv("CONSOLE_PLAYWRIGHT_MODULES", "local-modules")
    assert acceptance_environment(Path("runtime"))["CONSOLE_PLAYWRIGHT_MODULES"] == "local-modules"


def test_label_and_metric_edges() -> None:
    """未知标注保持独立、错误极性不计为命中，重复来源不抬高召回。"""
    case = EvaluationCase(case_id="test", text="不详", expected_current={"thirst": "unknown"})
    assert case.expected_current == {"thirst": "unknown"}
    assert classification_metrics(1, 1, 1)["f1"] == 0.5
    assert ranking_metrics(["wrong", "a", "a"], ["a", "b"], 3) == {
        "recall": 0.5,
        "mrr": 0.5,
        "ndcg": (1 / math.log2(3)) / (1 + 1 / math.log2(3)),
    }
    assert ranking_metrics(["a", "a", "b"], ["b"], 3)["mrr"] == pytest.approx(1 / 3)
    metrics = path_label_metrics([["太阳", "太阴"], ["少阳"], ["阳明"]], ["太阳", "太阴"])
    assert metrics["top1_recall"] == 1.0 and metrics["top3_recall"] == 1.0


def test_dataset_rejects_duplicate_cases(tmp_path: Path) -> None:
    """重复编号或空集必须明确拒绝，避免扩大样本数制造虚假通过。"""
    path = tmp_path / "cases.jsonl"
    path.write_text('{"case_id":"a","text":"发热"}\n' * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="唯一"):
        load_cases(path)
