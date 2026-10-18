"""在线测评断点与公开进度测试，依赖 HTTP 夹具，不访问真实模型或用户数据库。"""

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from eval.v2_online_eval import run


def make_dataset(tmp_path: Path, count: int = 1) -> Path:
    """生成只供调度测试的合成病例。"""
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"case_id": f"SC{i:03}", "text": "合成输入"}) for i in range(1, count + 1)
        ),
        encoding="utf-8",
    )
    return dataset


def mock_client(monkeypatch: pytest.MonkeyPatch, handle: Any) -> None:
    """注入 HTTP 边界，测试仍执行真实报告读写与调度逻辑。"""
    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )


def terminal_response() -> httpx.Response:
    """提供无需模型的医生审核暂停快照。"""
    return httpx.Response(
        200,
        json={
            "consultation_id": "consultation-test",
            "status": "awaiting_doctor_review",
            "completed_stages": ["s1_extract", "s2_localize", "s3_retrieve", "s4_compare"],
        },
    )


def test_resume_skips_existing_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """旧版四例报告可接着跑第五例；失败例也保留且不重新消耗。"""
    dataset = make_dataset(tmp_path, 6)
    output = tmp_path / "report.json"
    old_rows = [
        {
            "case_id": f"SC{i:03}",
            "error": "EXTRACTION_INVALID" if i == 2 else None,
            "metrics": {},
            "latency_ms": 1,
        }
        for i in range(1, 5)
    ]
    output.write_text(
        json.dumps(
            {
                "protocol": "clinical-v2.0",
                "dataset_digest": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "rows": old_rows,
                "config": {"llm_model": "fixture"},
            }
        ),
        encoding="utf-8",
    )
    submissions = []

    def handle(request: httpx.Request) -> httpx.Response:
        """只允许剩余两例被提交。"""
        if request.url.path.endswith("/config"):
            return httpx.Response(200, json={"llm_model": "fixture"})
        if request.method == "POST":
            submissions.append(request)
            return terminal_response()
        return httpx.Response(200, json=[])

    mock_client(monkeypatch, handle)
    report = run(dataset, "http://test", output, resume=True)
    assert len(submissions) == 2
    assert report["rows"][:4] == old_rows
    assert [row["case_id"] for row in report["rows"]] == [f"SC{i:03}" for i in range(1, 7)]
    assert report["summary"]["failed"] == 1
    assert "[5/6] SC005 开始" in capsys.readouterr().out


def test_interrupted_case_recovers_same_consultation_and_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """中断已受理病例后只查询旧 ID；周期显示公开阶段且不重复 POST。"""
    dataset = make_dataset(tmp_path)
    output = tmp_path / "report.json"
    submissions = 0
    interrupt = True
    queries = 0

    def handle(request: httpx.Request) -> httpx.Response:
        """模拟受理、第一次轮询被中断、恢复后完成。"""
        nonlocal submissions, queries
        if request.url.path.endswith("/config"):
            return httpx.Response(200, json={})
        if request.method == "POST":
            submissions += 1
            return httpx.Response(
                200, json={"consultation_id": "consultation-test", "status": "running"}
            )
        if request.url.path.endswith("/events/page"):
            return httpx.Response(200, json=[])
        if interrupt:
            raise KeyboardInterrupt
        queries += 1
        if queries <= 2:
            return httpx.Response(
                200,
                json={
                    "consultation_id": "consultation-test",
                    "status": "running",
                    "completed_stages": ["s1_extract"],
                },
            )
        return terminal_response()

    mock_client(monkeypatch, handle)
    monkeypatch.setattr("eval.v2_online_eval.time.sleep", lambda _: None)
    with pytest.raises(KeyboardInterrupt):
        run(dataset, "http://test", output)
    interrupted = json.loads(output.read_text(encoding="utf-8"))
    assert interrupted["pending"]["consultation_id"] == "consultation-test"
    assert interrupted["rows"] == []
    interrupt = False
    report = run(dataset, "http://test", output, resume=True, progress_seconds=0.000001)
    assert submissions == 1
    assert report["summary"]["count"] == 1 and "pending" not in report
    printed = capsys.readouterr().out
    assert "恢复查询已有会诊" in printed
    assert "最近完成=s1_extract" in printed and "已等待" in printed


def test_resume_replays_same_key_if_creation_response_was_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """受理响应未保存时也复用已落盘幂等键，不生成第二个测评请求身份。"""
    dataset = make_dataset(tmp_path)
    output = tmp_path / "report.json"
    keys = []

    def handle(request: httpx.Request) -> httpx.Response:
        """第一次提交模拟进程中断，第二次同键完成。"""
        if request.url.path.endswith("/config"):
            return httpx.Response(200, json={})
        if request.method == "POST":
            keys.append(request.headers["Idempotency-Key"])
            if len(keys) == 1:
                raise KeyboardInterrupt
            return terminal_response()
        return httpx.Response(200, json=[])

    mock_client(monkeypatch, handle)
    with pytest.raises(KeyboardInterrupt):
        run(dataset, "http://test", output)
    run(dataset, "http://test", output, resume=True)
    assert len(keys) == 2 and keys[0] == keys[1]


def test_existing_report_requires_explicit_resume_and_matching_dataset(tmp_path: Path) -> None:
    """拒绝覆盖原报告及混合不同题库，原文件必须保持完整。"""
    dataset = make_dataset(tmp_path)
    output = tmp_path / "report.json"
    original = '{"protocol":"clinical-v2.0","dataset_digest":"other","rows":[]}'
    output.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="--resume"):
        run(dataset, "http://test", output)
    with pytest.raises(ValueError, match="摘要不一致"):
        run(dataset, "http://test", output, resume=True)
    assert output.read_text(encoding="utf-8") == original


def test_completed_resume_does_not_contact_backend(tmp_path: Path) -> None:
    """已全部记录的报告无需后端也能查看汇总，失败仍保留。"""
    dataset = make_dataset(tmp_path)
    output = tmp_path / "report.json"
    output.write_text(
        json.dumps(
            {
                "protocol": "clinical-v2.0",
                "dataset_digest": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "rows": [{"case_id": "SC001", "error": None, "metrics": {}, "latency_ms": 1}],
            }
        ),
        encoding="utf-8",
    )
    assert run(dataset, "http://test", output, resume=True)["summary"]["count"] == 1


def test_event_collection_failure_preserves_terminal_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """日志接口故障不覆盖临床失败码、终态快照或研发指标。"""
    dataset = make_dataset(tmp_path)
    output = tmp_path / "report.json"

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/config"):
            return httpx.Response(200, json={})
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "consultation_id": "test-id",
                    "status": "failed",
                    "failure_code": "EXTRACTION_INVALID",
                },
            )
        return httpx.Response(503, text="SECRET_BACKEND_RESPONSE")

    mock_client(monkeypatch, handle)
    report = run(dataset, "http://test", output)
    row = report["rows"][0]
    assert row["error"] == "EXTRACTION_INVALID" and row["result"]["status"] == "failed"
    assert row["event_capture"]["status"] == "partial_or_unavailable"
    assert row["diagnostics"]["status"] == "not_configured"
    assert "SECRET" not in Path(report["sessions"][0]["records_file"]).read_text(encoding="utf-8")


def test_network_loss_keeps_pending_and_same_idempotency_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """响应丢失不是临床终态失败，续跑复用已保存请求身份。"""
    dataset = make_dataset(tmp_path)
    output = tmp_path / "report.json"
    keys: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/config"):
            return httpx.Response(200, json={})
        if request.method == "POST":
            keys.append(request.headers["Idempotency-Key"])
            if len(keys) == 1:
                raise httpx.ReadTimeout("SECRET_URL", request=request)
            return terminal_response()
        return httpx.Response(200, json=[])

    mock_client(monkeypatch, handle)
    interrupted = run(dataset, "http://test", output)
    assert interrupted["run_state"] == "interrupted" and interrupted["rows"] == []
    assert interrupted["pending"]["last_client_error"]["error_type"] == "ReadTimeout"
    complete = run(dataset, "http://test", output, resume=True)
    assert keys[0] == keys[1] and len(complete["rows"]) == 1
    assert len(complete["sessions"]) == 2


def test_partial_event_pages_survive_disconnect_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """首100事件落盘后断线，恢复从游标继续而不丢失或重复记录。"""
    from eval.experiment_records import ExperimentRecords
    from eval.v2_online_eval import capture_events, load_report

    dataset = make_dataset(tmp_path)
    output = tmp_path / "report.json"
    report = load_report(
        output,
        hashlib.sha256(dataset.read_bytes()).hexdigest(),
        [{"case_id": "SC001"}],
        resume=False,
    )
    pending: dict[str, Any] = {
        "case_id": "SC001",
        "idempotency_key": "key",
        "consultation_id": "cid",
    }
    report["pending"] = pending
    records = ExperimentRecords(output, None, None)
    requests = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(
                200,
                json=[
                    {
                        "event_id": f"cid:{i}",
                        "consultation_id": "cid",
                        "event": "model_started",
                        "sequence": i,
                        "payload": {"stage": "s1_extract"},
                    }
                    for i in range(1, 101)
                ],
            )
        assert request.url.params["after"] == "cid:100"
        if requests == 2:
            raise httpx.ReadTimeout("synthetic", request=request)
        return httpx.Response(
            200,
            json=[
                {
                    "event_id": "cid:101",
                    "consultation_id": "cid",
                    "event": "stage_completed",
                    "payload": {"stage": "s1_extract"},
                }
            ],
        )

    # 此测试直接注入HTTP transport，不建立真实会诊。
    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle)) as client:
        assert not capture_events(client, "cid", pending, records, output, report)
        saved = json.loads(output.read_text(encoding="utf-8"))
        assert len(saved["pending"]["events"]) == 100
        assert capture_events(client, "cid", saved["pending"], records, output, saved)
        assert len(saved["pending"]["events"]) == 101


def test_evaluation_joins_safe_backend_error_by_consultation_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """完整测评调度关联服务端规则与代码位置，不把另一会诊错误合入。"""
    dataset = make_dataset(tmp_path)
    output = tmp_path / "report.json"
    backend_log = tmp_path / "backend.jsonl"
    backend_log.write_text(
        json.dumps(
            {
                "event": "v2_runtime_error",
                "consultation_id": "test-id",
                "failure_code": "LOCALIZATION_INVALID",
                "validation_rule": "unknown_as_counter",
                "candidate_index": 1,
                "repair_attempts": 1,
                "code_location": "backend/app/clinical_v2/channel_localization.py:51",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/config"):
            return httpx.Response(200, json={})
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "consultation_id": "test-id",
                    "status": "failed",
                    "failure_code": "LOCALIZATION_INVALID",
                },
            )
        return httpx.Response(200, json=[])

    mock_client(monkeypatch, handle)
    report = run(dataset, "http://test", output, backend_log=backend_log)
    diagnostic = report["rows"][0]["diagnostics"]
    assert diagnostic["status"] == "matched"
    assert diagnostic["records"][0]["validation_rule"] == "unknown_as_counter"
    assert diagnostic["records"][0]["repair_attempts"] == 1
