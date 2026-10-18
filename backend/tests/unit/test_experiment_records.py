"""实验诊断的隐私、日志落盘和能力口径测试；不启动应用或在线能力。"""

import json
from pathlib import Path
from uuid import uuid4

import httpx
from pydantic import BaseModel, ValidationError

from backend.app.core.exceptions import ApplicationError
from backend.app.core.logging import (
    configure_diagnostic_file,
    log_runtime_error,
    logger,
    runtime_error_record,
)
from eval.experiment_records import ExperimentRecords, safe_event, usage_summary


def test_runtime_schema_log_excludes_input_and_exception_message(tmp_path: Path) -> None:
    """字段路径和错误类型可查，病例原文、异常正文及ctx不进入日志。"""

    class Sample(BaseModel):
        count: int

    identifier = uuid4()
    try:
        Sample.model_validate({"count": "SECRET_PATIENT_TOKEN"})
    except ValidationError as cause:
        try:
            raise ApplicationError(
                "SECRET_PASSWORD", details={"failure_code": "S2_RESPONSE_INVALID"}
            ) from cause
        except ApplicationError as error:
            payload = runtime_error_record(identifier, error)
    assert payload["failure_code"] == "S2_RESPONSE_INVALID"
    assert payload["schema_errors"] == [{"loc": ["count"], "type": "int_parsing"}]
    serialized = json.dumps(payload)
    assert "SECRET" not in serialized
    assert payload["exception_chain"] == ["ApplicationError", "ValidationError"]


def test_optional_diagnostic_file_has_only_explicit_safe_records(tmp_path: Path) -> None:
    """UTF-8文件只收明确签发记录，普通日志和堆栈不被文件处理器复制。"""
    path = tmp_path / "backend.jsonl"
    handler = configure_diagnostic_file(path, tmp_path)
    identifier = uuid4()
    try:
        logger.error("SECRET_GENERIC_LOG")
        log_runtime_error(
            identifier,
            ApplicationError(
                "SECRET_BODY",
                details={
                    "validation_rule": "support_counter_overlap",
                    "candidate_index": 1,
                    "repair_attempts": 1,
                    "provider": "SECRET_PROVIDER",
                    "api_key": "SECRET_KEY",
                },
            ),
        )
    finally:
        logger.removeHandler(handler)
        handler.close()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["event"] == "backend_diagnostics_started"
    assert records[1]["consultation_id"] == str(identifier)
    assert records[1]["validation_rule"] == "support_counter_overlap"
    assert isinstance(records[1]["process_id"], int)
    assert "SECRET" not in path.read_text(encoding="utf-8")


def test_backend_join_rotations_missing_and_redaction(tmp_path: Path) -> None:
    """轮转日志只关联当前CID，其他会诊及未知字段不流入实验日志。"""
    path = tmp_path / "backend.jsonl"
    record = {
        "event": "v2_runtime_error",
        "consultation_id": "wanted",
        "validation_rule": "duplicate_candidate",
        "raw_input": "SECRET_PATIENT",
    }
    path.with_suffix(".jsonl.1").write_text(json.dumps(record) + "\n", encoding="utf-8")
    path.write_text(
        '{"event":"v2_runtime_error","consultation_id":"other"}\nBAD\n', encoding="utf-8"
    )
    records = ExperimentRecords(tmp_path / "report.json", None, path)
    joined = records.backend_records("wanted")
    assert joined["status"] == "matched" and len(joined["records"]) == 1
    assert joined["coverage"] == "partial" and joined["invalid_lines"] == 1
    assert "SECRET" not in json.dumps(joined)
    assert records.backend_records("absent")["status"] == "no_matching_record"
    assert (
        ExperimentRecords(tmp_path / "b.json", None, tmp_path / "missing").backend_records("x")[
            "status"
        ]
        == "file_missing"
    )
    assert (
        ExperimentRecords(tmp_path / "b.json", None, None).backend_records("x")["status"]
        == "not_configured"
    )


def test_events_and_http_hooks_never_record_bodies_or_credentials(tmp_path: Path) -> None:
    """即使服务器加了敏感payload或请求含认证，日志仍仅包含允许元数据。"""
    event = safe_event(
        {
            "event_id": "cid:1",
            "consultation_id": "cid",
            "event": "stage_failed",
            "payload": {
                "stage": "s1_extract",
                "raw_input": "SECRET_PATIENT",
                "error_type": "SECRET NATURAL LANGUAGE",
                "duration_ms": 10,
            },
        }
    )
    assert event["payload"] == {"stage": "s1_extract", "error_type": "redacted", "duration_ms": 10}
    records = ExperimentRecords(tmp_path / "report.json", None, None)
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(500, text="SECRET_RESPONSE")),
        event_hooks={"request": [records.request_hook], "response": [records.response_hook]},
    ) as client:
        client.post(
            "http://user:SECRET_PASSWORD@test/api/v2/consultations?key=SECRET_QUERY",
            json={"text": "SECRET_PATIENT"},
            headers={"Authorization": "SECRET_KEY"},
        )
    saved = records.path.read_text(encoding="utf-8")
    assert "SECRET" not in saved and "http://" not in saved
    assert json.loads(saved)["status_code"] == 500


def test_usage_distinguishes_capabilities_and_missing_s4_tokens() -> None:
    """流式心跳不计调用，Embedding/Reranker独立且缺失usage不填零。"""
    events = [
        {
            "event": "model_completed",
            "payload": {
                "capability": "clinical_stream",
                "prompt_tokens": 10,
                "completion_tokens": 4,
            },
        },
        {"event": "model_failed", "payload": {"capability": "embedding"}},
        {"event": "model_started", "payload": {"capability": "clinical_stream"}},
        {"event": "model_started", "payload": {"capability": "clinical_stream"}},
        {"event": "s4_stream_started", "payload": {}},
    ]
    usage = usage_summary(events)
    assert usage["generic_model_terminal_events"] == 2
    assert usage["clinical_terminal_events"] == 1 and usage["s4_stream_sessions"] == 1
    assert usage["by_capability"]["embedding"]["input_tokens_observed"] is None
    assert usage["s4_token_counts"] is None and usage["estimated_cost"] is None
    assert usage["usage_status"] == "partial_or_unavailable"


def test_runtime_location_identifies_existing_s1_validation_branch() -> None:
    """无需模型或原文日志，通过代码位置定位未结构化的S1对齐错误分支。"""
    from backend.app.clinical_v2.verification import ExtractionInvalidError, align_source

    try:
        align_source("合成输入", "不存在的合成片段")
    except ExtractionInvalidError as error:
        record = runtime_error_record(uuid4(), error)
    assert record["failure_code"] == "EXTRACTION_INVALID"
    assert record["code_location"].startswith("backend/app/clinical_v2/verification.py:")
    assert "合成" not in json.dumps(record, ensure_ascii=False)
