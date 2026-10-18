"""实验记录边界：安全事件日志、版本指纹、用量覆盖与后端诊断关联。

只处理已有文件和测评HTTP元数据，不启动服务，不读取.env，不保存患者/模型正文。
"""

import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

import httpx

PROJECT = Path(__file__).resolve().parents[1]
PAYLOAD_FIELDS = {
    "stage",
    "attempt",
    "capability",
    "duration_ms",
    "elapsed_ms",
    "ttft_ms",
    "status",
    "error_type",
    "error_category",
    "failure_code",
    "validation_rule",
    "candidate_index",
    "conflicting_reference_count",
    "repair_attempts",
    "http_status",
    "usage_status",
    "prompt_tokens",
    "completion_tokens",
    "output_chars",
    "content_chunks",
    "received_content_chars",
    "first_content_seen",
    "prompt_digest",
    "request_payload_digest",
    "candidate_count",
    "fact_count",
    "count",
    "knowledge_type",
    "method",
    "reranked",
    "localization_status",
    "result_presentation",
    "revision",
    "paused_at",
    "stream_status",
}


def utc_now() -> str:
    """生成显式UTC时间，不依赖主机本地时区。"""
    return datetime.now(UTC).isoformat()


def safe_scalar(value: Any) -> Any:
    """日志字符串只接受元数据标识符，任意自然语言或URL不得落盘。"""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:+-]{1,160}", value):
        return value
    return "redacted"


def safe_event(event: dict[str, Any]) -> dict[str, Any]:
    """保留事件定位与已审核元数据，不复制未来新增的任意payload字段。"""
    return {
        key: safe_scalar(event.get(key))
        for key in ("event_id", "consultation_id", "event", "turn", "sequence", "created_at")
    } | {
        "payload": {
            key: safe_scalar(value)
            for key, value in event.get("payload", {}).items()
            if key in PAYLOAD_FIELDS
        }
    }


def local_fingerprint() -> dict[str, Any]:
    """记录本地代码/Prompt摘要，不冒充远程运行进程版本。"""
    source: dict[str, Any] = {"commit": None, "branch": None, "dirty": None}
    for key, args in (
        ("commit", ["rev-parse", "HEAD"]),
        ("branch", ["branch", "--show-current"]),
        ("dirty", ["status", "--porcelain", "--untracked-files=no"]),
    ):
        try:
            result = subprocess.run(
                ["git", "-C", str(PROJECT), *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                source[key] = (
                    bool(result.stdout.strip()) if key == "dirty" else result.stdout.strip()
                )
        except (OSError, subprocess.TimeoutExpired):
            source[key] = None
    paths = [Path(__file__).resolve(), PROJECT / "eval/v2_online_eval.py"]
    paths += sorted((PROJECT / "backend/app/clinical_v2/prompts").glob("*.md"))
    source["file_sha256"] = {
        p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    return source


def usage_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """按能力区分终止事件与S4会话；无Token值保持null并披露覆盖。"""
    terminal = [e for e in events if e["event"] in {"model_completed", "model_failed"}]
    by_capability: dict[str, Any] = {}
    for capability in sorted({e["payload"].get("capability", "unknown") for e in terminal}):
        group = [
            e["payload"]
            for e in terminal
            if e["payload"].get("capability", "unknown") == capability
        ]
        tokens = [
            p
            for p in group
            if isinstance(p.get("prompt_tokens"), int)
            and isinstance(p.get("completion_tokens"), int)
        ]
        by_capability[capability] = {
            "terminal_events": len(group),
            "with_token_counts": len(tokens),
            "input_tokens_observed": sum(p["prompt_tokens"] for p in tokens) if tokens else None,
            "output_tokens_observed": (
                sum(p["completion_tokens"] for p in tokens) if tokens else None
            ),
        }
    s4 = [e for e in events if e["event"] == "s4_stream_started"]
    return {
        "generic_model_terminal_events": len(terminal),
        "by_capability": by_capability,
        "clinical_terminal_events": sum(
            e["payload"].get("capability") in {"clinical_stream", "chat", "clinical_chat"}
            for e in terminal
        ),
        "s4_stream_sessions": len(s4),
        "s4_token_counts": None,
        "usage_status": (
            "partial_or_unavailable"
            if s4
            or any(v["with_token_counts"] != v["terminal_events"] for v in by_capability.values())
            or not terminal
            else "available"
        ),
        "estimated_cost": None,
        "count_semantics": "observed_lifecycles_not_provider_http_requests",
    }


def stage_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """去掉外层无stage重复失败事件；完整保留各次阶段耗时。"""
    return {
        "stage_events": [
            e
            for e in events
            if e["event"] in {"stage_started", "stage_completed", "stage_failed"}
            and e["payload"].get("stage")
        ],
        "model_terminal_counts": dict(
            Counter(
                f"{e['payload'].get('stage')}:{e['payload'].get('capability')}:{e['event']}"
                for e in events
                if e["event"] in {"model_completed", "model_failed"}
            )
        ),
    }


class ExperimentRecords:
    """每次进程独立JSONL日志；每行刷新同步，续跑不覆盖历史日志。"""

    def __init__(self, output: Path, directory: Path | None, backend_log: Path | None) -> None:
        """项目内报告的辅助日志默认放上级工程目录，避免污染代码工作区。"""
        resolved = output.resolve()
        default_parent = (
            PROJECT.parent / "实验记录" if resolved.is_relative_to(PROJECT) else resolved.parent
        )
        self.directory = (directory or default_parent / (output.stem + ".records")).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session_id = str(uuid4())
        self.path = self.directory / f"session-{self.session_id}.jsonl"
        self.backend_log = backend_log.resolve() if backend_log else None
        if self.backend_log == resolved:
            raise ValueError("后端日志不能与实验报告使用同一文件")
        self.case_id: str | None = None
        self.consultation_id: str | None = None

    def append(self, event: str, **metadata: Any) -> None:
        """逐条追加已投影元数据；不接受原始HTTP请求/响应对象。"""
        record = {
            "recorded_at_utc": utc_now(),
            "session_id": self.session_id,
            "event": event,
            "case_id": self.case_id,
            "consultation_id": self.consultation_id,
            **metadata,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def response_hook(self, response: httpx.Response) -> None:
        """只记路由类别/状态码/安全request ID，不记录URL、头、请求体或响应体。"""
        path = response.request.url.path
        route = (
            "config"
            if path.endswith("/config")
            else (
                "events_page"
                if path.endswith("/events/page")
                else "events" if path.endswith("/events") else "consultation"
            )
        )
        request_id = response.headers.get("x-request-id")
        try:
            request_id = str(UUID(request_id)) if request_id else None
        except ValueError:
            request_id = None
        self.append(
            "http_response",
            method=response.request.method,
            route=route,
            status_code=response.status_code,
            request_id=request_id,
            duration_ms=round(
                (perf_counter() - response.request.extensions.get("eval_started", perf_counter()))
                * 1000,
                1,
            ),
        )

    def request_hook(self, request: httpx.Request) -> None:
        """只在内存记录单调计时，不复制请求头或正文。"""
        request.extensions["eval_started"] = perf_counter()

    def backend_records(self, identifier: str) -> dict[str, Any]:
        """关联显式指定的安全JSONL及轮转文件，缺失/损坏明确标记，不读任意原始日志。"""
        result: dict[str, Any] = {"status": "not_configured", "records": [], "startup": None}
        if self.backend_log is None:
            return result
        result["status"] = "no_matching_record"
        files = [
            self.backend_log.with_name(self.backend_log.name + f".{i}") for i in range(5, 0, -1)
        ] + [self.backend_log]
        invalid = 0
        found_file = False
        for path in files:
            if not path.is_file():
                continue
            found_file = True
            try:
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            record = json.loads(line)
                        except ValueError:
                            invalid += 1
                            continue
                        if not isinstance(record, dict):
                            invalid += 1
                            continue
                        if record.get("event") == "backend_diagnostics_started":
                            source = record.get("source", {})
                            if not isinstance(source, dict):
                                invalid += 1
                                continue
                            result["startup"] = {
                                "recorded_at_utc": safe_scalar(record.get("recorded_at_utc")),
                                "commit": safe_scalar(source.get("commit")),
                                "dirty": safe_scalar(source.get("dirty")),
                                "process_id": safe_scalar(record.get("process_id")),
                            }
                        if (
                            record.get("event") == "v2_runtime_error"
                            and record.get("consultation_id") == identifier
                        ):
                            projected = {
                                key: safe_scalar(value)
                                for key, value in record.items()
                                if key
                                in PAYLOAD_FIELDS
                                | {
                                    "recorded_at_utc",
                                    "consultation_id",
                                    "cause_type",
                                    "failure_code",
                                    "process_id",
                                }
                            }
                            location = record.get("code_location", "")
                            if isinstance(location, str) and re.fullmatch(
                                r"backend/app/[A-Za-z0-9_/]+\.py:\d+", location
                            ):
                                projected["code_location"] = location
                            projected["schema_errors"] = [
                                {
                                    "loc": [safe_scalar(p) for p in error.get("loc", [])],
                                    "type": safe_scalar(error.get("type")),
                                }
                                for error in (
                                    record.get("schema_errors", [])
                                    if isinstance(record.get("schema_errors", []), list)
                                    else []
                                )[:30]
                                if isinstance(error, dict)
                                and isinstance(error.get("loc", []), list)
                            ]
                            projected["backend_startup"] = result["startup"]
                            chain = record.get("exception_chain", [])
                            projected["exception_chain"] = (
                                [safe_scalar(value) for value in chain[:12]]
                                if isinstance(chain, list)
                                else []
                            )
                            result["records"].append(projected)
            except (OSError, UnicodeError) as exc:
                result["read_error"] = type(exc).__name__
        result["invalid_lines"] = invalid
        result["status"] = (
            "matched"
            if result["records"]
            else "no_matching_record" if found_file else "file_missing"
        )
        if result.get("read_error") or invalid:
            result["coverage"] = "partial"
        return result
