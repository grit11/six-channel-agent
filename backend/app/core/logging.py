"""结构化基础设施日志，只记录安全字段，不输出密钥、连接 URL 或病例文本。"""

import hashlib
import json
import logging
import os
import re
import subprocess
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import ValidationError

logger = logging.getLogger("six_channel")


def source_identity(root: Path) -> dict[str, object]:
    """只读取本地 Git 标识，不读取远端地址、配置、环境或差异正文。"""
    identity: dict[str, object] = {"commit": None, "branch": None, "dirty": None}
    for key, args in (
        ("commit", ["rev-parse", "HEAD"]),
        ("branch", ["branch", "--show-current"]),
        ("dirty", ["status", "--porcelain", "--untracked-files=no"]),
    ):
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                identity[key] = (
                    bool(result.stdout.strip()) if key == "dirty" else result.stdout.strip()
                )
        except (OSError, subprocess.TimeoutExpired):
            identity[key] = None
    paths = [root / "backend/app/services/runtime_service.py", root / "backend/app/core/llm.py"]
    paths += sorted((root / "backend/app/clinical_v2").glob("*.py"))
    paths += sorted((root / "backend/app/clinical_v2/prompts").glob("*.md"))
    identity["file_sha256"] = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }
    identity["meaning"] = "startup_checkout_not_proof_of_loaded_module_bytes"
    return identity


def runtime_error_record(identifier: UUID | None, error: BaseException | None) -> dict[str, Any]:
    """投影异常链的安全元数据；永不保存异常正文、input、ctx 或局部变量。"""
    record: dict[str, Any] = {
        "event": "v2_runtime_error",
        "consultation_id": str(identifier) if identifier else None,
        "thread_id": f"clinical-v2:{identifier}" if identifier else None,
        "cause_type": type(error).__name__,
        "schema_errors": [],
        "exception_chain": [],
    }
    current = error
    visited: set[int] = set()
    safe_keys = {
        "failure_code",
        "error_category",
        "stage",
        "validation_rule",
        "field_path",
        "item_index",
        "schema_type",
        "schema_errors",
        "candidate_index",
        "conflicting_reference_count",
        "repair_attempts",
        "http_status",
        "content_chars",
        "finish_reason",
        "saw_done",
        "response_format_enabled",
        "stream_options_enabled",
    }
    while current is not None and id(current) not in visited and len(visited) < 12:
        visited.add(id(current))
        record["exception_chain"].append(type(current).__name__)
        details = getattr(current, "details", {})
        if isinstance(details, dict):
            for key in safe_keys:
                value = details.get(key)
                if key == "schema_errors" and isinstance(value, list):
                    safe_errors = []
                    for item in value[:30]:
                        if not isinstance(item, dict):
                            continue
                        location = item.get("loc", [])
                        if not isinstance(location, list):
                            location = []
                        safe_errors.append(
                            {
                                "loc": [
                                    (
                                        part
                                        if isinstance(part, int)
                                        else (
                                            part
                                            if isinstance(part, str)
                                            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", part)
                                            else "<redacted>"
                                        )
                                    )
                                    for part in location[:20]
                                ],
                                "type": str(item.get("type", "unknown"))[:100],
                            }
                        )
                    record["schema_errors"] = safe_errors
                    continue
                if isinstance(value, (bool, int, float)) or (
                    isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", value)
                ):
                    record.setdefault(key, value)
        code = getattr(current, "code", None)
        if isinstance(code, str) and re.fullmatch(r"[A-Z0-9_]{1,100}", code):
            record.setdefault("failure_code", code)
        trace = current.__traceback__
        while trace is not None:
            filename = Path(trace.tb_frame.f_code.co_filename).as_posix()
            if "/backend/app/" in filename:
                record["code_location"] = (
                    "backend/app/" + filename.split("/backend/app/", 1)[1] + f":{trace.tb_lineno}"
                )
            trace = trace.tb_next
        if isinstance(current, ValidationError):
            # Pydantic 的 input/context/msg 可包含患者或模型正文，只取 loc/type。
            for item in current.errors(
                include_url=False, include_context=False, include_input=False
            )[:30]:
                location = [
                    (
                        part
                        if isinstance(part, int)
                        else (
                            part
                            if isinstance(part, str)
                            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", part)
                            else "<redacted>"
                        )
                    )
                    for part in item.get("loc", ())
                ]
                record["schema_errors"].append({"loc": location, "type": item["type"]})
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return record


class _DiagnosticFormatter(logging.Formatter):
    """只格式化明确签发的结构化记录，不展开 LogRecord 或堆栈。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "diagnostic_record", {})
        return json.dumps(
            {
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "process_id": os.getpid(),
                **payload,
            },
            ensure_ascii=False,
        )


class _DiagnosticFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return isinstance(getattr(record, "diagnostic_record", None), dict)


def configure_diagnostic_file(path: Path, root: Path) -> logging.Handler:
    """显式启用 UTF-8 JSONL 轮转；启动失败明确报错，不静默丢日志。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    # 添加文件handler后lastResort不再触发，因此显式保留原有终端输出。
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())
    handler = RotatingFileHandler(path, maxBytes=20_000_000, backupCount=5, encoding="utf-8")
    handler.setFormatter(_DiagnosticFormatter())
    handler.addFilter(_DiagnosticFilter())
    logger.addHandler(handler)
    logger.warning(
        "backend_diagnostics_started",
        extra={
            "diagnostic_record": {
                "event": "backend_diagnostics_started",
                "source": source_identity(root),
            }
        },
    )
    return handler


def log_runtime_error(identifier: UUID | None, error: BaseException | None) -> None:
    """同时向终端及已启用的诊断文件发布同一安全错误记录。"""
    payload = runtime_error_record(identifier, error)
    logger.error(json.dumps(payload, ensure_ascii=False), extra={"diagnostic_record": payload})


def log_backend_event(event: str, backend: str, cause_type: str) -> None:
    """记录后端选择及异常类别，不序列化原始异常以防凭据泄露。"""
    logger.warning(
        json.dumps(
            {
                "event": event,
                "backend": backend,
                "cause_type": cause_type,
                "request_id": None,
                "consultation_id": None,
                "thread_id": None,
                "stage": "infrastructure_startup",
            },
            ensure_ascii=False,
        )
    )
