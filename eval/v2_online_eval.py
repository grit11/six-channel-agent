"""V2 在线实验：调用真实会诊 API，记录暂停、来源、usage 和研发银标指标。

不自动审核或补造患者事实；暂停是合法结果，RAGAS 与临床参考指标独立记录。
断点仅恢复测评调度和已受理会诊；不自动重试失败阶段，不改变临床服务配置。
"""

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

# 兼容直接执行 python eval/v2_online_eval.py 与包导入测试，不导入临床应用。
if __package__:
    from .experiment_records import (
        ExperimentRecords,
        local_fingerprint,
        safe_event,
        stage_summary,
        usage_summary,
        utc_now,
    )
else:
    from experiment_records import (  # type: ignore[no-redef]
        ExperimentRecords,
        local_fingerprint,
        safe_event,
        stage_summary,
        usage_summary,
        utc_now,
    )


def collect_events(
    client: httpx.Client,
    identifier: str,
    pending: dict[str, Any],
    records: ExperimentRecords,
    output: Path,
    report: dict[str, Any],
) -> None:
    """运行中分页拉取公开事件并逐页保存；断线不丢已收到的事件，不阻塞等待SSE终态。"""
    events = pending.setdefault("events", [])
    seen = {e["event_id"] for e in events}
    cursor = pending.get("event_cursor")
    # 限制单次读取，避免异常服务端重复页面造成无限循环。
    for _ in range(100):
        response = client.get(
            f"/api/v2/consultations/{identifier}/events/page",
            params={"after": cursor} if cursor else None,
        )
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, list):
            raise ValueError("公开事件分页响应不是列表")
        added = 0
        for item in page:
            if not isinstance(item, dict) or not isinstance(item.get("event_id"), str):
                raise ValueError("公开事件缺少身份")
            if item.get("consultation_id") != identifier:
                raise ValueError("公开事件会诊身份不匹配")
            if not isinstance(item.get("payload", {}), dict):
                raise ValueError("公开事件payload不是对象")
            if item["event_id"] not in seen:
                event = safe_event(item)
                events.append(event)
                seen.add(item["event_id"])
                records.append("backend_event", backend_event=event)
                added += 1
            cursor = item["event_id"]
        pending["event_cursor"] = cursor
        save_report(output, report)
        if len(page) < 100:
            return
        if not added:
            raise ValueError("公开事件游标未前进")
    raise ValueError("公开事件分页超过安全上限，保留已读取部分")


def capture_events(
    client: httpx.Client,
    identifier: str,
    pending: dict[str, Any],
    records: ExperimentRecords,
    output: Path,
    report: dict[str, Any],
) -> bool:
    """采集故障仅标记观测缺口，不覆盖临床终态或已成功保存的结果。"""
    try:
        collect_events(client, identifier, pending, records, output, report)
        return True
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        error = {"recorded_at_utc": utc_now(), "error_type": type(exc).__name__}
        pending.setdefault("observation_errors", []).append(error)
        records.append("event_collection_failed", **error)
        save_report(output, report)
        return False


def load_cases(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，拒绝空数据、空原文和重复 ID；保持参考标签不变。"""
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    ids = [c.get("case_id") for c in cases]
    if not cases or any(not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("数据集必须非空且 case_id 唯一")
    if any(not isinstance(c.get("text"), str) or not c["text"].strip() for c in cases):
        raise ValueError("病例 text 必须非空")
    return cases


def percentile(values: list[float], fraction: float) -> float | None:
    """按最近秩计算延迟分位数，空实验不伪造零延迟。"""
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def recall(predicted: list[str], expected: list[str]) -> float | None:
    """按去重标签计算研发召回；无参考不计临床准确率。"""
    return len(set(predicted) & set(expected)) / len(set(expected)) if expected else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """阶段、失败和研发召回单独统计，失败仍留在整体样本中。"""
    latency = [float(r["latency_ms"]) for r in rows]
    metrics = {}
    for name in (
        "channel_top1",
        "channel_top3",
        "formula_top1",
        "formula_top3",
        "retrieval_formula_top5",
    ):
        values = [r["metrics"][name] for r in rows if r["metrics"].get(name) is not None]
        metrics[name] = sum(values) / len(values) if values else None
    return {
        "count": len(rows),
        "failed": sum(bool(r["error"]) for r in rows),
        "p50_ms": percentile(latency, 0.5),
        "p95_ms": percentile(latency, 0.95),
        "metrics": metrics,
        "label_status": "研发银标；未作独立专家临床验证",
    }


def save_report(output: Path, report: dict[str, Any]) -> None:
    """先写同目录临时文件再替换，避免中断写入破坏已有实验记录。"""
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)


def load_report(
    output: Path, digest: str, cases: list[dict[str, Any]], *, resume: bool
) -> dict[str, Any]:
    """只恢复同一数据集的唯一病例记录；普通运行禁止覆盖已有结果。"""
    if not resume:
        if output.exists():
            raise ValueError("输出报告已存在；续跑请加 --resume，重新实验请使用新的 --output")
        return {
            "protocol": "clinical-v2.0",
            "dataset_digest": digest,
            "label_status": "研发银标",
            "rows": [],
        }
    if not output.exists():
        raise ValueError("续跑报告不存在，请检查 --output")
    report = json.loads(output.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict) or report.get("dataset_digest") != digest:
        raise ValueError("报告与当前数据集摘要不一致，不能合并实验")
    if report.get("protocol") != "clinical-v2.0" or not isinstance(report.get("rows"), list):
        raise ValueError("不是可恢复的 clinical-v2.0 报告")
    rows = report["rows"]
    identifiers = [row.get("case_id") for row in rows if isinstance(row, dict)]
    allowed = {case["case_id"] for case in cases}
    if (
        len(identifiers) != len(rows)
        or len(set(identifiers)) != len(identifiers)
        or any(identifier not in allowed for identifier in identifiers)
    ):
        raise ValueError("报告病例 ID 重复、缺失或不属于数据集")
    pending = report.get("pending")
    if pending and (
        not isinstance(pending, dict)
        or pending.get("case_id") not in allowed
        or not pending.get("idempotency_key")
        or pending.get("case_id") in identifiers
    ):
        raise ValueError("报告中的未完成病例记录无效")
    return report


def progress_snapshot(result: dict[str, Any]) -> str:
    """只显示公开状态和已完成阶段，不输出病例或模型正文。"""
    completed = result.get("completed_stages") or []
    stage = completed[-1] if completed else "无"
    return f"状态={result.get('status', 'unknown')}，最近完成={stage}"


def run(
    dataset: Path,
    base_url: str,
    output: Path,
    *,
    limit: int | None = None,
    timeout_seconds: float = 360,
    stop_after_failures: int = 3,
    resume: bool = False,
    progress_seconds: float = 5,
    records_dir: Path | None = None,
    backend_log: Path | None = None,
) -> dict[str, Any]:
    """逐例保存报告，受理后轮询权威快照，网络错误保留类别不伪装成功。"""
    cases = load_cases(dataset)
    if timeout_seconds <= 0 or progress_seconds <= 0 or (limit is not None and limit <= 0):
        raise ValueError("超时、进度间隔和 limit 必须大于零")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    report = load_report(output, digest, cases, resume=resume)
    total_dataset = len(cases)
    if limit:
        cases = cases[:limit]
    completed = {row["case_id"] for row in report["rows"]}
    remaining = [case for case in cases if case["case_id"] not in completed]
    if report.get("pending") and report["pending"]["case_id"] not in {
        case["case_id"] for case in remaining
    }:
        raise ValueError("--limit 排除了待恢复病例，请扩大范围或移除 limit")
    print(
        f"[测评] 数据集 {total_dataset} 例，本次范围 {len(cases)} 例；"
        f"已记录 {len(completed)} 例（含失败），待执行 {len(remaining)} 例。",
        flush=True,
    )
    if not remaining:
        report["summary"] = summarize(report["rows"])
        print("[测评] 所选病例均已记录，无需调用后端。", flush=True)
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    records = ExperimentRecords(output, records_dir, backend_log)
    session = {
        "session_id": records.session_id,
        "started_at_utc": utc_now(),
        "records_file": str(records.path),
        "local_source": local_fingerprint(),
        "runtime_commit": "unconfirmed;see_backend_diagnostics",
        "python_version": platform.python_version(),
        "httpx_version": httpx.__version__,
        "parameters": {
            "limit": limit,
            "timeout_seconds": timeout_seconds,
            "stop_after_failures": stop_after_failures,
            "resume": resume,
            "progress_seconds": progress_seconds,
        },
    }
    report.setdefault("run_id", str(uuid4()))
    report.setdefault("sessions", []).append(session)
    report["record_schema"] = "experiment-records/v1"
    records.append(
        "session_started",
        run_id=report["run_id"],
        dataset_digest=digest,
        local_source=session["local_source"],
        parameters=session["parameters"],
    )
    save_report(output, report)
    print(f"[测评] 安全运行日志：{records.path}", flush=True)
    failures = 0
    with httpx.Client(
        base_url=base_url,
        timeout=timeout_seconds,
        event_hooks={"request": [records.request_hook], "response": [records.response_hook]},
    ) as client:
        try:
            config = client.get("/api/console/config", timeout=min(timeout_seconds, 10))
            config.raise_for_status()
            current_config = config.json()
            if not isinstance(current_config, dict):
                raise ValueError("配置响应不是对象")
        except (httpx.HTTPError, ValueError) as exc:
            report["run_state"] = "interrupted"
            report["summary"] = summarize(report["rows"])
            session["ended_at_utc"] = utc_now()
            records.append("config_failed", error_type=type(exc).__name__)
            save_report(output, report)
            raise
        previous_config = report.get("config") or {}
        identity_keys = ("llm_model", "knowledge_version", "retrieval_mode", "reranker_enabled")
        if resume and any(
            key in previous_config and previous_config[key] != current_config.get(key)
            for key in identity_keys
        ):
            report["run_state"] = "interrupted"
            session["ended_at_utc"] = utc_now()
            records.append("config_mismatch")
            save_report(output, report)
            raise ValueError("服务模型、知识版本或检索配置已变化，请使用新的报告分别实验")
        report["config"] = current_config
        report.pop("stopped_reason", None)
        for position, case in enumerate(cases, start=1):
            if case["case_id"] in completed:
                continue
            started = time.perf_counter()
            print(
                f"[{position}/{len(cases)}] {case['case_id']} 开始，已记录 {len(completed)} 例",
                flush=True,
            )
            # 创建前保存稳定幂等键；若响应到达前退出，续跑可重放同一受理请求。
            pending = report.get("pending") or {
                "case_id": case["case_id"],
                "idempotency_key": "eval-" + str(uuid4()),
            }
            if pending["case_id"] != case["case_id"]:
                raise ValueError("待恢复病例与数据集执行顺序不一致")
            report["pending"] = pending
            records.case_id = case["case_id"]
            records.consultation_id = pending.get("consultation_id")
            pending.setdefault("started_at_utc", utc_now())
            pending["resume_attempts"] = pending.get("resume_attempts", 0) + 1
            records.append("case_started", resumed=bool(pending.get("consultation_id")))
            report["run_state"] = "running"
            save_report(output, report)
            row: dict[str, Any] = {
                "case_id": case["case_id"],
                "result": None,
                "error": None,
                "metrics": {},
            }
            try:
                if pending.get("consultation_id"):
                    response = client.get("/api/v2/consultations/" + pending["consultation_id"])
                    row["consultation_id"] = pending["consultation_id"]
                    print("  恢复查询已有会诊，不重新提交病例。", flush=True)
                else:
                    response = client.post(
                        "/api/v2/consultations",
                        json={"clinical_text": case["text"]},
                        headers={"Idempotency-Key": pending["idempotency_key"]},
                    )
                response.raise_for_status()
                result = response.json()
                identifier = result["consultation_id"]
                row["consultation_id"] = identifier
                pending["consultation_id"] = identifier
                records.consultation_id = identifier
                save_report(output, report)
                print("  " + progress_snapshot(result), flush=True)
                deadline = time.monotonic() + timeout_seconds
                next_progress = time.monotonic() + progress_seconds
                while result["status"] in {"created", "running"}:
                    if time.monotonic() > deadline:
                        raise TimeoutError("会诊仍在运行，保留 ID 供继续查询")
                    time.sleep(0.25)
                    response = client.get("/api/v2/consultations/" + identifier)
                    response.raise_for_status()
                    result = response.json()
                    if time.monotonic() >= next_progress:
                        capture_events(client, identifier, pending, records, output, report)
                        records.append(
                            "case_progress",
                            status=result.get("status"),
                            completed_stages=result.get("completed_stages", []),
                        )
                        print(
                            f"  {case['case_id']} 已等待 {time.perf_counter() - started:.1f} 秒，"
                            + progress_snapshot(result),
                            flush=True,
                        )
                        next_progress = time.monotonic() + progress_seconds
                row["result"] = result
                row["status"] = result["status"]
                row["consultation_id"] = identifier
                if result["status"] == "failed":
                    row["error"] = result.get("failure_code") or "RUN_FAILED"
                captured = capture_events(client, identifier, pending, records, output, report)
                row["events"] = pending.get("events", [])
                row["event_capture"] = {
                    "status": "complete" if captured else "partial_or_unavailable",
                    "errors": pending.get("observation_errors", []),
                    "last_cursor": pending.get("event_cursor"),
                }
                if row["event_capture"]["errors"]:
                    row["event_capture"]["history"] = (
                        "collection_failed_then_recovered"
                        if captured
                        else "unrecovered_collection_failure"
                    )
                row["diagnostics"] = records.backend_records(identifier)
                records.append("backend_diagnostics", diagnostics=row["diagnostics"])
                row["stage_observability"] = stage_summary(row["events"])
                row["usage_by_capability"] = usage_summary(row["events"])
                paths = (result.get("comparison") or {}).get("ordered_paths", [])
                hypotheses = (result.get("localization") or {}).get("candidates", [])
                expected_channels = (
                    case.get("expected_channels") or case.get("reference_channels") or []
                )
                expected_formulas = (
                    case.get("expected_formulas") or case.get("reference_formulas") or []
                )
                row["metrics"] = {
                    "channel_top1": recall(
                        hypotheses[0]["six_channels"] if hypotheses else [], expected_channels
                    ),
                    "channel_top3": recall(
                        [c for h in hypotheses[:3] for c in h["six_channels"]], expected_channels
                    ),
                    "formula_top1": recall(
                        [p["formula_name"] for p in paths[:1]], expected_formulas
                    ),
                    "formula_top3": recall(
                        [p["formula_name"] for p in paths[:3]], expected_formulas
                    ),
                    "retrieval_formula_top5": recall(
                        [
                            c["formula_name"]
                            for c in (result.get("retrieval") or {}).get("formula_candidates", [])
                        ],
                        expected_formulas,
                    ),
                }
                model_events = [
                    e for e in row["events"] if e["event"] in {"model_completed", "model_failed"}
                ]
                row["usage"] = {
                    "count_semantics": "legacy_all_capabilities;see_usage_by_capability",
                    "llm_calls": len(model_events),
                    "input_tokens": sum(
                        int(e["payload"].get("prompt_tokens", 0)) for e in model_events
                    ),
                    "output_tokens": sum(
                        int(e["payload"].get("completion_tokens", 0)) for e in model_events
                    ),
                    "usage_status": (
                        "available"
                        if model_events
                        and all(
                            e["payload"].get("usage_status") == "available" for e in model_events
                        )
                        else "partial_or_unavailable"
                    ),
                    "estimated_cost": "unavailable",
                }
            except (httpx.HTTPError, TimeoutError, KeyError, ValueError) as exc:
                records.append("client_error", error_type=type(exc).__name__)
                # 受理状态不确定或仍运行时保留幂等身份，下次仅恢复，不追加假终态失败。
                report["run_state"] = "interrupted"
                report["summary"] = summarize(report["rows"])
                report["stopped_reason"] = "客户端请求或响应失败；保留pending供恢复"
                pending["last_client_error"] = {
                    "error_type": type(exc).__name__,
                    "at_utc": utc_now(),
                }
                if pending.get("consultation_id"):
                    capture_events(
                        client, pending["consultation_id"], pending, records, output, report
                    )
                save_report(output, report)
                session["ended_at_utc"] = utc_now()
                records.append("session_interrupted", error_type=type(exc).__name__)
                save_report(output, report)
                return report
            except KeyboardInterrupt:
                report["run_state"] = "interrupted"
                session["ended_at_utc"] = utc_now()
                records.append("session_interrupted", error_type="KeyboardInterrupt")
                save_report(output, report)
                print("\n[测评] 已中断并保存进度；使用相同命令加 --resume 继续。", flush=True)
                raise
            row["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            row["started_at_utc"] = pending["started_at_utc"]
            row["ended_at_utc"] = utc_now()
            row["session_id"] = records.session_id
            row["latency_scope"] = "current_process_attempt_including_observation"
            row["client_attempts"] = pending["resume_attempts"]
            report["rows"].append(row)
            report.pop("pending", None)
            completed.add(case["case_id"])
            failures = failures + 1 if row["error"] else 0
            report["summary"] = summarize(report["rows"])
            save_report(output, report)
            records.append(
                "case_finished",
                status=row.get("status"),
                failure_code=row["error"],
                latency_ms=row["latency_ms"],
                event_capture=row.get("event_capture"),
            )
            print(
                f"[{position}/{len(cases)}] {case['case_id']} "
                f"{'失败=' + row['error'] if row['error'] else row.get('status', 'unknown')}；"
                f"耗时 {row['latency_ms'] / 1000:.1f} 秒；"
                f"累计记录 {len(completed)} 例，失败 {report['summary']['failed']} 例。",
                flush=True,
            )
            if failures >= stop_after_failures:
                report["stopped_reason"] = (
                    "连续病例技术失败，停止额外在线消耗（含结构与引用校验失败）"
                )
                print("[测评] " + report["stopped_reason"], flush=True)
                break
    report["run_state"] = "stopped" if report.get("stopped_reason") else "finished"
    session["ended_at_utc"] = utc_now()
    records.append("session_finished", run_state=report["run_state"])
    save_report(output, report)
    print(f"[测评] 本次运行结束，报告：{output}", flush=True)
    return report


def main() -> int:
    """提供真实 API 在线实验参数，失败返回非零退出码。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=360)
    parser.add_argument("--resume", action="store_true", help="恢复相同输出报告，跳过已记录病例")
    parser.add_argument("--progress-seconds", type=float, default=5, help="终端进度输出间隔")
    parser.add_argument(
        "--records-dir", type=Path, help="辅助日志目录；默认保存在报告旁或上级工程目录"
    )
    parser.add_argument("--backend-log", type=Path, help="已由后端--diagnostic-log生成的安全JSONL")
    args = parser.parse_args()
    report = run(
        args.dataset,
        args.base_url,
        args.output,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
        resume=args.resume,
        progress_seconds=args.progress_seconds,
        records_dir=args.records_dir,
        backend_log=args.backend_log,
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    return int(bool(report["summary"]["failed"]) or report.get("run_state") == "interrupted")


if __name__ == "__main__":
    raise SystemExit(main())
