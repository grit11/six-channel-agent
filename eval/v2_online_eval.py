"""V2 在线实验：调用真实会诊 API，记录暂停、来源、usage 和研发银标指标。

不自动审核或补造患者事实；暂停是合法结果，RAGAS 与临床参考指标独立记录。
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx


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


def run(
    dataset: Path,
    base_url: str,
    output: Path,
    *,
    limit: int | None = None,
    timeout_seconds: float = 360,
    stop_after_failures: int = 3,
) -> dict[str, Any]:
    """逐例保存报告，受理后轮询权威快照，网络错误保留类别不伪装成功。"""
    cases = load_cases(dataset)
    if limit:
        cases = cases[:limit]
    report: dict[str, Any] = {
        "protocol": "clinical-v2.0",
        "dataset_digest": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "label_status": "研发银标",
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with httpx.Client(base_url=base_url, timeout=timeout_seconds) as client:
        config = client.get("/api/console/config")
        config.raise_for_status()
        report["config"] = config.json()
        for case in cases:
            started = time.perf_counter()
            row: dict[str, Any] = {
                "case_id": case["case_id"],
                "result": None,
                "error": None,
                "metrics": {},
            }
            try:
                response = client.post(
                    "/api/v2/consultations",
                    json={"clinical_text": case["text"]},
                    headers={
                        "Idempotency-Key": "eval-"
                        + report["dataset_digest"][:12]
                        + "-"
                        + case["case_id"]
                        + "-"
                        + str(time.time_ns())
                    },
                )
                response.raise_for_status()
                result = response.json()
                identifier = result["consultation_id"]
                row["consultation_id"] = identifier
                deadline = time.monotonic() + timeout_seconds
                while result["status"] in {"created", "running"}:
                    if time.monotonic() > deadline:
                        raise TimeoutError("会诊仍在运行，保留 ID 供继续查询")
                    time.sleep(0.25)
                    response = client.get("/api/v2/consultations/" + identifier)
                    response.raise_for_status()
                    result = response.json()
                row["result"] = result
                row["status"] = result["status"]
                row["consultation_id"] = identifier
                # SSE 包含真实 Provider usage；响应原文只在受控本地实验报告保存。
                with client.stream(
                    "GET", "/api/v2/consultations/" + identifier + "/events"
                ) as events:
                    events.raise_for_status()
                    row["events"] = [
                        json.loads(line[6:])
                        for line in events.iter_lines()
                        if line.startswith("data: ")
                    ]
                if result["status"] == "failed":
                    row["error"] = result.get("failure_code") or "RUN_FAILED"
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
                row["error"] = type(exc).__name__
            row["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            report["rows"].append(row)
            failures = failures + 1 if row["error"] else 0
            report["summary"] = summarize(report["rows"])
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            if failures >= stop_after_failures:
                report["stopped_reason"] = "连续 Provider/运行错误，停止额外在线消耗"
                break
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    """提供真实 API 在线实验参数，失败返回非零退出码。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=360)
    args = parser.parse_args()
    report = run(
        args.dataset,
        args.base_url,
        args.output,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    return int(bool(report["summary"]["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
