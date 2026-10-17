"""QA V2.1 独立在线评测：调用现有 QA API，记录真实来源、用量、延迟和多轮结果。

不生成医学参考答案、不改临床图或知识库；RAGAS 使用独立 Judge 配置。
题库仅研发用例，未经过专家审核；所有错误仍纳入报告，不把空证据标为回答成功。
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
from pathlib import Path
from uuid import uuid4

import httpx


def percentile(values: list[float], fraction: float) -> float | None:
    """最近秩分位数只取实际运行延迟，空运行返回不可用。"""
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


async def run(args: argparse.Namespace) -> dict[str, object]:
    """逐会话运行多轮问题，每题落盘；只核验编号身份，不冒充医学忠实性。"""
    judge_config = tuple(
        os.getenv(name, "")
        for name in ("RAGAS_JUDGE_MODEL", "RAGAS_JUDGE_BASE_URL", "RAGAS_JUDGE_API_KEY")
    )
    if args.ragas and not all(judge_config):
        raise ValueError("RAGAS 必须提供独立 Judge MODEL/BASE_URL/API_KEY")
    cases = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("题库必须非空且 case_id 唯一")
    report: dict[str, object] = {
        "dataset_digest": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "label_status": "研发题库，未独立专家审核",
        "rows": [],
        "ragas_scored": False,
    }
    rows: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:
        for case in cases[: args.limit] if args.limit else cases:
            session = None
            session_error = None
            try:
                created = await client.post(
                    "/api/v2/knowledge/sessions", json={"title": case["case_id"]}
                )
                created.raise_for_status()
                session = created.json()["session_id"]
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                session_error = type(exc).__name__
            for index, question in enumerate(case["questions"]):
                started = time.perf_counter()
                row = {
                    "case_id": case["case_id"],
                    "round": index + 1,
                    "question": question,
                    "response": None,
                    "error": None,
                }
                try:
                    if session_error:
                        raise ValueError("QA 会话创建失败：" + session_error)
                    response = await client.post(
                        "/api/v2/knowledge/chat",
                        json={
                            "session_id": session,
                            "message": question,
                            "client_message_id": str(uuid4()),
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    originals = []
                    for source in data["sources"]:
                        result = await client.get(
                            "/api/v1/knowledge/evidence/" + source["evidence_id"]
                        )
                        result.raise_for_status()
                        item = result.json()
                        if item["evidence_id"] != source["evidence_id"]:
                            raise ValueError("精确来源身份不匹配")
                        originals.append(item["content"])
                    row["response"] = data
                    row["retrieved_contexts"] = originals
                    row["source_identity_verified"] = True
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    row["error"] = type(exc).__name__
                    if isinstance(exc, httpx.HTTPStatusError):
                        # 只保留服务端安全业务码，不把请求头、密钥或原始 provider 正文落盘。
                        row["http_status"] = exc.response.status_code
                        try:
                            row["error_code"] = exc.response.json().get("error", {}).get("code")
                        except (ValueError, AttributeError):
                            row["error_code"] = None
                row["latency_ms"] = int((time.perf_counter() - started) * 1000)
                rows.append(row)
                report["rows"] = rows
                args.output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
    if args.ragas:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import Faithfulness

        model, url, key = judge_config
        async with AsyncOpenAI(base_url=url, api_key=key) as judge_client:
            metric = Faithfulness(llm=llm_factory(model, client=judge_client, temperature=0))
            for row in rows:
                response = row["response"]
                if (
                    row["error"]
                    or not row.get("retrieved_contexts")
                    or response["status"] != "answered"
                ):
                    row["faithfulness"] = None
                    continue
                score = await metric.ascore(
                    user_input=row["question"],
                    response=response["answer"],
                    retrieved_contexts=row["retrieved_contexts"],
                )
                row["faithfulness"] = float(score.value)
            report["ragas_scored"] = True
    latency = [float(r["latency_ms"]) for r in rows]
    report["summary"] = {
        "count": len(rows),
        "failures": sum(bool(r["error"]) for r in rows),
        "answered": sum(
            bool(r["response"] and r["response"]["status"] == "answered") for r in rows
        ),
        "p50_ms": percentile(latency, 0.5),
        "p95_ms": percentile(latency, 0.95),
        "source_identity_note": "身份有效不代表每个医学断言均忠实于来源",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """明确选择在线目标，错误保留非零退出码，不自动修改服务配置。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("eval/datasets/qa_v21.jsonl"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ragas", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(run(args))
    summary = report["summary"]
    print(json.dumps(summary, ensure_ascii=False))
    if isinstance(summary, dict) and summary.get("failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
