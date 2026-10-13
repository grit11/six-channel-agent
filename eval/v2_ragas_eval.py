"""V2 RAGAS 辅助实验：执行独立 Judge 的忠实度评分，不代表临床准确率。

上下文只来自真实检索来源，输入来自患者事实原文；参考不足时不计算临床正确性。
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any


def build_rows(source: Path) -> list[dict[str, Any]]:
    """仅纳入有真实来源及比较结果的样本，不把空结果当忠实成功。"""
    payload: Any = json.loads(source.read_text(encoding="utf-8-sig"))
    raw_records: Any = payload.get("rows", []) if isinstance(payload, dict) else payload
    records: list[dict[str, Any]] = [item for item in raw_records if isinstance(item, dict)]
    rows = []
    for record in records:
        result = record.get("result") or {}
        comparison = result.get("comparison") or {}
        evidence = result.get("clinical_evidence") or result.get("evidence") or {}
        sources = [
            item["excerpt"]
            for candidate in (result.get("retrieval") or {}).get("formula_candidates", [])
            for item in candidate["source_evidence"]
        ]
        sources += [item["excerpt"] for item in comparison.get("source_evidence", [])]
        paths = comparison.get("ordered_paths") or []
        if record.get("error") or not sources or not paths:
            continue
        rows.append(
            {
                "case_id": record["case_id"],
                "user_input": evidence["raw_text"],
                "response": "\n".join(
                    p["brief_rationale"] + " " + p["differential_summary"] for p in paths
                ),
                "retrieved_contexts": list(dict.fromkeys(sources)),
            }
        )
    return rows


async def evaluate_rows(
    rows: list[dict[str, Any]], model: str, base_url: str, key: str
) -> list[dict[str, Any]]:
    """使用独立 Judge 真正执行 RAGAS，配置缺失或错误不转为通过。"""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    async with AsyncOpenAI(base_url=base_url, api_key=key) as client:
        judge = llm_factory(model, client=client, temperature=0)
        metric = Faithfulness(llm=judge)
        results = []
        for row in rows:
            try:
                score = await metric.ascore(
                    user_input=row["user_input"],
                    response=row["response"],
                    retrieved_contexts=row["retrieved_contexts"],
                )
                results.append(
                    {"case_id": row["case_id"], "faithfulness": float(score.value), "error": None}
                )
            except Exception as exc:  # noqa: BLE001 - Judge SDK exceptions vary by provider.
                results.append(
                    {"case_id": row["case_id"], "faithfulness": None, "error": type(exc).__name__}
                )
        return results


def main() -> int:
    """Judge 身份与被测模型配置分开；仅显式 export-only 时不评分。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--judge-model", default=os.getenv("RAGAS_JUDGE_MODEL", ""))
    parser.add_argument("--judge-base-url", default=os.getenv("RAGAS_JUDGE_BASE_URL", ""))
    args = parser.parse_args()
    rows = build_rows(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.export_only:
        report = {"status": "export_only", "rows": rows, "scored": False}
    else:
        key = os.getenv("RAGAS_JUDGE_API_KEY", "")
        if not key or not args.judge_model or not args.judge_base_url:
            parser.error(
                "请设置独立 RAGAS_JUDGE_MODEL/BASE_URL/API_KEY；不能用被测模型密钥隐式替代"
            )
        if not rows:
            parser.error("没有含真实来源与比较结果的可评分样本")
        result = asyncio.run(evaluate_rows(rows, args.judge_model, args.judge_base_url, key))
        report = {
            "status": "scored",
            "judge_model": args.judge_model,
            "temperature": 0,
            "metric": "faithfulness",
            "clinical_accuracy": None,
            "rows": result,
            "excluded_metrics": ["Context Recall / Factual Correctness 缺少专家审核参考"],
        }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    scored_rows: Any = report.get("rows", [])
    return int(any(isinstance(r, dict) and r.get("error") for r in scored_rows))


if __name__ == "__main__":
    raise SystemExit(main())
