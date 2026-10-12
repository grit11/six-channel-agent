"""六经公开结果的 RAGAs 评测入口。

职责：从真实 system_eval 快照建立评分样本，输出逐案指标、分母与失败状态。
依赖 KnowledgeService 核验引用；语义评分经现有配置/模型工厂注入评判器。
边界：上下文是最终 Top-3 引用并集，非完整原始检索排名；不修改推理和银标。
"""

# 直接执行 eval 脚本前需加入仓库根目录，随后才可导入 backend 和 eval。

import argparse
import asyncio
import json
import math
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from typing import cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.config import Settings
from backend.app.core.dependencies import open_llm
from backend.app.core.exceptions import ApplicationError
from backend.app.knowledge.service import KnowledgeService
from backend.app.schemas.reasoning_api import ReasoningResponse
from eval.benchmark_eval import load_details
from eval.evaluation_tools import digest
from eval.system_eval import load_cases


class ScoringSample(BaseModel):
    """可序列化的评分输入，case_id 与分组不传入业务推理。"""

    model_config = ConfigDict(extra="forbid")
    case_id: str
    split: str
    user_input: str
    response: str
    retrieved_contexts: list[str]
    retrieved_context_ids: list[str]
    reference_contexts: list[str]
    reference_context_ids: list[str]
    reference: str | None = None
    context_meta: dict[str, object] = Field(default_factory=dict)


def public_response(result: ReasoningResponse) -> str:
    """从 Top-3 DTO 构造公开候选摘要，不复制来源正文冒充生成答案。"""
    lines = ["以下是待医生核验的候选路径，排序不代表临床概率。"]
    for path in result.reasoning_paths[:3]:
        lines.extend(
            [
                (
                    f"候选{path.rank}：{'、'.join(path.six_channels)}；{path.syndrome_name}；"
                    f"方证参考：{path.formula_pattern or '未确定'}；"
                    f"关联层级：{path.association_scope}"
                ),
                "支持：" + "、".join(path.supporting_findings),
                "反证：" + "、".join(path.contradicting_findings),
                "缺失：" + "、".join(path.missing_key_findings),
            ]
        )
    if not result.reasoning_paths:
        lines.append("无足够依据构造候选。")
    return "\n".join(lines)


def reference_answer(detail: dict[str, object]) -> str | None:
    """只把明确银标组成评判参考；欠定病例不制造唯一标准答案。"""
    labels = []
    for field, name in (("reference_channels", "六经候选"), ("reference_formulas", "方证参考")):
        values = detail.get(field)
        if isinstance(values, list) and values:
            labels.append(name + "：" + "、".join(str(value) for value in values))
    return "；".join(labels) if labels else None


async def prepare_samples(
    report_path: Path,
    folder: Path,
    knowledge: KnowledgeService,
    *,
    allow_partial: bool = False,
    context_scope: str = "citations",
    max_contexts: int | None = None,
    max_context_chars: int | None = None,
    include_clinical_input: bool = False,
) -> list[ScoringSample]:
    """核验全量身份、原文和引用正文；参考来源只在结果生成后用于评分。"""
    cases = await asyncio.to_thread(load_cases, folder / "cases_eval_compatible.jsonl")
    details = await asyncio.to_thread(load_details, folder / "cases_detailed.jsonl")
    report = json.loads(await asyncio.to_thread(report_path.read_text, encoding="utf-8"))
    if report.get("status", "completed") != "completed":
        raise ValueError("源报告尚未完成，不能进入评分")
    if report.get("expected_sample_count", len(cases)) != len(cases):
        raise ValueError("源报告的期望分母与输入集不一致")
    if context_scope not in {"citations", "retrieved"}:
        raise ValueError("上下文范围必须为 citations/retrieved")
    records = report["cases"]
    if allow_partial:
        ids = {row["case_id"] for row in records}
        all_ids = {case.case_id for case in cases}
        if not records or len(ids) != len(records) or not ids.issubset(all_ids):
            raise ValueError("部分报告必须包含唯一且属于数据集的病例")
        if report.get("status") != "completed" or report.get("complete_benchmark") is not False:
            raise ValueError("部分报告必须已结束且明确标记非完整运行")
        cases = [case for case in cases if case.case_id in ids]
    if len(records) != len(cases) or {record["case_id"] for record in records} != {
        case.case_id for case in cases
    }:
        raise ValueError("评分报告必须包含完整输入集且不能重复 ID")
    indexed = {row["case_id"]: row for row in records}
    samples = []
    for case in cases:
        if (
            indexed[case.case_id].get("input_text") != case.text
            or details[case.case_id]["text"] != case.text
        ):
            raise ValueError("报告与参考输入不一致")
        if indexed[case.case_id].get("result") is None:
            if not indexed[case.case_id].get("execution_error"):
                raise ValueError("缺少结果的病例必须具有明确执行失败记录")
            continue
        result = ReasoningResponse.model_validate(indexed[case.case_id]["result"])
        detail = details[case.case_id]
        if indexed[case.case_id].get("input_text") != case.text or detail["text"] != case.text:
            raise ValueError("报告与参考输入不一致")
        catalog = {item.evidence_id: item for item in result.evidence_catalog}
        # 最终引用只按公开路径遍历排序；不把它冒充 QueryBuilder 的召回排名。
        identifiers = list(
            dict.fromkeys(
                identifier
                for path in result.reasoning_paths[:3]
                for identifier in (
                    path.framework_evidence
                    + path.classic_evidence
                    + path.formula_evidence
                    + path.differential_evidence
                    + path.case_evidence
                )
            )
        )
        references = list(
            dict.fromkeys(
                str(source["evidence_id"])
                for source in cast(list[dict[str, object]], detail["source_refs"])
            )
        )
        if not set(identifiers).issubset(catalog):
            raise ValueError("路径引用缺少公开目录")
        exact = await knowledge.get_evidence_by_ids(identifiers)
        if any(item.content != catalog[item.evidence_id].content for item in exact):
            raise ValueError("输出引用正文与知识版本不一致")
        reference_evidence = await knowledge.get_evidence_by_ids(references)
        contexts = [catalog[item].content for item in identifiers]
        if context_scope == "retrieved":
            if "retrieved_contexts" not in indexed[case.case_id]:
                raise ValueError("旧报告缺少真实检索追踪，不能补造召回上下文")
            observed = indexed[case.case_id]["retrieved_contexts"]
            identifiers = [item["evidence_id"] for item in observed]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("检索追踪包含重复来源")
            verified = await knowledge.get_evidence_by_ids(identifiers)
            if [item.content for item in verified] != [item["content"] for item in observed]:
                raise ValueError("检索正文与知识版本不一致")
            contexts = [item.content for item in verified]
        total = len(contexts)
        contexts = contexts[:max_contexts]
        identifiers = identifiers[:max_contexts]
        shortened = (
            sum(len(text) > max_context_chars for text in contexts) if max_context_chars else 0
        )
        if max_context_chars:
            contexts = [text[:max_context_chars] for text in contexts]
        meta = {
            "scope": context_scope,
            "total": total,
            "used": len(contexts),
            "omitted": total - len(contexts),
            "shortened": shortened,
            "clinical_input_included": include_clinical_input,
        }
        samples.append(
            ScoringSample(
                case_id=case.case_id,
                split=str(detail["split"]),
                user_input=case.text,
                response=public_response(result),
                retrieved_contexts=contexts,
                retrieved_context_ids=identifiers,
                reference_contexts=[item.content for item in reference_evidence],
                reference_context_ids=references,
                reference=reference_answer(detail),
                context_meta=meta,
            )
        )
    return samples


def export(output: Path, report: dict[str, object]) -> None:
    """以标准 JSON 保存逐条结果，失败/NaN 保持 null 并记录状态。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def metric_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """每项独立记录成功、未标注、失败分母，不给无法评分项补零。"""
    names = {name for row in rows for name in cast(dict[str, object], row["metrics"])}
    output: dict[str, object] = {}
    for name in sorted(names):
        values = [cast(dict[str, dict[str, object]], row["metrics"])[name] for row in rows]
        scores = [cast(float, value["score"]) for value in values if value["status"] == "scored"]
        output[name] = {
            "mean": mean(scores) if scores else None,
            "scored": len(scores),
            "failed": sum(value["status"] == "failed" for value in values),
            "unavailable": sum(value["status"] == "unavailable" for value in values),
        }
    return output


async def score_samples(
    samples: list[ScoringSample],
    output: Path,
    settings: Settings,
    *,
    semantic: bool,
    context_precision: bool = False,
    provenance: dict[str, object] | None = None,
    expected_sample_count: int = 48,
    failed_case_ids: list[str] | None = None,
    allow_silver_reference: bool = False,
) -> bool:
    """运行真实 RAGAs 指标；ID 指标离线，语义指标显式开启且逐项异常可审计。"""
    # 固定 0.4 评测 API，避免未来版本改用不同工厂而悄然绑定默认厂商。
    from ragas import SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        IDBasedContextPrecision,
        IDBasedContextRecall,
        LLMContextPrecisionWithoutReference,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.run_config import RunConfig

    metrics = [IDBasedContextPrecision(), IDBasedContextRecall()]
    semantic_names = [
        "faithfulness",
        "llm_context_precision_with_reference",
        "context_recall",
        "context_utilization",
    ]
    rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "version": "ragas-six-channel/v1.1",
        "ragas_version": version("ragas"),
        "context_scope": (
            samples[0].context_meta.get("scope", "citations") if samples else "unavailable"
        ),
        "label_type": "silver_reference_pending_expert_review",
        "limitations": "ID 参考集合并非穷尽相关来源；模型忠实度仅为公开候选摘要代理指标。",
        "semantic_enabled": semantic,
        "silver_reference_enabled": allow_silver_reference,
        "judge_model": settings.llm_model if semantic else None,
        "judge_provider": settings.llm_provider if semantic else None,
        "sample_count": len(samples),
        "source_report": provenance or {},
        "expected_sample_count": expected_sample_count,
        "complete_benchmark": len(samples) == expected_sample_count,
        "source_execution_failed_ids": failed_case_ids or [],
        "status": "running",
        "aggregate": {},
        "cases": rows,
    }

    async def run_metrics() -> None:
        """顺序评判控制调用成本，每条完成即落盘，终止时保留未完成状态。"""
        for sample in samples:
            inputs = SingleTurnSample(
                **sample.model_dump(exclude={"case_id", "split", "context_meta"}, exclude_none=True)
            )
            values: dict[str, dict[str, object]] = {}
            for metric in metrics:
                metric_inputs = inputs
                if metric.name == "faithfulness" and sample.context_meta.get(
                    "clinical_input_included"
                ):
                    # 仅改变忠实度支撑范围，病例原文不能混入召回ID和上下文利用率。
                    metric_inputs = inputs.model_copy(
                        update={
                            "retrieved_contexts": [
                                "医生输入（病例事实，非文献）：" + sample.user_input,
                                *sample.retrieved_contexts,
                            ]
                        }
                    )
                missing = (
                    not sample.reference_context_ids
                    if metric.name.startswith("id_based")
                    else not sample.retrieved_contexts
                    or (
                        metric.name in {"context_recall", "llm_context_precision_with_reference"}
                        and (sample.reference is None or not allow_silver_reference)
                    )
                )
                if missing:
                    values[metric.name] = {"score": None, "status": "unavailable"}
                    continue
                try:
                    score = await metric.single_turn_ascore(
                        metric_inputs, timeout=settings.llm_timeout_seconds * 4
                    )
                    values[metric.name] = {
                        "score": score if math.isfinite(score) else None,
                        "status": "scored" if math.isfinite(score) else "failed",
                    }
                except Exception as exc:  # noqa: BLE001 -- 第三方指标异常须逐项记账，不能丢病例。
                    # Provider 异常可能带 URI、响应或输入；报告仅保留类型和安全业务码。
                    values[metric.name] = {
                        "score": None,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_code": exc.code if isinstance(exc, ApplicationError) else None,
                    }
            for name in semantic_names:
                if name not in values:
                    values[name] = {"score": None, "status": "unavailable"}
            rows.append(
                {
                    "case_id": sample.case_id,
                    "split": sample.split,
                    "metrics": values,
                    "context_meta": sample.context_meta,
                }
            )
            report["aggregate"] = metric_summary(rows)
            await asyncio.to_thread(export, output, report)
            print(f"RAGAs {len(rows)}/{len(samples)} {sample.case_id}", flush=True)

    if semantic and samples:
        from eval.ragas_adapters import ProjectRagasLLM

        async with open_llm(settings) as provider:
            judge = ProjectRagasLLM(provider)
            judge.set_run_config(
                RunConfig(max_retries=1, timeout=math.ceil(settings.llm_timeout_seconds))
            )
            metrics.extend(
                [
                    Faithfulness(llm=judge),
                    LLMContextRecall(llm=judge),
                    LLMContextPrecisionWithoutReference(llm=judge, name="context_utilization"),
                ]
            )
            if context_precision:
                metrics.append(LLMContextPrecisionWithReference(llm=judge))
            await run_metrics()
    else:
        await run_metrics()
    failed = any(
        metric["status"] == "failed"
        for row in rows
        for metric in cast(dict[str, dict[str, object]], row["metrics"]).values()
    )
    failed = failed or bool(failed_case_ids)
    report["status"] = "failed" if failed else "completed"
    if not samples:
        report["aggregate"] = {
            name: {"mean": None, "scored": 0, "failed": 0, "unavailable": expected_sample_count}
            for name in ["id_based_context_precision", "id_based_context_recall", *semantic_names]
        }
        report["unavailable_reason"] = "source_report_has_no_successful_reasoning_results"
    report["groups"] = {
        split: metric_summary([row for row in rows if row["split"] == split])
        for split in sorted({sample.split for sample in samples})
    }
    await asyncio.to_thread(export, output, report)
    return not failed


async def run(args: argparse.Namespace) -> int:
    """准备评分输入后执行，limit 是显式部分评测且不改变完整输入报告验证。"""
    if args.context_precision and not args.semantic:
        raise ValueError("--context-precision 需要同时指定 --semantic")
    if args.context_precision and not args.allow_silver_reference:
        raise ValueError("--context-precision 需要 --allow-silver-reference")
    if any(
        value is not None and value < 1 for value in (args.max_contexts, args.max_context_chars)
    ):
        raise ValueError("上下文数量和字符上限须大于零")
    if await asyncio.to_thread(args.output.exists):
        raise FileExistsError("评分报告已存在，请指定新的 --output")
    report_bytes = await asyncio.to_thread(args.report.read_bytes)
    source_report = json.loads(report_bytes)
    provenance = {
        "sha256": sha256(report_bytes).hexdigest(),
        "mode": source_report.get("mode"),
        "knowledge_version": source_report.get("knowledge_version"),
        "version": source_report.get("version"),
        "runtime": source_report.get("runtime"),
    }
    settings = Settings(_env_file=args.env_file or ROOT / ".env")  # type: ignore[call-arg]
    knowledge = await KnowledgeService.from_directory(
        settings.knowledge_directory, settings.knowledge_base_version
    )
    samples = await prepare_samples(
        args.report,
        args.folder,
        knowledge,
        allow_partial=args.allow_partial,
        context_scope=args.context_scope,
        max_contexts=args.max_contexts,
        max_context_chars=args.max_context_chars,
        include_clinical_input=args.include_clinical_input,
    )
    if args.judge_env_file:
        settings = Settings(_env_file=args.judge_env_file)  # type: ignore[call-arg]
    provenance["judge_configuration"] = {
        "endpoint_sha256": digest(settings.llm_base_url),
        "temperature": settings.llm_temperature,
        "timeout_seconds": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
        "context_scope": args.context_scope,
        "max_contexts": args.max_contexts,
        "max_context_chars": args.max_context_chars,
        "include_clinical_input": args.include_clinical_input,
        "independent_config": args.judge_env_file is not None,
        "bridge_sha256": sha256(
            await asyncio.to_thread((ROOT / "eval/ragas_adapters.py").read_bytes)
        ).hexdigest(),
    }
    failed_case_ids = [
        row["case_id"] for row in source_report["cases"] if row.get("result") is None
    ]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit 必须大于零")
        samples = samples[: args.limit]
    await asyncio.to_thread(
        export,
        args.output.with_suffix(".samples.json"),
        {"samples": [sample.model_dump(mode="json") for sample in samples]},
    )
    if args.prepare_only:
        print(f"Prepared {len(samples)} samples; no model calls")
        return 0
    passed = await score_samples(
        samples,
        args.output,
        settings,
        semantic=args.semantic,
        context_precision=args.context_precision,
        provenance=provenance,
        expected_sample_count=source_report.get(
            "expected_sample_count", len(source_report["cases"])
        ),
        failed_case_ids=failed_case_ids,
        allow_silver_reference=args.allow_silver_reference,
    )
    return 0 if passed else 1


def main() -> int:
    """默认在线语义评判，提供显式离线 ID 开关，不假报未执行的分数。"""
    parser = argparse.ArgumentParser(description="公开 Top-3 引用的 RAGAs 评测")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--folder", type=Path, default=ROOT / "eval/datasets/benchmark48")
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmark48/ragas_online.json")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--judge-env-file", type=Path, help="独立裁判配置，复用项目模型工厂")
    parser.add_argument("--context-scope", choices=["citations", "retrieved"], default="citations")
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument("--max-context-chars", type=int)
    parser.add_argument("--include-clinical-input", action="store_true")
    parser.add_argument(
        "--allow-silver-reference",
        action="store_true",
        help="显式开启银标 ContextRecall/有参Precision",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--semantic", dest="semantic", action="store_true", help="调用已配置模型评判语义（默认）"
    )
    mode.add_argument("--offline", dest="semantic", action="store_false", help="只运行 ID 指标")
    parser.set_defaults(semantic=True)
    parser.add_argument(
        "--context-precision", action="store_true", help="对每个最终引用另行评分，增加调用次数"
    )
    parser.add_argument("--prepare-only", action="store_true", help="仅核验评分样本")
    parser.add_argument("--limit", type=int, help="显式仅评分前 N 条，报告标明实际数量")
    parser.add_argument(
        "--allow-partial", action="store_true", help="显式接收已完成的部分源报告，仍标记非全量"
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
