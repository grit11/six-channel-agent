"""Parser/Reasoning/Differential/RAG 在线评测执行器与标注契约。

复用模型与检索依赖工厂，默认在线解析和推理；显式 offline 用于工程回归。
不修改知识与部署配置、不写业务会诊和 checkpoint，不生成专家金标准或隐藏推理。
"""

# 直接执行入口必须先配置仓库导入路径。

import argparse
import asyncio
import json
import math
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Literal, cast
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.config import Settings
from backend.app.core.dependencies import configure_knowledge_service, open_llm
from backend.app.core.exceptions import ApplicationError
from backend.app.graph.six_channel_graph import STAGES
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.nodes.differential_reasoner import compare_formula_candidates
from backend.app.observability.progress import progress_scope
from backend.app.rag.vector_store import build_milvus_store
from backend.app.schemas.clinical import Polarity, SixChannel
from backend.app.schemas.events import StreamEventName
from backend.app.schemas.reasoning import FormulaCandidate
from backend.app.schemas.reasoning_api import ReasoningResponse
from backend.app.services.reasoning_service import ReasoningService
from eval.evaluation_tools import (
    KnowledgeObserver,
    case_trace,
    digest,
    experiment_identity,
    observed_contexts,
    set_trace_stage,
)


def safe_execution_error(exc: Exception) -> dict[str, object]:
    """保留错误链类型、HTTP 状态和 Schema 字段，不保存响应、原文和凭据。"""
    chain: list[dict[str, object]] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        item: dict[str, object] = {"error_type": type(current).__name__}
        if isinstance(current, httpx.HTTPStatusError):
            item["http_status"] = current.response.status_code
        if isinstance(current, ValidationError):
            item["validation_fields"] = [
                {"location": list(error["loc"]), "type": error["type"]}
                for error in current.errors(include_input=False, include_context=False)
            ]
        chain.append(item)
        current = current.__cause__
    return {
        "error_type": type(exc).__name__,
        "error_code": (
            exc.code if isinstance(exc, ApplicationError) else "EVALUATION_EXECUTION_ERROR"
        ),
        "cause_chain": chain,
    }


class RetrievalLabel(BaseModel):
    """医生核对的查询相关来源 ID；分级在真实来源中核验，不从候选答案反造标签。"""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["classic", "formula", "framework", "case", "differential"]
    query: str = Field(min_length=1)
    clause_id: str | None = None
    candidates: list[str] = Field(default_factory=list)
    relevant_ids: list[str] = Field(min_length=1)


class EvaluationCase(BaseModel):
    """单条独立评测输入；空列表表示明确无当前事实，None 表示该类未标注。"""

    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=20_000)
    expected_current: dict[str, Polarity] | None = None
    expected_unknown: list[str] = Field(default_factory=list)
    conflicted_keys: list[str] = Field(default_factory=list)
    expected_channels: list[SixChannel] | None = None
    expected_formulas: list[str] | None = None
    retrieval: list[RetrievalLabel] = Field(default_factory=list)


def load_cases(path: Path) -> list[EvaluationCase]:
    """严格校验 JSONL 和唯一身份，拒绝空集和重复编号以免污染统计。"""
    cases = [
        EvaluationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("评测集不能为空，case_id 必须唯一")
    return cases


def classification_metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    """按显式标签计算微平均；零分母记零并保留原始计数。"""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def ranking_metrics(ids: list[str], relevant: list[str], k: int) -> dict[str, float]:
    """计算二元相关 Recall/MRR/NDCG，去重命中避免重复来源虚增指标。"""
    expected = set(relevant)
    seen: set[str] = set()
    hits: list[int] = []
    for position, item in enumerate(ids[:k], 1):
        if item in expected and item not in seen:
            hits.append(position)
        seen.add(item)
    dcg = sum(1 / math.log2(position + 1) for position in hits)
    ideal = sum(1 / math.log2(position + 1) for position in range(1, min(k, len(expected)) + 1))
    return {
        "recall": len(hits) / len(expected) if expected else 0.0,
        "mrr": 1 / hits[0] if hits else 0.0,
        "ndcg": dcg / ideal if ideal else 0.0,
    }


def path_label_metrics(groups: list[list[str]], relevant: Sequence[str]) -> dict[str, object]:
    """以路径为 Top-K 单位计算多标签覆盖率；同一路径两经均属于该条路径。"""
    expected = set(relevant)
    first = set(groups[0]) if groups else set()
    top3 = {label for group in groups[:3] for label in group}
    return {
        "top1_recall": len(first & expected) / len(expected) if expected else 0.0,
        "top3_recall": len(top3 & expected) / len(expected) if expected else 0.0,
        "top1_path_coverage": expected.issubset(first) if expected else None,
        "top3_path_coverage": (
            any(expected.issubset(group) for group in groups[:3]) if expected else None
        ),
        "top3_exact_path": (
            any(expected == set(group) for group in groups[:3]) if expected else None
        ),
        "top3_multilabel": classification_metrics(
            len(top3 & expected), len(top3 - expected), len(expected - top3)
        ),
    }


def usage_summary(telemetry: dict[str, object]) -> dict[str, object]:
    """聚合每次真实 Provider 尝试；部分缺 usage 时保留缺失，不伪造完整成本。"""
    events = cast(list[dict[str, object]], telemetry.get("events", []))
    completed = [
        item for item in events if item.get("event") in {"model_completed", "model_failed"}
    ]
    known = [item for item in completed if item.get("usage_status") == "available"]
    return {
        "status": (
            "available"
            if completed and len(known) == len(completed)
            else "partial" if known else "unavailable" if completed else "not_applicable"
        ),
        "llm_calls": len(completed),
        "retry_attempts": sum(cast(int, item.get("attempt", 0)) > 0 for item in completed),
        "calls_with_usage": len(known),
        "prompt_tokens": sum(cast(int, item["prompt_tokens"]) for item in known) if known else None,
        "completion_tokens": (
            sum(cast(int, item["completion_tokens"]) for item in known) if known else None
        ),
    }


async def verify_result(result: ReasoningResponse, knowledge: KnowledgeService) -> list[str]:
    """核验公开阶段、路径排名和精确引用正文；不把来源存在等同医学忠实度。"""
    errors: list[str] = []
    if (
        result.completed_stages != STAGES
        or [item.stage for item in result.stage_snapshots] != STAGES
    ):
        errors.append("六阶段方案的五个自动阶段快照不完整")
    catalog = {item.evidence_id: item for item in result.evidence_catalog}
    if len(catalog) != len(result.evidence_catalog):
        errors.append("引用目录存在重复 ID")
    exact = await knowledge.get_evidence_by_ids(list(catalog))
    if any(catalog[item.evidence_id].content != item.content for item in exact):
        errors.append("来源正文与知识库不一致")
    for rank, path in enumerate(result.reasoning_paths, 1):
        refs = (
            path.framework_evidence
            + path.classic_evidence
            + path.formula_evidence
            + path.differential_evidence
            + path.case_evidence
        )
        if not set(refs).issubset(catalog):
            errors.append("路径引用未进入精确目录")
        if path.rank != rank or path.review_status != "pending":
            errors.append("路径排名或人工审核状态异常")
        detail = path.completeness_detail
        if detail and detail.total_count != (
            detail.supported_count + detail.contradicted_count + detail.unknown_count
        ):
            errors.append("来源完整度计数不可复算")
    return errors


async def differential_regression(knowledge: KnowledgeService) -> dict[str, int]:
    """逐条执行全部来源规则；缺失患者事实时不得制造决定性正反证。"""
    passed = 0
    for number in range(1, 54):
        rule = await knowledge.get_differential_by_id(f"diff_v12_{number:03}")
        if rule is None:
            raise ValueError(f"缺少真实来源规则 {number}")
        formulas = [
            FormulaCandidate(
                formula_name=name,
                association_scope="classic_clause_based",
                evidence_ids=[rule.evidence_id],
                rank_score=0.5,
            )
            for name in rule.candidates
        ]
        results = await compare_formula_candidates(
            parse_clinical_text("尚未录入脉证"), formulas, knowledge
        )
        relevant = [
            item
            for item in results
            if any(d.source_rule_id == rule.rule_id for d in item.dimensions)
        ]
        passed += int(
            len(relevant) == len(rule.candidates) * (len(rule.candidates) - 1) // 2
            and all(
                not item.decisive_support
                and not item.decisive_contradictions
                and {d.dimension for d in item.dimensions} == set(rule.key_dimensions)
                and all(d.status == "unknown" for d in item.dimensions)
                for item in relevant
            )
        )
    return {"rules": 53, "passed": passed}


async def retrieve(label: RetrievalLabel, knowledge: KnowledgeService) -> list[str]:
    """所有检索和标签来源检查均通过 KnowledgeService，不直接读取医学数据。"""
    await knowledge.get_evidence_by_ids(label.relevant_ids)
    if label.clause_id:
        item = await knowledge.get_clause_by_id(label.clause_id)
        return [item.evidence_id] if item else []
    if label.kind == "differential":
        rules = await knowledge.search_differential(label.candidates, query=label.query, top_k=10)
        return [item.evidence_id for item in rules]
    else:
        if label.kind == "classic":
            items = await knowledge.search_classic(label.query, top_k=10)
        elif label.kind == "formula":
            items = await knowledge.search_formula(label.query, top_k=10)
        elif label.kind == "framework":
            items = await knowledge.search_core_framework(label.query, top_k=10)
        else:
            items = await knowledge.search_case(label.query, top_k=10)
    return [item.evidence_id for item in items]


def evaluation_settings(
    *, online: bool, env_file: Path | None = None, ablation: str | None = None
) -> Settings:
    """在线读取真实配置并明确启用 LLM；离线固定规则，均不改写 .env。"""
    if ablation not in {None, "A", "B", "C", "D"}:
        raise ValueError("ablation 必须为 A/B/C/D")
    if online or ablation is not None:
        group = ablation
        baseline = Settings(_env_file=env_file or ROOT / ".env")  # type: ignore[call-arg]
        return Settings(  # type: ignore[call-arg]
            _env_file=env_file or ROOT / ".env",
            clinical_parser_mode="rules" if group == "A" else "llm",
            reasoning_mode="rules" if group == "A" else "llm",
            retrieval_mode=(
                "bm25"
                if group in {"A", "B"}
                else "hybrid" if group in {"C", "D"} else baseline.retrieval_mode
            ),
            reranker_enabled=(group == "D" if group else baseline.reranker_enabled),
        )
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        clinical_parser_mode="rules",
        reasoning_mode="rules",
        retrieval_mode="bm25",
        reranker_enabled=False,
    )


@asynccontextmanager
async def evaluation_runtime(
    settings: Settings, *, online: bool
) -> AsyncIterator[tuple[KnowledgeService, ReasoningService, dict[str, object]]]:
    """复用生产模型/检索工厂，整个批次共享客户端，独立评测无需业务数据库。"""
    async with AsyncExitStack() as stack:
        knowledge = await KnowledgeService.from_directory(
            settings.knowledge_directory, settings.knowledge_base_version
        )
        backend: str | None = None
        degraded = False
        llm = None
        if online:
            if settings.reasoning_mode == "llm" or settings.clinical_parser_mode == "llm":
                llm = await stack.enter_async_context(open_llm(settings))
            vector = await build_milvus_store(settings)
            stack.push_async_callback(vector.close)
            await configure_knowledge_service(settings, knowledge, vector, stack)
            backend = vector.status.backend
            degraded = vector.status.degraded
        reasoning = ReasoningService(
            knowledge,
            llm=llm,
            reasoning_llm=llm,
            channel_weight=settings.topk_channel_weight,
            scoring_policy=settings.scoring_policy,
            evidence_gap_policy=settings.evidence_gap_policy,
            max_supplement_rounds=settings.max_supplement_rounds,
        )
        yield knowledge, reasoning, {
            "online": online,
            "clinical_parser_mode": settings.clinical_parser_mode,
            "reasoning_mode": settings.reasoning_mode,
            "retrieval_mode": settings.retrieval_mode,
            "llm_model": settings.llm_model if online else None,
            "embedding_model": settings.embedding_model if online else None,
            "reranker_model": (
                settings.reranker_model if online and settings.reranker_enabled else None
            ),
            "reranker_enabled": settings.reranker_enabled,
            "vector_backend": backend,
            "degraded": degraded,
            "checkpoint_backend": "none_evaluation",
            "consultation_storage": "none_evaluation",
            "scoring_policy": settings.scoring_policy.model_dump(mode="json"),
            "experiment_identity": await asyncio.to_thread(experiment_identity, ROOT, settings),
        }


async def evaluate(
    cases: list[EvaluationCase],
    output: Path,
    *,
    online: bool = True,
    env_file: Path | None = None,
    concurrency: int = 2,
    expected_count: int | None = None,
    case_timeout: float = 180,
    ablation: str | None = None,
    resume: bool = False,
) -> bool:
    """运行完整独立批次；部署资源初始化失败明确退出，不回退离线。"""
    if concurrency < 1 or concurrency > 8:
        raise ValueError("concurrency 必须在 1–8 之间")
    settings = evaluation_settings(online=online, env_file=env_file, ablation=ablation)
    async with evaluation_runtime(settings, online=online) as (knowledge, reasoning, runtime):
        runtime["ablation"] = ablation
        with KnowledgeObserver(knowledge):
            return await evaluate_with_runtime(
                cases,
                output,
                knowledge,
                reasoning,
                runtime,
                concurrency,
                expected_count=expected_count,
                case_timeout=case_timeout,
                resume=resume,
            )


async def evaluate_with_runtime(
    cases: list[EvaluationCase],
    output: Path,
    knowledge: KnowledgeService,
    reasoning: ReasoningService,
    runtime: dict[str, object],
    concurrency: int = 2,
    *,
    expected_count: int | None = None,
    case_timeout: float = 180,
    resume: bool = False,
) -> bool:
    """只将病例原文送入图；完成后评分，异常逐案落盘并保留失败分母。"""
    counts = [0, 0, 0]
    labelled_cases = 0
    negative_total = negative_correct = unknown_total = unknown_correct = 0
    records: list[dict[str, object]] = []
    channel_rows: list[dict[str, object]] = []
    formula_rows: list[dict[str, object]] = []
    retrieval_rows: list[dict[str, object]] = []
    telemetry_by_case: dict[str, dict[str, object]] = {}
    semaphore = asyncio.Semaphore(concurrency)
    input_digest = digest([case.model_dump(mode="json") for case in cases])
    signature = {
        "input_digest": input_digest,
        "runtime": runtime,
        "expected_sample_count": expected_count or len(cases),
        "case_timeout": case_timeout,
    }
    previous: dict[str, dict[str, object]] = {}
    if await asyncio.to_thread(output.exists):
        if not resume:
            raise FileExistsError("报告已存在，请使用 --resume 或选择新的 --output")
        old = json.loads(await asyncio.to_thread(output.read_text, encoding="utf-8"))
        if old.get("evaluation_signature") != signature:
            raise ValueError("输入、配置、代码、提示或知识身份改变，禁止混合续跑")
        old_rows = old["cases"]
        if len({row["case_id"] for row in old_rows}) != len(old_rows):
            raise ValueError("旧报告包含重复病例")
        previous = {row["case_id"]: row for row in old_rows}
        if not set(previous).issubset({case.case_id for case in cases}):
            raise ValueError("旧报告包含输入集之外的病例")

    async def analyze(
        case: EvaluationCase,
    ) -> tuple[
        EvaluationCase, ReasoningResponse | None, float, dict[str, object], list[dict[str, object]]
    ]:
        """限流在线调用；错误仅保存安全类型与业务码，不泄露请求正文或凭据。"""
        async with semaphore:
            saved = previous.get(case.case_id)
            if saved and saved.get("passed") is True and saved.get("result") is not None:
                if saved.get("input_text") != case.text:
                    raise ValueError("续跑原文不一致")
                telemetry_by_case[case.case_id] = cast(
                    dict[str, object], saved.get("telemetry", {})
                )
                return (
                    case,
                    ReasoningResponse.model_validate(saved["result"]),
                    float(cast(float, saved["runtime_seconds"])),
                    cast(dict[str, object], saved.get("telemetry", {})),
                    cast(list[dict[str, object]], saved.get("knowledge_calls", [])),
                )
            started = perf_counter()
            progress: dict[str, object] = {}
            calls: list[dict[str, object]] = []
            telemetry: dict[str, object] = {"events": []}
            telemetry_by_case[case.case_id] = telemetry

            async def track(event: StreamEventName, payload: dict[str, object]) -> None:
                """仅记录阶段与能力名称用于定位错误，事件不包含病例或模型正文。"""
                cast(list[dict[str, object]], telemetry["events"]).append(
                    {"event": event, **payload}
                )
                if payload.get("stage") is not None:
                    progress["stage"] = payload["stage"]
                    set_trace_stage(str(payload["stage"]))
                if event == "model_failed":
                    progress["capability"] = payload.get("capability")

            try:
                progress["stage"] = "clinical_parser"
                with case_trace(calls):
                    async with asyncio.timeout(case_timeout), progress_scope(track):
                        result = await reasoning.analyze_text(case.text)
                return case, result, perf_counter() - started, {}, calls
            except Exception as exc:  # noqa: BLE001 -- 批评测需保留异常病例并继续其余病例。
                return (
                    case,
                    None,
                    perf_counter() - started,
                    {**safe_execution_error(exc), **progress, "telemetry": telemetry},
                    calls,
                )

    tasks = [asyncio.create_task(analyze(case)) for case in cases]
    try:
        for task in asyncio.as_completed(tasks):
            case, result, runtime_seconds, failure, calls = await task
            status = "failed" if result is None else "completed"
            print(
                f"System {len(records)+1}/{len(cases)} {case.case_id}: {status}",
                flush=True,
            )
            if result is None:
                records.append(
                    {
                        "case_id": case.case_id,
                        "input_text": case.text,
                        "runtime_seconds": runtime_seconds,
                        "passed": False,
                        "errors": [failure["error_code"]],
                        "execution_error": failure,
                        "knowledge_calls": calls,
                        "telemetry": telemetry_by_case.get(
                            case.case_id, failure.get("telemetry", {})
                        ),
                        "token_usage": usage_summary(telemetry_by_case.get(case.case_id, {})),
                        "retrieved_contexts": observed_contexts(calls),
                        "result": None,
                    }
                )
                await asyncio.to_thread(
                    export_evaluation,
                    output,
                    {
                        "status": "running",
                        "complete_benchmark": False,
                        "passed": False,
                        "runtime": runtime,
                        "evaluation_signature": signature,
                        "cases": records,
                    },
                    records,
                    cases,
                )
                continue
            current = {
                item.key: item.polarity for item in result.clinical_evidence.current_findings()
            }
            explicit = {(key, value) for key, value in current.items() if value != "unknown"}
            errors = await verify_result(result, knowledge)
            if case.expected_current is not None:
                labelled_cases += 1
                expected = {
                    (key, value)
                    for key, value in case.expected_current.items()
                    if value != "unknown"
                }
                for index, amount in enumerate(
                    (len(explicit & expected), len(explicit - expected), len(expected - explicit))
                ):
                    counts[index] += amount
                if explicit != expected:
                    errors.append("当前事实与完整事实标签不符")
                for key, polarity in case.expected_current.items():
                    if polarity == "negative":
                        negative_total += 1
                        negative_correct += int(current.get(key) == "negative")
            unknown_keys = set(case.expected_unknown) | {
                key
                for key, polarity in (case.expected_current or {}).items()
                if polarity == "unknown"
            }
            unknown_total += len(unknown_keys)
            unknown_correct += sum(current.get(key) == "unknown" for key in unknown_keys)
            if any(current.get(key) != "unknown" for key in unknown_keys):
                errors.append("应保留未知的事实被补造或丢失")
            if not set(case.conflicted_keys).issubset(
                {item.key for item in result.clinical_evidence.contradictions}
            ) or any(key in current for key in case.conflicted_keys):
                errors.append("同时间冲突事实未隔离")
            paths = result.reasoning_paths
            for labels, predicted, rows in (
                (
                    case.expected_channels,
                    [path.six_channels for path in paths],
                    channel_rows,
                ),
                (
                    case.expected_formulas,
                    [[path.formula_pattern] if path.formula_pattern else [] for path in paths],
                    formula_rows,
                ),
            ):
                if labels:
                    rows.append({"case_id": case.case_id, **path_label_metrics(predicted, labels)})
            for label in case.retrieval:
                ids = await retrieve(label, knowledge)
                retrieval_rows.append(
                    {
                        "case_id": case.case_id,
                        "kind": label.kind,
                        "at_5": ranking_metrics(ids, label.relevant_ids, 5),
                        "at_10": ranking_metrics(ids, label.relevant_ids, 10),
                    }
                )
                if label.clause_id and not set(label.relevant_ids).issubset(ids):
                    errors.append("精确条文目标未命中")
            records.append(
                {
                    "case_id": case.case_id,
                    "input_text": case.text,
                    "runtime_seconds": runtime_seconds,
                    "passed": not errors,
                    "errors": errors,
                    "result": result.model_dump(mode="json"),
                    "knowledge_calls": calls,
                    "retrieved_contexts": observed_contexts(calls),
                    "telemetry": telemetry_by_case.get(case.case_id, {}),
                    "token_usage": usage_summary(telemetry_by_case.get(case.case_id, {})),
                    "stage_snapshots_bytes": len(
                        json.dumps(
                            [item.model_dump(mode="json") for item in result.stage_snapshots],
                            ensure_ascii=False,
                        ).encode()
                    ),
                    "formula_funnel": {
                        "candidate_pool": [item.formula_name for item in result.formula_candidates],
                        "branch_ids": [
                            ref for item in result.formula_candidates for ref in item.branch_ids
                        ],
                        "final_topk": [item.formula_pattern for item in result.reasoning_paths],
                    },
                }
            )
            await asyncio.to_thread(
                export_evaluation,
                output,
                {
                    "status": "running",
                    "complete_benchmark": False,
                    "passed": False,
                    "runtime": runtime,
                    "evaluation_signature": signature,
                    "cases": records,
                },
                records,
                cases,
            )
    finally:
        # 中断时释放未完成请求，最近落盘报告仍标记 running，不伪装全量。
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    order = {case.case_id: index for index, case in enumerate(cases)}
    records.sort(key=lambda row: order[str(row["case_id"])])
    differential = (
        await differential_regression(knowledge)
        if not runtime["online"]
        else {"status": "not_run_online", "reason": "独立规则回归不得调用在线检索干扰批次收尾"}
    )
    passed = all(item["passed"] for item in records) and (
        bool(runtime["online"]) or differential.get("passed") == 53
    )
    report = {
        "version": "system-evaluation/v1.1",
        "mode": (
            f"online/{runtime.get('reasoning_mode', 'llm')}/{runtime['retrieval_mode']}"
            if runtime["online"]
            else "rules/bm25/extractive"
        ),
        "runtime": runtime,
        "evaluation_signature": signature,
        "status": "completed",
        "complete_benchmark": len(records) == (expected_count or len(cases)),
        "expected_sample_count": expected_count or len(cases),
        "sample_count": len(records),
        "execution_failed": sum(row["result"] is None for row in records),
        "input_count": len(cases),
        "input_digest": input_digest,
        "scope": "工程语义与来源完整性；排序指标只有外部标签才可计算，非临床结论",
        "passed": passed,
        "knowledge_version": knowledge.version,
        "parser": classification_metrics(*counts) if labelled_cases else "unavailable",
        "parser_labelled_cases": labelled_cases,
        "negation": {
            "correct": negative_correct,
            "total": negative_total,
            "accuracy": negative_correct / negative_total if negative_total else None,
        },
        "unknown_preservation": {
            "correct": unknown_correct,
            "total": unknown_total,
            "accuracy": unknown_correct / unknown_total if unknown_total else None,
        },
        "differential": differential,
        "channel_top3": channel_rows or "unavailable",
        "formula_top3": formula_rows or "unavailable",
        "retrieval": retrieval_rows,
        "semantic_faithfulness": "unavailable:需专家审核",
        "cases": records,
    }
    await asyncio.to_thread(export_evaluation, output, report, records, cases)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False
        )
    )
    return passed


def export_evaluation(
    output: Path,
    report: dict[str, object],
    records: list[dict[str, object]],
    cases: list[EvaluationCase],
) -> None:
    """由工作线程写报告和空标签模板；失败快照只保留公开 DTO。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    # 原子替换防止读取者碰到半份 JSON；临时文件与目标位于同一文件系统。
    temporary = output.with_name(output.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    output.with_suffix(".failures.jsonl").write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n" for item in records if not item["passed"]
        ),
        encoding="utf-8",
    )
    output.with_suffix(".annotation.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "case_id": case.case_id,
                    "text": case.text,
                    "expected_current": None,
                    "expected_channels": None,
                    "expected_formulas": None,
                },
                ensure_ascii=False,
            )
            + "\n"
            for case in cases
        ),
        encoding="utf-8",
    )


def main() -> int:
    """默认运行固定工程集；外部数据只读取明确指定的已去标识化 JSONL。"""
    parser = argparse.ArgumentParser(description="在线系统评测和公开失败回放")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "eval/datasets/benchmark48/cases_eval_compatible.jsonl",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmark48/online.json")
    parser.add_argument("--offline", action="store_true", help="显式使用规则/BM25 工程回归")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int, help="仅在线验证前 N 条，报告记录真实输入数量")
    parser.add_argument("--case-timeout", type=float, default=180, help="单案总时限（秒）")
    parser.add_argument("--ablation", choices=["A", "B", "C", "D"], help="固定消融组")
    parser.add_argument("--resume", action="store_true", help="严格核验实验身份后复用成功病例")
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    if args.offline and args.ablation:
        parser.error("--offline 与 --ablation 不能同时使用")
    expected_count = len(cases)
    if args.case_timeout <= 0:
        parser.error("--case-timeout 必须大于零")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit 必须大于零")
        cases = cases[: args.limit]
    return (
        0
        if asyncio.run(
            evaluate(
                cases,
                args.output,
                online=not args.offline,
                env_file=args.env_file,
                concurrency=args.concurrency,
                expected_count=expected_count,
                case_timeout=args.case_timeout,
                ablation=args.ablation,
                resume=args.resume,
            )
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
