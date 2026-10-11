"""S7 离线 Parser/Reasoning/Differential/RAG 评测执行器与标注契约。

依赖现有规则图和 KnowledgeService；固定样例只验证工程语义，外部 JSONL 须事先去标识化。
不修改知识、不调用在线模型、不生成专家金标准；未提供的临床标签指标保持 unavailable。
"""

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.config import Settings
from backend.app.graph.six_channel_graph import STAGES
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.nodes.differential_reasoner import compare_formula_candidates
from backend.app.schemas.clinical import Polarity, SixChannel
from backend.app.schemas.reasoning import FormulaCandidate
from backend.app.schemas.reasoning_api import ReasoningResponse
from backend.app.services.reasoning_service import ReasoningService


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
        "top3_multilabel": classification_metrics(
            len(top3 & expected), len(top3 - expected), len(expected - top3)
        ),
    }


async def verify_result(result: ReasoningResponse, knowledge: KnowledgeService) -> list[str]:
    """核验公开阶段、路径排名和精确引用正文；不把来源存在等同医学忠实度。"""
    errors: list[str] = []
    if (
        result.completed_stages != STAGES
        or [item.stage for item in result.stage_snapshots] != STAGES
    ):
        errors.append("九阶段快照不完整")
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


async def evaluate(cases: list[EvaluationCase], output: Path) -> bool:
    """执行全部病例并导出失败公开快照、待专家填写模板；异常不静默丢弃。"""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    knowledge = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    reasoning = ReasoningService(knowledge)
    counts = [0, 0, 0]
    labelled_cases = 0
    negative_total = negative_correct = unknown_total = unknown_correct = 0
    records: list[dict[str, object]] = []
    channel_rows: list[dict[str, object]] = []
    formula_rows: list[dict[str, object]] = []
    retrieval_rows: list[dict[str, object]] = []
    for case in cases:
        result = await reasoning.analyze_text(case.text)
        current = {item.key: item.polarity for item in result.clinical_evidence.current_findings()}
        explicit = {(key, value) for key, value in current.items() if value != "unknown"}
        errors = await verify_result(result, knowledge)
        if case.expected_current is not None:
            labelled_cases += 1
            expected = {
                (key, value) for key, value in case.expected_current.items() if value != "unknown"
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
            key for key, polarity in (case.expected_current or {}).items() if polarity == "unknown"
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
                "passed": not errors,
                "errors": errors,
                "result": result.model_dump(mode="json"),
            }
        )
    differential = await differential_regression(knowledge)
    passed = all(item["passed"] for item in records) and differential["passed"] == 53
    report = {
        "version": "system-evaluation/v1.0",
        "mode": "rules/bm25/extractive",
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
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
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
    parser = argparse.ArgumentParser(description="离线系统评测和公开失败回放")
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "eval/datasets/engineering_cases.jsonl"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/system_evaluation.json")
    args = parser.parse_args()
    return 0 if asyncio.run(evaluate(load_cases(args.dataset), args.output)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
