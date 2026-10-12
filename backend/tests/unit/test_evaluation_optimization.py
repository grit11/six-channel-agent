"""评测优化的工程回归。

职责：验证并发追踪隔离、严格续跑、配对比较及真实上下文选择。
依赖本地知识与真实规则图，不连接外部模型、不评价临床正确性。
"""

import asyncio
import json
from pathlib import Path

import pytest
from eval.benchmark_eval import build_report
from eval.compare_eval import compare_reports
from eval.evaluation_tools import (
    KnowledgeObserver,
    case_trace,
    domain_metrics,
    experiment_identity,
    observed_contexts,
)
from eval.ragas_eval import ScoringSample, prepare_samples, score_samples
from eval.system_eval import (
    EvaluationCase,
    evaluate_with_runtime,
    evaluation_settings,
    path_label_metrics,
)

from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService
from backend.app.services.reasoning_service import ReasoningService

ROOT = Path(__file__).resolve().parents[3]


async def knowledge_fixture() -> KnowledgeService:
    """构造本地只读BM25知识服务，禁止测试访问线上服务。"""
    settings = evaluation_settings(online=False)
    return await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")


async def test_parallel_traces_restore_and_preserve_failed_calls() -> None:
    """两个任务的检索与精确访问不串案；失败调用仍保留安全审计。"""
    knowledge = await knowledge_fixture()
    original = knowledge.search_classic

    async def one(number: str) -> list[dict[str, object]]:
        """每条独立上下文包含异步让出，强制两个病例交错执行。"""
        calls: list[dict[str, object]] = []
        with case_trace(calls):
            await asyncio.sleep(0)
            await knowledge.search_classic("probe", clause_nos=[number], top_k=1)
            await knowledge.get_evidence_by_ids([f"1.2:clause_{number}"])
            with pytest.raises(ValueError):
                await knowledge.search_classic("secret-patient", top_k=0)
        return calls

    with KnowledgeObserver(knowledge):
        a, b = await asyncio.gather(one("1"), one("2"))
    assert knowledge.search_classic == original
    assert "search_classic" not in vars(knowledge)
    assert observed_contexts(a)[0]["evidence_id"] == "1.2:clause_1"
    assert observed_contexts(b)[0]["evidence_id"] == "1.2:clause_2"
    assert a[-1]["status"] == "failed"
    assert "secret-patient" not in json.dumps(a)


def test_combined_path_cannot_be_created_from_label_union() -> None:
    """路径覆盖和并集召回分开报告，不改变旧指标含义。"""
    score = path_label_metrics([["太阳"], ["少阳"]], ["太阳", "少阳"])
    assert score["top3_recall"] == 1
    assert score["top3_path_coverage"] is False
    assert score["top3_exact_path"] is False


async def test_resume_reuses_success_and_rejects_changed_identity(tmp_path: Path) -> None:
    """同身份复用成功结果，代码/配置身份改变与无resume覆盖均拒绝。"""
    knowledge = await knowledge_fixture()
    reasoning = ReasoningService(knowledge)
    cases = [EvaluationCase(case_id="one", text="发热、恶风、汗出、脉浮缓")]
    output = tmp_path / "resume.json"
    runtime = {"online": True, "retrieval_mode": "bm25", "experiment_identity": {"code": "a"}}
    with KnowledgeObserver(knowledge):
        assert await evaluate_with_runtime(cases, output, knowledge, reasoning, runtime)
    first = json.loads(output.read_text(encoding="utf-8"))
    assert first["cases"][0]["retrieved_contexts"]

    async def forbidden(text: str) -> object:
        """续跑若再次调用已成功病例，测试必须失败。"""
        raise AssertionError(text)

    reasoning.analyze_text = forbidden  # type: ignore[method-assign,assignment]
    assert await evaluate_with_runtime(cases, output, knowledge, reasoning, runtime, resume=True)
    with pytest.raises(FileExistsError):
        await evaluate_with_runtime(cases, output, knowledge, reasoning, runtime)
    with pytest.raises(ValueError, match="身份"):
        await evaluate_with_runtime(
            cases,
            output,
            knowledge,
            reasoning,
            {**runtime, "experiment_identity": {"code": "b"}},
            resume=True,
        )


def test_experiment_identity_binds_prompt_and_excludes_secrets(tmp_path: Path) -> None:
    """提示内容变化使身份变化；凭据轮换不影响医学实验身份也不泄露。"""
    (tmp_path / "backend/app/prompts").mkdir(parents=True)
    (tmp_path / "eval").mkdir()
    prompt = tmp_path / "backend/app/prompts/probe.md"
    prompt.write_text("one", encoding="utf-8")
    settings = evaluation_settings(online=False)
    first = experiment_identity(tmp_path, settings)
    from pydantic import SecretStr

    settings.llm_api_key = SecretStr("secret-fixture")
    assert first == experiment_identity(tmp_path, settings)
    assert "secret" not in json.dumps(first)
    prompt.write_text("two", encoding="utf-8")
    assert experiment_identity(tmp_path, settings)["prompt_sha256"] != first["prompt_sha256"]


async def test_retrieved_context_validation_and_partial_summary(tmp_path: Path) -> None:
    """真实召回经过知识边界核验，缺追踪不能从最终引用补造，截断披露。"""
    knowledge = await knowledge_fixture()
    folder = ROOT / "eval/datasets/benchmark48"
    from eval.system_eval import load_cases

    case = load_cases(folder / "cases_eval_compatible.jsonl")[0]
    result = await ReasoningService(knowledge).analyze_text(case.text)
    evidence = (await knowledge.get_evidence_by_ids(["1.2:clause_2"]))[0]
    row = {
        "case_id": case.case_id,
        "input_text": case.text,
        "result": result.model_dump(mode="json"),
        "retrieved_contexts": [{"evidence_id": evidence.evidence_id, "content": evidence.content}],
        "errors": [],
        "passed": True,
    }
    report = {
        "cases": [row],
        "status": "completed",
        "complete_benchmark": False,
        "expected_sample_count": 48,
        "passed": True,
        "mode": "fixture",
    }
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    sample = (
        await prepare_samples(
            path,
            folder,
            knowledge,
            allow_partial=True,
            context_scope="retrieved",
            max_context_chars=5,
        )
    )[0]
    assert sample.retrieved_contexts == [evidence.content[:5]]
    assert sample.context_meta["shortened"] == 1
    summary = build_report(path, folder, allow_partial=True)
    assert summary["expected_sample_count"] == 48
    assert summary["cases"][0]["domain_metrics"]
    row["retrieved_contexts"][0]["content"] = "forged"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="正文"):
        await prepare_samples(
            path, folder, knowledge, allow_partial=True, context_scope="retrieved"
        )
    del row["retrieved_contexts"]
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="追踪"):
        await prepare_samples(
            path, folder, knowledge, allow_partial=True, context_scope="retrieved"
        )


async def test_comparison_uses_same_successful_ids_and_rejects_prompt_change(
    tmp_path: Path,
) -> None:
    """各组相同输入却不同失败样本时，只对共同成功病例评分。"""
    knowledge = await knowledge_fixture()
    folder = ROOT / "eval/datasets/benchmark48"
    details = [
        json.loads(s)
        for s in (folder / "cases_detailed.jsonl").read_text(encoding="utf-8").splitlines()
    ][:2]
    result = await ReasoningService(knowledge).analyze_text(details[0]["text"])
    metrics = domain_metrics(details[0], result)
    assert "joint_channel_formula_relation_top3" in metrics
    identity = {
        key: "fixture"
        for key in (
            "code_sha256",
            "prompt_sha256",
            "knowledge_sha256",
            "comparison_configuration_sha256",
        )
    }
    reports = []
    for index in range(2):
        rows = [
            {
                "case_id": d["case_id"],
                "input_text": d["text"],
                "passed": j == 0 or index == 0,
                "result": result.model_dump(mode="json") if j == 0 or index == 0 else None,
            }
            for j, d in enumerate(details)
        ]
        path = tmp_path / f"group{index}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "input_digest": "same",
                    "cases": rows,
                    "runtime": {"experiment_identity": identity},
                    "complete_benchmark": False,
                }
            ),
            encoding="utf-8",
        )
        reports.append(path)
    comparison = compare_reports(reports, folder)
    assert comparison["paired_case_ids"] == [details[0]["case_id"]]
    assert comparison["groups"][1]["execution_failed"] == 1
    source = json.loads(reports[1].read_text(encoding="utf-8"))
    source["runtime"]["experiment_identity"]["prompt_sha256"] = "changed"
    reports[1].write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="身份"):
        compare_reports(reports, folder)


@pytest.mark.parametrize("group", ["A", "B", "C", "D"])
def test_ablation_settings_validate_and_preserve_env(tmp_path: Path, group: str) -> None:
    """固定消融只改四个开关，原配置不写回。"""
    env = tmp_path / ".env"
    text = (
        "RETRIEVAL_MODE=bm25\nLLM_BASE_URL=https://fixture.invalid\nLLM_MODEL=probe\n"
        "EMBEDDING_BASE_URL=https://fixture.invalid\nEMBEDDING_MODEL=probe\n"
        "EMBEDDING_MODEL_VERSION=1\nRERANKER_BASE_URL=https://fixture.invalid\n"
        "RERANKER_MODEL=probe\n"
    )
    env.write_text(text, encoding="utf-8")
    settings = evaluation_settings(online=True, env_file=env, ablation=group)
    assert settings.reasoning_mode == ("rules" if group == "A" else "llm")
    assert settings.retrieval_mode == ("bm25" if group in {"A", "B"} else "hybrid")
    assert settings.reranker_enabled == (group == "D")
    assert env.read_text(encoding="utf-8") == text


async def test_semantic_context_policy_and_nan_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实指标对象经注入裁判装配，输入仅给忠实度加病历，NaN明确失败。"""
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    from eval import ragas_eval
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference

    received: dict[str, list[str]] = {}

    async def faithfulness(self: object, sample: object, **kwargs: object) -> float:
        """替代网络评分但保留真实RAGAs对象与SingleTurnSample构造。"""
        received["faithfulness"] = sample.retrieved_contexts
        return float("nan")

    async def utilization(self: object, sample: object, **kwargs: object) -> float:
        """利用率不允许将病历原文冒充召回文档。"""
        received["context_utilization"] = sample.retrieved_contexts
        return 0.5

    @asynccontextmanager
    async def provider(settings: Settings) -> AsyncIterator[object]:
        """仅工厂生命周期占位，指标替身不会调用模型。"""
        yield object()

    monkeypatch.setattr(Faithfulness, "single_turn_ascore", faithfulness)
    monkeypatch.setattr(LLMContextPrecisionWithoutReference, "single_turn_ascore", utilization)
    monkeypatch.setattr(ragas_eval, "open_llm", provider)
    sample = ScoringSample(
        case_id="one",
        split="challenge",
        user_input="病例原文",
        response="公开答案",
        retrieved_contexts=["真实来源"],
        retrieved_context_ids=["fixture"],
        reference_contexts=[],
        reference_context_ids=[],
        reference="参考",
        context_meta={"scope": "retrieved", "clinical_input_included": True},
    )
    output = tmp_path / "semantic.json"
    assert not await score_samples(
        [sample], output, evaluation_settings(online=False), semantic=True
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["aggregate"]["faithfulness"]["mean"] is None
    assert report["aggregate"]["context_recall"]["unavailable"] == 1
    assert report["aggregate"]["context_utilization"]["mean"] == 0.5
    assert received["faithfulness"][0].startswith("医生输入")
    assert received["context_utilization"] == ["真实来源"]
    assert "NaN" not in output.read_text(encoding="utf-8")
