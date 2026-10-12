"""48 条评测与 RAGAs 桥接回归。

覆盖未标注分母、公开输入隔离、来源核验和异常记账；依赖 fake 模型，不调用在线服务。
不验证临床正确性、不更改测试标签和知识记录。
"""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from eval.benchmark_eval import coverage, summarize
from eval.ragas_adapters import ProjectRagasLLM
from eval.ragas_eval import (
    ScoringSample,
    metric_summary,
    prepare_samples,
    reference_answer,
    score_samples,
)
from eval.system_eval import (
    EvaluationCase,
    evaluate_with_runtime,
    evaluation_settings,
    load_cases,
    safe_execution_error,
)
from langchain_core.prompt_values import StringPromptValue

from backend.app.core.config import Settings
from backend.app.core.exceptions import ProviderUnavailableError
from backend.app.core.ports import StructuredResult
from backend.app.knowledge.service import KnowledgeService
from backend.app.services.reasoning_service import ReasoningService

ROOT = Path(__file__).resolve().parents[3]


def test_benchmark_schema_and_unlabelled_denominators() -> None:
    """48 个合法输入保留 None；未知标签不以零分摊入已标注平均。"""
    cases = load_cases(ROOT / "eval/datasets/benchmark48/cases_eval_compatible.jsonl")
    assert len(cases) == 48
    assert any(case.expected_channels is None for case in cases)
    labelled = coverage([["太阳"]], ["太阳"])
    report = summarize(
        [
            {"channel": labelled, "formula": coverage([], None)},
            {"channel": coverage([], None), "formula": coverage([], None)},
        ]
    )
    assert report["channel"] == {
        "labelled": 1,
        "unlabelled": 1,
        "top1_recall": 1.0,
        "top3_recall": 1.0,
    }
    assert report["formula"]["top3_recall"] is None


def test_silver_reference_does_not_invent_labels() -> None:
    """欠定输入不产生标准答案；存在一类标签时只描述已标注部分。"""
    assert reference_answer({"reference_channels": None, "reference_formulas": None}) is None
    assert reference_answer({"reference_channels": ["太阳"], "reference_formulas": None}) == (
        "六经候选：太阳"
    )


def test_metric_failure_not_averaged_as_zero() -> None:
    """失败和未标注不补零，提高或降低均不能掩盖实际评分分母。"""
    report = metric_summary(
        [
            {"metrics": {"x": {"score": 0.8, "status": "scored"}}},
            {"metrics": {"x": {"score": None, "status": "failed"}}},
            {"metrics": {"x": {"score": None, "status": "unavailable"}}},
        ]
    )
    assert report["x"] == {"mean": 0.8, "scored": 1, "failed": 1, "unavailable": 1}


class FakeJudge:
    """只返回裁判 JSON 的假端口，记录收到的公开评分消息。"""

    def __init__(self) -> None:
        """创建隔离消息列表，不读取真实配置。"""
        self.messages: list[list[dict[str, str]]] = []

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """记录传入文本并提供固定评分格式。"""
        self.messages.append(messages)
        return '{"score": 1}'

    async def ainvoke_structured(
        self, messages: list[dict[str, str]], schema: type[StructuredResult]
    ) -> StructuredResult:
        """本回归不需要结构化业务调用。"""
        return schema.model_validate({})


async def test_judge_adapter_uses_only_injected_port() -> None:
    """裁判生成复用端口，拒绝同步事件循环和空批次；不要求厂商 SDK。"""
    provider = FakeJudge()
    judge = ProjectRagasLLM(provider)
    prompt = StringPromptValue(text="public scoring input")
    result = await judge.agenerate_text(prompt, n=2)
    assert judge.is_finished(result)
    assert len(result.generations[0]) == len(provider.messages) == 2
    assert provider.messages[0][1]["content"] == "public scoring input"
    with pytest.raises(RuntimeError, match="异步"):
        judge.generate_text(prompt)


async def test_real_ragas_id_metrics_and_semantic_unavailable(tmp_path: Path) -> None:
    """运行真实 RAGAs ID 指标，去重和参考集合不等同语义相关性。"""
    sample = ScoringSample(
        case_id="fixture",
        split="reference",
        user_input="病例",
        response="公开候选",
        retrieved_contexts=["a", "b"],
        retrieved_context_ids=["1.2:a", "1.2:b"],
        reference_contexts=["a", "c"],
        reference_context_ids=["1.2:a", "1.2:c"],
    )
    output = tmp_path / "ragas.json"
    settings = Settings(_env_file=None, retrieval_mode="bm25", reranker_enabled=False)
    assert await score_samples([sample], output, settings, semantic=False)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["complete_benchmark"] is False
    assert report["aggregate"]["id_based_context_precision"]["mean"] == 0.5
    assert report["aggregate"]["id_based_context_recall"]["mean"] == 0.5
    assert report["aggregate"]["faithfulness"]["unavailable"] == 1


async def test_prepare_rejects_incomplete_report(tmp_path: Path) -> None:
    """少病例或重复 ID 的报告拒绝进入评分，防止子集伪装全量。"""
    report = tmp_path / "incomplete.json"
    report.write_text('{"cases": []}', encoding="utf-8")
    settings = Settings(_env_file=None, retrieval_mode="bm25", reranker_enabled=False)
    knowledge = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    with pytest.raises(ValueError, match="完整"):
        await prepare_samples(report, ROOT / "eval/datasets/benchmark48", knowledge)


async def test_prepare_checks_input_and_reference_contents(tmp_path: Path) -> None:
    """真实图只收到病例原文；伪造输出正文与错配输入都被评分入口拒绝。"""
    settings = Settings(_env_file=None, retrieval_mode="bm25", reranker_enabled=False)
    knowledge = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    text = "发热、恶风、汗出、脉浮缓"
    result = await ReasoningService(knowledge).analyze_text(text)
    case = {"case_id": "one", "text": text}
    detail = {
        **case,
        "split": "reference",
        "reference_channels": ["太阳"],
        "reference_formulas": ["此标签不能进入系统响应"],
        "source_refs": [{"evidence_id": "1.2:clause_2"}],
    }
    (tmp_path / "cases_eval_compatible.jsonl").write_text(
        json.dumps(case, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "cases_detailed.jsonl").write_text(
        json.dumps(detail, ensure_ascii=False), encoding="utf-8"
    )
    record = {"case_id": "one", "input_text": text, "result": result.model_dump(mode="json")}
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"cases": [record]}, ensure_ascii=False), encoding="utf-8")
    sample = (await prepare_samples(path, tmp_path, knowledge))[0]
    assert "此标签不能进入系统响应" not in sample.response
    assert sample.reference and "此标签不能进入系统响应" in sample.reference
    assert sample.user_input == text
    assert sample.reference_context_ids == ["1.2:clause_2"]
    second = {"case_id": "two", "text": "其他病例"}
    (tmp_path / "cases_eval_compatible.jsonl").write_text(
        json.dumps(case, ensure_ascii=False) + "\n" + json.dumps(second, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="完整"):
        await prepare_samples(path, tmp_path, knowledge)
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "complete_benchmark": False,
                "expected_sample_count": 2,
                "cases": [record],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert len(await prepare_samples(path, tmp_path, knowledge, allow_partial=True)) == 1
    (tmp_path / "cases_eval_compatible.jsonl").write_text(
        json.dumps(case, ensure_ascii=False), encoding="utf-8"
    )
    record["input_text"] = "wrong input"
    path.write_text(json.dumps({"cases": [record]}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="不一致"):
        await prepare_samples(path, tmp_path, knowledge)
    record["input_text"] = text
    record["result"]["evidence_catalog"][0]["content"] = "伪造正文"
    path.write_text(json.dumps({"cases": [record]}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="正文"):
        await prepare_samples(path, tmp_path, knowledge)


def test_online_settings_enable_models_without_changing_deployment(tmp_path: Path) -> None:
    """真实配置规则模式也启用两个在线入口；部署文件不改写。"""
    env_file = tmp_path / ".env"
    content = (
        "CLINICAL_PARSER_MODE=rules\nREASONING_MODE=rules\nRETRIEVAL_MODE=bm25\n"
        "LLM_BASE_URL=https://example.invalid/v1\nLLM_MODEL=fixture-model\n"
    )
    env_file.write_text(content, encoding="utf-8")
    settings = evaluation_settings(online=True, env_file=env_file)
    assert settings.clinical_parser_mode == settings.reasoning_mode == "llm"
    assert settings.retrieval_mode == "bm25"
    assert env_file.read_text(encoding="utf-8") == content
    offline = evaluation_settings(online=False)
    assert offline.clinical_parser_mode == offline.reasoning_mode == "rules"
    assert offline.retrieval_mode == "bm25" and not offline.reranker_enabled


async def test_online_batch_preserves_provider_failure_and_label_isolation(tmp_path: Path) -> None:
    """失败病例必须保留，其余继续；推理只收到原文而不收到银标。"""
    settings = evaluation_settings(online=False)
    knowledge = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    reasoning = ReasoningService(knowledge)
    original_analyze = reasoning.analyze_text
    received = []

    async def injected_analyze(text: str) -> object:
        """模拟 Provider 失败，原始错误不得流入报告。"""
        received.append(text)
        if text == "失败输入":
            raise ProviderUnavailableError("敏感地址及密钥不应落盘")
        return await original_analyze(text)

    reasoning.analyze_text = injected_analyze  # type: ignore[method-assign,assignment]
    cases = [
        EvaluationCase(case_id="fail", text="失败输入"),
        EvaluationCase(
            case_id="ok", text="发热、恶风、汗出、脉浮缓", expected_formulas=["不能泄漏的标签"]
        ),
    ]
    output = tmp_path / "online.json"
    assert not await evaluate_with_runtime(
        cases, output, knowledge, reasoning, {"online": True, "retrieval_mode": "bm25"}
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert received == [case.text for case in cases]
    assert report["complete_benchmark"] and report["execution_failed"] == 1
    assert [row["case_id"] for row in report["cases"]] == ["fail", "ok"]
    assert report["cases"][0]["execution_error"]["error_code"] == "PROVIDER_UNAVAILABLE"
    assert report["cases"][1]["result"] is not None
    assert "敏感地址" not in output.read_text(encoding="utf-8")
    assert "不能泄漏的标签" not in json.dumps(report["cases"][1]["result"], ensure_ascii=False)


def test_failed_execution_is_separate_from_missing_labels() -> None:
    """失败不算成无标注，也不悄然混入有效评分分母。"""
    report = summarize(
        [
            {"channel": {"status": "execution_failed"}, "formula": {"status": "execution_failed"}},
            {"channel": coverage([], None), "formula": coverage([], None)},
        ]
    )
    assert report["channel"]["execution_failed"] == 1
    assert report["channel"]["unlabelled"] == 1
    assert report["channel"]["labelled"] == 0


async def test_no_successful_sources_cannot_complete_ragas(tmp_path: Path) -> None:
    """源模型全部失败时不能报告评分成功或把无评分伪装零分。"""
    output = tmp_path / "failed-ragas.json"
    assert not await score_samples(
        [],
        output,
        evaluation_settings(online=False),
        semantic=False,
        expected_sample_count=2,
        failed_case_ids=["a", "b"],
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed" and not report["complete_benchmark"]
    assert report["aggregate"]["faithfulness"]["mean"] is None
    assert report["aggregate"]["faithfulness"]["unavailable"] == 2


def test_safe_error_chain_does_not_expose_provider_payload() -> None:
    """错误链只保存状态码，异常 URI、正文和凭据不进入报告。"""
    response = httpx.Response(401, request=httpx.Request("POST", "https://fixture.invalid/secret"))
    cause = httpx.HTTPStatusError("secret-token", request=response.request, response=response)
    error = ProviderUnavailableError("secret-patient")
    error.__cause__ = cause
    value = safe_execution_error(error)
    assert value["cause_chain"][1]["http_status"] == 401
    assert "secret" not in json.dumps(value)


async def test_case_timeout_and_subset_are_reported(tmp_path: Path) -> None:
    """单案超时保存失败记录，部分病例不能冒充完整 48 条运行。"""
    settings = evaluation_settings(online=False)
    knowledge = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    reasoning = ReasoningService(knowledge)

    async def delayed(text: str) -> object:
        """模拟尚未完成的外部模型调用。"""
        await asyncio.sleep(1)
        raise AssertionError(text)

    reasoning.analyze_text = delayed  # type: ignore[method-assign,assignment]
    output = tmp_path / "timeout.json"
    assert not await evaluate_with_runtime(
        [EvaluationCase(case_id="one", text="发热")],
        output,
        knowledge,
        reasoning,
        {"online": True, "retrieval_mode": "bm25"},
        expected_count=48,
        case_timeout=0.01,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert not report["complete_benchmark"] and report["expected_sample_count"] == 48
    assert report["cases"][0]["execution_error"]["error_type"] == "TimeoutError"
    assert report["cases"][0]["execution_error"]["stage"] == "clinical_parser"
    assert report["differential"]["status"] == "not_run_online"
