"""真实图 S1/S2 失败恢复测试；不重写图、不改变既有医学期望。"""

from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.app.clinical_v2.schemas import ChannelLocalizationV2
from backend.app.core.llm import ClinicalStageStreamError
from backend.app.core.ports import StructuredResult
from backend.app.persistence.repositories import InMemoryConsultationRepository
from backend.app.services.clinical_v2_service import ClinicalV2Service
from backend.app.services.consultation_service import ConsultationService
from backend.app.services.runtime_service import ReliableRuntime
from backend.tests.v2.test_end_to_end_workflow import FullLLM


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_schema,stage",
    [("S1ExtractionDraft", "s1_extract"), ("ChannelLocalizationV2", "s2_localize")],
)
async def test_safe_retry_after_sqlite_restart(
    tmp_path: Path, failed_schema: str, stage: str
) -> None:
    """关闭并重开 saver 后继续失败阶段，S2 不再计一次 S1 调用；收据拒绝重复请求。"""

    class FailsOnce(FullLLM):
        """模拟一次阶段级技术失败，之后沿用完整合成结果。"""

        failed = False

        async def ainvoke_structured(
            self, messages: list[dict[str, str]], schema: type[StructuredResult]
        ) -> StructuredResult:
            """仅第一次目标节点失败，不制造医学 unknown。"""
            if schema.__name__ == failed_schema and not self.failed:
                self.failed = True
                raise ClinicalStageStreamError(
                    "S1" if stage == "s1_extract" else "S2", "STAGE_TIMEOUT"
                )
            return await super().ainvoke_structured(messages, schema)

    llm = FailsOnce()
    repo = InMemoryConsultationRepository()
    path = str(tmp_path / "restart.sqlite")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        service = ConsultationService(repo, ClinicalV2Service(llm, saver), run_mode="sync")
        service.runtime = ReliableRuntime(service)
        failed = await service.create("发热")
        assert failed.status == "failed" and failed.failed_stage == stage and failed.retryable
        assert bool(failed.clinical_evidence) == (stage == "s2_localize")
        before = llm.calls.count("S1ExtractionDraft")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        service = ConsultationService(repo, ClinicalV2Service(llm, saver), run_mode="sync")
        service.runtime = ReliableRuntime(service)
        kwargs = dict(
            expected_turn=failed.turn, expected_revision=failed.revision, idempotency_key="once"
        )
        restored = await service.retry_clinical_stage(failed.consultation_id, **kwargs)
        assert restored.status == "awaiting_doctor_review"
        assert llm.calls.count("S1ExtractionDraft") == before + (stage == "s1_extract")
        count = len(llm.calls)
        await service.retry_clinical_stage(failed.consultation_id, **kwargs)
        assert len(llm.calls) == count


@pytest.mark.asyncio
async def test_invalid_s2_preserves_s1_and_explicit_retry_does_not_extract_again(
    tmp_path: Path,
) -> None:
    """引用修正仍失败时只发布已核验 S1，重开 SQLite 后可只重试 S2。"""

    class InvalidLocalizationLLM(FullLLM):
        """让两次 S2 草稿都重复引用，之后显式重试才返回合法结果。"""

        invalid_calls = 0

        async def ainvoke_structured(
            self, messages: list[dict[str, str]], schema: type[StructuredResult]
        ) -> StructuredResult:
            """在真实合成候选上制造来源矛盾，不干预医学排序。"""
            # FullLLM 原夹具只识别首次请求的末条事实 JSON；修正对话仍从原始事实取 ID。
            source_messages = messages[:2] if schema is ChannelLocalizationV2 else messages
            result = await super().ainvoke_structured(source_messages, schema)
            if isinstance(result, ChannelLocalizationV2) and self.invalid_calls < 2:
                self.invalid_calls += 1
                result.candidates[0].counter_fact_ids = list(result.candidates[0].support_fact_ids)
            return result

    llm = InvalidLocalizationLLM()
    repo = InMemoryConsultationRepository()
    path = str(tmp_path / "invalid-s2.sqlite")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        service = ConsultationService(repo, ClinicalV2Service(llm, saver), run_mode="sync")
        service.runtime = ReliableRuntime(service)
        failed = await service.create("发热")
        assert failed.status == "failed"
        assert failed.failure_code == "LOCALIZATION_INVALID"
        assert failed.failed_stage == "s2_localize" and failed.retryable
        assert failed.clinical_evidence and len(failed.clinical_evidence.findings) == 1
        assert failed.completed_stages == ["s1_extract"]
        assert (
            failed.localization is None and failed.retrieval is None and failed.comparison is None
        )
        assert llm.calls.count("ChannelLocalizationV2") == 2
        before = llm.calls.count("S1ExtractionDraft")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        service = ConsultationService(repo, ClinicalV2Service(llm, saver), run_mode="sync")
        service.runtime = ReliableRuntime(service)
        kwargs = dict(
            expected_turn=failed.turn, expected_revision=failed.revision, idempotency_key="s2-only"
        )
        result = await service.retry_clinical_stage(failed.consultation_id, **kwargs)
        assert result.status == "awaiting_doctor_review"
        assert llm.calls.count("S1ExtractionDraft") == before
        calls = len(llm.calls)
        await service.retry_clinical_stage(failed.consultation_id, **kwargs)
        assert len(llm.calls) == calls
