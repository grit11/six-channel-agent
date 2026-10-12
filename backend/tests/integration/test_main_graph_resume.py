"""临床主图真实恢复：补证与审核 interrupt、saver 重开、输入保留及终态。

业务 Repository 使用显式内存替身跨应用对象复用，不宣称 PostgreSQL 重启验收。
"""

from pathlib import Path

import pytest
from langchain_core.runnables import RunnableConfig

from backend.app.core.config import Settings
from backend.app.graph.graph_factory import build_consultation_workflow
from backend.app.knowledge.service import KnowledgeService
from backend.app.persistence.checkpoint import build_checkpoint_runtime
from backend.app.persistence.repositories import InMemoryConsultationRepository
from backend.app.schemas.clinical import ClinicalEvidence, TreatmentEvent
from backend.app.schemas.consultation import (
    ConsultationCreateRequest,
    ConsultationEvidenceRequest,
    ConsultationRerunRequest,
)
from backend.app.schemas.review import DoctorReviewRequest
from backend.app.services.consultation_service import ConsultationService
from backend.app.services.reasoning_service import ReasoningService
from backend.app.services.review_service import ReviewService


@pytest.mark.asyncio
@pytest.mark.integration
async def test_business_interrupt_survives_saver_restart_and_review_loop(tmp_path: Path) -> None:
    """真实临床状态暂停后关闭/重开 saver，以同 thread 补证及审核，原建议保留。"""
    settings = Settings(
        _env_file=None,
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "main.db"),
    )
    knowledge = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    repository = InMemoryConsultationRepository()
    first = await build_checkpoint_runtime(settings)
    try:
        workflow = build_consultation_workflow(
            knowledge,
            first.checkpointer,
            channel_weight=0.6,
            max_supplement_rounds=3,
        )
        service = ConsultationService(
            repository,
            ReasoningService(knowledge),
            protocol_version="1.0",
            knowledge_base_version="1.2",
            scoring_config_version="1.0",
            workflow=workflow,
        )
        initial = await service.create(
            ConsultationCreateRequest(text="发热，无汗，恶寒，身痛，喘，脉浮")
        )
        config: RunnableConfig = {"configurable": {"thread_id": initial.thread_id}}
        state = await workflow.graph.aget_state(config)
        assert state.next == ("doctor_review",)
        assert state.tasks[0].interrupts[0].value["phase"] == "supplement"
        assert initial.status == "needs_evidence"
        assert initial.reasoning is not None
        unavailable_ids = [
            gap.question_id for gap in initial.reasoning.evidence_gaps if gap.status == "pending"
        ]
        initial = await service.add_evidence(
            initial.consultation_id,
            ConsultationEvidenceRequest(
                unavailable_question_ids=unavailable_ids,
                expected_turn=initial.turn,
                clinical_evidence=ClinicalEvidence(
                    course="医生已核对病程",
                    treatment_events=[
                        TreatmentEvent(
                            event_id="history1", description="已用治疗记录", source_text="前后待补"
                        )
                    ],
                ),
                replace_context_fields=["course", "treatment_events"],
            ),
        )
        assert initial.status == "awaiting_doctor_review"
        assert initial.reasoning.evidence_gap_decision.reason == "all_unavailable"
        assert initial.evidence_gap_context.unavailable_question_ids == unavailable_ids
        assert initial.supplementation_round == 1
        assert initial.clinical_evidence.findings
    finally:
        await first.close()
    second = await build_checkpoint_runtime(settings)
    try:
        resumed = build_consultation_workflow(
            knowledge,
            second.checkpointer,
            channel_weight=0.6,
            max_supplement_rounds=3,
        )
        service = ConsultationService(
            repository,
            ReasoningService(knowledge),
            protocol_version="1.0",
            knowledge_base_version="1.2",
            scoring_config_version="1.0",
            workflow=resumed,
        )
        saved = await service.add_evidence(
            initial.consultation_id,
            ConsultationEvidenceRequest(
                clinical_text="无汗，脉浮紧",
                resume=False,
                expected_turn=initial.turn,
                available_question_ids=unavailable_ids,
            ),
        )
        state = await resumed.graph.aget_state(config)
        assert state.values["clinical_evidence"]["course"] == "医生已核对病程"
        assert state.values["clinical_evidence"]["treatment_events"][0]["event_id"] == "history1"
        assert state.next == ("doctor_review",) and state.values["result"] is None
        assert state.tasks[0].interrupts
        result = await service.rerun(
            initial.consultation_id, ConsultationRerunRequest(expected_turn=saved.turn)
        )
        assert result.thread_id == initial.thread_id
        assert result.reasoning is not None
        assert result.clinical_evidence.course == "医生已核对病程"
        assert (
            result.reasoning.syndrome_relationships[0].temporal_assessments[0].status
            == "incomplete"
        )
        assert result.evidence_gap_context.unavailable_question_ids == []
        assert result.evidence_gap_context.asked_counts
        assert {
            item.key for item in result.clinical_evidence.findings if item.polarity != "unknown"
        }.issuperset({"fever", "sweating", "pulse_tight"})
        paths = result.reasoning.reasoning_paths
        reviewed = await ReviewService(service).review(
            initial.consultation_id,
            DoctorReviewRequest(
                action="note",
                note="人工核对",
                expected_turn=result.turn,
            ),
        )
        state = await resumed.graph.aget_state(config)
        assert state.tasks[0].interrupts
        assert state.values["review"]["action"] == "note"
        final = await ReviewService(service).review(
            initial.consultation_id,
            DoctorReviewRequest(
                action="accept",
                selected_path_id=paths[0].path_id,
                expected_turn=reviewed.turn,
            ),
        )
        assert final.status == "completed"
        assert final.reasoning.reasoning_paths == paths
        state = await resumed.graph.aget_state(config)
        assert state.next == () and state.values["phase"] == "completed"
        other = await resumed.graph.aget_state({"configurable": {"thread_id": "isolated"}})
        assert other.values == {}
    finally:
        await second.close()
