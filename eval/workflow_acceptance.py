"""S7 离线纵向验收：真实知识、临床主图、SQLite 重开、补证和医生审核闭环。

依赖应用服务，业务 Repository 明确使用 memory 并跨对象复用，只验收 checkpoint 重开。
不连接 PostgreSQL/在线模型，不操作既有数据，不把此流程称为业务进程重启持久化。
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.agents.knowledge_qa_agent import KnowledgeQAAgent
from backend.app.core.config import Settings
from backend.app.graph.graph_factory import build_consultation_workflow
from backend.app.knowledge.service import KnowledgeService
from backend.app.persistence.checkpoint import build_checkpoint_runtime
from backend.app.persistence.repositories import InMemoryConsultationRepository
from backend.app.schemas.consultation import (
    ConsultationCreateRequest,
    ConsultationEvidenceRequest,
    ConsultationRerunRequest,
)
from backend.app.schemas.knowledge import KnowledgeQARequest
from backend.app.schemas.reasoning_api import (
    RetrospectiveReplayRequest,
    RetrospectiveReplayStep,
)
from backend.app.schemas.review import DoctorReviewRequest
from backend.app.services.consultation_service import ConsultationService
from backend.app.services.knowledge_qa_service import KnowledgeQAService
from backend.app.services.reasoning_service import ReasoningService
from backend.app.services.review_service import ReviewService


async def run(workdir: Path) -> dict[str, object]:
    """独立 SQLite 生命周期完成主链路，所有断言失败直接产生非零退出码。"""
    await asyncio.to_thread(workdir.mkdir, parents=True, exist_ok=True)
    database = workdir / ("checkpoint_" + uuid4().hex + ".sqlite3")
    settings = Settings(  # type: ignore[call-arg]  # BaseSettings 运行参数未进入插件签名。
        _env_file=None, checkpoint_backend="sqlite", langgraph_sqlite_path=str(database)
    )
    knowledge = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    repository = InMemoryConsultationRepository()
    runtime = await build_checkpoint_runtime(settings)
    try:
        workflow = build_consultation_workflow(
            knowledge, runtime.checkpointer, channel_weight=0.6, max_supplement_rounds=3
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
        assert initial.status == "needs_evidence" and initial.reasoning is not None
        question_ids = [
            gap.question_id for gap in initial.reasoning.evidence_gaps if gap.status == "pending"
        ]
        assert question_ids
        reviewing = await service.add_evidence(
            initial.consultation_id,
            ConsultationEvidenceRequest(
                unavailable_question_ids=question_ids, expected_turn=initial.turn
            ),
        )
        assert reviewing.status == "awaiting_doctor_review"
        assert (
            reviewing.reasoning is not None
            and reviewing.reasoning.evidence_gap_decision is not None
        )
        assert reviewing.reasoning.evidence_gap_decision.reason == "all_unavailable"
    finally:
        await runtime.close()
    runtime = await build_checkpoint_runtime(settings)
    try:
        workflow = build_consultation_workflow(
            knowledge, runtime.checkpointer, channel_weight=0.6, max_supplement_rounds=3
        )
        service = ConsultationService(
            repository,
            ReasoningService(knowledge),
            protocol_version="1.0",
            knowledge_base_version="1.2",
            scoring_config_version="1.0",
            workflow=workflow,
        )
        config: RunnableConfig = {"configurable": {"thread_id": initial.thread_id}}
        state = await workflow.graph.aget_state(config)
        assert state.next == ("doctor_wait",) and state.tasks[0].interrupts
        supplemented = await service.add_evidence(
            initial.consultation_id,
            ConsultationEvidenceRequest(
                clinical_text="无汗，脉浮紧",
                available_question_ids=question_ids,
                expected_turn=reviewing.turn,
            ),
        )
        assert supplemented.thread_id == initial.thread_id and supplemented.reasoning is not None
        assert supplemented.clinical_evidence is not None
        assert any(
            fact.key == "pulse_tight" and fact.polarity == "positive"
            for fact in supplemented.clinical_evidence.current_findings()
        )
        # QA 使用独立图，知识问题和其背景不写入患者事实或临床 checkpoint。
        qa = KnowledgeQAService(knowledge, KnowledgeQAAgent(knowledge))
        answer = await qa.ask(
            KnowledgeQARequest(query="第1条", query_type="exact_clause", clause_id="1")
        )
        assert answer.status == "answered" and answer.evidence_catalog
        assert await service.get_snapshot(initial.consultation_id) == supplemented
        pending = await service.request_review(
            initial.consultation_id, ConsultationRerunRequest(expected_turn=supplemented.turn)
        )
        assert pending.reasoning is not None and pending.reasoning.reasoning_paths
        original_paths = pending.reasoning.reasoning_paths
        final = await ReviewService(service).review(
            initial.consultation_id,
            DoctorReviewRequest(
                action="accept",
                selected_path_id=original_paths[0].path_id,
                expected_turn=pending.turn,
            ),
        )
        assert final.status == "completed" and final.reasoning is not None
        assert final.reasoning.reasoning_paths == original_paths
        assert final.review_history[-1].doctor_adopted
        state = await workflow.graph.aget_state(config)
        assert state.next == ()
        replay = await ReasoningService(knowledge).replay(
            RetrospectiveReplayRequest(
                case_evidence_id="1.2:case_0001",
                steps=[
                    RetrospectiveReplayStep(label="初始", text="发热"),
                    RetrospectiveReplayStep(label="补充", text="恶寒，无汗，脉浮紧"),
                ],
            )
        )
        first, second = [step.result for step in replay.steps]
        assert first.thread_id != second.thread_id
        assert not any(
            f.key == "pulse_tight" and f.polarity == "positive"
            for f in first.clinical_evidence.findings
        )
        assert any(
            f.key == "pulse_tight" and f.polarity == "positive"
            for f in second.clinical_evidence.findings
        )
        report = {
            "passed": True,
            "knowledge_version": knowledge.version,
            "checks": [
                "create_and_interrupt",
                "unavailable_preserves_unknown",
                "sqlite_reopen",
                "same_thread_supplement",
                "qa_isolation",
                "doctor_accept_preserves_original",
                "terminal_graph",
                "historical_replay_no_future_leak",
            ],
            "business_storage": "memory_reused_across_objects",
            "postgres_restart": "not_run",
            "sqlite_path": str(database),
        }
        await asyncio.to_thread(
            (workdir / "workflow.json").write_text,
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report
    finally:
        await runtime.close()


def main() -> None:
    """从命令行指定隔离输出目录，不覆盖既有运行数据。"""
    parser = argparse.ArgumentParser(description="S7 离线纵向工程验收")
    parser.add_argument("--workdir", type=Path, default=ROOT / "data/s7_workflow")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.workdir)), ensure_ascii=False))


if __name__ == "__main__":
    main()
