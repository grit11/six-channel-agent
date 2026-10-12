"""图工厂：装配临床主图和无独立 saver 子图，统一真实 interrupt/resume 入口。

Factory 只接收生命周期资源，不创建 Provider，不直接写会诊数据库。
业务 Repository 是事实与审核的权威；checkpoint 恢复仍需要核对业务版本。
"""

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import TypeAdapter

from backend.app.agents.six_channel_agent import SixChannelAgent
from backend.app.core.exceptions import InvalidStateTransitionError
from backend.app.core.ports import StructuredLLM
from backend.app.graph.main_graph import HumanGraphCommand, MainGraphState, build_main_graph
from backend.app.graph.six_channel_graph import build_six_channel_graph
from backend.app.knowledge.service import KnowledgeService
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.consultation import ConsultationSnapshot, ReasoningStage
from backend.app.schemas.reasoning import EvidenceGapPolicy, ScoringPolicy
from backend.app.schemas.reasoning_api import ReasoningResponse


def workflow_config(thread_id: str) -> RunnableConfig:
    """新图使用独立物理线程，绝不读取旧 doctor_wait 悬挂状态。"""
    physical = (
        thread_id if thread_id.startswith("six-stage-v1.2:") else "six-stage-v1.2:" + thread_id
    )
    return {"configurable": {"thread_id": physical}}


class ConsultationWorkflow:
    """服务层可调用的主图适配；同线程暂停恢复，不改变业务 Repository 契约。"""

    def __init__(self, graph: CompiledStateGraph) -> None:
        """绑定应用级已编译主图。"""
        self.graph = graph

    async def analyze(
        self,
        snapshot: ConsultationSnapshot,
        evidence: ClinicalEvidence,
    ) -> ReasoningResponse:
        """首轮运行，已暂停线程通过 Command 恢复分析；图状态不当作业务历史。"""
        config = workflow_config(snapshot.thread_id)
        state = await self.graph.aget_state(config)
        command = HumanGraphCommand(
            kind="reason",
            run_turn=snapshot.turn,
            clinical_evidence=evidence,
            top_k=snapshot.top_k,
            supplementation_round=snapshot.supplementation_round,
            evidence_gap_context=snapshot.evidence_gap_context,
            from_stage=TypeAdapter(ReasoningStage).validate_python(snapshot.rerun_from_stage),
            previous_result=(
                snapshot.reasoning.model_dump(mode="json") if snapshot.reasoning else None
            ),
        )
        if state.next == ("doctor_review",):
            result = await self.graph.ainvoke(
                Command(resume=command.model_dump(mode="json")), config
            )
        else:
            # 失败后存在待重试的节点时用医生权威事实重新启动，不复用残留中间结果。
            inputs: MainGraphState = {
                "intent": "clinical",
                "run_turn": snapshot.turn,
                "from_stage": command.from_stage,
                "previous_result": command.previous_result,
                "thread_id": snapshot.thread_id,
                "clinical_evidence": evidence.model_dump(mode="json"),
                "top_k": snapshot.top_k,
                "supplementation_round": snapshot.supplementation_round,
                "evidence_gap_context": snapshot.evidence_gap_context.model_dump(mode="json"),
                "result": None,
                "review": None,
            }
            result = await self.graph.ainvoke(inputs, config)
        return ReasoningResponse.model_validate(result["result"])

    async def human_action(
        self,
        snapshot: ConsultationSnapshot,
        command: HumanGraphCommand,
    ) -> None:
        """提交已通过业务校验的人工控制；缺少业务暂停点时明确失败。"""
        config = workflow_config(snapshot.thread_id)
        state = await self.graph.aget_state(config)
        if state.next != ("doctor_review",):
            raise InvalidStateTransitionError("图线程未处于医生暂停点，请先重新推理")
        await self.graph.ainvoke(Command(resume=command.model_dump(mode="json")), config)

    async def recover_result(self, snapshot: ConsultationSnapshot) -> ReasoningResponse | None:
        """恢复计算已落 checkpoint 但业务发布未完成的相同轮次，不使用旧轮结果。"""
        config = workflow_config(snapshot.thread_id)
        state = await self.graph.aget_state(config)
        # 进程可能在 reason 提交完成、doctor_review 尚未建立 interrupt 时退出。
        if state.next == ("doctor_review",) and not any(task.interrupts for task in state.tasks):
            await self.graph.ainvoke(None, config)
            state = await self.graph.aget_state(config)
        if (
            state.next == ("doctor_review",)
            and state.values.get("result_turn") == snapshot.turn
            and state.values.get("result")
        ):
            return ReasoningResponse.model_validate(state.values["result"])
        return None

    async def reconcile(
        self, snapshot: ConsultationSnapshot, command: HumanGraphCommand, action_id: str
    ) -> None:
        """业务 outbox 的幂等人工控制；已执行标记防止重复 resume/重复审核。"""
        config = workflow_config(snapshot.thread_id)
        state = await self.graph.aget_state(config)
        if (
            state.values.get("action_id") == action_id
            or state.values.get("action_turn", 0) >= snapshot.turn
        ):
            # 崩溃可能发生在人工节点完成与下一 doctor_review 之间，恢复剩余节点。
            if state.next and (
                state.next != ("doctor_review",) or not any(task.interrupts for task in state.tasks)
            ):
                await self.graph.ainvoke(None, config)
            return
        if state.next == ("doctor_review",):
            await self.graph.ainvoke(
                Command(
                    resume=command.model_copy(
                        update={"action_id": action_id, "action_turn": snapshot.turn}
                    ).model_dump(mode="json")
                ),
                config,
            )
            return
        if command.kind == "cancel":
            # 取消可能发生在未完成推理节点，业务终态为权威，不继续医学节点。
            await self.graph.aupdate_state(
                config,
                {
                    "action_id": action_id,
                    "action_turn": snapshot.turn,
                    "phase": "cancelled",
                    "route": "finish",
                    "result": None,
                },
                as_node="doctor_review",
            )
            return
        # checkpoint 丢失或只写到半途时从已提交业务快照重建人工控制状态；
        # 不重跑医学节点、不重新生成建议，也不重复医生采用记录。
        update: MainGraphState = {
            "intent": "clinical",
            "thread_id": snapshot.thread_id,
            "action_id": action_id,
            "action_turn": snapshot.turn,
            "clinical_evidence": (
                snapshot.clinical_evidence.model_dump(mode="json")
                if snapshot.clinical_evidence
                else {}
            ),
            "top_k": snapshot.top_k,
            "supplementation_round": snapshot.supplementation_round,
            "evidence_gap_context": snapshot.evidence_gap_context.model_dump(mode="json"),
            "result": snapshot.reasoning.model_dump(mode="json") if snapshot.reasoning else None,
            "review": command.review,
            "phase": (
                "completed"
                if command.kind == "finish"
                else "review" if snapshot.status == "awaiting_doctor_review" else "supplement"
            ),
            "route": "finish" if command.kind == "finish" else "wait",
        }
        await self.graph.aupdate_state(config, update, as_node="doctor_review")
        if command.kind != "finish":
            await self.graph.ainvoke(None, config)


def build_consultation_workflow(
    knowledge: KnowledgeService,
    checkpointer: BaseCheckpointSaver,
    *,
    channel_weight: float,
    max_supplement_rounds: int,
    formula_branch_recall_top_n: int = 12,
    reasoning_llm: StructuredLLM | None = None,
    scoring_policy: ScoringPolicy | None = None,
    evidence_gap_policy: EvidenceGapPolicy | None = None,
) -> ConsultationWorkflow:
    """应用启动构造主图与子图，复用同一知识服务，子图显式关闭独立 checkpoint。"""
    child = build_six_channel_graph(
        knowledge,
        checkpointer=False,
        channel_weight=channel_weight,
        reasoning_llm=reasoning_llm,
        scoring_policy=scoring_policy,
        evidence_gap_policy=evidence_gap_policy,
        max_supplement_rounds=max_supplement_rounds,
        formula_branch_recall_top_n=formula_branch_recall_top_n,
    )
    agent = SixChannelAgent(child)
    return ConsultationWorkflow(
        build_main_graph(
            agent,
            checkpointer,
            max_supplement_rounds=max_supplement_rounds,
        )
    )
