"""六经 Agent 边界：运行协议子图并返回结构化结果，不管理 HTTP 或人工业务记录。

主图使用无独立 saver 的子图，业务暂停点由主图统一持有，避免同 thread 两套图冲突。
原型 ReasoningService 仍可独立运行带 checkpoint 的子图，两种入口不共享上下文。
"""

from typing import cast

from langgraph.graph.state import CompiledStateGraph

from backend.app.graph.six_channel_graph import SixChannelState, response_from_state, stage_seed
from backend.app.knowledge.run_cache import run_cache_scope
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.consultation import ReasoningStage
from backend.app.schemas.reasoning import EvidenceGapContext
from backend.app.schemas.reasoning_api import ReasoningResponse


class SixChannelAgent:
    """封装六经子图端口；结果是可审计 DTO，不返回模型隐藏推理。"""

    def __init__(self, graph: CompiledStateGraph) -> None:
        """接收工厂编译的子图，不在每次请求创建图或 Provider。"""
        self.graph = graph

    async def analyze(
        self,
        evidence: ClinicalEvidence,
        *,
        thread_id: str,
        top_k: int,
        evidence_gap_context: EvidenceGapContext | None = None,
        from_stage: ReasoningStage = "clinical_parser",
        previous_result: ReasoningResponse | None = None,
    ) -> ReasoningResponse:
        """以医生已核对事实运行子图，保留来源和三值语义。"""
        seed = (
            stage_seed(previous_result, from_stage)
            if previous_result is not None and from_stage != "clinical_parser"
            else {}
        )
        with run_cache_scope():
            state = await self.graph.ainvoke(
                {
                    **seed,
                    "from_stage": from_stage,
                    "clinical_evidence": evidence,
                    "reset_clinical": False,
                    "thread_id": thread_id,
                    "top_k": top_k,
                    "evidence_gap_context": evidence_gap_context or EvidenceGapContext(),
                }
            )
        return response_from_state(cast(SixChannelState, state), thread_id)
