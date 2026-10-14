"""退役 QA Graph 兼容占位。

V2 QA 已迁移到 LangChain create_agent；本模块不再编排 StateGraph、工具循环或临床状态。
冻结的 graph_factory 仍保留历史导出名，因此这里只保留显式失败的兼容函数。
"""

from typing import NoReturn


class KnowledgeGraphState(dict[str, object]):
    """旧 QA Graph 状态类型，仅为兼容旧导入，不再用于运行。"""


def build_knowledge_graph(agent: object) -> NoReturn:
    """拒绝旧 Graph 构造，调用方必须迁移到 KnowledgeQAService.chat。"""
    raise RuntimeError("知识 QA Graph 已退役，请使用 LangChain create_agent")
