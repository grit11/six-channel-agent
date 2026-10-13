"""图装配公共边界：唯一临床图由 clinical_v2_graph 构建，知识 QA 独立。"""

from backend.app.graph.clinical_v2_graph import build_clinical_v2_graph
from backend.app.graph.knowledge_graph import build_knowledge_graph

__all__ = ["build_clinical_v2_graph", "build_knowledge_graph"]
