# 六经辨证临床辅助系统 V2 仓库约束

本文件约束 V2 分支 V2全新推理内核重建。V2 运行时代码尚未在 P0 创建。

## 唯一临床架构
V2 只有一个 clinical-v2.0 临床 LangGraph：S1 LLM 原文事实抽取、S2 LLM 六经定位与澄清 HITL、S3 六经引导 Hybrid RAG、S4 LLM 证据比较、S5 来源核验与医生审核 HITL。独立知识问答图不是第二条临床链。

V2 不用医学规则分类、关键词正则或固定权重决定六经、方证和排序；不保留 ScoringPolicy、七因子评分、rules fallback 或旧 FPEU Grader。Pydantic、原文 span、Evidence ID、版本、幂等和状态路由属于软件完整性控制，必须保留。

## 分期和删除边界
按 P0～P7 分期实施，每期通过后才能进入下一期。P0 只冻结 V1.2、完成依赖审计和文档治理，不修改医学运行代码。消费者迁移和测试回归通过后，才物理删除 backend/app/nodes 及 V1.2 旧 Graph、Schema、Service、Prompt、Grader 和派生索引。禁止创建 legacy、v12_fallback 或复制旧评分器。

最终临床代码只放在 backend/app/clinical_v2/；唯一临床图为 backend/app/graph/clinical_v2_graph.py。QA 必须拆为独立 knowledge_graph.py。

## 医学与工程约束
Unknown 不等于 Negative；未提及不生成占位 Finding。Finding 保留连续原文 span、极性、时点和来源状态；词表外事实不得丢弃。六经允许多候选、ambiguous、conflicting 和 undetermined。S2 不调用在线 RAG；S3 只返回真实 Evidence 和最多五个不同方证；S4 由一次批量 LLM 比较，程序只核验来源，不产生诊断分；S5 医生拥有最终决策权。

Router 不写医学逻辑；领域代码不 import FastAPI；Provider、Embedding、Reranker、Milvus、PostgreSQL 和 Checkpoint 均经工厂注入。新建或实质修改 Python 文件必须有中文模块说明、类型标注和公开 docstring；使用 Black 100、Ruff、mypy。不得记录密钥、密码、完整患者原文或隐藏推理。

当前基线提交为 17e303d9988708572631c9feb77638ddaafff5dc，冻结标签为 v1.2-before-clinical-v2。不改 main/V1.2 历史，不自动 push；用户明确要求后才提交或推送。每期报告必须列实际退出码、模型调用/失败和遗留风险。
