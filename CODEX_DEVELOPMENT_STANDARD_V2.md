# 六经辨证临床辅助系统工程规范 V2.0

状态：V2 实施规范，基于开发手册；P0 后才允许进入运行时代码重构。

协议为 clinical-v2.0，线程命名空间为 clinical-v2:。五业务阶段是 S1 事实抽取、S2 六经定位、S3 Hybrid RAG、S4 证据比较、S5 医生审核；s2_clarification_gate 是技术控制节点。唯一医学包为 backend/app/clinical_v2/，唯一临床图为 backend/app/graph/clinical_v2_graph.py。新文件必须在手册白名单内。

V2 不使用词表、正则、IF-THEN、固定权重或规则回退决定辨证。S1 允许 unmapped finding；unknown、否定和未提及保持区分。S2 用待专家复核的静态六经指导卡，不在线检索；unknown/conflicting 持久暂停。S3 采用 BM25、Dense、RRF 和真实 Reranker，最多五个不同方证。S4 排序来自模型，不得有 ranking_score、score_components 或诊断概率。S5 医生拥有最终决策权。

Python 文件必须有中文模块说明、类型标注和公开 docstring；Black 100、Ruff、mypy、pytest 是最低门禁。Provider 失败必须明确失败。Milvus Remote/Lite、Checkpoint PostgreSQL/SQLite 保持同一契约并报告真实后端。Repository 是业务状态权威，Checkpoint 只负责图恢复。

P0 冻结、审计、基线、治理文档；P1 S1；P2 S2/HITL；P3 S3；P4 S4；P5 API/服务/Console；P6 删除旧代码；P7 评测发布。历史 48 条数据是研发银标，RAGAS 是检索辅助指标，不等于临床准确率。
