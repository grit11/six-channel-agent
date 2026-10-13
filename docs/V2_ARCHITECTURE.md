# V2 架构基线与目标

本文件由 P0 建立，当前 V2 运行时代码已按后续阶段接入。基线为 17e303d9988708572631c9feb77638ddaafff5dc，分支为 V2全新推理内核重建，冻结标签为 v1.2-before-clinical-v2。

目标链：S1 LLM事实抽取 -> S2 LLM六经定位 -> s2_clarification_gate -> S3 六经引导 Hybrid(BM25 + Dense + RRF + Reranker) -> S4 LLM批量方证证据比较 -> 程序来源核验 -> S5医生审核。

S2 unknown/conflicting 时，S3/S4 必须为零调用并持久 interrupt；医生补证使旧检索和比较结果失效。S2 ambiguous 若依据充分可带多经进入 S3。S5 可 accept、modify、reject、note、add_evidence 或 defer；医生未审核不得 completed。

V2 DTO 集中于 backend/app/clinical_v2/schemas.py，核心对象为 FindingV2、ClinicalEvidenceV2、ChannelLocalizationV2、RetrievalBundleV2、ComparisonV2 和 ClinicalV2RunOutcome。clinical_v2 是唯一临床推理包；clinical_v2_graph.py 是唯一可恢复临床图；knowledge_graph.py 承载独立 QA。KnowledgeService 只读五个原始 JSONL 与通用索引，不执行患者匹配评分。RAG 只提供证据。

保留并通过工厂注入 LLM/Embedding/Reranker、Milvus、PostgreSQL、Checkpoint、Repository、SSE 和原生 Console。原 Alembic 0001/0002 迁移链与五类原始知识只读保留。业务 Repository 是快照权威，运行任务保存提交意图，Checkpoint 仅用于图恢复。
