# V2 迁移与 P0 依赖审计报告

## 审计结论
审计对象是 Console 分支 V2全新推理内核重建，HEAD 17e303d9988708572631c9feb77638ddaafff5dc。AST 扫描 Git 跟踪的 162 个 Python 文件，665 条仓内模块边，0 个解析错误；旧医学模块候选 33 个，71 个 Python 文件存在直接旧模块边，123 个文件可传递到旧模块。关键符号命中 59 个文件。

主要消费者：core/config.py 仍有 ScoringPolicy/EvidenceGapPolicy 配置；core/dependencies.py 和 graph/graph_factory.py 装配旧 ReasoningService 与旧图；knowledge_qa_service.py 依赖旧混合 main_graph；schemas/consultation.py 和 persistence/checkpoint.py 依赖旧 ClinicalEvidence/Reasoning DTO；knowledge/service.py 依赖旧 nodes 词表、Grader、branch/run cache/differential。故 P0 不删除旧代码。

## 保留、重写、删除
保留并适配：core/llm.py、core/ports.py、通用 RAG、PostgreSQL/Repository/runtime、Checkpoint、SSE、knowledge QA、classic、v1 knowledge API、五类原始 JSONL、Alembic 0001/0002、FastAPI/Console 壳。
P1-P5 重写：core/config.py、core/dependencies.py、graph/graph_factory.py、knowledge/service.py、consultation/review/runtime/stream 服务、schemas consultation/events/review、checkpoint、main 和 Console。
P6 删除：backend/app/nodes/ 全目录；旧 graph、six_channel_agent、reasoning_service、旧 clinical/reasoning Schema、旧临床 v1 路由、旧 Prompt、branch_index/run_cache/differential、rag/grader/query_builder、knowledge/derived/fpeu_v12.jsonl 和旧 V1.2 评测/架构文档。原始五库、迁移链、通用 QA/RAG/DB/Checkpoint 不删除。

## P0 基线
Console 198 个 Git 跟踪文件与隔离副本 data/v12_release 逐项字节一致。Python 3.13.15，Conda 环境 langgraph。五个 JSONL 条目数为 398、128、53、69、11，本次未修改。已创建本地标签 v1.2-before-clinical-v2，指向基线提交，不推送远程。
隔离验收报告位于 data/v2_p0_acceptance/reports/20261012T095833Z_17b17bf6；Ruff、Black、mypy、pytest、知识校验、检索、离线评测、工作流、Console/labels Node 语法和 E2E 共 11 项均通过。该结果是 V1.2 基线工程通过，不代表 V2 已实现。真实 .env、数据库、Milvus 索引、在线模型和用户进程未修改。

## 手册审查
主手册引用了不存在的 05_V2_CODEX_PHASED_TASKS.md，实际阶段文件是 04_V2_CODEX_PHASED_IMPLEMENTATION.md；标签命名在 README 与阶段手册不一致，本次采用 v1.2-before-clinical-v2；示例分支名与用户指定分支冲突，本次保留用户分支。

## 下一阶段
P1 只新增 V2 schemas/extraction/verification、S1 Prompt 和对应测试，完成离线 mock 与至少七类语义回归；不得同时删除旧代码。指导卡仍为 pending_expert_review，不能作为已审核临床标准发布。
