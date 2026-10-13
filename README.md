# 六经辨证临床辅助系统 V2

本分支重建基于 LangGraph、结构化 LLM、KnowledgeService RAG 和《伤寒论》来源证据的医生侧辅助系统。系统不使用 V1.2 规则打分、固定权重或方剂硬编码；模型输出必须绑定当前事实和真实 Evidence ID，医生审核是完成入口。

## 运行阶段

S1 事实抽取 → S2 六经定位 → S3 Hybrid RAG → S4 证据比较 → S5 医生审核。S2 unknown/conflicting 会暂停并请求补证，补证后以新 revision 重跑；S5 的人工动作通过同一会诊线程恢复。

业务 Repository 保存权威快照，Checkpoint 只保存 LangGraph 恢复状态。KnowledgeService 是唯一医学知识访问边界，QA 使用独立 knowledge graph。日志不保存密钥、密码、完整病例或隐藏推理。

本地验证命令见 [eval/README_V2.md](eval/README_V2.md)。工程验收入口为 `conda run -n langgraph python scripts/engineering_acceptance.py --output-dir data/v2_acceptance_final`，报告中的任何失败或未运行项都不算通过。
