# V2.1 基线审计

基线：9be4567499131102835a853117322cd21dd63683；目标分支：S1-S2工程可靠性+QA优化。
工作区初始无修改，已包含 QA 祖先 53b20d6。用户授权依次执行所有阶段，覆盖手册逐期审批要求；不提交或推送。

临床冻结清单最初保存在验证工作区 data/v21_freeze.json，最终完整 SHA256 清单归档于 P7_ACCEPTANCE_REPORT.md，不把运行缓存复制到项目。S1 保留原 JSON 输入及一次来源修复；S2 保留完整 ClinicalEvidence JSON；医学 Prompt、Schema、verification、S3/S4/S5 和 Graph 均冻结。
现有 checkpoint.next 可直接证明 S1/S2 失败节点，采用 Service 验证并继续 ainvoke(None)，无需 Graph 改造。
模型/Prompt 变化后拒绝复用技术失败恢复凭据；业务版本、原文、上游事实及并发幂等必须核对。

基线测试：Conda langgraph Python 3.13.15；44 passed，退出码 0。范围为 v2、qa、structured_llm 和 QA API；仅缓存写权限警告。
源码 Alembic head 为 0002_reliable_runtime，新 QA revision 接在其后。QA 当前仅内存 TTL，无可信生产身份；新持久历史必须按可信服务端主体过滤，生产无主体拒绝访问。

实现顺序：P1 S1 独立流协议，P2 S2 相同输入接入，P3 checkpoint 恢复，P4 QA 独立表/事务/历史，P5 有界多轮/工具/usage，P6 SSE/知识页面，P7 冻结、故障、数据库和浏览器验收。
主要文件：core ports/llm/config/dependencies、S1/S2 调用模块、临床恢复 Service/Router/非医学 DTO、QA agent/tools/service/repository/DTO/Router、QA ORM/迁移、知识页及专项测试。

保留用户全局及 S4 配置，不修改 .env。缺少在线 P95 数据，初始 S1/S2 采用保守可配置预算，不新增输出 token 限额；实测性能与临床准确性另行记录，不伪造验收。
