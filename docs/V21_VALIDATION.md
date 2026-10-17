# V2.1 验证与运行说明

本轮只改变 S1/S2 传输、技术恢复及独立 QA；不改变医学输入、Prompt、DTO、临床 Graph、S3/S4/S5、知识主库和其他页面。保留真实 .env 的模型、全局超时和 S4 设置。

## 后端与页面

### S2 引用完整性与有限修正

S2 同一候选的支持与反证引用必须互斥；跨候选允许同一事实承担不同角色。
提示词明确区分过去、当前及变化趋势，不把减轻当消失。引用校验失败后，模型最多
修正一次完整草稿，两次调用共享原 S2 阶段预算；程序不删除反证、不替模型分类。
再次失败仍返回 `LOCALIZATION_INVALID`，日志只记录规则名、候选索引、冲突数量和
修正次数，不保存无效模型草稿或患者原文。失败快照保留 checkpoint 中已完成的
S1 事实，允许显式重试 S2；未经验证的 S2 结果不得进入 S3。

### GLM 推理强度

GLM-5.3 / Flash 系列通过 `.env` 的 `LLM_REASONING_EFFORT=low` 设置推理强度，
可选 `low`、`high`、`max`；空值不发送该参数，沿用供应商默认（官方默认为 `max`）。
共享模型适配层在初始化时读取一次，临床 S1/S2/S4、非流式调用及 QA 的 ChatOpenAI
统一使用；不再在 S4 写死 `low`。该系列必须保持思考开启，系统不采集隐藏推理。
其他型号不注入这组 GLM 参数；修改后需重启后端，已启动请求不会自动改变配置。
切换推理强度会更新临床技术恢复指纹，避免旧配置的失败任务直接复用新配置恢复。
这项调整不改变现有阶段、全局超时、医学提示词或 QA 工具与回答流程。

```powershell
conda run -n langgraph python scripts/engineering_acceptance.py --output-dir data/v21_acceptance
node scripts/qa_e2e.cjs
conda run -n langgraph python scripts/qa_postgres_probe.py --env-file .env
```

Playwright 使用已安装的模块，必要时将 CONSOLE_PLAYWRIGHT_MODULES 指向已有 node_modules；不要求安装到运行环境。qa_e2e 验证真实页面与受控 HTTP/SSE 夹具，真实 Agent 和 PostgreSQL 由各自专项测试证明，不能把夹具回答当医学评测。PostgreSQL probe 仅创建随机 qa_v21_test_* schema，并验证跨子进程恢复后删除，不接触临床 schema。

QA 表迁移已在本地数据库执行，当前 revision 为 `0003_qa_history`。启动服务前关闭旧进程，再从 Console 项目根目录启动，使新代码和原 `.env` 一起生效；不需要修改模型、全局超时或 S4 参数。

## 在线研发评测

QA 题库为 36 组、含多轮追问，未独立专家审核；不编造参考医学结论。临床沿用原 48 条研发银标，不根据测试反向修改 Prompt。

按用户最新要求，真实全量模型和 RAGAS 实验由用户本地执行，本次不将它们作为已通过项。

```powershell
conda run -n langgraph python eval/v2_online_eval.py --dataset eval/datasets/benchmark48/cases_eval_compatible.jsonl --base-url http://127.0.0.1:8000 --output data/v21_eval/clinical48.json --timeout-seconds 1800
conda run -n langgraph python eval/qa_v21_eval.py --base-url http://127.0.0.1:8000 --output data/v21_eval/qa.json
```

测试目标应是本版本服务。QA 需现有 KNOWLEDGE_QA_MODE=llm 且 API 支持 tool calling。题库中的空来源、对抗问题本来应拒答；不能只按 answered 数量评价质量。

可加 --ragas，必须显式配置独立 RAGAS_JUDGE_MODEL/BASE_URL/API_KEY。Judge 只测真实原文与回答的忠实度，不能等同临床诊断准确率；缺少 Judge 不伪造分数。临床运行的 S1/S2 请求指纹、ttft、用量、重试和 stage 状态来自真实事件，报告要单独区分传输成功、医学 unknown/HITL 与 Provider 故障。

## 回滚

代码可切回 9be4567，临床数据与 Graph 无迁移。QA 表是独立新增表，可保留而不调用；如明确丢弃 QA 历史，可执行 alembic downgrade 0002_reliable_runtime，此动作会删除 QA 消息。不要回滚临床 migration 或修改原知识文件。CLINICAL_STREAM_ENABLED=false 是显式非流式兼容开关；旧传输会沿用既有 Provider 超时/重试语义。
