# V2.1 S1/S2 工程可靠性与 QA 优化验收

验收日期：2026-10-14。源码实现与离线工程验收完成；整体上线/医学验收为 False。按用户最新要求，真实全量模型与 RAGAS 实验交由用户本地执行，不再由本轮继续运行。未自动提交或推送。

## 基线与范围

工作区：`D:\Codex\伤寒论-六经辨证\六经辨证临床辅助agent\SixChannel_Agent_Console`；分支：`S1-S2工程可靠性+QA优化`；基线/当前 HEAD：`9be4567499131102835a853117322cd21dd63683`。QA 和 S4 既有实现作为祖先保留。实现先在可写 staging 完成验证，再按指纹保护清单同步；实际 .env 不修改。

P0 审计、P1 S1 流式、P2 S2 流式、P3 阶段恢复、P4 QA PostgreSQL、P5 QA 工具/上下文、P6 QA SSE/UI 均已实现并通过对应工程检查。P7 工程证明已归档，在线 48 例/QA 36 组/独立 RAGAS 和专家验收仍待用户运行。

S1 保留 `{clinical_text, approved_terminology}`、原来源校验与一次来源修复；S2 保留完整 `evidence.model_dump_json()`。传输 messages、Schema、模型及温度通过相同响应夹具逐字等价验证，不引入医学上下文投影、输出 token 截断或医学评分。

新增阶段使能和首正文/空闲/总预算，S1 修复共用总预算；连接建立确定失败最多重试一次，正文开始后不自动整份重发。完整 SSE、[DONE]、stop、JSON/Schema 及原医学验证完成后才提交；阶段进度仅发布计数/时间，不发布症状流片段或隐藏推理。

S1/S2 技术失败保留有效上游状态。恢复核对 checkpoint.next、原文/revision、上游事实、运行资源指纹，再通过 owner/turn/revision/action 幂等与已有会话锁继续。真实 SQLite saver 关闭重开后 S2 重试不增加 S1 调用，重复幂等请求不再调用模型。无安全 checkpoint 则拒绝恢复。

QA 保留 LangChain create_agent() 和四个只读 KnowledgeService 工具；仅新增独立 qa_sessions/qa_messages。生产无可信服务端主体拒绝历史访问，客户端任意身份 header 无效；开发合成主体明确区分。完整答案与来源原子提交，完成后发布失败不覆盖终态或重复调用。

QA 每轮最多4次模型、5次工具；相同快照/参数缓存，30秒工具预算，明确连接/429/5xx至多一次工具层重试。tool_audit 记录工具次数、缓存和工具读取尝试，不代表底层 Provider 所有网络重试。最多六轮已核验完整问答进入软字符预算，旧引用编号移除，新轮来源重新注册。真实 provider usage 汇总当轮所有 AIMessage，缺失/部分独立标记。

知识页支持真正历史分页/搜索/切换/重命名/归档/恢复/删除、SSE停止/断线回查/显式重试及每轮来源展开；保留条文速查。增量为临时生成状态，最终来源核验结果覆盖临时正文，浏览器只保存定位标识。

## 实际测试与退出码

命令使用既有 Conda langgraph，解释器为 `D:/Anaconda/envs/langgraph/python.exe`。

| 检查 | 退出码 | 结果 |
|---|---:|---|
| ruff | 0 | 通过 |
| black | 0 | 通过 |
| mypy | 0 | 通过 |
| pytest | 0 | 171 passed，失败/跳过均为0 |
| knowledge | 0 | 通过 |
| retrieval | 0 | 通过 |
| config | 0 | 通过 |
| console_syntax | 0 | 通过 |
| labels_syntax | 0 | 通过 |
| console_e2e | 0 | 通过 |

`conda run -n langgraph python scripts/engineering_acceptance.py --output-dir data/v21_acceptance` 最终10/10通过。早期格式、模块路径及新增测试 import 排序失败已修正，早期失败报告保留，最终未以跳过代替通过。

`conda run -n langgraph python -m mypy --follow-imports=silent --ignore-missing-imports eval/qa_v21_eval.py scripts/qa_postgres_probe.py`：退出0，2文件无错误；补测 Black/Ruff 和 QA JS syntax 均退出0。仓库当前为原生静态 Console，没有可运行的 React package.json，未虚构 npm lint/typecheck/test。

`conda run -n langgraph python scripts/qa_postgres_probe.py --env-file .env`：退出0；实际 PostgreSQL 随机隔离 schema 内 migration up/down、CRUD、幂等、双 owner 隔离、子进程恢复全部通过，测试 schema 已清除，不访问临床数据。

本地部署数据库先确认0002_reliable_runtime和QA表不存在，再仅执行0003_qa_history升级；退出0。升级后仅多出qa_sessions/qa_messages，alembic_version更新到0003，原临床/checkpoint表保留。应用不会隐式迁移。

`node scripts/qa_e2e.cjs`：退出0；1440/1024/390均通过，每个尺寸含正常完成、EOF后回查不重发、主动停止三次运行，并覆盖刷新/来源/历史管理/XSS防护/无横向溢出。使用真实页面和受控HTTP/SSE夹具，不能作为在线医学质量证明。原临床Console E2E也通过。

最终工程原始报告与日志保留于 `D:\Codex\伤寒论-六经辨证\six-channel-agent\data\v21_work\data\v21_acceptance\reports\20261014T135634Z_76e532d3`。QA浏览器结果：`{"passed":true,"results":[{"width":1440,"passed":true,"runs":3},{"width":1024,"passed":true,"runs":3},{"width":390,"passed":true,"runs":3}]}`。

失败注入包含首正文心跳不延长预算、半截/缺DONE/length/损坏JSON拒绝、HTTP429/503不整份重发、取消传播、S1/S2重启恢复和幂等、QA工具503一次重试/缓存/假ID拒绝、引用失败、完成后发布失败、预留后取消、过期运行解占用及生产身份隔离。原S4流式、只读兜底、阶段重试、医生审核和接口测试全部保留。

## 在线实验的实际边界

在用户要求暂不运行真实全量实验前，启动过使用现有gpt-5.5/hybrid配置的隔离服务；临床测试状态使用独立内存与SQLite，不改变真实env/global/S4参数。此服务与所有实验进程已停止，8041端口已释放。

中止前临床仅完成3例，均S2失败；安全事件实际标为HTTP403（S2_PROVIDER_UNAVAILABLE），不是阶段等待超时。旧48例脚本按连续3次失败停止，未完成48例；这些记录不能证明医学质量通过或退化。本轮不擅自更换API或修改医学输入。QA也仅部分题目完成，存在HTTP失败，原脚本记录尚不足以确定每条provider原因，因此不宣称QA36组通过或给出完整p95。

部分原始报告只保留于 `D:\Codex\伤寒论-六经辨证\six-channel-agent\data\v21_work\data\v21_online`。独立RAGAS Judge未配置、未评分；专家审查未完成。新评测脚本补齐安全业务码、会话失败继续留档、错误非零退出；这只通过离线HTTP夹具验证，未重新发起在线实验。

## 冻结证明

18文件SHA256均与基线一致。原S4 ainvoke_structured_stream方法AST一致，原临床ORM类AST一致，workbench/history/trace DOM一致。原Graph、S3/S4/S5医学逻辑、主库、Provider基础设施文件及真实env均未修改。

| 文件 | SHA256（基线=交付） |
|---|---|
| `backend/app/clinical_v2/comparison.py` | `3d48930a0a6a90c69a33355f64eb497317d5c8942ed8f320f884a3d18b3fe681` |
| `backend/app/clinical_v2/prompts/formula_discrimination_guide.md` | `e54aab75fe6c5f184411b01ccf1df44b6a2bdbb1b528d6b1e20cd59a3d526650` |
| `backend/app/clinical_v2/prompts/s1_clinical_extract.md` | `5c11b88508c460e740150363ed2c23c3434034313a7d2348b7c88fe1325c934d` |
| `backend/app/clinical_v2/prompts/s2_channel_localize.md` | `e28e37bec9d8190e9ffc6a3d504398c3fd5f177b3d3431059450ac4a51d8e466` |
| `backend/app/clinical_v2/prompts/s4_formula_comparison.md` | `c40f18f3b22a1cb360ae59c21a6c260543b3edb201b82f44381bd5f7e13c367e` |
| `backend/app/clinical_v2/prompts/six_channel_reference.md` | `dcd37dd5371f341ff7f6d7ee65bf33aacddb54032f91eeb56bb924de5fdf0d49` |
| `backend/app/clinical_v2/retrieval.py` | `5ce736f387c63dbfc5d36a2cde350ef44dc0d121094291a7d24169d6c4ee2a47` |
| `backend/app/clinical_v2/schemas.py` | `c68571ff224f8fd46da8ad24d065e5de202c5c239426be1b37419b39678fc462` |
| `backend/app/clinical_v2/verification.py` | `de959b04cd65bcd1644797971fee7ceb46fb191c5616ed36ba524b87d16c4370` |
| `backend/app/graph/clinical_v2_graph.py` | `6dfc86ef21663b198af848f0c6a36aca4e99c6976e3cdb7355b6ebf4e559eb71` |
| `backend/tests/unit/test_structured_llm.py` | `130f0270f6e9843aec407299c04bcfad76a9d1265bc8fc237114ed874b145b59` |
| `backend/tests/v2/test_end_to_end_workflow.py` | `0e7d43fa54ad90a1d8a39ec48e8eab7dd5ba5fc486422db3f1d9b9019bce0d5b` |
| `backend/tests/v2/test_s4_comparison.py` | `ac2f3b2ae02fb34c667a17159908a90589cc6b8450eb0c820f36fd8d341f481c` |
| `knowledge/v1.2/01_伤寒论经典条文库_398条_V1.2.jsonl` | `bfd97d30f23bdcdafeec53091b487258f79e6063d1b9ead6984f51ef5db2e451` |
| `knowledge/v1.2/02_六经方证知识库_V1.2.jsonl` | `8861cec9e6929567e868b45b50a0bca8dcc25d7363aa6b9c4712f4088f4ad942` |
| `knowledge/v1.2/03_六经鉴别规则库_53条_V1.2.jsonl` | `73e9ea33d231029e56c111026cbc3318bfba9c654db37f39d1f2dace1dc07eba` |
| `knowledge/v1.2/04_六经医案知识库_范中林_V1.2.jsonl` | `ad753ca3e68b1002d082be7e0a9923079bba2a2822e6b2ebaa2bff42ce4e356e` |
| `knowledge/v1.2/05_六经辨证核心框架库_V1.2.jsonl` | `8b897073a70dfb62cc0a429e2ebb80e29879c8a9071b447e2a50f61c0b87ac95` |

## git diff --name-status 交付清单

以下为最终同步后的工作树相对HEAD清单，A包含尚未git add的新文件；没有删除文件、没有提交。

```text
M	.env.example
M	backend/app/agents/knowledge_qa_agent.py
M	backend/app/agents/knowledge_qa_tools.py
M	backend/app/agents/prompts/knowledge_qa_v2.md
M	backend/app/api/v2/consultation.py
M	backend/app/api/v2/knowledge.py
M	backend/app/clinical_v2/channel_localization.py
M	backend/app/clinical_v2/extraction.py
M	backend/app/core/config.py
M	backend/app/core/dependencies.py
M	backend/app/core/llm.py
M	backend/app/core/ports.py
M	backend/app/persistence/models.py
M	backend/app/schemas/consultation.py
M	backend/app/schemas/knowledge.py
M	backend/app/services/clinical_v2_service.py
M	backend/app/services/consultation_service.py
M	backend/app/services/knowledge_qa_service.py
M	backend/app/services/runtime_service.py
M	backend/app/static/dashboard.html
M	backend/app/static/knowledge_qa.js
M	docs/QA_V2_DEVELOPMENT.md
A	backend/alembic/versions/0003_qa_history.py
A	backend/app/persistence/qa_repository.py
A	backend/app/services/qa_context.py
A	backend/tests/api/test_qa_history_api.py
A	backend/tests/api/test_qa_owner_isolation.py
A	backend/tests/qa/test_qa_context_and_eval.py
A	backend/tests/qa/test_qa_persistence.py
A	backend/tests/qa/test_qa_reliability.py
A	backend/tests/v2/test_clinical_stream.py
A	backend/tests/v2/test_clinical_transport_equivalence.py
A	backend/tests/v2/test_stage_recovery.py
A	docs/P0_BASELINE_AUDIT.md
A	docs/P7_ACCEPTANCE_REPORT.md
A	docs/V21_VALIDATION.md
A	eval/datasets/qa_v21.jsonl
A	eval/qa_v21_eval.py
A	scripts/qa_e2e.cjs
A	scripts/qa_postgres_probe.py
```

## 本地运行与回滚

完整测试命令、QA36组和48例运行入口见V21_VALIDATION.md。先关闭旧服务，再从Console根目录重启加载代码；真实env无需本轮调整。在线结果由用户本地测试后补入报告。

回滚代码到9be4567时，QA独立表可保留而不访问，不影响临床。仅明确丢弃QA历史时才降至0002（会删QA表）；不要回滚临床迁移。CLINICAL_STREAM_ENABLED=false可作显式非流式对照，此模式沿用旧Provider超时/重试语义。任何已有用户未提交改动须保留，不执行reset/clean。

生产上线仍需可信认证主体接入和独立实机/Remote Milvus/医学质量验收。当前内存QA替身重启会丢失，不能作为生产权威；成功来源编号只证明身份，不等同医学忠实度。
