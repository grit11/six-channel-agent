# SixChannel Agent 会话交接

更新时间：2026-10-17。用途：让新的 Codex 会话快速接手当前项目。本文是截至该日期的状态快照，不能代替读取最新 Git、代码和实验报告；其中“建议”不代表用户已授权实施。

## 1. 新会话先做什么

1. 确认工作目录是下面的 **Console 项目**，不要误用同级旧工程。
2. 阅读该项目 `AGENTS.md`、工程规范 V1.1/V2，以及与任务有关的现行文档、代码和测试。
3. 执行只读的 `git status --short --branch`、`git log -3 --oneline`，核对本文的分支与未提交改动。
4. 读取最新 `data/eval/clinical42.json`。若已续跑，本文的 19 例统计即为历史快照。
5. 先向用户简要复述接手状态，再处理用户新任务；不要自动开启全量实验、修改临床代码或提交推送。

## 2. 项目位置与运行环境

- **实际开发/运行项目**：`D:\Codex\伤寒论-六经辨证\六经辨证临床辅助agent\SixChannel_Agent_Console`。
- 原会话工具默认工作区：`D:\Codex\伤寒论-六经辨证\six-channel-agent`。其中 `data/` 保存了多次临时实现、测试和旧版副本，**不能把这些副本当作现行源码**。
- 新会话应直接在 Console 项目打开，并确认写入权限。旧会话默认工作区和实际项目不一致，曾需通过工具审批把已验证文件同步到 Console。
- Windows / PowerShell；复用 Conda 环境 `langgraph`，不擅建环境、不向 base 安装依赖。
- 运行配置是 Console 根目录 `.env`；改配置后必须重启对应后端。不要读取并转录密钥、数据库密码或完整患者信息到本文、日志或提交中。
- GitHub SSH remote：`git@github.com:grit11/six-channel-agent.git`。本机曾需要系统代理才能连接 GitHub SSH；实际端口、代理状态和现有 Git SSH 配置应现场核对，不能覆盖为猜测值。

## 3. Git 状态（本次已重新核对）

- 当前分支：`S1-S2工程可靠性+QA优化`。
- HEAD：`0e56405cd0f4284b153a857b846cbf18f8c59a10`，提交标题 `feat: complete S1 S2 reliability and QA optimization`。
- 本地跟踪引用 `origin/S1-S2工程可靠性+QA优化` 指向相同提交；本次未重新查询远程服务器。
- 前一提交：`9be4567 feat: finalize S4 streaming compatibility and fallback validation`。
- **本地尚未提交的内容**：
  - 修改 `eval/v2_online_eval.py`：断点续跑、原子保存、幂等恢复、终端进展。
  - 修改 `eval/README_V2.md`：续跑说明和命令。
  - 新增 `backend/tests/unit/test_v2_online_eval.py`：离线调度与恢复测试。
  - 删除 `eval/datasets/benchmark48/cases_detailed_42_v2.jsonl`：用户已有删除，**不要擅自恢复或认领为助手改动**。
  - 本交接文档 `handoff.md`：为换会话新增，尚未提交。
- `data/` 被 Git 忽略，实验报告在本地，不会随普通 Git 提交上传。
- 本轮交接不提交、不推送、不更换分支。

## 4. 用户目标与架构底线

用户要的是 LangGraph + LLM + RAG +《伤寒论》知识引导的辨证系统，明确拒绝回到固定医学规则打分、固定权重、关键词分类或个别方剂的硬编码补丁。

当前 clinical-v2.0 主链：

`S1 原文事实抽取 → S2 六经定位/澄清 → S3 Hybrid RAG → S4 证据比较与排序 → S5 医生审核`

- 临床包：`backend/app/clinical_v2/`。
- 临床图：`backend/app/graph/clinical_v2_graph.py`。
- Unknown 不等于 Negative；未提及不自动变成阴性；保留多候选、补问和人工审核。
- Pydantic、连续原文 span、事实引用、Evidence ID、候选身份、版本、幂等、排序格式核验是软件完整性控制，不是用户拒绝的医学规则库，不能为提高通过率直接取消。
- KnowledgeService 是医学知识访问边界；Provider 与基础设施通过现有配置/工厂注入。
- QA 与临床运行隔离；QA 已重构并经后续修复，涉及 LangChain `create_agent()`、四个只读工具、来源展示、会话历史等。不要把早期“QA 仅临时内存”的限制照搬到现有代码。
- 当前前端是 `backend/app/static/` 原生 Console，不能凭旧 React 文档引入新的前端栈。
- 源码有中文模块说明、公开 docstring、类型注解；避免无关重构。

### 文档存在历史残留

Console 的 `AGENTS.md` 仍有“V2 运行时代码尚未在 P0 创建”等重构初期描述；V1.1 工程规范仍引用 V1.2 六阶段或已清理路径。用户在旧会话明确完成并批准了 V2、QA、S4、S1/S2 的后续开发。不要据旧描述回退现行代码、恢复旧规则链或删除当前 QA 持久化。

发现冲突先说明，再按用户最新明确要求、适用工程约束、现行设计与代码证据处理；本交接文档本身不覆盖用户后续要求。

## 5. 当前任务与已完成内容

最近工作集中在 **在线临床评测脚本和实验结果分析**，没有因本轮分析改动临床 S1/S2/S3/S4。

用户先后要求：讲解评测脚本、分析失败、支持关闭后续跑并显示进展、解释 S1 校验、分析实验为什么自动停止。用户之前说过“那先不用管”，因此不能把分析问题自动扩展为临床修复授权。

### 评测续跑功能已实现，但未提交

- `--resume` 加载已有报告，校验数据集 SHA256、协议、病例 ID，跳过所有已记录病例，**包括失败病例**。
- 开始提交前保存稳定幂等键，收到响应后保存 consultation ID。
- 中断时保存 `pending`；续跑优先查询同一会诊，若提交响应未保存则复用原幂等键。
- 报告使用同目录临时文件 + replace 原子保存，避免中断破坏已有文件。
- 普通运行拒绝覆盖既有报告；新实验换 `--output`。
- 校验报告公开的模型、知识版本、检索模式和重排器使能配置，避免明显混合实验；这不是对全部 `.env` 参数的完整一致性保证。
- 默认每 5 秒显示公开状态、最近完成阶段和等待时长，每例显示完成数、失败数和耗时。
- `--limit N` 是题库前 N 例，不是再额外跑 N 例。
- 用户本地执行真实全量实验与 RAGAS；助手不要擅自发起付费全量运行。

### 验证记录的可信范围

前序会话记录：新增 5 项离线调度测试与原有 2 项相关测试合计 7 项通过，相关 Ruff、Black、mypy 检查通过。曾有 Black 缓存权限警告。**本次仅整理文档，没有重新执行这些检查，也没有全量 pytest/在线模型/RAGAS 通过的结论。**

可查 `backend/tests/unit/test_v2_online_eval.py`；隔离工作副本在旧工作区 `data/eval_resume_task/`。不得因为局部测试通过就写“全量验收通过”。

## 6. 最新实验快照：为什么停止

报告：Console 项目 `data/eval/clinical42.json`。

数据集：`eval/datasets/benchmark48/cases_detailed.jsonl`，实际 **48 例**。输出文件名 clinical42 不决定样本量。

已核对报告：`run_state=stopped`，原因是“连续 Provider/运行错误，停止额外在线消耗”，没有待恢复 `pending`。

**直接原因：SC017、SC018、SC019 连续三例 row.error 非空，触发测评调度的默认停止保护。不是 1800 秒超时，也不是终端崩溃。**

停止代码在 `eval/v2_online_eval.py`：

```python
# run() 的参数，当前未暴露 CLI 选项
stop_after_failures: int = 3

failures = failures + 1 if row["error"] else 0
if failures >= stop_after_failures:
    report["stopped_reason"] = "连续 Provider/运行错误，停止额外在线消耗"
    break
```

每次调用 run（包括续跑）重新从零累计连续失败。**任何 row.error 都算**，不仅网络/Provider 故障；现有停止文案过于笼统。这次全部九例失败没有被报告为客户端超时。

| 病例 | 状态/错误码 | 整例耗时（秒） |
| --- | --- | ---: |
| SC001 | awaiting_doctor_review | 86.47 |
| SC002 | EXTRACTION_INVALID | 12.06 |
| SC003 | awaiting_doctor_review | 73.72 |
| SC004 | awaiting_doctor_review | 54.06 |
| SC005 | S2_RESPONSE_INVALID | 17.44 |
| SC006 | LOCALIZATION_INVALID | 35.60 |
| SC007 | awaiting_doctor_review | 96.92 |
| SC008 | awaiting_doctor_review | 117.32 |
| SC009 | COMPARISON_UNGROUNDED | 101.27 |
| SC010 | awaiting_doctor_review | 104.33 |
| SC011 | awaiting_doctor_review | 172.04 |
| SC012 | COMPARISON_UNGROUNDED | 118.03 |
| SC013 | needs_evidence | 23.04 |
| SC014 | COMPARISON_UNGROUNDED | 73.27 |
| SC015 | awaiting_doctor_review | 52.23 |
| SC016 | awaiting_doctor_review | 78.53 |
| SC017 | COMPARISON_UNGROUNDED | 44.12 |
| SC018 | LOCALIZATION_INVALID | 36.54 |
| SC019 | EXTRACTION_INVALID | 21.71 |

合计：19/48 已记录，9 例进入医生审核、1 例正常补问暂停、9 例技术失败，剩余 SC020～SC048 共 29 例。等待医生审核不等于医生已经批准。

失败分布：S1 两例；S2 三例；S4 四例。SC005 的 `model_failed` 明确是 `ValidationError` / `schema_validation`，不是仅凭外层 `ClinicalLLMUnavailableError` 就能判成 API 网络故障。

| 当前指标 | 数值 |
| --- | ---: |
| channel_top1 | 63.2%（12/19） |
| channel_top3 | 68.4%（13/19） |
| formula_top1 | 29.4%（5/17） |
| formula_top3 | 29.4%（5/17） |
| retrieval_formula_top5 | 41.2%（7/17） |
| 整例延迟 P50 / P95 | 73.27 秒 / 172.04 秒 |

SC009、SC017 没有方剂参考标签，因此方证相关指标不计入分母。指标按去重参考标签召回计算，失败/暂停的可计分项也进入整体统计；不要称为专家验证的临床准确率。48 例是研发银标，尚未独立专家验证。

报告公开配置快照：模型 `glm-5.3`；hybrid + reranker；vector=lite、checkpoint=sqlite、business DB=postgres；knowledge_version=1.2。**这只说明实验当时配置，用户运行后可能修改了 .env。** Lite/SQLite 的 degraded 状态不是本次九例失败的已证实原因。

## 7. 已定位的问题与未确定的部分

### S1：输出来源核验不稳定

- 入口与核验：`backend/app/clinical_v2/extraction.py`、`verification.py`、`schemas.py`，提示词在 `clinical_v2/prompts/`。
- 先由 LLM 输出结构化草稿，再核验连续原文 source_text、出现次数、身份/引用、冲突关系等，程序计算 span 与事实身份。
- 来源不合规时可让模型完整修正一次，两次调用共享 S1 阶段预算；不能靠程序伪造原文片段过关。
- 工作台同一输入后来成功，不能反证 SC002 当时没有失败；模型每次生成可能不同，预解析也不能假定被后续启动直接复用。
- 报告没有保存当时无效草稿与具体校验细则，不能猜测是哪一个词或 fact_id 导致错误。

### S2：结构/事实引用校验失败

- 代码：`backend/app/clinical_v2/channel_localization.py`。
- 核验候选 ID 不重复、事实存在、继续检索前没有未解决冲突、同一候选支持与反证互斥、Unknown 不作反证、补问引用有效。
- 同一事实跨候选可以扮演不同角色，不能误判为全局互斥。
- 引用失败已有一次受限模型修正，两次共享阶段预算；不能随意删反证来提高通过率。

### S4：四次失败是 grounding/输出完整性拦截

- 代码：`backend/app/clinical_v2/comparison.py` 与相关核验。
- `COMPARISON_UNGROUNDED` 可能是排名不连续/重复、候选或事实/证据不存在、方名不一致、缺少当前候选自身来源支持、路径 ID 重复等。
- **目前报告只有阶段和错误类型，缺少实际失败校验规则，不能把四例都断言成同一种 rank 错误或模型医学幻觉。**
- 四例均为 `result_presentation=s3_retrieval_only`，只是保留检索结果，不是已验证的鉴别结论。

### S3：部分参考方剂未进入候选池

- 在已经进入医生审核的 SC008、SC010、SC011、SC015，参考方剂均未出现在 S3 返回的候选池中。
- 参考分别是小陷胸汤、大承气汤、小承气汤、猪苓汤；当前这些例的方证 Top-1/Top-3 均未命中。
- 应先检查检索与候选选择，不能只归因 S4 排序。
- SC012、SC014 已检得参考方剂却在 S4 校验失败，应分开分析。
- 以上是报告现象，不足以证明知识主库缺条目；需进一步追踪召回、重排、去重与截断。

### 评测脚本仍有统计/调度局限

- 三次连续错误的停止条件过宽，且没有 CLI 配置；不要告诉用户存在尚未实现的禁用参数。
- `llm_calls` 从通用 model_completed/model_failed 事件计数，可能混入 embedding/reranker，而遗漏 S4 专项流事件；不是可靠的纯 LLM 调用数。
- token 汇总存在同样覆盖率问题，不能当完整账单；无 usage 需如实保留 unavailable。
- latency 是整例客户端耗时，不是单独某个阶段的耗时。
- 续跑跳过已记录失败例，重新测失败例需要独立安排，不可暗中混入原基线。
- 跨配置完整一致性依赖用户保持其他参数不变，现有公开 config 校验不是完整运行指纹。

## 8. 下一步建议（尚未实施）

1. 根据用户新指令选择：先继续完成基线，或先修复输出可靠性。不要自行选择改变实验条件。
2. 若只续跑，原命令会从 SC020 开始；再遇连续三例失败仍会停止。
3. 若用户要求全量不中断，可仅修改测评调度：开放停止阈值及明确的关闭语义、区分错误类型、保留逐例报告与进度；不要为了续跑同时改临床 Prompt/模型参数。
4. 若要修复临床失败，先补充安全的校验规则、字段路径、索引等诊断信息，保留异常链；不要记录无效完整模型草稿、患者全文或隐藏推理。
5. 再按证据分别处理 S1 来源、S2 schema/引用、S4 grounding 和 S3 候选召回，避免一次改变多条链路而无法归因。
6. 变更后跑对应离线测试与静态检查；用户负责的真实全量模型/RAGAS实验不自动代跑。
7. 根据用户明确要求才整理 Git 提交/推送，保留原有未提交改动和用户删除。

## 9. 常用命令

以下全部在 Console 项目根目录执行。

```powershell
# 终端一：先启动后端，确保旧服务已关闭
conda run --no-capture-output -n langgraph python scripts/start_console.py

# 终端二：续跑原报告，不重新测试已有失败例
conda run --no-capture-output -n langgraph python -u eval/v2_online_eval.py --dataset eval/datasets/benchmark48/cases_detailed.jsonl --output data/eval/clinical42.json --timeout-seconds 1800 --resume --progress-seconds 5

# 离线验证续跑功能；真实服务与模型不参与
conda run --no-capture-output -n langgraph python -m pytest backend/tests/unit/test_v2_online_eval.py -c backend/pyproject.toml -q
```

`--timeout-seconds 1800` 是测评客户端等待，不会替后端提高 S1/S2/S4 或全局预算。Conda 输出缓冲时使用 `--no-capture-output` 和 `python -u`。

## 10. 踩过的坑与工作习惯

- **先启动后端**：之前调用 `127.0.0.1:8000/api/console/config` 502 时，用户确认后端没有启动。先查服务、端口、进程，不要直接认定模型坏了。
- **确认加载的是哪个 .env、哪份源码**：用户多次切换供应商、模型和 reasoning 参数；运行后改配置不能用来反推旧运行配置。修改后务必重启。
- **不要仅看外层异常名**：ClinicalLLMUnavailableError 可能包装超时、schema 校验或其他适配错误，应看 failure_code、error_category、异常链和阶段事件。
- **流式兼容与 token 限制曾是独立问题**：GLM 推理参数集中在工厂/适配层初始化注入，不应每阶段硬编码；只提高 max tokens 或时限不是全部错误的通用修复。
- **重试会重新发起请求**：超时后的重新调用不能默认接上供应商之前的生成。已有阶段共享预算，不擅自增加重试导致费用和延迟倍增。
- **rank 是模型排名**：合法唯一连续排名可以按 rank 序列化排序；不能用程序替模型补排名或重新打医学分。
- **失败与补问分开**：needs_evidence 是合法 HITL；awaiting_doctor_review 是待审核，不能算已批准。
- **运行输出与源码分开**：不提交 .env、数据库、索引、缓存、患者快照或 data 报告。
- **Windows 临时目录/文件锁问题确实出现过**：隔离测试环境、核对具体报错，不把后续所有 LLM 问题都归因 Milvus 或权限；不宣称未跑的全量测试通过。
- **保持用户最新范围**：倾向完成已授权工作，避免反复审批；仅分析时不顺带改临床、配置和其他页面；受工具沙箱限制需要审批时如实说明。

## 11. 继续维护这份交接

完成一项实质任务后更新：日期、Git 分支/提交、未提交文件、报告路径、已验证结论、未验证假设、下一步。保留必要决定和经验，过期状态标明日期，不把所有终端输出复制进来。

`handoff.md` 不会仅凭文件名保证被新会话自动读取。最稳妥的方式是在新会话第一条消息明确要求先阅读本文，并指定后续任务。若后续用户希望自动加载，可在适用 AGENTS.md 中加入简短读取入口，不把实验流水账塞进仓库开发约束。
