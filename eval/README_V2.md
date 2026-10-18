# V2 评测说明

`v2_online_eval.py` 调用 V2 API 保存每条病例的公开结果、暂停状态、错误类别和延迟；`v2_ragas_eval.py` 生成 RAGAS 输入，`v2_compare.py` 比较工程指标。

```powershell
conda run -n langgraph python eval/v2_online_eval.py --dataset eval/datasets/benchmark48/cases_detailed.jsonl --base-url http://127.0.0.1:8000 --output data/v2_online.json
conda run -n langgraph python eval/v2_ragas_eval.py --input data/v2_online.json --output data/v2_ragas.json --export-only
conda run -n langgraph python eval/v2_compare.py data/v2_online.json data/v2_online.json
```

在线实验不会自动提交医生审核；`failed`、`needs_evidence` 和 `awaiting_doctor_review` 都保留为真实状态。RAGAS 评分必须显式提供独立 Judge 的模型、地址和密钥。

## 中断续跑与终端进度

先启动后端；测评默认每 5 秒打印当前病例、等待时长、公开状态与最近完成的阶段，
每例结束打印累计完成数、失败数和耗时。Conda 下使用 `--no-capture-output` 避免缓冲。

```powershell
conda run --no-capture-output -n langgraph python -u eval/v2_online_eval.py --dataset eval/datasets/benchmark48/cases_detailed.jsonl --output data/eval/clinical42.json --timeout-seconds 1800 --resume --progress-seconds 5
```

`--resume` 必须使用原数据集与原输出报告；脚本校验整个数据集文件摘要并跳过所有
已记录病例（包括失败病例），旧版报告也兼容。例如已有 SC001～SC004 会从 SC005
继续。输出文件叫 clinical42 并不决定题库数量；上述数据集仍为 48 例。

新脚本在提交前保存幂等键、受理后保存会诊 ID，Ctrl+C 后可继续查询同一会诊；
强制关闭终端也可读取最后一次原子保存。旧脚本没有保存运行中病例，因此旧版中断时
尚未写入报告的病例可能会重新提交。恢复不会自动执行临床失败阶段重试或医生动作。

普通运行不覆盖已有报告；要重新实验请更换输出文件。`--limit N` 表示题库前 N 例，
不是额外运行 N 例。续跑拒绝混合不同模型、知识版本、检索模式和重排器使能配置；
仍应自行保证其他运行参数不变。`--timeout-seconds` 只控制客户端等待，不修改后端预算。

RAGAS 只用于检索忠实度、上下文相关性等辅助分析。没有独立专家审核的参考标准时，不把六经或方证指标称为临床准确率；48 条研发银标仅作研发对照。

## 实验记录与后端诊断日志（新实验建议）

评测客户端不能凭空取得另一进程的 stderr。要记录具体校验位置及 Schema 字段，
后端必须在**用户下一次主动启动时**显式启用安全诊断文件；仅更新评测脚本、
仅启用客户端日志或仅对旧报告加 `--resume`，都不能补回过去未保存的错误详情。
助手不会自动重启后端或发起在线实验。

以下命令均在 `SixChannel_Agent_Console` 目录执行。实验输出放上一级工程目录，
与项目代码分开；每轮使用新的目录和报告，续跑才复用原报告。

```powershell
# 终端一：用户按正常流程启动后端，同时开启安全JSONL日志
conda run --no-capture-output -n langgraph python scripts/start_console.py --diagnostic-log "../实验记录/run_001/backend.jsonl"

# 终端二：用户自行开始实验；首次运行不要加--resume
conda run --no-capture-output -n langgraph python -u eval/v2_online_eval.py --dataset eval/datasets/benchmark48/cases_detailed.jsonl --output "../实验记录/run_001/report.json" --records-dir "../实验记录/run_001/client" --backend-log "../实验记录/run_001/backend.jsonl" --timeout-seconds 1800
```

在相同测评命令末尾加 `--resume` 继续；不要把后端日志路径填成报告路径。
后端在远程机器时，客户端不能直接读取远程文件：应先让远程启动入口记录同格式日志，
再经授权的文件挂载/复制提供本地可读文件。没有文件时仍能评测，但报告明确显示缺口，
不会假称已拿到服务端根因。不新增公开诊断API，不从配置文件获取SSH或数据库凭据。

保存内容：

- `report.json`：兼容原来的病例结果、研发指标和旧版续跑；新增稳定 `run_id`、每次
  `sessions`、UTC起止、脚本/Prompt SHA、本地Git身份、显式参数、Python/HTTPX版本。
  本地版本不是服务器实际加载代码的证明，后端启动日志另外记录本地checkout指纹。
- `client/session-<UUID>.jsonl`：每次启动一个新文件，追加并逐条刷新；记录病例与会诊ID、
  HTTP方法/路由类别/状态码/响应头request ID（若为UUID）/响应头到达耗时、阶段事件、
  事件采集和客户端中断类别，不保存请求URL、认证头、响应正文或异常全文。
- `backend.jsonl`：可选、安全结构化后端错误；异常链类型、源代码位置、已有校验规则、
  候选索引/修复次数及Pydantic的`loc/type`，不写异常message/input/context、患者原文、
  模型JSON或堆栈局部变量。20MB轮转，最多5个备份；外部应按实验归档并控制访问。
  每个文件只供一个后端进程写入；不支持多个worker共享轮转文件，日志另含PID便于核对。
- 每行结果新增 `diagnostics`（按同一consultation ID关联）、`event_capture`、
  `stage_observability`、`usage_by_capability`。缺少日志/没有匹配/坏行/读失败分别披露。
  后端异常没有结构化rule时，可利用安全代码位置定位分支；不编造规则名称。

事件改用已有 `/events/page` 按游标分页：运行中每个进度周期采集、每页保存；终态再补齐。
中途断网、超时或Ctrl+C保留已收到事件和稳定pending身份，后续不会重提新会诊。
终态事件采集失败不覆盖既有临床结果或失败码，而单独记录partial/unavailable；
后续成功读取可补齐并保留曾失败的历史。强制结束进程只能保留最后一次落盘内容。

`usage` 保留历史兼容口径（通用model终止事件，含非Chat能力），请使用
`usage_by_capability`区分临床流、Embedding和Reranker，心跳不计调用。
S4专项流单列会话数；现有公开事件没有S4 Token数，因此新脚本仍标null/partial，
**不会把状态available推算成Token或费用**。客户端HTTP次数也不是供应商模型调用次数。
本轮不改后端S4流式usage协议或任何临床判断/校验逻辑。

原研发召回指标和默认连续三例失败停止策略保持；停止文案明确包括结构/引用失败。
`latency_ms`含客户端本次尝试与观测开销，不是纯推理耗时；跨进程续跑另标attempt次数，
阶段耗时应从阶段事件读取。原始`result`仍可能含完整病例，属于受控实验数据，
不得把报告作为“脱敏日志”公开分享；辅助JSONL只保存安全投影。已有临床报告不自动改写。
