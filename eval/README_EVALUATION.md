# V1.2 六经辨证测评脚本使用说明

本说明对应当前 `六经辨证推理优化` 分支的测评工具。Benchmark48 包含 48 条同源合成银标病例，标签仍需专家审查；工程通过、标签命中和 RAGAs 分数分别反映不同问题，不能互相替代。

## 1. 保留的脚本与数据

| 文件 | 职责 |
| --- | --- |
| `eval/system_eval.py` | 执行病例解析与辨证，记录公开结构化结果、工程断言、耗时、失败及真实知识访问追踪 |
| `eval/benchmark_eval.py` | 读取推理报告，输出六经、方证、关系、联合命中、分组及鉴别对照指标 |
| `eval/ragas_eval.py` | 核验来源与上下文，运行在线 RAGAs 语义评分或离线 ID 指标 |
| `eval/compare_eval.py` | 在共同成功病例上比较消融组；可配对比较同口径 RAGAs 报告 |
| `eval/evaluation_tools.py` | 并发隔离的知识访问观察器、实验身份与报告校验工具，供入口调用 |
| `eval/ragas_adapters.py` | 将项目模型端口接入 RAGAs，供评分脚本调用 |
| `eval/retrieval_eval.py` | 398 条经典精确读取、53 条鉴别检索的工程回归，统一验收仍调用此入口 |
| `eval/workflow_acceptance.py` | 离线主图、补证、SQLite checkpoint 重开与医生审核闭环验收 |

后两项为独立工程验收入口，并非 `system_eval.py` 的旧版本。它们及 `backend/tests/` 保留，避免破坏 `scripts/engineering_acceptance.py` 的执行契约。

`eval/datasets/benchmark48/` 只包含以下四个数据文件：

- `cases_eval_compatible.jsonl`：系统测评默认输入。
- `cases_detailed.jsonl`：详细银标，供结果汇总与评分使用。
- `challenge_pairs.csv`：成对鉴别病例。
- `SixChannel_Benchmark48_Review.xlsx`：专家审核表。

`eval/datasets/engineering_cases.jsonl` 为另外 6 条工程回归输入。标签只用于评分，不作为业务模型推理输入。

## 2. 本地运行准备

在 PowerShell 中进入 Console 项目，使用已有 Conda 环境：

```powershell
Set-Location -LiteralPath 'D:\Codex\伤寒论-六经辨证\六经辨证临床辅助agent\SixChannel_Agent_Console'
$env:PYTHONUTF8 = '1'
conda run -n langgraph python --version
```

依赖声明位于 `backend/pyproject.toml`；若尚未安装评测依赖：

```powershell
conda run -n langgraph python -m pip install -e './backend[dev,eval]'
```

当前评测依赖为 RAGAs 0.4.x、langchain-core 1.x。在线推理读取项目 `.env`，需配置可用 LLM、Embedding、Reranker 及向量后端。模型与服务均通过项目现有工厂创建。`MILVUS_MODE=auto` 在允许降级时可使用 Lite，实际后端及 degraded 状态记录在报告中。

无需先启动 FastAPI。此测评直接调用项目服务，不写业务会诊数据库或 LangGraph checkpoint；在线模式可能读取或自动建立 Lite 向量索引。使用 Lite 时建议先停止占用同一索引的 Console 服务，再依次执行测评。

## 3. 全量 48 条：推理 → 汇总 → RAGAs

以下三个命令按顺序执行。全量命令不设置 `--limit`，读取整个默认数据集。

### 第一步：全量 D 组在线推理

```powershell
conda run --no-capture-output -n langgraph python eval/system_eval.py --dataset eval/datasets/benchmark48/cases_eval_compatible.jsonl --ablation D --concurrency 2 --case-timeout 180 --output data/benchmark48/online_D_full.json
```

D 组固定为 LLM 解析/推理 + Hybrid 检索 + 重排。`--concurrency` 范围为 1–8；服务限流时可改为 1。`--case-timeout` 是单案总时限，超时逐案记录为失败；若需提高时限，应在首次全量运行前调整。

输出包括：

- `online_D_full.json`：完整结果及实际配置、代码/提示/知识身份。
- `online_D_full.failures.jsonl`：执行失败或工程断言失败的病例。
- `online_D_full.annotation.jsonl`：供人工审核的摘要。

进度中的 `completed` 表示该案生成结果；整个报告的 `status=completed` 仅表示所选批次结束。全量工程通过须同时核对 `expected_sample_count=48`、`sample_count=48`、`complete_benchmark=true`、`execution_failed=0` 和 `passed=true`。退出码 0 表示此次选择的病例工程检查通过，不能单独证明完成全量或达到医学质量标准。

### 第二步：标签指标汇总

```powershell
conda run --no-capture-output -n langgraph python eval/benchmark_eval.py --report data/benchmark48/online_D_full.json --output data/benchmark48/online_D_summary.json
```

输出 `online_D_summary.json` 和同名 `.md`，包含 Top-1/Top-3 标签覆盖、单路径覆盖、精确路径、同路径六经/方证/关系联合命中、禁止方证、Unknown/冲突检查与鉴别对照。旧的 Top-3 标签并集覆盖与单路径指标口径不同，应分别查看。

命令正常生成摘要不等于源推理全部通过；执行失败单列，缺少标签保持 unlabelled，部分运行保留原集数量。具体指标须结合报告中的有效样本数和分母解释。

### 第三步：全量 RAGAs

```powershell
conda run --no-capture-output -n langgraph python eval/ragas_eval.py --report data/benchmark48/online_D_full.json --context-scope citations --output data/benchmark48/ragas_D_full.json
```

默认在线评分公开 Top-3 摘要，使用最终引用并集作为上下文。默认指标为 Faithfulness、ContextUtilization 和 IDBasedContextPrecision/Recall；不默认开启银标 ContextRecall 或有参 ContextPrecision。输出 `ragas_D_full.json` 与 `ragas_D_full.samples.json`，逐指标记录均值、有效评分数、failed 和 unavailable。

源推理失败病例不构造答案，记录在 `source_execution_failed_ids`。评分异常或 NaN 保持失败，不补零。应检查 `status`、`complete_benchmark`、实际样本数及各指标有效评分数；48 条都尝试过不等于48条都有可评分结果。

## 4. 中断恢复与重复运行

`system_eval.py` 拒绝覆盖已有报告。相同实验可在第一步命令中追加 `--resume`：

```powershell
conda run --no-capture-output -n langgraph python eval/system_eval.py --dataset eval/datasets/benchmark48/cases_eval_compatible.jsonl --ablation D --concurrency 2 --case-timeout 180 --output data/benchmark48/online_D_full.json --resume
```

续跑严格校验病例、配置、代码、提示和知识身份，复用工程检查成功的病例并重新核验引用，失败病例重新执行。缺少身份签名的旧报告或身份不一致的报告不能续跑。改动配置、输入或实现后使用新的输出文件名；不删除失败记录来伪装一次运行全部成功。

RAGAs 暂不支持评分续跑，重复评分需指定新的输出路径。推理续跑改变源报告后，应重新生成指标汇总和 RAGAs 报告，避免将旧评分绑定到新结果。

## 5. 可选评分口径

独立裁判通过 `--judge-env-file .env.judge` 指定配置，沿用项目 Settings 和模型工厂；项目知识配置仍来自 `.env` 或 `--env-file`。裁判配置文件不提交 Git。

真实召回上下文示例：

```powershell
conda run --no-capture-output -n langgraph python eval/ragas_eval.py --report data/benchmark48/online_D_full.json --context-scope retrieved --max-contexts 30 --max-context-chars 2000 --output data/benchmark48/ragas_D_retrieved.json
```

`retrieved` 仅使用新报告中业务执行时真实记录的召回，并经 KnowledgeService 核验正文。多阶段访问按调用顺序汇总，并非单次全局排名；显式预算会记录省略和截短情况。默认不截断，完整上下文可能增加评分成本。

`--include-clinical-input` 只将病例原文加入 Faithfulness 的支撑范围，不加入检索 ID 或 ContextUtilization。不同上下文范围、预算或是否加入病历原文的分数不可直接比较。

显式开启银标参考指标：

```powershell
conda run --no-capture-output -n langgraph python eval/ragas_eval.py --report data/benchmark48/online_D_full.json --allow-silver-reference --context-precision --output data/benchmark48/ragas_D_silver.json
```

`--context-precision` 必须与 `--allow-silver-reference` 一起使用，会增加逐篇裁判调用。银标参考和证据 ID 不穷尽所有正确证据，这些分数为探索性评价。

`--prepare-only` 仅核验评分样本，不运行裁判；`--offline` 仅运行 ID 指标。用 `--limit 2` 运行冒烟时，后续汇总/评分需加 `--allow-partial`，此选项只接受已结束的部分批次，不将部分结果变为全量。

## 6. 消融与配对比较

| 组 | 解析/推理 | 检索 | 重排 |
| --- | --- | --- | --- |
| A | 规则 | BM25 | 关闭 |
| B | LLM | BM25 | 关闭 |
| C | LLM | Hybrid | 关闭 |
| D | LLM | Hybrid | 开启 |

A 组仍通过实际基础设施工厂建立运行环境，不调用业务 LLM。`--offline` 是另一个工程模式，不能与 `--ablation` 同时指定。不指定消融组时，在线模式强制 LLM 解析/推理，检索与重排沿用 `.env`。

将第一步 `--ablation D` 改为 A/B/C，并分别使用独立输出路径。得到 A、D 报告后：

```powershell
conda run -n langgraph python eval/compare_eval.py --reports data/benchmark48/online_A_full.json data/benchmark48/online_D_full.json --output data/benchmark48/compare_AD_full.json
```

比较要求相同病例、输入标签、代码/提示/知识身份；执行与工程失败分别披露，领域指标只比较共同成功病例。可按同序提供 `--ragas-reports`，另校验源报告 SHA、裁判和上下文口径。

## 7. 离线工程回归

六条固定工程病例及 53 条鉴别规则：

```powershell
conda run --no-capture-output -n langgraph python eval/system_eval.py --offline --dataset eval/datasets/engineering_cases.jsonl --output data/benchmark48/engineering_current.json
```

离线工程回归固定规则/BM25，不是 48 条在线模型测评。`retrieval_eval.py`、`workflow_acceptance.py` 及后端 pytest 分别覆盖检索来源、主图审核流程和工程行为；统一入口为 `scripts/engineering_acceptance.py`。

## 8. 结果解释与当前验证范围

Faithfulness 衡量公开摘要的上下文支撑；ContextUtilization 评价上下文使用；银标命中评价与参考标签的一致性。均不能作为临床准确率或自动处方依据。Token usage 从真实 Provider 响应聚合，缺失时保持 unavailable，并披露有效调用覆盖率；不以估算替代供应商 usage。

当前发布的工程验证与已知限制见 [V1.2 发布说明](../docs/RELEASE_V1_2.md)。历史在线结果不能自动代表当前代码质量；本次发布整理未运行当前版本完整在线 48 条或 RAGAs，需使用新的实验身份重新测评。

历史生成结果位于 Git 忽略的 `data/benchmark48/`，全量命令使用新的 `*_full` 文件名。此目录可以生成结果 Markdown；`eval/` 仅保留这一份人工维护的测评说明。新结果以本次实际运行报告为准。
