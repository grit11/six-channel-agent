# V1.2 项目结构

仓库根目录为 `SixChannel_Agent_Console/`。原 React 项目、升级资料包及父目录测试归档不属于本仓库。

```text
SixChannel_Agent_Console/
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI、生命周期、原生页面
│   │   ├── core/               # 配置、端口、依赖工厂、日志、异常
│   │   ├── api/v1/             # HTTP/SSE 边界，不写医学规则
│   │   ├── services/           # 会诊、任务、恢复、审核、知识 QA
│   │   ├── graph/              # 五自动节点子图与人工审核主图
│   │   ├── agents/             # 临床和独立 QA 的编排
│   │   ├── nodes/              # 阶段实现及共享规则模块
│   │   ├── knowledge/          # 唯一知识边界、分支索引、请求缓存
│   │   ├── rag/                # BM25/Dense/融合/重排/适用性核验
│   │   ├── persistence/        # 业务 ORM/Repository 与独立 Saver
│   │   ├── schemas/            # 三值事实、来源、候选、会诊和事件契约
│   │   ├── prompts/            # clinical_parser.md、channel_candidate.md
│   │   ├── observability/      # 公开进度、真实请求 usage，不保存隐藏推理
│   │   ├── safety/             # 保留目录契约，安全分诊未实现
│   │   └── static/             # HTML/JS/SVG 原生 Console
│   ├── tests/                  # unit、integration、api
│   ├── alembic/                # 0001、0002 业务迁移链
│   └── pyproject.toml          # 依赖、包版本和质量工具
├── knowledge/
│   ├── v1.2/                   # 五类原始知识 JSONL
│   └── derived/fpeu_v12.jsonl  # 来源约束的方证分支索引
├── eval/                       # 当前评测入口、适配器和数据集
├── scripts/                    # 启动、索引、校验、验收、浏览器与真实联调
├── docs/                       # 当前架构、结构、发布状态
├── README.md
├── AGENTS.md
├── CODEX_DEVELOPMENT_STANDARD_V1.1.md
├── .env.example                # 可分发开发模板；真实 .env 不提交
├── docker-compose.yml
├── Makefile
└── start_console_windows.bat
```

## 阶段与模块映射

| 阶段 | 内部组件 |
| --- | --- |
| S1 | `clinical_parser.py`，医生结构化事实优先 |
| S2 | `eight_principles.py`、`channel_candidate.py`、`reasoning_llm.py` |
| S3 | `formula_candidate.py`、`knowledge/branch_index.py`、KnowledgeService |
| S4 | `syndrome_relation.py`、`relationship_rules.py`、`temporal_evidence.py`、`differential_reasoner.py` |
| S5 | `topk_builder.py`、`candidate_score.py`、`source_conditions.py`、`path_summary.py`、`evidence_gap.py` |
| S6 | `main_graph.py` 的 `doctor_review` 及审核/会诊服务 |

文件数不等于节点数。删除共享模块会破坏 V1.2。API、业务存储、后台恢复和独立知识 QA 同样属于当前版本，不能按旧文件名或测试阶段编号判为无关。

## scripts

| 文件 | 职责 |
| --- | --- |
| `start_console.py` | 固定根目录配置，兼容 Windows PostgreSQL 异步循环 |
| `index_knowledge.py` | 索引校验、dry-run、创建和显式重建 |
| `validate_knowledge.py` | 主库与派生索引结构、来源核验 |
| `bootstrap_dev.py` | 环境及可选基础设施探测 |
| `engineering_acceptance.py`、`ci.ps1` | 11 项统一离线工程验收 |
| `console_e2e.cjs`、`run_e2e_backend.py` | 隔离后端与浏览器流程回归 |
| `verify_real_hybrid.py` | 真实 Hybrid 联调；`--probe-reranker` 独立验证配置重排器 |

`eval/` 入口职责与全量命令见 [测评说明](../eval/README_EVALUATION.md)。`test_similar_cases.py` 验证历史医案不自动检索、不影响排名，应保留。名字中的旧阶段号不代表测试无效。

## 本地与版本管理

- `data/` 存 Lite 索引、SQLite Checkpoint、评测结果，整体忽略，不清空运行数据。
- `.env`、`.env.judge` 等保存凭据，整体忽略；模板只含占位值。
- `.vscode/` 保留个人设置，不上传。
- `__pycache__`、工具缓存和临时报告可再生，不发布。
- 数据库迁移链、包标记、两个运行 Prompt、评测数据和有效测试必须保留。

清理结果与删除理由见 [V1.2 发布说明](RELEASE_V1_2.md)。
