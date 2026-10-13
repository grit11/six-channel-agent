# V2 评测说明

`v2_online_eval.py` 调用 V2 API 保存每条病例的公开结果、暂停状态、错误类别和延迟；`v2_ragas_eval.py` 生成 RAGAS 输入，`v2_compare.py` 比较工程指标。

```powershell
conda run -n langgraph python eval/v2_online_eval.py --dataset eval/datasets/benchmark48/cases_detailed.jsonl --base-url http://127.0.0.1:8000 --output data/v2_online.json
conda run -n langgraph python eval/v2_ragas_eval.py --input data/v2_online.json --output data/v2_ragas.json --export-only
conda run -n langgraph python eval/v2_compare.py data/v2_online.json data/v2_online.json
```

在线实验不会自动提交医生审核；`failed`、`needs_evidence` 和 `awaiting_doctor_review` 都保留为真实状态。RAGAS 评分必须显式提供独立 Judge 的模型、地址和密钥。

RAGAS 只用于检索忠实度、上下文相关性等辅助分析。没有独立专家审核的参考标准时，不把六经或方证指标称为临床准确率；48 条研发银标仅作研发对照。
