## 结构示例（仅教学，不可复用其中 ID）

以下示例只说明 `ComparisonDraftV2` 的字段边界和未决策略。示例中的 `cand_*`、`fact_*`、`ev_*` 均为虚构占位符；真实输出必须从当前输入包逐字选择 ID，不得复制这些 ID，也不得从示例推断医学结论。

### FS4-01：证据充分时形成有限排名

```json
{
  "status": "ranked",
  "ordered_paths": [
    {
      "candidate_id": "cand_A",
      "six_channels": ["太阳"],
      "rank": 1,
      "support_fact_ids": ["fact_A", "fact_B"],
      "counter_fact_ids": [],
      "missing_dimensions": [],
      "support_evidence_ids": ["ev_A", "ev_diff"],
      "counter_evidence_ids": [],
      "brief_rationale": "已核对的关键表现与该候选的来源鉴别维度更一致。",
      "differential_summary": "相对另一候选，当前已确认的区分事实和来源支持更完整；不依据检索分数断言临床适用性。"
    },
    {
      "candidate_id": "cand_B",
      "six_channels": ["太阳"],
      "rank": 2,
      "support_fact_ids": ["fact_B"],
      "counter_fact_ids": [],
      "missing_dimensions": ["关键鉴别表现尚未核对"],
      "support_evidence_ids": ["ev_B"],
      "counter_evidence_ids": [],
      "brief_rationale": "有部分共同表现，但关键鉴别事实尚未同等确认。",
      "differential_summary": "未询问的表现保持未知，不能自动当作阴性。"
    }
  ],
  "next_questions": [],
  "unresolved_reason": null
}
```

### FS4-02：关键事实不足，先补问

```json
{
  "status": "needs_clarification",
  "ordered_paths": [],
  "next_questions": [
    {
      "question_id": "question_A",
      "question": "请核对能够区分现有候选的关键表现及其出现时间。",
      "targets": ["cand_A", "cand_B"],
      "source_fact_ids": ["fact_A"],
      "expected_effect": "补充事实后再判断现有候选之间的相对优先级。"
    }
  ],
  "unresolved_reason": "关键鉴别事实尚未明确，不能发布正式排名。"
}
```

### FS4-03：候选或来源整体不足

```json
{
  "status": "insufficient",
  "ordered_paths": [],
  "next_questions": [],
  "unresolved_reason": "当前候选或真实来源不足，无法形成可核验的相对比较。"
}
```

禁止从这些示例复制方名、事实、Evidence、路径身份、排名或文字结论；不得输出 `path_id`、`formula_name`、`verification_status`、`source_evidence` 或 Schema 外字段。
