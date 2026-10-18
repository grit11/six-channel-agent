## V2.1 S1 Few-shot（仅示范抽取契约）

以下样例只示范如何保留原文、否定、时点、未知、重复出现和冲突；不得把样例中的症状、概念或组合当作固定词典，也不得补造原文没有的事实。输出仍必须严格遵循 `S1ExtractionDraft`。

### 例1：复合脉象与局部否定

输入：`今天怕冷明显，身上没有出汗，咳嗽时稍有气喘；不发热，脉浮而紧。`

```json
{"findings":[{"local_id":"f1","concept":"恶寒","polarity":"positive","time_scope":"current","source_text":"今天怕冷明显","occurrence":0,"needs_clarification":false},{"local_id":"f2","concept":"汗出","polarity":"negative","time_scope":"current","source_text":"身上没有出汗","occurrence":0,"needs_clarification":false},{"local_id":"f3","concept":"咳嗽","polarity":"positive","time_scope":"current","source_text":"咳嗽","occurrence":0,"needs_clarification":false},{"local_id":"f4","concept":"喘","polarity":"positive","time_scope":"current","source_text":"稍有气喘","occurrence":0,"needs_clarification":false},{"local_id":"f5","concept":"发热","polarity":"negative","time_scope":"current","source_text":"不发热","occurrence":0,"needs_clarification":false},{"local_id":"f6","concept":"脉浮","polarity":"positive","time_scope":"current","source_text":"脉浮而紧","occurrence":0,"needs_clarification":false},{"local_id":"f7","concept":"脉紧","polarity":"positive","time_scope":"current","source_text":"脉浮而紧","occurrence":0,"needs_clarification":false}],"chief_complaint_text":null,"unresolved_phrases":[],"conflicts":[]}
```

同一复合片段可支持不同概念；不能把“没有出汗”再拆成额外的双重否定事实。

### 例2：跨时点与治疗后变化

输入：`三日前有腹泻，昨日服药后开始腹胀；今天腹泻已停止，但腹胀仍在，腹胀程度比昨日减轻。`

```json
{"findings":[{"local_id":"f1","concept":"下利","polarity":"positive","time_scope":"past","source_text":"三日前有腹泻","occurrence":0,"needs_clarification":false},{"local_id":"f2","concept":"腹胀","polarity":"positive","time_scope":"after_treatment","source_text":"昨日服药后开始腹胀","occurrence":0,"needs_clarification":false},{"local_id":"f3","concept":"下利","polarity":"negative","time_scope":"current","source_text":"今天腹泻已停止","occurrence":0,"needs_clarification":false},{"local_id":"f4","concept":"腹胀","polarity":"positive","time_scope":"current","source_text":"腹胀仍在","occurrence":0,"needs_clarification":false},{"local_id":"f5","concept":"腹胀程度减轻","polarity":"positive","time_scope":"current","source_text":"腹胀程度比昨日减轻","occurrence":0,"needs_clarification":false}],"chief_complaint_text":null,"unresolved_phrases":[],"conflicts":[]}
```

不同时间的相反极性不构成同一时点冲突；治疗后出现也不代表当前已经消失。

### 例3：相同原文的 occurrence

输入：`昨日病历写“头痛”；今天复诊又写“头痛”，现在无恶寒。`

```json
{"findings":[{"local_id":"f1","concept":"头痛","polarity":"positive","time_scope":"past","source_text":"头痛","occurrence":0,"needs_clarification":false},{"local_id":"f2","concept":"头痛","polarity":"positive","time_scope":"current","source_text":"头痛","occurrence":1,"needs_clarification":false},{"local_id":"f3","concept":"恶寒","polarity":"negative","time_scope":"current","source_text":"现在无恶寒","occurrence":0,"needs_clarification":false}],"chief_complaint_text":null,"unresolved_phrases":[],"conflicts":[]}
```

`occurrence` 是相同 `source_text` 在原文中的零起始出现序号，不是 `local_id` 序号。

### 例4：明确未知与无法解释的短语

输入：`目前是否口渴还不清楚，夜间盗汗尚未询问；胸口总觉得怪怪的，具体感觉说不清。`

```json
{"findings":[{"local_id":"f1","concept":"口渴","polarity":"unknown","time_scope":"current","source_text":"目前是否口渴还不清楚","occurrence":0,"needs_clarification":true},{"local_id":"f2","concept":"夜间盗汗","polarity":"unknown","time_scope":"unspecified","source_text":"夜间盗汗尚未询问","occurrence":0,"needs_clarification":true}],"chief_complaint_text":null,"unresolved_phrases":["胸口总觉得怪怪的，具体感觉说不清"],"conflicts":[]}
```

明确未问/不清楚可抽取为 `unknown`；无法确定概念的片段必须逐字放入 `unresolved_phrases`，不可改写或丢弃。

### 例5：同一时点冲突

输入：`今天同一时点两份记录：一份记“有汗”，另一份记“无汗”，实际情况待核实。`

```json
{"findings":[{"local_id":"f1","concept":"汗出","polarity":"positive","time_scope":"current","source_text":"有汗","occurrence":0,"needs_clarification":false},{"local_id":"f2","concept":"汗出","polarity":"negative","time_scope":"current","source_text":"无汗","occurrence":0,"needs_clarification":false}],"chief_complaint_text":null,"unresolved_phrases":["实际情况待核实"],"conflicts":[["f1","f2"]]}
```

冲突两侧都要保留；`conflicts` 引用草稿 `local_id`，不能擅自选一侧为真。

### 例6：主诉连续原文与复合否定

输入：`主诉：胸闷两日。查脉记“并非浮紧脉”，另见脉沉；是否有口苦尚未询问。`

```json
{"findings":[{"local_id":"f1","concept":"胸闷","polarity":"positive","time_scope":"current","source_text":"胸闷两日","occurrence":0,"needs_clarification":false},{"local_id":"f2","concept":"脉浮紧","polarity":"negative","time_scope":"current","source_text":"并非浮紧脉","occurrence":0,"needs_clarification":false},{"local_id":"f3","concept":"脉沉","polarity":"positive","time_scope":"current","source_text":"脉沉","occurrence":0,"needs_clarification":false},{"local_id":"f4","concept":"口苦","polarity":"unknown","time_scope":"unspecified","source_text":"是否有口苦尚未询问","occurrence":0,"needs_clarification":true}],"chief_complaint_text":"胸闷两日","unresolved_phrases":[],"conflicts":[]}
```

复合否定只否定复合概念，不自动拆成“脉浮阴性”和“脉紧阴性”；主诉必须是原文连续子串。
