/** V2 工作台依赖权威快照、公开事件和知识 API；不计算医学分，不存病例缓存。
 * 确认后保留病例，点击下一次问诊才重置；历史详情不覆盖当前会诊。
 */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const esc = (v) =>
    String(v ?? '').replace(
      /[&<>"']/g,
      (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
    );
  const stages = [
    ['s1_extract', 'S1 事实抽取'],
    ['s2_localize', 'S2 六经定位'],
    ['s3_retrieve', 'S3 来源检索'],
    ['s4_compare', 'S4 证据比较'],
    ['s5_doctor_review', 'S5 医生审核'],
  ];
  const terminal = new Set(['completed', 'undetermined', 'cancelled', 'failed']);
  const labels = {
    running: '执行中',
    created: '已创建',
    needs_evidence: '等待补证',
    awaiting_doctor_review: '等待医生审核',
    completed: '已完成',
    undetermined: '未决',
    cancelled: '已取消',
    failed: '执行失败',
    s3_retrieval_only: 'S3 检索兜底',
    positive: '阳性',
    negative: '阴性',
    unknown: '未知',
    current: '当前',
    past: '既往',
    before_treatment: '治疗前',
    after_treatment: '治疗后',
    unspecified: '时点未说明',
    grounded: '来源已核验',
    pending: '来源待核验',
    needs_clinician_review: '需医生核查',
    accept: '采用',
    modify: '修订后采用',
    reject: '驳回',
    note: '记录备注',
    resolved: '定位明确',
    ambiguous: '多经候选',
    conflicting: '依据冲突',
  };
  const label = (v) => labels[v] || v || '暂无';
  const state = {
    snapshot: null,
    facts: null,
    selected: null,
    source: new Map(),
    events: new Map(),
    stream: null,
    timer: null,
    epoch: 0,
    request: 0,
    applied: 0,
    busy: false,
    replay: null,
    config: null,
    historyTicket: 0,
    historySnapshot: null,
    view: 'workbench',
  };
  /** 统一同源 API 边界，只展示安全错误，不记录病例。 */
  async function api(path, body, headers = {}) {
    const response = await fetch(
      path,
      body === undefined
        ? {}
        : {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ...headers },
            body: JSON.stringify(body),
          },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || data.detail?.[0]?.msg || '请求失败');
    return data;
  }
  /** 提示与页面既有 visible 样式保持一致。 */
  function toast(message) {
    $('toast').textContent = message;
    $('toast').classList.add('visible');
    setTimeout(() => $('toast').classList.remove('visible'), 4500);
  }
  /** 极性、时点均来自原事实；反证不自动转换为阴性。 */
  function chip(f) {
    return (
      '<span class="chip ' +
      esc(f.polarity) +
      '" title="' +
      esc(f.source_span?.text) +
      '">' +
      esc(f.concept) +
      ' · ' +
      esc(label(f.polarity)) +
      ' · ' +
      esc(label(f.time_scope)) +
      '</span>'
    );
  }
  /** 结束后保留只读事实；暂停时才允许提交修订。 */
  function renderFacts(facts) {
    state.facts = facts;
    const findings = facts?.findings || [];
    $('findingsCount').textContent = findings.length + ' 项';
    $('unknownCount').textContent =
      '明确未知 ' + findings.filter((f) => f.polarity === 'unknown').length + ' 项';
    $('clinicalChips').innerHTML =
      findings.map(chip).join('') || '<span class="chip unknown">等待提取事实</span>';
    $('factRows').innerHTML = findings
      .map(
        (f, i) =>
          '<div class="fact-row"><strong>' +
          esc(f.concept) +
          '</strong><div><select aria-label="事实极性" data-polarity="' +
          i +
          '">' +
          ['positive', 'negative', 'unknown']
            .map(
              (p) =>
                '<option value="' +
                p +
                '" ' +
                (p === f.polarity ? 'selected' : '') +
                '>' +
                label(p) +
                '</option>',
            )
            .join('') +
          '</select><select aria-label="事实时点" data-time="' +
          i +
          '">' +
          ['current', 'past', 'before_treatment', 'after_treatment', 'unspecified']
            .map(
              (t) =>
                '<option value="' +
                t +
                '" ' +
                (t === f.time_scope ? 'selected' : '') +
                '>' +
                label(t) +
                '</option>',
            )
            .join('') +
          '</select><small>' +
          esc(f.source_span.text) +
          '</small></div></div>',
      )
      .join('');
    $('conflictNotice').textContent = facts?.conflicts?.length
      ? '存在事实冲突，请核对原文后修订。'
      : '';
    controls();
  }
  /** S3 和 S4 的真实来源合并，保证鉴别证据也可回查。 */
  function sources(s) {
    return new Map(
      [
        ...(s.retrieval?.formula_candidates || []).flatMap((c) => c.source_evidence),
        ...(s.comparison?.source_evidence || []),
      ].map((e) => [e.evidence_id, e]),
    );
  }
  /** 来源仅显示真实摘录，所有插值转义。 */
  function sourceCards(ids, source) {
    return (
      [...new Set(ids)]
        .map(
          (id) =>
            '<button class="source-item" data-source="' +
            esc(id) +
            '"><span class="source-top">' +
            esc(source.get(id)?.clause_no ? '第 ' + source.get(id).clause_no + ' 条 · ' + id : id) +
            ' ↗</span><span class="source-text">' +
            esc(source.get(id)?.excerpt || '点击回查完整来源') +
            '</span></button>',
        )
        .join('') || '<p class="small-empty">暂无引用来源</p>'
    );
  }
  /** 排名来自 LLM，视觉卡片不计算分数或概率。 */
  function card(p, i, interactive = true) {
    return (
      '<' +
      (interactive ? 'button type="button"' : 'article') +
      ' class="result-card" ' +
      (interactive ? 'data-path="' + i + '"' : '') +
      '><div class="result-top"><span class="result-rank">候选 ' +
      esc(p.rank ?? '未决') +
      '</span><span class="chip channel">' +
      esc(label(p.verification_status)) +
      '</span></div><h3 class="result-name">' +
      esc(p.formula_name || '未决') +
      '</h3><p class="result-channels">' +
      esc(p.six_channels.join(' / ')) +
      '</p><p class="result-rationale">' +
      esc(p.brief_rationale) +
      '</p><p class="result-desc">' +
      esc(p.differential_summary) +
      '</p><div class="result-footer"><span>支持 ' +
      p.support_fact_ids.length +
      ' · 反证 ' +
      p.counter_fact_ids.length +
      ' · 缺口 ' +
      p.missing_dimensions.length +
      '</span><strong>' +
      (interactive ? '查看证据 →' : '历史候选') +
      '</strong></div></' +
      (interactive ? 'button' : 'article') +
      '>'
    );
  }
  /** 选择只影响展示；不会修改模型排序或已保存审核。 */
  function selectPath(path) {
    state.selected = path;
    const paths = state.snapshot.comparison.ordered_paths;
    $('results')
      .querySelectorAll('[data-path]')
      .forEach((el) => {
        const active = paths[Number(el.dataset.path)].path_id === path.path_id;
        el.classList.toggle('selected', active);
        el.setAttribute('aria-pressed', String(active));
      });
    const facts = new Map(
      (state.snapshot.clinical_evidence?.findings || []).map((f) => [f.fact_id, f]),
    );
    for (const [id, ids] of [
      ['supportList', path.support_fact_ids],
      ['conflictList', path.counter_fact_ids],
    ])
      $(id).innerHTML =
        ids.map((key) => (facts.has(key) ? chip(facts.get(key)) : esc(key))).join('') ||
        '<span class="small-empty">暂无</span>';
    $('missingList').innerHTML =
      path.missing_dimensions
        .map((t) => '<span class="chip unknown">' + esc(t) + '</span>')
        .join('') || '<span class="small-empty">暂无</span>';
    $('selectedSyndrome').textContent = path.formula_name || '未决';
    $('supportCount').textContent = path.support_fact_ids.length + '项';
    $('conflictCount').textContent = path.counter_fact_ids.length + '项';
    $('missingCount').textContent = path.missing_dimensions.length + '项';
    $('pathAudit').textContent = JSON.stringify(path, null, 2);
    $('sourceList').innerHTML = sourceCards(
      [...path.support_evidence_ids, ...path.counter_evidence_ids],
      state.source,
    );
    bindSources($('sourceList'));
  }
  /** 无候选或重置时清理全部旧证据，不残留上次病例。 */
  function clearPath() {
    state.selected = null;
    ['supportList', 'conflictList', 'missingList', 'sourceList'].forEach(
      (id) => ($(id).innerHTML = '<span class="small-empty">暂无</span>'),
    );
    ['supportCount', 'conflictCount', 'missingCount'].forEach((id) => ($(id).textContent = '0项'));
    $('selectedSyndrome').textContent = '尚未选择候选';
    $('pathAudit').textContent = '尚无路径审计';
  }
  /** 定位公开解释来自快照，仅翻译状态名称。 */
  function localizationText(l) {
    return l
      ? label(l.status) +
          '\n' +
          l.candidates.map((c) => c.six_channels.join(' / ') + '：' + c.short_basis).join('\n') +
          '\n' +
          l.note
      : '等待六经定位';
  }
  /** 每轮每阶段独立累计真实调用，usage 缺失保持未知；S5 只计算人工等待。 */
  function metrics(events, snapshot) {
    const rows = new Map(),
      starts = new Map();
    let waiting = null,
      waitMs = 0,
      reviewCount = 0;
    for (const e of [...events.values()].sort((a, b) => a.sequence - b.sequence)) {
      const p = e.payload,
        key = e.turn + ':' + p.stage;
      if (stages.some(([id]) => id === p.stage)) {
        if (!rows.has(key))
          rows.set(key, {
            stage: p.stage,
            turn: e.turn,
            ms: 0,
            timed: false,
            calls: 0,
            known: 0,
            input: 0,
            output: 0,
            failed: 0,
          });
        const r = rows.get(key);
        if (e.event === 'stage_started') starts.set(key, Date.parse(e.created_at));
        if (['stage_completed', 'stage_failed'].includes(e.event)) {
          const ms = Number.isFinite(p.duration_ms)
            ? p.duration_ms
            : starts.has(key)
              ? Date.parse(e.created_at) - starts.get(key)
              : null;
          if (ms !== null) {
            r.ms += Math.max(0, ms);
            r.timed = true;
          }
          starts.delete(key);
          if (e.event === 'stage_failed') r.failed++;
        }
        if (['model_completed', 'model_failed'].includes(e.event)) {
          r.calls++;
          if (
            p.usage_status === 'available' &&
            Number.isInteger(p.prompt_tokens) &&
            p.prompt_tokens >= 0 &&
            Number.isInteger(p.completion_tokens) &&
            p.completion_tokens >= 0
          ) {
            r.known++;
            r.input += p.prompt_tokens;
            r.output += p.completion_tokens;
          }
        }
      }
      if (
        waiting &&
        ['snapshot_updated', 'consultation_completed', 'failed'].includes(e.event) &&
        (e.turn !== waiting.turn || terminal.has(p.status) || e.event !== 'snapshot_updated')
      ) {
        waitMs += Math.max(0, Date.parse(e.created_at) - waiting.time);
        waiting = null;
      }
      if (e.event === 'review_requested' && !waiting) {
        waiting = { time: Date.parse(e.created_at), turn: e.turn };
        reviewCount++;
      }
    }
    for (const [key, start] of starts) {
      const r = rows.get(key);
      r.ms += Math.max(
        0,
        (snapshot.status === 'running' ? Date.now() : Date.parse(snapshot.updated_at)) - start,
      );
      r.timed = true;
      r.incomplete = true;
    }
    if (waiting)
      waitMs += Math.max(
        0,
        (terminal.has(snapshot.status) ? Date.parse(snapshot.updated_at) : Date.now()) -
          waiting.time,
      );
    return { rows: [...rows.values()], waitMs, reviewCount };
  }
  /** 使用毫秒展示短阶段，避免虚构为零秒。 */
  function duration(ms) {
    return ms < 1000 ? Math.round(ms) + ' ms' : (ms / 1000).toFixed(1) + ' s';
  }
  /** 部分 usage 显示覆盖率，已知数字不能被解释为完整账单。 */
  function tokenText(r) {
    return !r.calls
      ? '无模型调用记录'
      : !r.known
        ? 'Token：未提供'
        : (r.known < r.calls ? '已知 ' : '') +
          'Token ' +
          (r.input + r.output) +
          '（入 ' +
          r.input +
          ' / 出 ' +
          r.output +
          '）' +
          (r.known < r.calls ? ' · 用量覆盖 ' + r.known + '/' + r.calls : '');
  }
  /** 五阶段累计卡片与各执行轮次共用事件集合，包含重试且不重复计数。 */
  function metricMarkup(events, s) {
    const data = metrics(events, s);
    const steps = stages
      .map(([id, name], i) => {
        const parts = data.rows.filter((r) => r.stage === id),
          sum = parts.reduce(
            (a, r) => ({
              ms: a.ms + r.ms,
              timed: a.timed || r.timed,
              calls: a.calls + r.calls,
              known: a.known + r.known,
              input: a.input + r.input,
              output: a.output + r.output,
            }),
            { ms: 0, timed: false, calls: 0, known: 0, input: 0, output: 0 },
          );
        const timing =
          i === 4
            ? data.reviewCount
              ? '医生等待 ' + duration(data.waitMs)
              : '尚未转交医生'
            : sum.timed
              ? '累计 ' + duration(sum.ms)
              : '耗时：待记录';
        return (
          '<div class="step ' +
          (s.completed_stages.includes(id)
            ? 'finished'
            : parts.length || (i === 4 && s.paused_at === 's5_doctor_review')
              ? 'active'
              : '') +
          '"><span class="step-dot">' +
          (i + 1) +
          '</span><span class="step-label">' +
          name +
          '</span><span class="stage-metric">' +
          esc(timing) +
          '</span><span class="stage-metric">' +
          esc(i === 4 ? '人工审核 · 无模型调用' : tokenText(sum)) +
          '</span></div>'
        );
      })
      .join('');
    const table = data.rows.length
      ? '<details class="metric-details"><summary>各执行轮次 · 含重试实际调用</summary><div class="metric-table-wrap"><table class="metric-table"><thead><tr><th>执行轮次 / 阶段</th><th>耗时</th><th>Token 用量</th></tr></thead><tbody>' +
        data.rows
          .map(
            (r) =>
              '<tr><td>第 ' +
              r.turn +
              ' 轮 · ' +
              esc(stages.find(([id]) => id === r.stage)?.[1]) +
              (r.failed ? ' · 失败' : '') +
              '</td><td>' +
              (r.timed ? duration(r.ms) : '未记录') +
              (r.incomplete ? '（未完成）' : '') +
              '</td><td>' +
              esc(tokenText(r)) +
              '</td></tr>',
          )
          .join('') +
        '</tbody></table></div></details>'
      : '';
    return { steps, table };
  }
  /** 工作台只渲染当前事件集合，历史详情有独立数据。 */
  function renderMetrics() {
    if (!state.snapshot) {
      $('stageRow').innerHTML = '';
      $('stageRounds').innerHTML = '';
      return;
    }
    const m = metricMarkup(state.events, state.snapshot);
    const expanded = $('stageRounds').querySelector('details')?.open;
    $('stageRow').innerHTML = m.steps;
    $('stageRounds').innerHTML = m.table;
    if (expanded && $('stageRounds').querySelector('details')) {
      $('stageRounds').querySelector('details').open = true;
    }
    $('traceEventCount').textContent = state.events.size;
    $('traceSessionCount').textContent = '1';
    $('traces').innerHTML = [...state.events.values()]
      .sort((a, b) => b.sequence - a.sequence)
      .map(
        (e) =>
          '<div class="trace"><span class="trace-time">' +
          esc(new Date(e.created_at).toLocaleTimeString()) +
          '</span><span class="trace-title">' +
          esc(e.payload.stage || '会诊') +
          ' · ' +
          esc(e.event) +
          '</span><span class="trace-status">第 ' +
          e.turn +
          ' 轮</span></div>',
      )
      .join('');
  }
  /** 分页回放与 SSE 按 event_id 去重；新问诊令旧回放失效。 */
  async function readEvents(id, events, valid) {
    let cursor = null;
    while (valid()) {
      const page = await api(
        '/api/v2/consultations/' +
          encodeURIComponent(id) +
          '/events/page' +
          (cursor ? '?after=' + encodeURIComponent(cursor) : ''),
      );
      if (!valid()) return;
      page.forEach((e) => events.set(e.event_id, e));
      if (page.length < 100) return;
      cursor = page[page.length - 1].event_id;
    }
  }
  /** 同会诊只保留一个进行中的回放任务。 */
  async function replay(id, epoch) {
    if (state.replay?.epoch === epoch) return state.replay.promise;
    const task = { epoch, promise: null };
    state.replay = task;
    task.promise = readEvents(
      id,
      state.events,
      () => epoch === state.epoch && state.snapshot?.consultation_id === id,
    )
      .then(() => {
        if (epoch === state.epoch) renderMetrics();
      })
      .finally(() => {
        if (state.replay === task) state.replay = null;
      });
    return task.promise;
  }
  /** 按状态开放写入；不得绕过待审核会诊开始新病例。 */
  function controls() {
    const s = state.snapshot,
      review = s?.paused_at === 's5_doctor_review' && s.status === 'awaiting_doctor_review',
      paused = Boolean(s?.paused_at),
      fallback = s?.result_presentation === 's3_retrieval_only',
      busy = state.busy;
    $('startBtn').disabled = busy || Boolean(s) || !state.config?.consultation_enabled;
    $('parseBtn').disabled = busy || Boolean(s) || !state.config?.agent_connected;
    $('clinicalText').readOnly = Boolean(s);
    $('fillSample').disabled = busy || Boolean(s);
    ['nextConsultationTop', 'nextConsultationReview'].forEach((id) => {
      $(id).disabled = busy || Boolean(s && !terminal.has(s.status));
      $(id).title =
        s && !terminal.has(s.status) ? '请先完成或取消本次问诊' : '清空工作台，开始新的问诊';
    });
    ['acceptBtn', 'modifyBtn'].forEach(
      (id) => ($(id).disabled = busy || !review || fallback || !s?.comparison?.ordered_paths.length),
    );
    ['rejectBtn', 'noteBtn'].forEach((id) => ($(id).disabled = busy || !review || fallback));
    $('retryS4Btn').disabled = busy || !fallback || !s?.retryable;
    $('addEvidence').disabled = busy || !paused;
    $('deferBtn').disabled = busy || s?.paused_at !== 's2_clarification';
    $('cancelBtn').disabled = busy || !paused;
    $('extraEvidence').readOnly = busy || !paused;
    $('reviewNote').readOnly = busy || !review;
    $('saveFacts').disabled = busy || !paused || !state.facts;
    $('factRows')
      .querySelectorAll('select')
      .forEach((el) => (el.disabled = busy || !paused));
  }
  /** 权威快照保留最终病例与备注；旧轮响应不能覆盖新轮。 */
  function render(s) {
    const old = state.snapshot;
    if (
      old?.consultation_id === s.consultation_id &&
      (s.turn < old.turn ||
        s.revision < old.revision ||
        (s.turn === old.turn && Date.parse(s.updated_at) < Date.parse(old.updated_at)))
    )
      return;
    state.snapshot = s;
    state.source = sources(s);
    const url = new URL(location.href);
    url.searchParams.set('consultation', s.consultation_id);
    history.replaceState(null, '', url);
    $('clinicalText').value = s.raw_input;
    renderFacts(s.clinical_evidence);
    $('resultState').dataset.status = s.status;
    $('resultState').textContent = label(s.status);
    $('sessionInfo').textContent =
      s.consultation_id + ' · revision ' + s.revision + ' · turn ' + s.turn;
    $('pipelineState').textContent = label(s.status);
    $('eightSummary').textContent = localizationText(s.localization);
    const questions =
      s.paused_at === 's2_clarification'
        ? s.localization?.clarification_questions
        : s.comparison?.next_questions;
    $('suggestedQuestion').textContent =
      (questions || []).map((q) => q.question).join('\n') ||
      s.comparison?.unresolved_reason ||
      '当前无补问';
    const paths = s.comparison?.ordered_paths || [],
      candidates = s.retrieval?.formula_candidates || [];
    const fallback = s.result_presentation === 's3_retrieval_only';
    $('resultNotice').textContent = fallback
      ? 'S4 大模型方证鉴别未完成，以下为 S3 检索候选，并非医学辨证优先级。'
      : '检索候选 ' +
        candidates.length +
        ' 个：' +
        candidates.map((c) => c.formula_name).join(' / ') +
        '。比较顺序来自 LLM；来源核验不代表临床正确性。';
    $('results').innerHTML =
      (fallback
        ? candidates
            .map(
              (c, i) =>
                '<article class="result-card"><div class="result-top"><span class="result-rank">检索候选 ' +
                (i + 1) +
                '</span><span class="chip unknown">S3 只读</span></div><h3 class="result-name">' +
                esc(c.formula_name) +
                '</h3><p class="result-channels">' +
                esc(c.six_channel_tags.join(' / ')) +
                '</p><p class="result-rationale">' +
                esc(c.coverage_notes.join('；') || '来源检索候选') +
                '</p><div class="result-footer"><span>真实来源 ' +
                c.source_evidence.length +
                ' 条</span><strong>只读回查</strong></div></article>',
            )
            .join('')
        : paths.map((p, i) => card(p, i)).join('')) ||
      '<div class="empty-state"><div class="empty-ring">六</div><strong>' +
        esc(label(s.status)) +
        '</strong><p>尚无可靠比较结果</p></div>';
    $('results')
      .querySelectorAll('[data-path]')
      .forEach((el) => (el.onclick = () => selectPath(paths[Number(el.dataset.path)])));
    const adopted = paths.find((p) => p.path_id === s.doctor_review?.selected_path_id);
    if (paths.length)
      selectPath(paths.find((p) => p.path_id === state.selected?.path_id) || adopted || paths[0]);
    else clearPath();
    $('reviewState').textContent = fallback
      ? 'S4 尚未完成，仅展示 S3 真实检索候选。可点击“仅重试鉴别”，成功后再进行医生审核。'
      : s.doctor_review
      ? '本次问诊已完成 · 医生' +
        label(s.doctor_review.action) +
        ' · ' +
        (s.doctor_review.modified_formula_pattern || adopted?.formula_name || '') +
        '。可继续查看，点击“下一次问诊”重置。'
      : '等待医生确认';
    if (s.doctor_review) $('reviewNote').value = s.doctor_review.note || '';
    controls();
    renderMetrics();
    if (s.status === 'running') watch(s.consultation_id);
    else stopWatch();
  }
  /** 世代和请求序号共同阻断旧页面、迟到响应。 */
  async function load(id = state.snapshot?.consultation_id) {
    if (!id) return;
    const epoch = state.epoch,
      ticket = ++state.request,
      s = await api('/api/v2/consultations/' + encodeURIComponent(id));
    if (epoch !== state.epoch || ticket < state.applied) return;
    state.applied = ticket;
    render(s);
    await replay(id, epoch);
  }
  /** SSE 与轮询仅更新当前会诊；每条事件先去重再统计。 */
  function watch(id) {
    if (state.stream) return;
    const epoch = state.epoch;
    state.stream = new EventSource('/api/v2/consultations/' + encodeURIComponent(id) + '/events');
    for (const name of [
      'snapshot_updated',
      'stage_started',
      'stage_completed',
      'stage_failed',
      'model_started',
      'model_completed',
      'model_failed',
      's4_stream_started',
      's4_stream_progress',
      's4_stream_failed',
      's4_retrieval_fallback_ready',
      's4_retry_started',
      'retrieval_started',
      'retrieval_completed',
      'clarification_requested',
      'review_requested',
      'consultation_completed',
      'failed',
    ])
      state.stream.addEventListener(name, (event) => {
        if (epoch !== state.epoch || state.snapshot?.consultation_id !== id) return;
        try {
          const e = JSON.parse(event.data);
          state.events.set(e.event_id, e);
          renderMetrics();
          if (
            [
              'snapshot_updated',
              'clarification_requested',
              'review_requested',
              'consultation_completed',
              'failed',
            ].includes(name)
          )
            load(id).catch((error) => {
              if (epoch === state.epoch) toast(error.message);
            });
        } catch (error) {
          toast('执行事件读取失败：' + error.message);
        }
      });
    state.timer = setInterval(
      () =>
        load(id).catch((error) => {
          if (epoch === state.epoch) toast(error.message);
        }),
      2500,
    );
  }
  /** 关闭订阅不等于取消已发送请求，epoch 负责后者。 */
  function stopWatch() {
    state.stream?.close();
    state.stream = null;
    clearInterval(state.timer);
    state.timer = null;
  }
  /** 只有主动下一次问诊才清空输入、结果、事件、地址与抽屉。 */
  function nextConsultation() {
    if (state.busy || (state.snapshot && !terminal.has(state.snapshot.status))) return;
    stopWatch();
    state.epoch++;
    state.request = 0;
    state.applied = 0;
    state.replay = null;
    state.snapshot = null;
    state.source = new Map();
    state.events = new Map();
    const url = new URL(location.href);
    url.searchParams.delete('consultation');
    url.searchParams.delete('history');
    history.replaceState(null, '', url);
    ['clinicalText', 'extraEvidence', 'reviewNote'].forEach((id) => ($(id).value = ''));
    $('factsEditor').open = false;
    renderFacts(null);
    clearPath();
    $('resultState').textContent = '尚未开始';
    delete $('resultState').dataset.status;
    $('results').innerHTML =
      '<div class="empty-state"><div class="empty-ring">六</div><strong>开始新的问诊</strong><p>录入四诊信息后启动辨证分析</p></div>';
    $('resultNotice').textContent = '候选顺序来自 LLM；来源核验仅证明引用可回查。';
    $('sessionInfo').textContent = '尚未创建';
    $('pipelineState').textContent = '等待启动';
    $('eightSummary').textContent = '等待六经定位';
    $('suggestedQuestion').textContent = '运行后展示补问建议';
    $('reviewState').textContent = '等待医生确认';
    $('traceEventCount').textContent = '0';
    $('traceSessionCount').textContent = '0';
    $('traces').innerHTML = '<p class="small-empty">运行会诊后显示公开执行事件</p>';
    $('drawer').classList.remove('visible');
    $('overlay').classList.remove('visible');
    closeHistory();
    renderMetrics();
    controls();
    showView('workbench');
    $('clinicalText').focus();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
  /** 人工动作携带版本与幂等键，未提交的输入保留在本轮。 */
  async function command(kind, extra = {}) {
    const s = state.snapshot;
    if (!s) return;
    const epoch = state.epoch,
      ticket = ++state.request;
    const body = {
      kind,
      consultation_id: s.consultation_id,
      revision: s.revision,
      action_turn: s.turn,
      action_id: crypto.randomUUID(),
      ...extra,
    };
    const suffix = kind === 'review' ? 'review' : kind === 'clarify' ? 'clarify' : 'resume',
      result = await api('/api/v2/consultations/' + s.consultation_id + '/' + suffix, body);
    if (epoch !== state.epoch) return;
    state.applied = ticket;
    render(result);
    await replay(s.consultation_id, epoch);
  }
  /** 审核后继续保存病例展示，不自动开始新会诊。 */
  async function review(action) {
    await command('review', {
      review: {
        action,
        selected_path_id: state.selected?.path_id || null,
        note: $('reviewNote').value || null,
      },
    });
  }
  /** 精确回查来源，不替换缺失引用。 */
  async function openSource(id) {
    const e = await api('/api/v1/knowledge/evidence/' + encodeURIComponent(id));
    $('drawerTitle').textContent = e.evidence_id;
    $('drawerBody').innerHTML = '<div class="original">' + esc(e.content) + '</div>';
    $('drawer').classList.add('visible');
    $('overlay').classList.add('visible');
  }
  /** 来源按钮限定本容器，历史不会改变当前路径选择。 */
  function bindSources(container) {
    container
      .querySelectorAll('[data-source]')
      .forEach(
        (el) => (el.onclick = () => openSource(el.dataset.source).catch((e) => toast(e.message))),
      );
  }
  /** 独立知识查询不进入会诊状态。 */
  async function searchKnowledge() {
    const query = $('knowledgeQuery').value.trim();
    if (!query) return;
    const result = await api(
      '/api/v1/knowledge/search/classic',
      /^\d+$/.test(query) ? { clause_nos: [query], top_k: 12 } : { query, top_k: 12 },
    );
    $('knowledgeStatus').textContent = result.length + ' 条来源';
    $('knowledgeResults').innerHTML = result
      .map(
        (e) =>
          '<button class="k-card" data-source="' +
          esc(e.evidence_id) +
          '"><strong>' +
          esc(e.metadata.clause_no || e.evidence_id) +
          '</strong><p>' +
          esc(e.content) +
          '</p></button>',
      )
      .join('');
    bindSources($('knowledgeResults'));
  }
  /** 历史列表仅进入专用详情，保留当前工作台状态。 */
  async function archive() {
    const page = await api('/api/v2/consultations');
    $('historyStatus').textContent = page.total + ' 条 · 最近 ' + page.items.length + ' 条';
    $('historyList').innerHTML =
      page.items
        .map(
          (s) =>
            '<button class="k-card" data-history="' +
            esc(s.consultation_id) +
            '"><span class="clause-tag">' +
            esc(label(s.status)) +
            '</span><strong>' +
            esc(s.topk_syndromes.join(' / ') || '未形成候选') +
            '</strong><p>' +
            esc(new Date(s.created_at).toLocaleString()) +
            '</p><p>' +
            esc(s.consultation_id) +
            '</p></button>',
        )
        .join('') || '<p class="small-empty">暂无历史会诊</p>';
    $('historyList')
      .querySelectorAll('[data-history]')
      .forEach(
        (el) =>
          (el.onclick = () => historyDetail(el.dataset.history).catch((e) => toast(e.message))),
      );
  }
  /** 历史路径呈现支持、反证、缺口及真实引用，不提供写入按钮。 */
  function historyPath(p, i, s, source) {
    const facts = new Map((s.clinical_evidence?.findings || []).map((f) => [f.fact_id, f]));
    return (
      '<div>' +
      card(p, i, false) +
      '<div class="history-path-evidence"><strong>支持事实</strong><div class="chips">' +
      p.support_fact_ids.map((id) => (facts.has(id) ? chip(facts.get(id)) : esc(id))).join('') +
      '</div><strong>反证事实</strong><div class="chips">' +
      p.counter_fact_ids.map((id) => (facts.has(id) ? chip(facts.get(id)) : esc(id))).join('') +
      '</div><strong>缺失信息</strong><p>' +
      esc(p.missing_dimensions.join(' / ') || '暂无') +
      '</p>' +
      sourceCards([...p.support_evidence_ids, ...p.counter_evidence_ids], source) +
      '</div></div>'
    );
  }
  /** 详情用独立快照和事件容器；URL 可刷新恢复，不调用工作台 render。 */
  async function historyDetail(id) {
    const ticket = ++state.historyTicket,
      s = await api('/api/v2/consultations/' + encodeURIComponent(id)),
      events = new Map();
    await readEvents(id, events, () => ticket === state.historyTicket);
    if (ticket !== state.historyTicket) return;
    state.historySnapshot = s;
    const paths = s.comparison?.ordered_paths || [],
      review = s.doctor_review,
      selected = paths.find((p) => p.path_id === review?.selected_path_id),
      m = metricMarkup(events, s),
      source = sources(s);
    $('historyDetailTitle').textContent = '会诊详情 · ' + label(s.status);
    $('historyDetailBody').innerHTML =
      '<div class="history-meta">' +
      esc(s.consultation_id) +
      ' · revision ' +
      s.revision +
      ' · ' +
      esc(new Date(s.created_at).toLocaleString()) +
      '</div><div class="history-block"><strong>病例原文与补充信息</strong><p>' +
      esc(s.raw_input) +
      '</p></div><div class="history-block"><strong>结构化事实</strong><div class="chips">' +
      (s.clinical_evidence?.findings || []).map(chip).join('') +
      '</div></div><div class="history-block"><strong>六经定位</strong><p>' +
      esc(localizationText(s.localization)) +
      '</p></div><div class="history-block review-summary"><strong>医生最终审核</strong><p>' +
      (review
        ? esc(label(review.action)) +
          ' · ' +
          esc(review.modified_formula_pattern || selected?.formula_name || '未采用候选') +
          '\n' +
          esc(review.note || '无审核备注')
        : '尚未完成医生审核') +
      '</p></div><h3>候选与证据</h3><div class="history-paths">' +
      (paths.map((p, i) => historyPath(p, i, s, source)).join('') || '<p>尚无候选</p>') +
      '</div><h3>Agent 执行统计</h3><div class="step-row">' +
      m.steps +
      '</div>' +
      m.table;
    bindSources($('historyDetailBody'));
    $('historyDetail').hidden = false;
    $('historyList').hidden = true;
    const url = new URL(location.href);
    url.searchParams.set('history', id);
    history.replaceState(null, '', url);
    showView('history');
    $('historyDetail').scrollIntoView({ block: 'start' });
  }
  /** 关闭详情只调整历史页面，不停止当前会诊。 */
  function closeHistory() {
    state.historyTicket++;
    state.historySnapshot = null;
    $('historyDetail').hidden = true;
    $('historyList').hidden = false;
    const url = new URL(location.href);
    url.searchParams.delete('history');
    history.replaceState(null, '', url);
  }
  /** 导航与详情隔离，离开历史时令未完成的详情请求失效。 */
  function showView(name) {
    state.view = name;
    document
      .querySelectorAll('[data-view]')
      .forEach((el) => el.classList.toggle('active', el.dataset.view === name));
    document
      .querySelectorAll('[data-nav]')
      .forEach((el) => el.classList.toggle('active', el.dataset.nav === name));
    $('crumbPage').textContent = {
      workbench: '临床辨证工作台',
      history: '会诊历史',
      knowledge: '经典知识速查',
      trace: 'Agent 执行轨迹',
    }[name];
  }
  /** 统一错误边界和写入锁，防止重复提交及提交时重置。 */
  function bind(id, action, clinical = false) {
    $(id).onclick = async () => {
      if (clinical && state.busy) return;
      if (clinical) {
        state.busy = true;
        controls();
      }
      try {
        await action();
      } catch (e) {
        toast(e.message);
      } finally {
        if (clinical) {
          state.busy = false;
          controls();
        }
      }
    };
  }
  bind(
    'parseBtn',
    async () => {
      renderFacts(await api('/api/v2/clinical/parse', { text: $('clinicalText').value }));
      $('factsEditor').open = true;
    },
    true,
  );
  bind(
    'startBtn',
    async () => {
      if (state.snapshot) return;
      const text = $('clinicalText').value.trim();
      if (!text) throw new Error('请先录入四诊信息');
      const s = await api(
        '/api/v2/consultations',
        { clinical_text: text },
        { 'Idempotency-Key': crypto.randomUUID() },
      );
      render(s);
      await replay(s.consultation_id, state.epoch);
    },
    true,
  );
  bind('nextConsultationTop', nextConsultation);
  bind('nextConsultationReview', nextConsultation);
  bind(
    'addEvidence',
    async () => {
      const text = $('extraEvidence').value.trim();
      if (!text) throw new Error('请填写补充事实');
      await command(state.snapshot.paused_at === 's2_clarification' ? 'clarify' : 'review', {
        clinical_text: state.snapshot.raw_input + '\n医生补充：' + text,
        ...(state.snapshot.paused_at === 's5_doctor_review'
          ? { review: { action: 'add_evidence' } }
          : {}),
      });
      $('extraEvidence').value = '';
    },
    true,
  );
  bind('deferBtn', () => command('clarify', { no_additional_information: true }), true);
  bind(
    'saveFacts',
    async () => {
      const facts = structuredClone(state.facts);
      $('factRows')
        .querySelectorAll('[data-polarity]')
        .forEach((el) => {
          facts.findings[Number(el.dataset.polarity)].polarity = el.value;
        });
      $('factRows')
        .querySelectorAll('[data-time]')
        .forEach((el) => {
          facts.findings[Number(el.dataset.time)].time_scope = el.value;
        });
      facts.findings.forEach((f) => (f.review_status = 'doctor_verified'));
      facts.doctor_revision = state.snapshot.revision + 1;
      facts.source = 'doctor_edit';
      await command(state.snapshot.paused_at === 's2_clarification' ? 'clarify' : 'review', {
        clinical_evidence: facts,
        ...(state.snapshot.paused_at === 's5_doctor_review'
          ? { review: { action: 'add_evidence' } }
          : {}),
      });
    },
    true,
  );
  bind('acceptBtn', () => review('accept'), true);
  bind('rejectBtn', () => review('reject'), true);
  bind('noteBtn', () => review('note'), true);
  bind(
    'modifyBtn',
    async () => {
      const formula = prompt('医生修订后的方证名称');
      if (!formula?.trim()) return;
      await command('review', {
        review: {
          action: 'modify',
          selected_path_id: state.selected?.path_id,
          modified_formula_pattern: formula.trim(),
          note: $('reviewNote').value || null,
        },
      });
    },
    true,
  );
  bind('cancelBtn', () => command('cancel'), true);
  bind(
    'retryS4Btn',
    async () => {
      const s = state.snapshot;
      if (!s?.retryable || s.result_presentation !== 's3_retrieval_only') return;
      const result = await api('/api/v2/consultations/' + encodeURIComponent(s.consultation_id) + '/retry-s4', {
        expected_turn: s.turn,
        expected_revision: s.revision,
        idempotency_key: crypto.randomUUID(),
      });
      render(result);
      await replay(s.consultation_id, state.epoch);
    },
    true,
  );
  bind('knowledgeSearch', searchKnowledge);
  bind('refreshHistory', archive);
  bind('closeHistoryDetail', closeHistory);
  bind('refreshButton', () =>
    state.view === 'history'
      ? state.historySnapshot
        ? historyDetail(state.historySnapshot.consultation_id)
        : archive()
      : load(),
  );
  bind('fillSample', () => {
    $('clinicalText').value = '恶寒发热，汗出，头痛，脉浮缓。';
  });
  bind('openDiff', () => {
    $('drawerTitle').textContent = '当前候选鉴别摘要';
    $('drawerBody').innerHTML =
      '<p>' +
      esc(state.selected?.differential_summary || '暂无鉴别结果') +
      '</p>' +
      sourceCards(state.selected?.counter_evidence_ids || [], state.source);
    bindSources($('drawerBody'));
    $('drawer').classList.add('visible');
    $('overlay').classList.add('visible');
  });
  bind('closeDrawer', () => {
    $('drawer').classList.remove('visible');
    $('overlay').classList.remove('visible');
  });
  document.querySelectorAll('[data-nav]').forEach(
    (el) =>
      (el.onclick = () => {
        if (el.dataset.nav !== 'history') closeHistory();
        showView(el.dataset.nav);
        if (el.dataset.nav === 'history') archive().catch((e) => toast(e.message));
        if (el.dataset.nav === 'knowledge') searchKnowledge().catch((e) => toast(e.message));
      }),
  );
  window.addEventListener('beforeunload', stopWatch);
  setInterval(() => {
    if (state.view === 'workbench' && state.snapshot && !terminal.has(state.snapshot.status))
      renderMetrics();
  }, 1000);
  api('/api/console/config')
    .then((c) => {
      state.config = c;
      $('runtimeInfo').textContent = c.text;
      $('modeBadge').textContent = c.clinical_engine;
      controls();
    })
    .catch((e) => toast(e.message));
  const url = new URL(location.href),
    id = url.searchParams.get('consultation'),
    historyId = url.searchParams.get('history');
  if (id) load(id).catch((e) => toast(e.message));
  if (historyId) {
    showView('history');
    archive().catch((e) => toast(e.message));
    historyDetail(historyId).catch((e) => toast(e.message));
  }
  controls();
})();
