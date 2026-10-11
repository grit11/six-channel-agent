/**
 * 六经辨证 Console 的原生交互层。
 * 依赖同源会诊、知识、解析和 SSE API；只映射公开 DTO，不推断病证或重算评分。
 * 患者事实不存浏览器本地存储；URL 仅保留会诊 ID，用于刷新恢复权威快照。
 */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const SAMPLE = '患者恶寒发热，无汗，头项疼痛，周身酸痛，脉浮紧。';
  const STAGES = [['build_evidence','脉证整理'],['eight_principles','八纲分析'],['channel_candidates','六经候选'],['syndrome_relationship','病证关系'],['formula_candidates','方证召回'],['differential','鉴别反证'],['evidence_gap','补证评估'],['topk','Top-K 路径'],['similar_cases','医案参考']];
  const STATUS = {created:'已创建',running:'运行中',needs_evidence:'等待补证',awaiting_doctor_review:'等待审核',completed:'已审核完成',failed:'运行失败',cancelled:'已取消',safety_escalated:'待进一步核验'};
  const ACTIONS = {accept:'采用',modify:'修改后采用',reject:'驳回',note:'备注',reorder:'调整查看顺序',add_evidence:'补证'};
  const TYPES = {classic_clause:'经典条文',formula_pattern:'方证',differential:'鉴别规则',medical_case:'历史医案',six_channel_framework:'核心框架'};
  const state = {snapshot:null,selected:null,config:null,busy:false,stream:null,poll:null,cursor:0,traces:[],catalog:new Map(),stageDone:new Set(),stageActive:null,parsed:null};

  /** 将不可信来源文字作为文本转义，避免 DOM 注入。 */
  function esc(value) { return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
  /** 只翻译既有字段词典，未知字段保留原名。 */
  function text(value) { return String(value ?? '').replace(/\b[a-z]+(?:_[a-z]+)*\b/g, key => window.CONSOLE_LABELS?.[key] || key); }
  /** 显示操作提示，不改变会诊数据。 */
  function toast(message) { $('toast').textContent=message; $('toast').classList.add('visible'); clearTimeout(toast.timer); toast.timer=setTimeout(()=>$('toast').classList.remove('visible'),4500); }
  /** HTTP 错误同时支持主工程业务错误和 FastAPI 校验错误。 */
  async function api(url,options={}) {
    const response=await fetch(url,{...options,headers:{'Content-Type':'application/json',...(options.headers||{})}});
    let data; try { data=await response.json(); } catch { throw new Error(`服务响应不是 JSON（HTTP ${response.status}）`); }
    if(!response.ok) throw new Error(data.error?.message || (typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||`HTTP ${response.status}`)));
    return data;
  }
  /** 写操作带幂等键与服务端版本；重复点击由 busy 和后端共同保护。 */
  function post(url,body) { return api(url,{method:'POST',headers:{'Idempotency-Key':crypto.randomUUID()},body:JSON.stringify(body)}); }
  /** 页面切换只改变展示，当前会诊继续接收事件。 */
  function navigate(name) {
    document.querySelectorAll('[data-nav]').forEach(el=>el.classList.toggle('active',el.dataset.nav===name));
    document.querySelectorAll('[data-view]').forEach(el=>el.classList.toggle('active',el.dataset.view===name));
    $('crumbPage').textContent={workbench:'临床辨证工作台',knowledge:'知识中心',trace:'Agent 执行轨迹',history:'会诊历史'}[name];
    if(name==='trace') renderTraces();
    if(name==='history') history();
    if(name==='knowledge' && !$('knowledgeResults').children.length) searchKnowledge();
  }
  /** 运行能力来自应用容器，规则模式与模型模式分别标明。 */
  async function config() {
    try {
      state.config=await api('/api/console/config'); const cfg=state.config;
      $('modeBadge').textContent=cfg.agent_connected?`◉ ${cfg.reasoning_mode==='llm'?'模型':'规则'}推理 · 服务在线`:'会诊存储尚未启用';
      $('sideStatus').textContent=`${cfg.vector_backend.toUpperCase()} · ${cfg.checkpoint_backend.toUpperCase()}`;
      $('traceMode').textContent='真实 SSE · 公开阶段';
      $('runtimeInfo').textContent=`解析 ${cfg.clinical_parser_mode} / 推理 ${cfg.reasoning_mode} / 检索 ${cfg.retrieval_mode} / 会诊 ${cfg.business_database==='postgres'?'PostgreSQL':'内存（重启丢失）'}`;
      $('disclaimer').textContent='研究与工程验证版本，最终由医生审核；排序分不是诊断概率。规则、来源适用性及模型质量仍需专家校准。';
      syncControls();
    } catch(error) { $('modeBadge').textContent='服务连接异常'; toast(error.message); }
  }
  /** 按真实九阶段事件显示进度，无事件时不补造完成。 */
  function renderStages() {
    $('stageRow').innerHTML=STAGES.map(([key,label],index)=>`<div class="step ${state.stageDone.has(key)?'finished':state.stageActive===key?'active':''}"><div class="step-dot">${state.stageDone.has(key)?'✓':index+1}</div><div class="step-label">${label}</div></div>`).join('');
  }
  /** 基于会诊状态启用合法动作，终态不继续修改。 */
  function syncControls() {
    const s=state.snapshot, editable=s && ['needs_evidence','awaiting_doctor_review'].includes(s.status);
    const selected=paths().find(p=>p.path_id===state.selected);
    const excluded=s?.excluded_path_ids?.includes(state.selected);
    $('startBtn').disabled=state.busy || !state.config?.agent_connected || s?.status==='running';
    $('parseBtn').disabled=state.busy || (s && !['completed','cancelled','failed'].includes(s.status));
    $('addEvidence').disabled=state.busy || !editable;
    ['acceptBtn','modifyBtn','rejectBtn'].forEach(id=>$(id).disabled=state.busy || !editable || !selected || excluded);
    $('noteBtn').disabled=state.busy || !editable;
    $('reviewReady').disabled=state.busy || s?.status!=='needs_evidence';
    $('cancelBtn').disabled=state.busy || !s || ['completed','cancelled'].includes(s.status);
    $('rerunBtn').disabled=state.busy || !s || !['needs_evidence','awaiting_doctor_review','failed'].includes(s.status);
    $('saveFacts').disabled=state.busy || !editable;
    $('replayBtn').disabled=!s || state.busy;
  }
  /** 获取服务端路径，不在界面生成候选。 */
  function paths() { return state.snapshot?.reasoning?.reasoning_paths || []; }
  /** 阳性、阴性、未知全部保留；未知默认折叠，避免挤占临床事实。 */
  function renderFacts(evidence) {
    const findings=evidence?.findings || [],known=findings.filter(f=>f.polarity!=='unknown'),unknown=findings.filter(f=>f.polarity==='unknown');
    const chip=f=>`<span class="chip ${esc(f.polarity)}" title="${esc(f.source_text || '')}">${f.polarity==='positive'?'＋':f.polarity==='negative'?'－':'?'} ${esc(text(f.key))}${f.time_scope?` · ${esc(f.time_scope)}`:''}</span>`;
    $('clinicalChips').innerHTML=known.map(chip).join('') || '<span class="chip unknown">尚无已知事实</span>';
    $('unknownChips').innerHTML=unknown.map(chip).join(''); $('unknownCount').textContent=`未知 ${unknown.length} 项`;
    $('findingsCount').textContent=`已知 ${known.length} / 未知 ${unknown.length}`;
    // 相同事实的周期快照不得覆盖医生尚未提交的选择；事实变更后才重建编辑器。
    const fingerprint=JSON.stringify(evidence);
    if(state.factFingerprint!==fingerprint) {
      $('factRows').innerHTML=findings.map((f,index)=>`<label class="fact-row"><span>${esc(text(f.key))}${f.time_scope?` · ${esc(f.time_scope)}`:''}</span><select data-fact-index="${index}" aria-label="${esc(text(f.key))}状态"><option value="positive" ${f.polarity==='positive'?'selected':''}>阳性</option><option value="negative" ${f.polarity==='negative'?'selected':''}>阴性</option><option value="unknown" ${f.polarity==='unknown'?'selected':''}>未知</option></select></label>`).join('');
      state.factFingerprint=fingerprint;
    }
    $('conflictNotice').textContent=evidence?.contradictions?.length?`存在 ${evidence.contradictions.length} 组事实冲突，请核对时点和原文。`:'';
  }
  /** 接受权威快照，URL 只保存 ID；原始病例不写入浏览器存储。 */
  function renderSnapshot(snapshot) {
    const differentConsultation=state.snapshot?.consultation_id!==snapshot.consultation_id;
    const factsChanged=JSON.stringify(state.snapshot?.clinical_evidence)!==JSON.stringify(snapshot.clinical_evidence);
    state.snapshot=snapshot;
    if(differentConsultation || factsChanged || !state.parsedText) state.parsed=snapshot.clinical_evidence;
    const url=new URL(location.href); url.searchParams.set('consultation',snapshot.consultation_id); window.history.replaceState(null,'',url);
    state.catalog=new Map((snapshot.reasoning?.evidence_catalog||[]).map(item=>[item.evidence_id,item]));
    if(differentConsultation || factsChanged || !state.parsedText) renderFacts(snapshot.clinical_evidence);
    renderCandidates();
    const gaps=snapshot.reasoning?.evidence_gaps || [];
    $('suggestedQuestion').textContent=gaps.length?gaps.map(g=>text(g.question)).join('\n'):'当前无关键补证问题';
    $('eightSummary').textContent=snapshot.reasoning?.eight_principles?.summary || '八纲倾向将在运行完成后展示';
    $('sessionInfo').textContent=`${STATUS[snapshot.status] || snapshot.status} · 第 ${snapshot.turn} 版 · ${snapshot.thread_id}`;
    $('reviewState').textContent=snapshot.doctor_review?`最近动作：${ACTIONS[snapshot.doctor_review.action]} · ${snapshot.doctor_review.note || '无备注'}`:'医生动作独立记录，不覆盖 Agent 原始建议。';
    $('traceSessionCount').textContent='1';
    if(snapshot.reasoning?.completed_stages) state.stageDone=new Set(snapshot.reasoning.completed_stages);
    $('pipelineState').textContent=STATUS[snapshot.status] || snapshot.status;
    if(snapshot.status!=='running') state.stageActive=null;
    renderStages();
    if(snapshot.status==='failed') $('stageDetail').textContent='本轮失败，可查看执行轨迹并重试。';
    syncControls();
    if(['completed','cancelled','failed'].includes(snapshot.status)) stopWatch();
  }
  /** 候选卡片展示后端排序分和医生驳回状态，比例换算只用于呈现。 */
  function renderCandidates() {
    const items=paths();
    if(!items.some(p=>p.path_id===state.selected)) state.selected=items[0]?.path_id || null;
    $('resultState').textContent=items.length?`Top ${items.length}`:state.snapshot?.status==='running'?'分析中':'尚无路径';
    $('results').innerHTML=items.length?items.map(p=>{
      const value=(p.ranking_score*100).toFixed(1),excluded=state.snapshot.excluded_path_ids.includes(p.path_id);
      return `<article class="result-card ${p.path_id===state.selected?'selected':''} ${excluded?'excluded':''}" data-path="${esc(p.path_id)}" tabindex="0" role="button" aria-pressed="${p.path_id===state.selected}"><div class="result-top"><span class="result-rank">RANK ${String(p.rank).padStart(2,'0')}${excluded?' · 已驳回':''}</span><strong class="result-score">${value}<span>/100 排序分</span></strong></div><div class="result-name">${esc(p.syndrome_name)}</div><p class="result-desc">${esc(p.six_channels.join(' / '))} · ${esc(p.formula_pattern || '暂无方证')} · ${esc(p.relationship || '关系未定')}</p><div class="progress-line"><div class="progress-fill" style="width:${value}%"></div></div><div class="result-footer"><span>支持 ${p.supporting_findings.length} · 反证 ${p.contradicting_findings.length} · 缺口 ${p.missing_key_findings.length}</span><strong>查看依据 →</strong></div></article>`;
    }).join(''):'<div class="empty-state"><div class="empty-ring">六</div><strong>等待真实分析结果</strong><p>录入脉证后启动会诊，候选与证据全部由后端返回。</p></div>';
    document.querySelectorAll('[data-path]').forEach(el=>{
      const select=()=>{state.selected=el.dataset.path;renderCandidates();syncControls();};
      el.onclick=select; el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}};
    });
    renderEvidence(items.find(p=>p.path_id===state.selected));
  }
  /** 精确引用从本轮目录展开，来源 ID 不做删除版本的模糊查询。 */
  function sourceCards(items,target) {
    $(target).innerHTML=items.length?items.map(e=>`<button class="source-item" data-source="${esc(e.evidence_id)}"><div class="source-top">${esc(TYPES[e.knowledge_type]||e.knowledge_type)} · ${esc(e.metadata?.clause_no || e.metadata?.formula_name || e.knowledge_id)} ↗</div><div class="source-text">${esc(e.content)}</div></button>`).join(''):'<p class="small-empty">本轮没有对应来源；未补造引用。</p>';
    $(target).querySelectorAll('[data-source]').forEach(el=>el.onclick=()=>openSource(state.catalog.get(el.dataset.source)));
  }
  /** 支持、反证、缺口和来源保留原始字段，且不把 Unknown 当阴性。 */
  function renderEvidence(path) {
    for(const [list,count,values,type] of [['supportList','supportCount',path?.supporting_findings,'positive'],['conflictList','conflictCount',path?.contradicting_findings,'negative'],['missingList','missingCount',path?.missing_key_findings,'unknown']]) {
      const items=values||[]; $(count).textContent=`${items.length} 项`; $(list).innerHTML=items.length?items.map(v=>`<span class="chip ${type}">${esc(text(v))}</span>`).join(''):'<span class="small-empty">暂无记录</span>';
    }
    $('selectedSyndrome').textContent=path?.syndrome_name || '尚未选择候选';
    const ids=path?[...path.classic_evidence,...path.formula_evidence,...path.framework_evidence]:[];
    sourceCards([...new Set(ids)].map(id=>state.catalog.get(id)).filter(Boolean),'sourceList');
    $('pathAudit').textContent=path?JSON.stringify({排序组成:path.score_components,来源条件覆盖率:path.evidence_completeness,警告:path.warnings,来源核验:path.evidence_assessments,与次选鉴别:path.comparison_with_next},null,2):'尚无路径审计';
    $('similarCases').textContent=state.snapshot?.reasoning?.similar_cases?.length?JSON.stringify(state.snapshot.reasoning.similar_cases,null,2):'当前无后置医案参考';
  }
  /** 按精确来源对象打开抽屉，外来文本均经转义。 */
  function openSource(item) {
    if(!item) return toast('该引用未出现在当前来源目录中');
    $('drawerEyebrow').textContent=TYPES[item.knowledge_type] || item.knowledge_type;
    $('drawerTitle').textContent=item.metadata?.title || item.metadata?.formula_name || `来源 ${item.knowledge_id}`;
    $('drawerBody').innerHTML=`<div class="original">${esc(item.content)}</div>${meta('引用 ID',item.evidence_id)}${meta('来源等级',item.source_level)}${meta('出处',item.source)}${meta('审核状态',item.metadata?.review_status || '未提供')}<details><summary>完整来源记录</summary><pre>${esc(JSON.stringify(item.metadata,null,2))}</pre></details>`;
    openDrawer();
  }
  /** 输出只读来源元信息。 */
  function meta(label,value) { return `<div class="metadata"><span>${esc(label)}</span><span>${esc(value ?? '未提供')}</span></div>`; }
  /** 打开抽屉并使关闭按钮获得键盘焦点。 */
  function openDrawer() { $('overlay').classList.add('visible'); $('drawer').classList.add('visible'); $('closeDrawer').focus(); }
  /** 关闭来源抽屉。 */
  function closeDrawer() { $('overlay').classList.remove('visible'); $('drawer').classList.remove('visible'); }
  /** 刷新精确会诊快照，过期会诊响应不更新当前界面。 */
  async function loadSnapshot(id=state.snapshot?.consultation_id) {
    if(!id) return; const snapshot=await api(`/api/v1/consultations/${encodeURIComponent(id)}`);
    if(state.snapshot && state.snapshot.consultation_id!==id) return;
    if(state.snapshot && state.snapshot.turn>snapshot.turn) return;
    renderSnapshot(snapshot);
  }
  /** 停止旧订阅与快照轮询，防止不同会诊结果混合。 */
  function stopWatch() { state.stream?.close(); state.stream=null; clearInterval(state.poll); state.poll=null; }
  /** 同源 EventSource 使用服务端游标自动重连；轮询是断流时的权威恢复。 */
  function watch(id) {
    stopWatch(); state.cursor=0;
    if(['completed','cancelled','failed'].includes(state.snapshot?.status)) return;
    state.stream=new EventSource(`/api/v1/consultations/${encodeURIComponent(id)}/stream`);
    for(const name of ['snapshot_updated','stage_started','stage_completed','stage_failed','retrieval_started','retrieval_completed','model_started','model_completed','model_failed','recovery_required','needs_evidence','topk_updated','review_required','completed','failed']) {
      state.stream.addEventListener(name,event=>{
        let item; try { item=JSON.parse(event.data); } catch { return trace('帧解析错误','事件格式异常',true); }
        if(item.consultation_id && item.consultation_id!==state.snapshot?.consultation_id) return;
        if(item.sequence && item.sequence<=state.cursor) return;
        state.cursor=item.sequence || state.cursor;
        const payload=item.payload||item;
        trace(name,`${payload.stage || ''} ${payload.status || ''} ${payload.count!==undefined?`来源 ${payload.count} 条`:''} ${payload.duration_ms!==undefined?`${payload.duration_ms} ms`:''}`,name.includes('failed'));
        if(item.turn >= (state.snapshot?.reasoning_run_turn || state.snapshot?.turn || 0)) {
          if(name==='stage_started') state.stageActive=payload.stage;
          if(name==='stage_completed') state.stageDone.add(payload.stage);
          if(name==='stage_failed') state.stageActive=null;
          renderStages(); $('stageDetail').textContent=`${name} · ${payload.stage || payload.status || ''}`;
        }
        if(['snapshot_updated','needs_evidence','topk_updated','review_required','completed','failed'].includes(name)) loadSnapshot(id).catch(e=>toast(e.message));
      });
    }
    state.stream.onerror=()=>{$('stageDetail').textContent='事件连接暂时中断，正在自动重连；快照恢复仍在进行。';};
    let fetching=false;
    state.poll=setInterval(async()=>{if(fetching)return;fetching=true;try{await loadSnapshot(id);}catch(error){$('stageDetail').textContent=error.message;}finally{fetching=false;}},2500);
  }
  /** 轨迹只保存公开事件名称与计数，不记录病例全文或模型正文。 */
  function trace(type,message,error=false) { state.traces.unshift({time:new Date().toLocaleTimeString('zh-CN'),type,message,error});state.traces=state.traces.slice(0,120);$('traceEventCount').textContent=state.traces.length;renderTraces(); }
  /** 展示公开执行日志。 */
  function renderTraces() { $('traces').innerHTML=state.traces.map(t=>`<div class="trace"><span class="trace-time">${esc(t.time)}</span><span><b>${esc(t.type)}</b><br>${esc(t.message)}</span><span class="trace-status">${t.error?'异常':'已接收'}</span></div>`).join('') || '<p class="small-empty">尚无真实事件</p>'; }
  /** 串行提交操作，并在失败时恢复按钮及读取可能已提交的快照。 */
  async function action(callback) { if(state.busy)return;state.busy=true;syncControls();try{await callback();}catch(error){toast(error.message);if(state.snapshot)await loadSnapshot().catch(()=>{});}finally{state.busy=false;syncControls();} }
  /** 创建自定义真实病例会诊，只有明确解析匹配时才提交医生编辑后的事实。 */
  async function start() {
    const input=$('clinicalText').value.trim(); if(!input)return toast('请填写四诊信息');
    await action(async()=>{
      const body={clinical_text:input,top_k:Number($('topK').value),doctor_id:'console_doctor',patient_ref:$('patientRef').value.trim()||null};
      if(state.parsed && state.parsedText===input) body.clinical_evidence=editedFacts();
      const snapshot=await post('/api/v1/consultations',body);
      stopWatch();state.selected=null;state.stageDone.clear();state.traces=[];state.factFingerprint=null;
      state.parsedText=null;renderSnapshot(snapshot);watch(snapshot.consultation_id);toast('会诊已创建，正在运行真实工作流');
    });
  }
  /** 单独解析让医生先核对事实，解析不等同于辨证结论。 */
  async function parse() {
    const input=$('clinicalText').value.trim();if(!input)return toast('请填写四诊信息');
    await action(async()=>{const reply=await post('/api/v1/clinical/parse',{text:input,source:'doctor_input'});state.parsed=reply.clinical_evidence;state.parsedText=input;renderFacts(state.parsed);$('factsEditor').open=true;toast('已解析，请核对三值事实后启动');});
  }
  /** 只修改医生明确选中的极性，来源与时间保持完整；服务端重新派生冲突。 */
  function editedFacts() {
    const evidence=structuredClone(state.parsed);const changed=[];
    document.querySelectorAll('[data-fact-index]').forEach(el=>{const f=evidence.findings[Number(el.dataset.factIndex)];if(f.polarity!==el.value){f.polarity=el.value;f.source='doctor_input';f.certainty=1;changed.push({key:f.key,time_scope:f.time_scope});}});
    state.changedFacts=changed;return evidence;
  }
  /** 已建会诊的事实修订使用精确键/时点组替换，避免旧极性残留。 */
  async function saveFacts() { await action(async()=>{const evidence=editedFacts();if(!state.changedFacts.length)return toast('没有修改事实');const s=state.snapshot;const next=await post(`/api/v1/consultations/${s.consultation_id}/evidence`,{clinical_evidence:evidence,replace_fact_groups:state.changedFacts,expected_turn:s.turn,resume:true});renderSnapshot(next);watch(next.consultation_id);toast('修订已提交，同线程重算');}); }
  /** 补充文本由后端解析合并并继续同一线程，不在浏览器推断。 */
  async function supplement() { const input=$('extraEvidence').value.trim();if(!input)return toast('请填写补充信息');await action(async()=>{const s=state.snapshot;const next=await post(`/api/v1/consultations/${s.consultation_id}/evidence`,{clinical_text:input,expected_turn:s.turn,resume:true});$('extraEvidence').value='';renderSnapshot(next);watch(next.consultation_id);toast('补证已提交');}); }
  /** 医生采用、驳回和备注均传递当前轮次，不覆盖 Agent 原始路径。 */
  async function review(kind) {
    await action(async()=>{
      const s=state.snapshot,body={action:kind,selected_path_id:state.selected,expected_turn:s.turn,note:$('reviewNote').value.trim()||null};
      if(kind==='modify') { const name=prompt('医生修订后的证型名称：',paths().find(p=>p.path_id===state.selected)?.syndrome_name||'');if(!name?.trim())return;body.modified_syndrome=name.trim(); }
      const next=await post(`/api/v1/consultations/${s.consultation_id}/review`,body);renderSnapshot(next);toast(`已记录：${ACTIONS[kind]}`);
    });
  }
  /** 当前结果可交医生审核，保留所有证据缺口，不宣称已经补齐。 */
  async function reviewReady() { await action(async()=>{const s=state.snapshot;renderSnapshot(await post(`/api/v1/consultations/${s.consultation_id}/review-ready`,{expected_turn:s.turn}));toast('已转交医生审核，缺口继续保留');}); }
  /** 从医生选择的已实现阶段重算；复用是否合法由服务端判定。 */
  async function rerun() { await action(async()=>{const s=state.snapshot;const next=await post(`/api/v1/consultations/${s.consultation_id}/rerun`,{expected_turn:s.turn,from_stage:$('rerunStage').value});renderSnapshot(next);watch(next.consultation_id);}); }
  /** 取消会诊交由运行服务停止任务与保存终态。 */
  async function cancel() { await action(async()=>{renderSnapshot(await post(`/api/v1/consultations/${state.snapshot.consultation_id}/cancel`,{}));toast('会诊已取消');}); }
  /** 数字查询走精确条文号，其余查询复用当前检索配置。 */
  async function searchKnowledge() {
    const query=$('knowledgeQuery').value.trim();if(!query)return;
    $('knowledgeStatus').textContent='检索中…';
    try {
      const body=/^\d+$/.test(query)?{clause_nos:[query],top_k:12}:{query,top_k:12};
      const items=await post('/api/v1/knowledge/search/classic',body);
      $('knowledgeStatus').textContent=`${items.length} 条来源 · ${/^\d+$/.test(query)?'精确查询':state.config?.retrieval_mode || 'KnowledgeService'}`;
      $('knowledgeResults').innerHTML=items.map((e,index)=>`<button class="k-card" data-knowledge="${index}"><span class="clause-tag">第 ${esc(e.metadata.clause_no)} 条 · ${esc(e.source_level)}</span><strong>${esc((e.metadata.related_formulas||[]).join(' / ') || e.metadata.section_title || '经典条文')}</strong><p>${esc(e.content)}</p></button>`).join('') || '<p class="small-empty">没有找到来源，未生成替代答案。</p>';
      document.querySelectorAll('[data-knowledge]').forEach(el=>el.onclick=()=>openSource(items[Number(el.dataset.knowledge)]));
    }catch(error){$('knowledgeStatus').textContent=error.message;}
  }
  /** 独立 QA 不使用会诊线程，答案仅来自返回的引用摘录。 */
  async function ask() {
    const query=$('qaQuery').value.trim();if(!query)return;
    $('askBtn').disabled=true;
    try { const reply=await post('/api/v1/knowledge/qa',{query,top_k:5});$('qaAnswer').innerHTML=reply.sections.map(section=>`<h3>${esc(section.label)}</h3>${section.items.map(item=>`<p>${esc(item.text)}<br><button class="text-link" data-qa-source="${esc(item.evidence_id)}">${esc(item.evidence_id)} ↗</button></p>`).join('')}`).join('')+`<p class="warning-bar">${esc(reply.notice)}</p>`;const sources=new Map(reply.evidence_catalog.map(e=>[e.evidence_id,e]));$('qaAnswer').querySelectorAll('[data-qa-source]').forEach(el=>el.onclick=()=>openSource(sources.get(el.dataset.qaSource))); }catch(error){toast(error.message);}finally{$('askBtn').disabled=false;}
  }
  /** 鉴别规则来自当前路径的精确引用，不内置固定候选对。 */
  function differential() {
    const path=paths().find(p=>p.path_id===state.selected),items=(path?.differential_evidence||[]).map(id=>state.catalog.get(id)).filter(Boolean);
    $('drawerTitle').textContent='当前路径鉴别依据';$('drawerEyebrow').textContent='DIFFERENTIAL / EXACT SOURCES';
    $('drawerBody').innerHTML=items.length?items.map(e=>`<h3>${esc(e.metadata.title||e.knowledge_id)}</h3>${meta('引用',e.evidence_id)}<pre>${esc(JSON.stringify({维度:e.metadata.key_dimensions,规则:e.metadata.decision_rules},null,2))}</pre>`).join(''):'<p>本轮未召回该路径的精确鉴别来源，没有补造规则。</p>';openDrawer();
  }
  /** 会诊历史读取业务 Repository 分页，不从 checkpoint 拼接病例。 */
  async function history() {
    try { const page=await api('/api/v1/consultations?page=1&page_size=50');$('historyList').innerHTML=page.items.map(s=>`<button class="k-card" data-history="${esc(s.consultation_id)}"><span class="clause-tag">${esc(STATUS[s.status])} · 第 ${s.turn} 版</span><strong>${esc(s.patient_ref||'未填写脱敏引用')}</strong><p>${esc(s.topk_syndromes.join(' / ')||'暂无候选')}</p><small>${esc(new Date(s.updated_at).toLocaleString('zh-CN'))}</small></button>`).join('') || '<p class="small-empty">暂无会诊历史；内存模式重启后不会保留。</p>';document.querySelectorAll('[data-history]').forEach(el=>el.onclick=async()=>{stopWatch();state.snapshot=null;await loadSnapshot(el.dataset.history);$('clinicalText').value=state.snapshot.raw_input;navigate('workbench');if(!['completed','cancelled','failed'].includes(state.snapshot.status))watch(state.snapshot.consultation_id);});$('historyStatus').textContent=`共 ${page.total} 条，显示最新 ${page.items.length} 条`; }catch(error){$('historyStatus').textContent=error.message;}
  }
  /** 阶段回放只展示后端已保存的公开 DTO。 */
  async function replay() { if(!state.snapshot)return toast('请先选择会诊');try{const reply=await api(`/api/v1/consultations/${state.snapshot.consultation_id}/replay`);$('drawerTitle').textContent='公开阶段回放';$('drawerEyebrow').textContent='STAGE REPLAY';$('drawerBody').innerHTML=reply.stages.map(s=>`<details><summary>${esc(s.stage)} · ${s.sequence}</summary><pre>${esc(JSON.stringify(s.payload,null,2))}</pre></details>`).join('') || `<p>${esc(reply.reason)}</p>`;openDrawer();}catch(error){toast(error.message);}}

  document.querySelectorAll('[data-nav]').forEach(el=>el.onclick=()=>navigate(el.dataset.nav));
  $('fillSample').onclick=()=>{$('clinicalText').value=SAMPLE;state.parsed=null;toast('已填入合成病例；候选将由真实后端生成');};
  $('startBtn').onclick=start; $('parseBtn').onclick=parse; $('saveFacts').onclick=saveFacts;
  $('addEvidence').onclick=supplement; $('reviewReady').onclick=reviewReady;
  $('acceptBtn').onclick=()=>review('accept'); $('rejectBtn').onclick=()=>review('reject'); $('noteBtn').onclick=()=>review('note'); $('modifyBtn').onclick=()=>review('modify');
  $('rerunBtn').onclick=rerun; $('cancelBtn').onclick=cancel;
  $('knowledgeSearch').onclick=searchKnowledge; $('knowledgeQuery').onkeydown=e=>{if(e.key==='Enter')searchKnowledge();}; $('askBtn').onclick=ask;
  $('openDiff').onclick=differential; $('replayBtn').onclick=replay; $('refreshHistory').onclick=history;
  $('refreshButton').onclick=()=>{config();if(state.snapshot)loadSnapshot().catch(e=>toast(e.message));};
  $('overlay').onclick=closeDrawer;$('closeDrawer').onclick=closeDrawer;document.addEventListener('keydown',e=>{if(e.key==='Escape')closeDrawer();});
  window.addEventListener('beforeunload',stopWatch);
  renderStages(); config().then(async()=>{const id=new URL(location.href).searchParams.get('consultation');if(id){try{await loadSnapshot(id);$('clinicalText').value=state.snapshot.raw_input;if(!['completed','cancelled','failed'].includes(state.snapshot.status))watch(id);}catch(error){toast(error.message);}}});
})();
