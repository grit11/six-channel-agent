/**
 * V2 医生工作台。通过业务快照和 SSE 展示五阶段、真实来源及人工审核。
 * 页面不计算医学分数，不存病例于浏览器缓存；刷新用 URL 中的会诊 ID 恢复。
 */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({
    '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
  })[c]);
  const stages = [
    ['s1_extract','S1 事实抽取'], ['s2_localize','S2 六经定位'],
    ['s3_retrieve','S3 来源检索'], ['s4_compare','S4 证据比较'], ['s5_doctor_review','S5 医生审核']
  ];
  const state = {snapshot:null, facts:null, selected:null, stream:null, timer:null, source:new Map()};
  const terminal = new Set(['completed','undetermined','cancelled','failed']);
  /** API 细节统一在此处理，业务错误直接展示安全消息。 */
  async function api(path, body, headers={}) {
    const response = await fetch(path, body === undefined ? {} : {
      method:'POST', headers:{'Content-Type':'application/json', ...headers}, body:JSON.stringify(body)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || data.detail?.[0]?.msg || '请求失败');
    return data;
  }
  /** 提示仅显示后端安全错误，不回显病例。 */
  function toast(message) {
    $('toast').textContent = message; $('toast').classList.add('show');
    setTimeout(()=>$('toast').classList.remove('show'), 4500);
  }
  /** 事实保留三值、时点和原文来源，医生可以逐条修订。 */
  function renderFacts(facts) {
    state.facts = facts;
    const findings = facts?.findings || [];
    $('findingsCount').textContent = findings.length + ' 项';
    $('unknownCount').textContent = '明确未知 ' + findings.filter(f=>f.polarity==='unknown').length + ' 项';
    $('clinicalChips').innerHTML = findings.map(f=>'<span class="chip '+(f.polarity==='unknown'?'unknown':'channel')+'">'+esc(f.concept)+' · '+esc(f.polarity)+'</span>').join('');
    $('factRows').innerHTML = findings.map((f,i)=>'<div class="input-row"><span>'+esc(f.concept)+'</span><select data-polarity="'+i+'">'+['positive','negative','unknown'].map(p=>'<option '+(p===f.polarity?'selected':'')+'>'+p+'</option>').join('')+'</select><select data-time="'+i+'">'+['current','past','before_treatment','after_treatment','unspecified'].map(t=>'<option '+(t===f.time_scope?'selected':'')+'>'+t+'</option>').join('')+'</select><small>'+esc(f.source_span.text)+'</small></div>').join('');
    $('conflictNotice').textContent = facts?.conflicts?.length ? '存在事实冲突，请核对原文后修订。' : '';
    $('saveFacts').disabled = !facts || !state.snapshot?.paused_at;
  }
  /** 选中路径只影响展示，不能改变后端排序或审核状态。 */
  function selectPath(path) {
    state.selected = path;
    const facts = new Map((state.snapshot.clinical_evidence?.findings || []).map(f=>[f.fact_id,f]));
    const list = (id, values) => $(id).innerHTML = values.map(v=>'<p>'+esc(v)+'</p>').join('') || '<p class="small-empty">暂无</p>';
    list('supportList',path.support_fact_ids.map(id=>facts.get(id)?.concept || id));
    list('conflictList',path.counter_fact_ids.map(id=>facts.get(id)?.concept || id));
    list('missingList',path.missing_dimensions);
    $('selectedSyndrome').textContent = path.formula_name || '未决';
    $('supportCount').textContent = path.support_fact_ids.length+'项';
    $('conflictCount').textContent = path.counter_fact_ids.length+'项';
    $('missingCount').textContent = path.missing_dimensions.length+'项';
    $('pathAudit').textContent = JSON.stringify(path,null,2);
    const ids = [...new Set([...path.support_evidence_ids,...path.counter_evidence_ids])];
    $('sourceList').innerHTML = ids.map(id=>'<button class="text-link" data-source="'+esc(id)+'">'+esc(id)+' ↗</button><p>'+esc(state.source.get(id)?.excerpt || '点击回查完整来源')+'</p>').join('');
    $('sourceList').querySelectorAll('[data-source]').forEach(el=>el.onclick=()=>openSource(el.dataset.source));
  }
  /** 业务快照是唯一状态权威，暂停时不虚构结果。 */
  function render(snapshot) {
    state.snapshot = snapshot;
    const url = new URL(location.href);
    url.searchParams.set('consultation',snapshot.consultation_id); history.replaceState(null,'',url);
    $('clinicalText').value = snapshot.raw_input;
    renderFacts(snapshot.clinical_evidence);
    state.source = new Map((snapshot.retrieval?.formula_candidates || []).flatMap(c=>c.source_evidence).map(e=>[e.evidence_id,e]));
    $('resultState').textContent = snapshot.status;
    $('sessionInfo').textContent = snapshot.consultation_id+' · revision '+snapshot.revision+' · turn '+snapshot.turn;
    $('pipelineState').textContent = snapshot.paused_at || snapshot.status;
    $('stageRow').innerHTML = stages.map(([id,label],i)=>'<div class="step '+(snapshot.completed_stages.includes(id)?'finished':'')+'"><span class="step-dot">'+(i+1)+'</span><span class="step-label">'+label+'</span></div>').join('');
    $('eightSummary').textContent = snapshot.localization ? JSON.stringify(snapshot.localization,null,2) : '等待六经定位';
    const questions = snapshot.paused_at==='s2_clarification' ? snapshot.localization?.clarification_questions : snapshot.comparison?.next_questions;
    $('suggestedQuestion').textContent = (questions || []).map(q=>q.question).join('\n') || snapshot.comparison?.unresolved_reason || '当前无补问';
    const paths = snapshot.comparison?.ordered_paths || [];
    const top5 = snapshot.retrieval?.formula_candidates || [];
    $('resultNotice').textContent = '检索候选 '+top5.length+' 个：'+top5.map(c=>c.formula_name).join(' / ')+'。比较顺序来自 LLM，来源核验不代表临床正确性。';
    $('results').innerHTML = paths.map((p,i)=>'<button class="k-card" data-path="'+i+'"><strong>排序 '+esc(p.rank ?? '未决')+' · '+esc(p.formula_name)+'</strong><p>'+esc(p.six_channels.join(' / '))+'</p><p>'+esc(p.brief_rationale)+'</p><p>'+esc(p.differential_summary)+'</p><small>'+esc(p.verification_status)+'</small></button>').join('') ||
      '<div class="empty-state"><strong>'+esc(snapshot.paused_at==='s2_clarification'?'六经依据不足，等待医生澄清':snapshot.failure_code || snapshot.status)+'</strong><p>尚无可靠比较结果</p></div>';
    $('results').querySelectorAll('[data-path]').forEach(el=>el.onclick=()=>selectPath(paths[Number(el.dataset.path)]));
    if (paths.length) selectPath(paths.find(p=>p.path_id===state.selected?.path_id) || paths[0]);
    else {
      state.selected=null;
      ['supportList','conflictList','missingList','sourceList'].forEach(id=>$(id).textContent='暂无');
    }
    const review = snapshot.paused_at==='s5_doctor_review' && snapshot.status==='awaiting_doctor_review';
    ['acceptBtn','modifyBtn'].forEach(id=>$(id).disabled=!review || !paths.length);
    ['rejectBtn','noteBtn'].forEach(id=>$(id).disabled=!review);
    $('addEvidence').disabled=!snapshot.paused_at;
    $('deferBtn').disabled=snapshot.paused_at!=='s2_clarification';
    $('cancelBtn').disabled=!snapshot.paused_at;
    $('startBtn').disabled=snapshot.status==='running';
    $('reviewState').textContent=snapshot.doctor_review ? '医生审核：'+snapshot.doctor_review.action : '等待医生确认';
    if (snapshot.status==='running') watch(snapshot.consultation_id);
    else stopWatch();
  }
  /** 刷新读取 Repository；事件本身不携带诊断。 */
  async function load(id=state.snapshot?.consultation_id) {
    if (id) render(await api('/api/v2/consultations/'+encodeURIComponent(id)));
  }
  /** 断线时轮询快照，避免浏览器静默停在运行中。 */
  function watch(id) {
    if (state.stream) return;
    state.stream = new EventSource('/api/v2/consultations/'+id+'/events');
    const names=['snapshot_updated','stage_started','stage_completed','stage_failed','model_started','model_completed','model_failed','retrieval_started','retrieval_completed','clarification_requested','review_requested','consultation_completed','failed'];
    for (const name of names) state.stream.addEventListener(name,event=>{
      const item=JSON.parse(event.data);
      const row=document.createElement('p');
      row.textContent=item.sequence+' · '+name+' · '+JSON.stringify(item.payload);
      $('traces').prepend(row);
      if (['snapshot_updated','clarification_requested','review_requested','consultation_completed','failed'].includes(name)) load(id).catch(e=>toast(e.message));
    });
    state.timer=setInterval(()=>load(id).catch(e=>toast(e.message)),2500);
  }
  function stopWatch() {
    state.stream?.close(); state.stream=null; clearInterval(state.timer); state.timer=null;
  }
  /** 每个人工动作有独立幂等键，并携带当前业务版本。 */
  async function command(kind, extra={}) {
    const s=state.snapshot; if (!s) return;
    const body={kind,consultation_id:s.consultation_id,revision:s.revision,
      action_turn:s.turn,action_id:crypto.randomUUID(),...extra};
    const suffix=kind==='review'?'review':kind==='clarify'?'clarify':'resume';
    render(await api('/api/v2/consultations/'+s.consultation_id+'/'+suffix,body));
  }
  async function review(action) {
    await command('review',{review:{action,selected_path_id:state.selected?.path_id || null,note:$('reviewNote').value || null}});
  }
  /** 回查精确来源，不把不存在的引用替换成模糊命中。 */
  async function openSource(id) {
    try {
      const evidence=await api('/api/v1/knowledge/evidence/'+encodeURIComponent(id));
      $('drawerTitle').textContent=evidence.evidence_id;
      $('drawerBody').textContent=evidence.content;
      $('drawer').classList.add('visible'); $('overlay').classList.add('visible');
    } catch(e) { toast(e.message); }
  }
  async function searchKnowledge() {
    const query=$('knowledgeQuery').value.trim(); if (!query) return;
    const result=await api('/api/v1/knowledge/search/classic',/^\d+$/.test(query)?{clause_nos:[query],top_k:12}:{query,top_k:12});
    $('knowledgeStatus').textContent=result.length+' 条来源';
    $('knowledgeResults').innerHTML=result.map(e=>'<button class="k-card" data-knowledge="'+esc(e.evidence_id)+'"><strong>'+esc(e.metadata.clause_no || e.evidence_id)+'</strong><p>'+esc(e.content)+'</p></button>').join('');
    $('knowledgeResults').querySelectorAll('[data-knowledge]').forEach(el=>el.onclick=()=>openSource(el.dataset.knowledge));
  }
  async function ask() {
    const reply=await api('/api/v1/knowledge/qa',{query:$('qaQuery').value,top_k:5});
    $('qaAnswer').innerHTML=reply.sections.map(s=>'<h3>'+esc(s.label)+'</h3>'+s.items.map(i=>'<p>'+esc(i.text)+'</p><button class="text-link" data-qa="'+esc(i.evidence_id)+'">'+esc(i.evidence_id)+'</button>').join('')).join('');
    $('qaAnswer').querySelectorAll('[data-qa]').forEach(el=>el.onclick=()=>openSource(el.dataset.qa));
  }
  async function archive() {
    const page=await api('/api/v2/consultations');
    $('historyStatus').textContent=page.total+' 条';
    $('historyList').innerHTML=page.items.map(s=>'<button class="k-card" data-history="'+s.consultation_id+'"><strong>'+esc(s.consultation_id)+'</strong><p>'+esc(s.status)+' · '+esc(s.topk_syndromes.join(' / '))+'</p></button>').join('');
    $('historyList').querySelectorAll('[data-history]').forEach(el=>el.onclick=async()=>{await load(el.dataset.history);showView('workbench');});
  }
  function showView(name) {
    document.querySelectorAll('[data-view]').forEach(el=>el.classList.toggle('active',el.dataset.view===name));
    document.querySelectorAll('[data-nav]').forEach(el=>el.classList.toggle('active',el.dataset.nav===name));
  }
  /** 所有交互统一捕捉错误，禁止未处理异步异常隐藏失败。 */
  function bind(id, action) { $(id).onclick=()=>Promise.resolve().then(action).catch(e=>toast(e.message)); }
  bind('parseBtn',async()=>{renderFacts(await api('/api/v2/clinical/parse',{text:$('clinicalText').value}));$('factsEditor').open=true;});
  bind('startBtn',async()=>{
    $('startBtn').disabled=true;
    try {render(await api('/api/v2/consultations',{clinical_text:$('clinicalText').value},{'Idempotency-Key':crypto.randomUUID()}));}
    finally {$('startBtn').disabled=state.snapshot?.status==='running';}
  });
  bind('addEvidence',async()=>{
    const text=$('extraEvidence').value.trim();
    if(!text) throw new Error('请填写补充事实');
    await command(state.snapshot.paused_at==='s2_clarification'?'clarify':'review',{
      clinical_text:state.snapshot.raw_input+'\n医生补充：'+text,
      ...(state.snapshot.paused_at==='s5_doctor_review'?{review:{action:'add_evidence'}}:{})
    });
    $('extraEvidence').value='';
  });
  bind('deferBtn',()=>command('clarify',{no_additional_information:true}));
  bind('saveFacts',async()=>{
    const facts=structuredClone(state.facts);
    $('factRows').querySelectorAll('[data-polarity]').forEach(el=>{facts.findings[Number(el.dataset.polarity)].polarity=el.value;});
    $('factRows').querySelectorAll('[data-time]').forEach(el=>{facts.findings[Number(el.dataset.time)].time_scope=el.value;});
    facts.findings.forEach(f=>f.review_status='doctor_verified');
    facts.doctor_revision=state.snapshot.revision+1; facts.source='doctor_edit';
    await command(state.snapshot.paused_at==='s2_clarification'?'clarify':'review',{
      clinical_evidence:facts,
      ...(state.snapshot.paused_at==='s5_doctor_review'?{review:{action:'add_evidence'}}:{})
    });
  });
  bind('acceptBtn',()=>review('accept')); bind('rejectBtn',()=>review('reject'));
  bind('noteBtn',()=>review('note')); bind('modifyBtn',async()=>{
    const formula=prompt('医生修订后的方证名称'); if (!formula?.trim()) return;
    await command('review',{review:{action:'modify',selected_path_id:state.selected?.path_id,
      modified_formula_pattern:formula.trim(),note:$('reviewNote').value || null}});
  });
  bind('cancelBtn',()=>command('cancel')); bind('knowledgeSearch',searchKnowledge);
  bind('askBtn',ask); bind('refreshHistory',archive);
  bind('refreshButton',()=>load()); bind('fillSample',()=>{$('clinicalText').value='恶寒发热，汗出，头痛，脉浮缓。';});
  bind('closeDrawer',()=>{$('drawer').classList.remove('visible');$('overlay').classList.remove('visible');});
  document.querySelectorAll('[data-nav]').forEach(el=>el.onclick=()=>{
    showView(el.dataset.nav);
    if (el.dataset.nav==='history') archive().catch(e=>toast(e.message));
    if (el.dataset.nav==='knowledge') searchKnowledge().catch(e=>toast(e.message));
  });
  window.addEventListener('beforeunload',stopWatch);
  api('/api/console/config').then(config=>{
    $('runtimeInfo').textContent=config.text;
    $('modeBadge').textContent=config.clinical_engine;
    $('startBtn').disabled=!config.consultation_enabled;
    $('parseBtn').disabled=!config.agent_connected;
  }).catch(e=>toast(e.message));
  const id=new URL(location.href).searchParams.get('consultation');
  if (id) load(id).catch(e=>toast(e.message));
})();


