/* 知识 QA V2 页面脚本：只管理 qa2-* 元素，不接触临床工作台、轨迹和历史状态。 */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const state = { sessionId: null, sessions: [], busy: false };
  const messages = $('qa2Messages');
  const input = $('qa2Input');
  const status = $('qa2Status');
  const sources = $('qa2Sources');

  function setStatus(value, stateName = 'idle') {
    status.textContent = value;
    status.dataset.state = stateName;
  }
  function addMessage(role, text) {
    const el = document.createElement('div');
    el.className = `qa2-message ${role}`;
    el.textContent = text;
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
  }
  function renderSources(items) {
    sources.textContent = '';
    if (!items || !items.length) {
      const empty = document.createElement('p'); empty.className = 'small-empty'; empty.textContent = '本轮尚未检索到来源'; sources.appendChild(empty); return;
    }
    items.forEach((item, index) => {
      const card = document.createElement('button'); card.type = 'button'; card.className = 'qa2-source-card';
      const title = document.createElement('strong'); title.textContent = `[${index + 1}] ${item.title}`;
      const meta = document.createElement('small'); meta.textContent = `${item.knowledge_type} · ${item.source_level} · ${item.source}`;
      const excerpt = document.createElement('p'); excerpt.textContent = item.excerpt;
      card.append(title, meta, excerpt);
      card.addEventListener('click', async () => {
        try {
          const response = await fetch('/api/v1/knowledge/evidence/' + encodeURIComponent(item.evidence_id));
          const full = await response.json();
          const detail = document.createElement('p'); detail.className = 'qa2-source-detail'; detail.textContent = full.content || item.excerpt;
          card.appendChild(detail);
        } catch (error) { setStatus('来源展开失败'); }
      });
      sources.appendChild(card);
    });
  }
  function rememberSession() {
    if (!state.sessionId) return;
    if (!state.sessions.find((item) => item.id === state.sessionId)) state.sessions.unshift({ id: state.sessionId, title: input.value.slice(0, 24) });
    const list = $('qa2SessionList'); list.textContent = '';
    state.sessions.slice(0, 8).forEach((item) => { const button = document.createElement('button'); button.type = 'button'; button.className = 'text-link'; button.textContent = item.title || '知识会话'; button.addEventListener('click', () => setStatus('临时会话只能在当前页面继续')); list.appendChild(button); });
  }
  async function send(message) {
    if (state.busy || !message.trim()) return;
    state.busy = true;
    $('qa2Send').disabled = true;
    addMessage('user', message.trim());
    setStatus('检索中…', 'retrieving');
    try {
      setStatus('生成回答中…', 'generating');
      const response = await fetch('/api/v2/knowledge/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: message.trim(), session_id: state.sessionId }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data?.error?.message || '知识问答暂不可用');
      state.sessionId = data.session_id; rememberSession(); const nextState = data.status === 'answered' ? 'answered' : data.status === 'partial' ? 'partial' : 'no_evidence'; setStatus(data.status === 'answered' ? '已回答' : data.status === 'partial' ? '部分证据' : '暂无证据', nextState); addMessage('assistant', data.answer); renderSources(data.sources);
    } catch (error) { setStatus('错误', 'error'); addMessage('assistant', error.message || '知识问答暂不可用'); }
    finally { state.busy = false; $('qa2Send').disabled = false; }
  }
  $('qa2Send').addEventListener('click', () => { const value = input.value; input.value = ''; send(value); });
  input.addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('qa2Send').click(); } });
  document.querySelectorAll('.qa2-example').forEach((button) => button.addEventListener('click', () => { input.value = button.textContent; input.focus(); }));
  $('qa2New').addEventListener('click', () => { state.sessionId = null; messages.textContent = ''; renderSources([]); setStatus('新建临时会话', 'idle'); });
  $('qa2SourceToggle').addEventListener('click', () => {
    const panel = document.querySelector('[data-view="knowledge"] .qa2-sources');
    if (panel) panel.classList.toggle('qa2-open');
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      const panel = document.querySelector('[data-view="knowledge"] .qa2-sources');
      if (panel) panel.classList.remove('qa2-open');
    }
  });
  document.querySelectorAll('[data-qa2-tab]').forEach((tab) => tab.addEventListener('click', () => { document.querySelectorAll('[data-qa2-tab]').forEach((item) => item.classList.toggle('active', item === tab)); const chat = tab.dataset.qa2Tab === 'chat'; document.querySelector('.qa2-layout').hidden = !chat; document.querySelector('.qa2-lookup').hidden = chat; }));
})();
