/**
 * QA 页面控制器：独立历史、只读来源、SSE 和断线回查，依赖 QA HTTP API。
 * 只保存会话/幂等标识，不保存问答内容，不访问临床图、审核或患者线程。
 */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const view = document.querySelector('[data-view="knowledge"]');
  const messages = $("qa2Messages");
  const input = $("qa2Input");
  const state = {
    session: null,
    busy: false,
    controller: null,
    page: 1,
    after: 0,
    pending: null,
    question: "",
    timer: null,
    selection: 0,
  };
  const root = "/api/v2/knowledge";
  const sessionKey = "sixchannel.qa.v21.session";
  const pendingKey = "sixchannel.qa.v21.pending";

  /** 统一解析安全错误，失败不能当成空历史。 */
  async function api(path, options = {}) {
    const response = await fetch(root + path, options);
    if (response.status === 204) return null;
    const data = await response.json();
    if (!response.ok) throw new Error(data?.error?.message || "知识请求失败");
    return data;
  }
  /** 只保存标识偏好，浏览器禁用存储不阻断问答。 */
  function remember(key, value) {
    try {
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      $("qa2Storage").textContent = "浏览器偏好不可保存，请从历史恢复";
    }
  }
  /** 浏览器值只用于定位服务端权威记录。 */
  function recall(key) {
    try {
      return JSON.parse(localStorage.getItem(key) || "null");
    } catch (error) {
      return null;
    }
  }
  /** 公开运行状态不包含隐藏推理。 */
  function status(text, name = "idle") {
    $("qa2Status").textContent = text;
    $("qa2Status").dataset.state = name;
  }
  /** 同轮运行期间禁止切换和重复提交。 */
  function busy(value) {
    state.busy = value;
    $("qa2Send").disabled = value || state.session?.status === "archived";
    $("qa2Stop").hidden = !value;
    $("qa2New").disabled = value;
    ["qa2Rename", "qa2Archive", "qa2Delete"].forEach(
      (id) => ($(id).disabled = value || !state.session),
    );
  }
  /** 问题和资料只以文本渲染，不能执行外部 HTML。 */
  function message(role, text, provisional = false) {
    const element = document.createElement("div");
    element.className = "qa2-message " + role;
    element.dataset.provisional = String(provisional);
    element.textContent = text;
    messages.appendChild(element);
    messages.scrollTop = messages.scrollHeight;
    return element;
  }
  /** 历史回答来源绑定该轮，不能串用不同轮的标记。 */
  function bindAnswer(element, response) {
    element.dataset.provisional = "false";
    element.textContent = response.answer;
    if (response.grounding_status === "ungrounded" && response.grounding_notice) {
      const notice = document.createElement("div");
      notice.className = "qa2-grounding-notice";
      notice.textContent = response.grounding_notice;
      element.appendChild(notice);
    } else if (response.grounding_status === "partial" && response.grounding_notice) {
      const notice = document.createElement("div");
      notice.className = "qa2-grounding-notice partial";
      notice.textContent = response.grounding_notice;
      element.appendChild(notice);
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "text-link qa2-answer-sources";
    button.textContent = "查看此回答来源";
    button.addEventListener("click", () => sources(response.sources));
    element.appendChild(button);
  }
  /** 展示真实来源，展开原文通过既有精确 ID API。 */
  function sources(items = []) {
    const target = $("qa2Sources");
    target.textContent = "";
    if (!items.length) {
      target.textContent = "本轮尚未检得可核验来源";
      return;
    }
    items.forEach((item, index) => {
      const card = document.createElement("div");
      card.className = "qa2-source-card";
      const title = document.createElement("strong");
      title.textContent = "[" + (index + 1) + "] " + item.title;
      const meta = document.createElement("small");
      meta.textContent = item.knowledge_type + " · " + item.source_level;
      const excerpt = document.createElement("p");
      excerpt.textContent = item.excerpt || "";
      const expand = document.createElement("button");
      expand.type = "button";
      expand.className = "text-link";
      expand.textContent = "展开原文";
      expand.addEventListener("click", () =>
        safely(async () => {
          expand.disabled = true;
          try {
            const response = await fetch(
              "/api/v1/knowledge/evidence/" +
                encodeURIComponent(item.evidence_id),
            );
            if (!response.ok) throw new Error("来源原文暂不可获取");
            const detail = await response.json();
            if (detail.evidence_id !== item.evidence_id)
              throw new Error("来源身份不匹配");
            const body = document.createElement("div");
            body.className = "qa2-source-detail";
            body.textContent = detail.content;
            card.appendChild(body);
            expand.hidden = true;
          } finally {
            expand.disabled = false;
          }
        }),
      );
      card.append(title, meta, excerpt, expand);
      target.appendChild(card);
    });
  }
  /** 用本地化时间帮助医生快速识别最近会话，异常时间回退为空。 */
  function formatSessionTime(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return new Intl.DateTimeFormat("zh-CN", {
      month: "numeric",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  }
  /** 服务端分页/搜索保持 owner 隔离，归档会话也可选取。 */
  async function sessions(reset = true) {
    if (reset) state.page = 1;
    const page = state.page;
    const query = $("qa2Search").value.trim();
    const data = await api(
      "/sessions?page=" + page + "&page_size=20&q=" + encodeURIComponent(query),
    );
    if (page !== state.page || query !== $("qa2Search").value.trim()) return;
    const list = $("qa2SessionList");
    if (reset) list.textContent = "";
    if (!data.length && reset) {
      const empty = document.createElement("p");
      empty.className = "qa2-session-empty";
      empty.textContent = query ? "没有匹配的知识会话" : "暂无知识会话";
      list.appendChild(empty);
    }
    data.forEach((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "qa2-session";
      button.title = item.title;
      const top = document.createElement("span");
      top.className = "qa2-session-top";
      const title = document.createElement("span");
      title.className = "qa2-session-title";
      title.textContent = item.title;
      const stateLabel = document.createElement("span");
      stateLabel.className = "qa2-session-status";
      stateLabel.textContent = item.status === "archived" ? "已归档" : "进行中";
      top.append(title, stateLabel);
      const time = document.createElement("time");
      time.className = "qa2-session-time";
      time.dateTime = item.updated_at || "";
      time.textContent = formatSessionTime(item.updated_at);
      button.append(top, time);
      button.classList.toggle(
        "active",
        item.session_id === state.session?.session_id,
      );
      button.addEventListener("click", () =>
        safely(() => select(item.session_id)),
      );
      list.appendChild(button);
    });
    $("qa2MoreSessions").hidden = data.length < 20;
  }
  /** 明确区分 PostgreSQL 持久化与研发内存替身。 */
  function showSession(item) {
    state.session = item;
    $("qa2Title").textContent = item?.title || "新知识会话";
    $("qa2Archive").textContent =
      item?.status === "archived" ? "恢复会话" : "归档";
    $("qa2Storage").textContent =
      item?.storage === "postgres"
        ? "QA 历史已持久化 · 不写入临床病历"
        : "研发临时内存历史 · 服务重启后不保留";
    remember(sessionKey, item?.session_id || null);
    busy(state.busy);
  }
  /** 顺序分页读取历史；失败消息只有安全状态而无半截回答。 */
  async function history(reset = false) {
    if (!state.session) return;
    const id = state.session.session_id;
    const rows = await api(
      "/sessions/" +
        id +
        "/messages?after=" +
        (reset ? 0 : state.after) +
        "&page_size=100",
    );
    if (id !== state.session?.session_id) return;
    if (reset) {
      messages.textContent = "";
      state.after = 0;
      sources();
    }
    rows.forEach((item) => {
      state.after = item.sequence;
      if (item.role === "user") {
        state.question = item.content;
        message("user", item.content);
      } else {
        const element = message(
          "assistant",
          item.content ||
            (item.status === "running"
              ? "此轮仍在运行，可查询状态"
              : "此轮未完成：" + (item.error_code || item.status)),
        );
        if (item.response) {
          bindAnswer(element, item.response);
          sources(item.response.sources);
        }
        if (item.status === "running") {
          const check = document.createElement("button");
          check.className = "text-link";
          check.textContent = "查询运行状态";
          check.addEventListener("click", () =>
            safely(() =>
              recover({
                sessionId: id,
                turnId: item.message_id,
                clientId: item.client_message_id,
              }),
            ),
          );
          element.appendChild(check);
        } else if (["failed", "interrupted"].includes(item.status)) {
          const retry = document.createElement("button");
          retry.className = "text-link";
          retry.textContent = "重新运行此问题";
          const question = state.question;
          retry.addEventListener("click", () => send(question));
          element.appendChild(retry);
        }
      }
    });
    $("qa2MoreMessages").hidden = rows.length < 100;
  }
  /** 仅 QA 内选择会话，不跳转到工作台。 */
  async function select(id) {
    if (state.busy) return;
    const version = ++state.selection;
    const item = await api("/sessions/" + encodeURIComponent(id));
    if (version !== state.selection) return;
    showSession(item);
    await history(true);
    status(
      item.status === "archived" ? "已归档 · 恢复后可提问" : "已恢复知识会话",
    );
    view.querySelector(".qa2-sessions").classList.remove("qa2-open");
    await sessions();
  }
  /** 只回查权威消息，不自动重新发送可能已完成的问题。 */
  async function recover(pending = state.pending) {
    if (!pending) return;
    let turn;
    if (pending.turnId) turn = await api("/turns/" + pending.turnId);
    else {
      let after = 0;
      while (true) {
        const rows = await api(
          "/sessions/" +
            pending.sessionId +
            "/messages?after=" +
            after +
            "&page_size=100",
        );
        turn = rows.find(
          (item) =>
            item.role === "assistant" &&
            item.client_message_id === pending.clientId,
        );
        if (turn || rows.length < 100) break;
        after = rows[rows.length - 1].sequence;
      }
    }
    if (!turn) {
      status("服务器尚无此轮记录，可重新发送", "error");
      $("qa2Retry").hidden = false;
      return;
    }
    if (turn.status === "running") {
      state.pending = { ...pending, turnId: turn.message_id };
      remember(pendingKey, state.pending);
      status("此轮仍在运行 · 点击查询状态", "generating");
      $("qa2Check").hidden = false;
      return;
    }
    state.pending = null;
    remember(pendingKey, null);
    $("qa2Check").hidden = true;
    if (state.session?.session_id === pending.sessionId) await history(true);
    if (turn.response) result(turn.response);
    else {
      status("运行未完成：" + (turn.error_code || turn.status), "error");
      $("qa2Retry").hidden = false;
    }
    showSession(await api("/sessions/" + pending.sessionId));
  }
  /** 完整 response 覆盖所有增量；无证据时也撤回未经核验的回答。 */
  function result(response, element = null) {
    if (element) bindAnswer(element, response);
    sources(response.sources);
    const label =
      response.grounding_status === "ungrounded"
        ? "已回答 · 未检得本地依据"
        : response.status === "answered"
          ? "已回答 · 引用已核验"
          : response.status === "no_evidence"
            ? "未检得充分证据"
            : "部分回答 · 引用待核验";
    status(label, response.status);
    const usage = response.usage;
    $("qa2Usage").textContent =
      Math.round(response.duration_ms / 100) / 10 +
      " 秒 · " +
      (usage.status === "unavailable"
        ? "Token 用量不可获取"
        : usage.input_tokens +
          " 输入 / " +
          usage.output_tokens +
          " 输出 Token" +
          (usage.status === "partial" ? "（部分调用）" : ""));
  }
  /** 首轮先取得服务端会话与稳定幂等键，随后执行受监督 SSE。 */
  async function send(question) {
    const text = question.trim();
    if (state.busy || !text || state.session?.status === "archived") return;
    busy(true);
    $("qa2Retry").hidden = $("qa2Check").hidden = true;
    state.question = text;
    let provisional;
    let terminal = false;
    try {
      if (!state.session) {
        showSession(
          await api("/sessions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: text.slice(0, 40) }),
          }),
        );
        messages.textContent = "";
      }
      state.pending = {
        sessionId: state.session.session_id,
        clientId: crypto.randomUUID(),
        turnId: null,
      };
      remember(pendingKey, state.pending);
      message("user", text);
      provisional = message("assistant", "正在检索资料…", true);
      sources();
      status("正在检索资料…", "retrieving");
      state.controller = new AbortController();
      const response = await fetch(root + "/chat/stream", {
        method: "POST",
        signal: state.controller.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          session_id: state.session.session_id,
          client_message_id: state.pending.clientId,
        }),
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data?.error?.message || "问答请求失败");
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let hasDelta = false;
      /** SSE 支持 CRLF、多行 data、注释，只有完整帧参与解码。 */
      function consume(frame) {
        let name = "";
        const lines = [];
        frame.split(/\r?\n/).forEach((line) => {
          if (line.startsWith("event:")) name = line.slice(6).trim();
          if (line.startsWith("data:"))
            lines.push(line.slice(5).replace(/^ /, ""));
        });
        if (!name || !lines.length) return;
        const data = JSON.parse(lines.join("\n"));
        if (name === "qa.started") {
          state.pending.turnId = data.turn_id;
          remember(pendingKey, state.pending);
        } else if (name === "qa.tool_started")
          status("检索资料：" + data.name, "retrieving");
        else if (name === "qa.answer_delta") {
          if (!hasDelta) {
            provisional.textContent = "";
            hasDelta = true;
          }
          provisional.textContent += data.text || "";
          status("生成中 · 内容尚未核验", "generating");
        } else if (name === "qa.completed" || name === "qa.partial") {
          terminal = true;
          result(data, provisional);
          state.pending = null;
          remember(pendingKey, null);
        } else if (name === "qa.failed")
          throw new Error(data.message || data.error_code || "问答未完成");
      }
      try {
        while (true) {
          const chunk = await reader.read();
          buffer += decoder.decode(chunk.value, { stream: !chunk.done });
          let boundary;
          while ((boundary = /\r?\n\r?\n/.exec(buffer)) !== null) {
            consume(buffer.slice(0, boundary.index));
            buffer = buffer.slice(boundary.index + boundary[0].length);
          }
          if (chunk.done) break;
        }
      } finally {
        reader.releaseLock();
      }
      if (!terminal) throw new Error("连接结束，尚未收到核验完成事件");
    } catch (error) {
      if (!terminal && provisional) {
        provisional.textContent = "此轮未完成，临时生成内容已撤回。";
        provisional.dataset.provisional = "false";
      }
      status(
        error.name === "AbortError" ? "已停止 · 查询服务器状态" : error.message,
        "error",
      );
      try {
        await recover();
      } catch (queryError) {
        status("状态回查暂不可用：" + queryError.message, "error");
        $("qa2Check").hidden = false;
      }
    } finally {
      state.controller = null;
      busy(false);
      await safely(async () => {
        if (state.session)
          showSession(await api("/sessions/" + state.session.session_id));
        await sessions();
      });
    }
  }
  /** 所有异步 UI 入口显示错误，避免未处理 rejection。 */
  async function safely(operation) {
    try {
      await operation();
    } catch (error) {
      status(error.message || "历史服务暂不可用", "error");
    }
  }
  $("qa2Send").addEventListener("click", () => {
    const text = input.value;
    input.value = "";
    send(text);
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!state.busy) $("qa2Send").click();
    }
  });
  $("qa2Stop").addEventListener("click", () => state.controller?.abort());
  $("qa2Retry").addEventListener("click", () => send(state.question));
  $("qa2Check").addEventListener("click", () => safely(() => recover()));
  $("qa2New").addEventListener("click", () => {
    if (state.busy) return;
    showSession(null);
    state.pending = null;
    remember(pendingKey, null);
    messages.textContent = "";
    input.value = "";
    $("qa2Usage").textContent = "";
    $("qa2Retry").hidden = $("qa2Check").hidden = true;
    sources();
    status("新知识会话");
  });
  $("qa2Rename").addEventListener("click", () =>
    safely(async () => {
      const title = window.prompt("输入会话标题", state.session.title);
      if (!title?.trim()) return;
      showSession(
        await api("/sessions/" + state.session.session_id, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title: title.trim(),
            expected_revision: state.session.revision,
          }),
        }),
      );
      await sessions();
    }),
  );
  $("qa2Archive").addEventListener("click", () =>
    safely(async () => {
      showSession(
        await api("/sessions/" + state.session.session_id, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            status: state.session.status === "archived" ? "active" : "archived",
            expected_revision: state.session.revision,
          }),
        }),
      );
      await sessions();
    }),
  );
  $("qa2Delete").addEventListener("click", () =>
    safely(async () => {
      if (!window.confirm("删除此知识会话及全部消息？")) return;
      await api(
        "/sessions/" +
          state.session.session_id +
          "?expected_revision=" +
          state.session.revision,
        { method: "DELETE" },
      );
      $("qa2New").click();
      await sessions();
    }),
  );
  $("qa2Search").addEventListener("input", () => {
    clearTimeout(state.timer);
    state.timer = setTimeout(() => safely(() => sessions()), 250);
  });
  $("qa2MoreSessions").addEventListener("click", () => {
    state.page += 1;
    safely(() => sessions(false));
  });
  $("qa2MoreMessages").addEventListener("click", () => safely(() => history()));
  /** 移动浮层内提供关闭入口，避免覆盖工具栏后无法收起。 */
  $("qa2HistoryClose").addEventListener("click", () =>
    view.querySelector(".qa2-sessions").classList.remove("qa2-open"),
  );
  $("qa2SourceClose").addEventListener("click", () =>
    view.querySelector(".qa2-sources").classList.remove("qa2-open"),
  );
  $("qa2HistoryToggle").addEventListener("click", () =>
    view.querySelector(".qa2-sessions").classList.toggle("qa2-open"),
  );
  $("qa2SourceToggle").addEventListener("click", () =>
    view.querySelector(".qa2-sources").classList.toggle("qa2-open"),
  );
  document.querySelectorAll(".qa2-example").forEach((button) =>
    button.addEventListener("click", () => {
      input.value = button.textContent;
      input.focus();
    }),
  );
  document.querySelectorAll("[data-qa2-tab]").forEach((tab) =>
    tab.addEventListener("click", () => {
      document
        .querySelectorAll("[data-qa2-tab]")
        .forEach((item) => item.classList.toggle("active", item === tab));
      const chat = tab.dataset.qa2Tab === "chat";
      view.querySelector(".qa2-layout").hidden = !chat;
      view.querySelector(".qa2-lookup").hidden = chat;
    }),
  );
  safely(async () => {
    await sessions();
    const last = recall(sessionKey);
    if (typeof last === "string") await select(last);
    state.pending = recall(pendingKey);
    if (state.pending) await recover();
    busy(false);
  });
})();
