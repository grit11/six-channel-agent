/**
 * QA-only 浏览器交互回归：真实页面脚本配合可控 HTTP/SSE 夹具验证历史和断线。
 * 不调用在线模型，不修改临床页面或运行数据；后端 Agent/PG 另由专项验收验证。
 */
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const root = path.resolve(__dirname, "..");
const modules = process.env.CONSOLE_PLAYWRIGHT_MODULES;
const { chromium } = require(
  require.resolve("playwright", {
    paths: [modules || path.resolve(root, "../../../frontend/node_modules")],
  }),
);
const html = fs.readFileSync(
  path.join(root, "backend/app/static/dashboard.html"),
  "utf8",
);
const js = fs.readFileSync(
  path.join(root, "backend/app/static/knowledge_qa.js"),
  "utf8",
);
const sessionId = "00000000-0000-0000-0000-000000000001";
const turnId = "00000000-0000-0000-0000-000000000002";
const evidence = {
  evidence_id: "1.2:clause_35",
  title: "第35条",
  knowledge_type: "classic_clause",
  source_level: "L1",
  excerpt: "真实来源由后端专项验证，此处只测试渲染",
};

/** 每个 viewport 使用独立上下文，模拟服务器分页和权威终态。 */
async function scenario(browser, width) {
  const context = await browser.newContext({
    viewport: { width, height: 1000 },
  });
  let session = null;
  let rows = [];
  let runs = 0;
  let disconnect = false;
  let slow = false;
  const errors = [];
  await context.route("https://qa.test/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const pathname = url.pathname;
    if (pathname === "/dashboard") {
      // 隔离 QA 脚本，临床 console.js 有单独的真实图 E2E，不混入 QA 夹具。
      const fixture = html.replace(
        /<script[^>]*src="[^"]*console\.js[^"]*"[^>]*><\/script>/g,
        "",
      );
      return route.fulfill({ contentType: "text/html", body: fixture });
    }
    if (pathname.endsWith("knowledge_qa.js"))
      return route.fulfill({ contentType: "text/javascript", body: js });
    const respond = (data, code = 200) =>
      route.fulfill({
        status: code,
        contentType: "application/json",
        body: JSON.stringify(data),
      });
    if (pathname.endsWith("/sessions") && request.method() === "POST") {
      session = {
        session_id: sessionId,
        title: request.postDataJSON().title,
        status: "active",
        revision: 0,
        storage: "postgres",
      };
      return respond(session);
    }
    if (pathname.endsWith("/sessions")) {
      const query = url.searchParams.get("q") || "";
      return respond(session && session.title.includes(query) ? [session] : []);
    }
    if (pathname.endsWith("/messages")) return respond(rows);
    if (pathname.endsWith("/sessions/" + sessionId)) {
      if (request.method() === "PATCH") {
        const change = request.postDataJSON();
        if (change.title) session.title = change.title;
        if (change.status) session.status = change.status;
        session.revision += 1;
      }
      if (request.method() === "DELETE") {
        session = null;
        rows = [];
        return route.fulfill({ status: 204 });
      }
      return respond(session);
    }
    if (pathname.endsWith("/turns/" + turnId)) return respond(rows.at(-1));
    if (pathname.startsWith("/api/v1/knowledge/evidence/")) {
      return respond({ ...evidence, content: "精确原文展示成功" });
    }
    if (pathname.endsWith("/chat/stream")) {
      runs += 1;
      const payload = request.postDataJSON();
      assert.equal(payload.session_id, sessionId);
      assert.ok(payload.client_message_id);
      const answer = {
        session_id: sessionId,
        turn_id: turnId,
        status: "answered",
        answer: "权威核验回答[1]",
        sources: [evidence],
        citations: [],
        duration_ms: 500,
        usage: { status: "available", input_tokens: 20, output_tokens: 5 },
      };
      rows = [
        {
          sequence: 1,
          role: "user",
          content: payload.message,
          status: "completed",
        },
        {
          sequence: 2,
          role: "assistant",
          content: answer.answer,
          status: "completed",
          message_id: turnId,
          client_message_id: payload.client_message_id,
          response: answer,
        },
      ];
      session.revision += 2;
      if (slow) {
        rows[1].response = null;
        rows[1].content = "";
        rows[1].status = "interrupted";
        rows[1].error_code = "QA_RUN_INTERRUPTED";
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
      const event = (name, data) =>
        "event: " + name + "\r\ndata: " + JSON.stringify(data) + "\r\n\r\n";
      const started = event("qa.started", {
        session_id: sessionId,
        turn_id: turnId,
      });
      const delta = event("qa.answer_delta", {
        text: "临时未核验文本",
        provisional: true,
      });
      return route.fulfill({
        contentType: "text/event-stream",
        body:
          ": heartbeat\r\n\r\n" +
          started +
          delta +
          (disconnect ? "" : event("qa.completed", answer)),
      });
    }
    return respond({});
  });
  const page = await context.newPage();
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("https://qa.test/dashboard");
  await page.evaluate(() => {
    document
      .querySelectorAll(".view")
      .forEach((v) => v.classList.remove("active"));
    document.querySelector('[data-view="knowledge"]').classList.add("active");
  });
  await page.locator("#qa2Input").fill("第35条");
  await page.locator("#qa2Send").click();
  await page.waitForFunction(
    () => document.getElementById("qa2Status").dataset.state === "answered",
  );
  assert.match(await page.locator("#qa2Messages").innerText(), /权威核验回答/);
  assert.doesNotMatch(
    await page.locator("#qa2Messages").innerText(),
    /临时未核验/,
  );
  if (width <= 700) await page.locator("#qa2SourceToggle").click();
  await page.locator("#qa2Sources button").first().click();
  await page.locator(".qa2-source-detail").waitFor();
  assert.match(
    await page.locator(".qa2-source-detail").innerText(),
    /精确原文/,
  );
  if (width <= 700) await page.locator("#qa2SourceClose").click();
  await page.reload();
  await page.evaluate(() => {
    document
      .querySelectorAll(".view")
      .forEach((v) => v.classList.remove("active"));
    document.querySelector('[data-view="knowledge"]').classList.add("active");
  });
  await page.locator(".qa2-message.assistant").waitFor();
  assert.match(await page.locator("#qa2Messages").innerText(), /权威核验回答/);
  page.once("dialog", (dialog) => dialog.accept("已重命名知识会话"));
  await page.locator("#qa2Rename").click();
  await page.waitForFunction(
    () =>
      document.getElementById("qa2Title").textContent === "已重命名知识会话",
  );
  await page.locator("#qa2Archive").click();
  await page.waitForFunction(() => document.getElementById("qa2Send").disabled);
  await page.locator("#qa2Archive").click();
  await page.waitForFunction(
    () => !document.getElementById("qa2Send").disabled,
  );
  if (width <= 1100) await page.locator("#qa2HistoryToggle").click();
  await page.locator("#qa2Search").fill("不存在");
  await page.waitForTimeout(350);
  assert.equal(await page.locator("#qa2SessionList button").count(), 0);
  await page.locator("#qa2Search").fill("");
  await page.waitForTimeout(350);
  assert.equal(await page.locator("#qa2SessionList button").count(), 1);
  if (width <= 1100) await page.locator("#qa2HistoryClose").click();
  disconnect = true;
  await page.locator("#qa2Input").fill("断线测试");
  await page.locator("#qa2Send").click();
  await page.waitForFunction(
    () => document.getElementById("qa2Status").dataset.state === "answered",
  );
  await page.waitForFunction(
    () => !document.getElementById("qa2Send").disabled,
  );
  assert.equal(runs, 2, "断线回查不能重发模型请求");
  assert.doesNotMatch(
    await page.locator("#qa2Messages").innerText(),
    /临时未核验/,
  );
  slow = true;
  await page.locator("#qa2Input").fill("停止测试");
  await page.locator("#qa2Send").click();
  await page.locator("#qa2Stop").click();
  await page.waitForFunction(
    () => !document.getElementById("qa2Send").disabled,
  );
  await page.locator("#qa2Retry").waitFor();
  assert.match(await page.locator("#qa2Status").innerText(), /未完成|停止/);
  assert.doesNotMatch(
    await page.locator("#qa2Messages").innerText(),
    /临时未核验/,
  );
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
    true,
  );
  page.once("dialog", (dialog) => dialog.accept());
  await page.locator("#qa2Delete").click();
  await page.waitForFunction(
    () => document.getElementById("qa2Title").textContent === "新知识会话",
  );
  assert.equal(await page.locator(".qa2-message").count(), 0);
  assert.deepEqual(errors, []);
  await context.close();
  return { width, passed: true, runs };
}

/** 全部三种尺寸真实浏览器运行，任意异常给出非零退出码。 */
async function main() {
  const browser = await chromium.launch({
    headless: true,
    ...(process.platform === "win32" ? { channel: "msedge" } : {}),
  });
  try {
    const results = [];
    for (const width of [1440, 1024, 390])
      results.push(await scenario(browser, width));
    console.log(JSON.stringify({ passed: true, results }));
  } finally {
    await browser.close();
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
