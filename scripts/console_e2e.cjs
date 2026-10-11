/** 原生 Console 浏览器回归：使用隔离真实图服务验证输入、补证、审核、来源和恢复。
 * 不调用在线模型，不安装依赖；Playwright 仅用于工程检查，不是运行时依赖。
 */
const {spawn} = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const port = Number(process.env.E2E_API_PORT || 8037);
const base = `http://127.0.0.1:${port}`;

/** 解析现有测试依赖；缺失则明确失败，不把跳过当通过。 */
function playwright() {
  try { return require('playwright'); }
  catch (error) {
    // 优先使用显式模块路径，其次查找同一项目树中已安装的测试工具；不安装依赖。
    const choices=[process.env.CONSOLE_PLAYWRIGHT_MODULES,
      path.resolve(root,'../../six-channel-agent/frontend/node_modules')].filter(Boolean);
    for(const modules of choices) {
      let modulePath;
      try { modulePath=require.resolve('playwright',{paths:[modules]}); }
      catch(resolveError) { if(resolveError.code==='MODULE_NOT_FOUND')continue;throw resolveError; }
      return require(modulePath);
    }
    throw new Error('Playwright not found. Set CONSOLE_PLAYWRIGHT_MODULES to an existing node_modules folder with Playwright.');
  }
}

/** 优先使用显式浏览器配置与本机 Edge/Chrome；缺失时由 Playwright 报告安装问题。 */
function browserOptions() {
  if(process.env.CONSOLE_BROWSER_EXECUTABLE) {
    if(!fs.existsSync(process.env.CONSOLE_BROWSER_EXECUTABLE)) {
      throw new Error('CONSOLE_BROWSER_EXECUTABLE does not point to an existing browser.');
    }
    return {headless:true,executablePath:process.env.CONSOLE_BROWSER_EXECUTABLE};
  }
  if(process.env.CONSOLE_BROWSER_CHANNEL) {
    return {headless:true,channel:process.env.CONSOLE_BROWSER_CHANNEL};
  }
  if(process.platform==='win32') {
    const folders=[process.env['PROGRAMFILES(X86)'],process.env.PROGRAMFILES,process.env.LOCALAPPDATA].filter(Boolean);
    for(const [channel,relative] of [['msedge','Microsoft/Edge/Application/msedge.exe'],['chrome','Google/Chrome/Application/chrome.exe']]) {
      if(folders.some(folder=>fs.existsSync(path.join(folder,relative)))) {
        return {headless:true,channel};
      }
    }
  }
  return {headless:true};
}

/** 等待真实服务就绪，超时返回失败。 */
async function ready() {
  for(let n=0;n<60;n++) {
    try { if((await fetch(`${base}/api/console/config`)).ok) return; } catch {}
    await new Promise(resolve=>setTimeout(resolve,500));
  }
  throw new Error('Isolated API did not start');
}

/** 补证同步/异步返回均须保持会诊线程、提升版本；202 只能代表运行中。 */
async function acceptedEvidence(response,previous) {
  assert.ok([200,202].includes(response.status()),`Evidence HTTP ${response.status()}`);
  const snapshot=await response.json();
  assert.equal(snapshot.consultation_id,previous.consultation_id);
  assert.equal(snapshot.thread_id,previous.thread_id);
  assert.ok(snapshot.turn>previous.turn,'Supplement must advance the snapshot version');
  if(response.status()===202) assert.equal(snapshot.status,'running');
  else assert.ok(['needs_evidence','awaiting_doctor_review'].includes(snapshot.status));
  return snapshot;
}

/** 等待新轮权威快照和页面版本同时就绪，避免审核旧轮候选或把受理当完成。 */
async function waitForEvidenceResult(page,id,minTurn) {
  const deadline=Date.now()+60000;
  let lastStatus='not_read';
  while(Date.now()<deadline) {
    // 网络轮询在 Node 侧完成，避免跨浏览器异步 JSHandle 返回 Promise 而非快照。
    const response=await page.request.get(`${base}/api/v1/consultations/${encodeURIComponent(id)}`,{timeout:10000});
    assert.equal(response.status(),200,'Authoritative snapshot must be available');
    const snapshot=await response.json();
    assert.equal(snapshot.consultation_id,id);
    lastStatus=`${snapshot.status} / turn ${snapshot.turn}`;
    if(['failed','cancelled','safety_escalated'].includes(snapshot.status)) throw new Error(`Supplement ended: ${lastStatus}`);
    if(snapshot.turn>=minTurn && ['needs_evidence','awaiting_doctor_review'].includes(snapshot.status)) {
      // DOM 检查同步返回布尔值；只有医生操作合法且页面已显示当前版时才交回快照。
      const displayed=await page.evaluate(turn=>{
        const session=document.getElementById('sessionInfo');
        const accept=document.getElementById('acceptBtn');
        return Boolean(session?.textContent.includes(`第 ${turn} 版`) && accept && !accept.disabled);
      },snapshot.turn);
      if(displayed) return snapshot;
    }
    await page.waitForTimeout(500);
  }
  throw new Error(`Supplement did not settle within 60 seconds: ${lastStatus}`);
}

/** 验证关键纵向流程并保存桌面/移动截图；所有候选来自真实后端。 */
async function main() {
  const {chromium}=playwright();
  const external=process.env.E2E_EXTERNAL_SERVERS==='true';
  const server=external?null:spawn(process.env.E2E_PYTHON || 'python',[path.join(root,'scripts/run_e2e_backend.py'),'--port',String(port)],{cwd:root,stdio:'inherit',windowsHide:true});
  let browser;
  try {
    await ready();
    const options=browserOptions();
    console.log(`Browser: ${options.channel || options.executablePath || 'Playwright Chromium'}`);
    browser=await chromium.launch(options);
    const page=await browser.newPage({viewport:{width:1600,height:1000}});
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.goto(`${base}/dashboard`);
    await page.waitForFunction(()=>!document.getElementById('startBtn').disabled);
    await page.locator('#clinicalText').fill('患者往来寒热，胸胁苦满，口苦，咽干，目眩，脉弦。');
    await page.locator('#parseBtn').click();
    await page.locator('[data-fact-index]').first().waitFor();
    await page.locator('#startBtn').click();
    await page.locator('.result-card').first().waitFor({timeout:60000});
    const id=new URL(page.url()).searchParams.get('consultation');assert.ok(id);
    await page.locator('.source-item').first().click();
    assert.ok(await page.locator('#drawerBody').innerText());await page.locator('#closeDrawer').click();
    await page.locator('#factsEditor').evaluate(el=>el.open=true);
    const first=page.locator('[data-fact-index]').first();await first.selectOption('unknown');
    await page.waitForTimeout(3000);assert.equal(await first.inputValue(),'unknown');
    await page.reload();await page.locator('.result-card').first().waitFor();
    const previous=await (await page.request.get(`${base}/api/v1/consultations/${id}`)).json();
    await page.locator('#extraEvidence').fill('无呕吐，脉弦，无汗。');
    const evidence=page.waitForResponse(r=>r.url().endsWith('/evidence')&&r.request().method()==='POST');
    await page.locator('#addEvidence').click();
    const accepted=await acceptedEvidence(await evidence,previous);
    assert.equal(new URL(page.url()).searchParams.get('consultation'),id);
    const settled=await waitForEvidenceResult(page,id,accepted.turn);
    assert.equal(settled.thread_id,previous.thread_id);
    assert.ok(settled.clinical_evidence.findings.some(f=>f.key==='vomiting'&&f.polarity==='negative'));
    const review=page.waitForResponse(r=>r.url().endsWith('/review')&&r.request().method()==='POST');
    await page.locator('#acceptBtn').click();
    const reviewResponse=await review;assert.equal(reviewResponse.status(),200);
    const reviewed=await reviewResponse.json();
    assert.equal(reviewed.consultation_id,id);
    assert.equal(reviewed.status,'completed');
    assert.equal(reviewed.doctor_review.action,'accept');
    // 全量验收将截图关联到本次报告，单独执行时写入统一测试资料目录。
    const artifacts=path.resolve(process.env.CONSOLE_TEST_ARTIFACTS || path.join(root,'..','Console版本测试文件'));
    const output=path.resolve(process.env.CONSOLE_E2E_OUTPUT || path.join(artifacts,'screenshots'));
    fs.mkdirSync(output,{recursive:true});
    await page.screenshot({path:path.join(output,'desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1));
    await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
    await page.locator('[data-nav="history"]').click();await page.locator('[data-history]').first().waitFor();
    assert.deepEqual(errors,[]);console.log('Native Console E2E passed');
  } finally {if(browser)await browser.close();if(server)server.kill();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
