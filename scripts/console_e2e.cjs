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

/** 等待业务快照进入指定暂停或终态，同时要求页面采用同一状态。 */
async function settled(page,id,status) {
  for(let n=0;n<120;n++) {
    const response=await page.request.get(`${base}/api/v2/consultations/${id}`);
    assert.equal(response.status(),200);
    const snapshot=await response.json();
    if(snapshot.status==='failed') throw new Error(`Clinical failed: ${snapshot.failure_code}`);
    if(snapshot.status===status) {
      await page.waitForFunction(expected=>document.getElementById('resultState').textContent===expected,status);
      return snapshot;
    }
    await page.waitForTimeout(500);
  }
  throw new Error(`Snapshot did not reach ${status}`);
}

/** 真实 V2 五阶段、补证、审核、来源、刷新与 S2 未决浏览器回归。 */
async function main() {
  const {chromium}=playwright();
  const external=process.env.E2E_EXTERNAL_SERVERS==='true';
  const server=external?null:spawn(process.env.E2E_PYTHON || 'python',[path.join(root,'scripts/run_e2e_backend.py'),'--port',String(port)],{cwd:root,stdio:'inherit',windowsHide:true});
  let browser;
  try {
    await ready();
    browser=await chromium.launch(browserOptions());
    const page=await browser.newPage({viewport:{width:1600,height:1000}});
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.goto(`${base}/dashboard`);
    await page.waitForFunction(()=>!document.getElementById('startBtn').disabled);
    await page.locator('#clinicalText').fill('发热，汗出，头痛，脉浮缓。');
    await page.locator('#parseBtn').click();
    await page.locator('[data-polarity]').first().waitFor();
    await page.locator('#startBtn').click();
    await page.locator('[data-path]').first().waitFor({timeout:60000});
    const id=new URL(page.url()).searchParams.get('consultation');assert.ok(id);
    const first=await settled(page,id,'awaiting_doctor_review');
    assert.equal(await page.locator('#stageRow .step').count(),5);
    assert.equal(await page.locator('#stageRow .step').last().locator('.step-dot').innerText(),'5');
    assert.equal(first.protocol_version,'clinical-v2.0');
    assert.equal(first.paused_at,'s5_doctor_review');
    assert.ok(first.retrieval.formula_candidates.length<=5);
    assert.ok(first.comparison.ordered_paths.length<=3);
    assert.equal(first.comparison.ordered_paths[0].verification_status,'grounded');
    await page.locator('[data-source]').first().click();
    await page.locator('#drawer.visible').waitFor();
    assert.ok((await page.locator('#drawerBody').innerText()).length>0);
    await page.locator('#closeDrawer').click();
    await page.reload();await page.locator('[data-path]').first().waitFor();
    assert.equal(new URL(page.url()).searchParams.get('consultation'),id);
    await page.locator('#extraEvidence').fill('今天仍发热。');
    await page.locator('#addEvidence').click();
    await page.waitForFunction(()=>document.getElementById('sessionInfo').textContent.includes('revision 1'));
    const supplemented=await settled(page,id,'awaiting_doctor_review');
    assert.equal(supplemented.thread_id,first.thread_id);
    assert.equal(supplemented.revision,1);
    assert.ok(supplemented.turn>first.turn);
    await page.locator('#acceptBtn').click();
    const complete=await settled(page,id,'completed');
    assert.equal(complete.doctor_review.action,'accept');
    assert.equal(complete.review_history.length,1);
    assert.equal(await page.locator('#stageRow .step.finished').count(),5);
    const artifacts=path.resolve(process.env.CONSOLE_TEST_ARTIFACTS || path.join(root,'..','Console版本测试文件'));
    const output=path.resolve(process.env.CONSOLE_E2E_OUTPUT || path.join(artifacts,'screenshots'));
    fs.mkdirSync(output,{recursive:true});
    await page.screenshot({path:path.join(output,'v2-desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1),'Mobile overflow');
    await page.screenshot({path:path.join(output,'v2-mobile.png'),fullPage:true});
    await page.locator('[data-nav="history"]').click();await page.locator('[data-history]').first().waitFor();
    await page.setViewportSize({width:1600,height:1000});
    await page.goto(`${base}/dashboard`);
    await page.waitForFunction(()=>!document.getElementById('startBtn').disabled);
    await page.locator('#clinicalText').fill('难受');await page.locator('#startBtn').click();
    await page.waitForFunction(()=>document.getElementById('resultState').textContent==='needs_evidence');
    const unknownId=new URL(page.url()).searchParams.get('consultation');assert.ok(unknownId);
    const unknown=await settled(page,unknownId,'needs_evidence');
    assert.equal(unknown.paused_at,'s2_clarification');assert.equal(unknown.retrieval,null);
    await page.reload();await page.waitForFunction(()=>!document.getElementById('deferBtn').disabled);
    await page.locator('#deferBtn').click();
    assert.equal((await settled(page,unknownId,'undetermined')).comparison,null);
    assert.deepEqual(errors,[]);console.log('V2 Native Console E2E passed');
  } finally {if(browser)await browser.close();if(server)server.kill();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});

