// Synthetic full-reference transport fixtures. These verify interaction, not model quality.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const root = path.resolve(__dirname, '..');
process.env.TEMP = process.env.TMP = path.join(root, '.tmp');

(async () => {
  const context = await chromium.launchPersistentContext(path.join(root, '.cache', 'browser-v3-test-profile'), {
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe', headless: true,
    viewport: { width: 1440, height: 1024 }, args: ['--disable-crash-reporter', '--no-first-run'],
  });
  const page = context.pages()[0], errors = [], checks = [], requests = [];
  page.on('pageerror', error => errors.push(error.message));
  let response = { serial: 0, status: 'idle', revision: 0, intent_revision: 0 }, exported = null, saved = null;
  await page.route('**/api/**', async route => {
    const request = route.request(), url = new URL(request.url());
    let body = { ok: true };
    if (url.pathname === '/api/status') body = { model: { ready: true, message: 'Synthetic full-reference fixture; no inference' } };
    else if (url.pathname === '/api/sessions') body = { id: 'v3-transport-fixture' };
    else if (url.pathname.endsWith('/result')) body = response;
    else if (url.pathname.endsWith('/suggest')) { requests.push(request.postDataJSON()); body = { job_id: 'fixture' }; }
    else if (url.pathname === '/api/exports') { exported = request.postDataJSON(); body = { name: 'synthetic.png' }; }
    else if (url.pathname === '/api/projects' && request.method() === 'POST') { saved = request.postDataJSON(); body = { name: 'synthetic.json' }; }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
  });
  try {
    await page.goto(process.env.DRAWINGLAB_TEST_URL || 'http://127.0.0.1:8765/?v=3');
    await page.waitForFunction(() => window.DrawingLab?.debugState().session);
    await page.locator('#auto-suggest').uncheck();
    await page.locator('#prompt').fill('a person with simple angular contours');
    const box = await page.locator('#ink').boundingBox();
    const move = (x, y) => page.mouse.move(box.x + x * box.width / 768, box.y + y * box.height / 768);
    const line = async (x, y, xx, yy) => { await move(x, y); await page.mouse.down(); await move(xx, yy); await page.mouse.up(); };
    const state = () => page.evaluate(() => window.DrawingLab.debugState());
    const snapshot = () => page.evaluate(() => {
      const pixels = document.querySelector('#ghost').getContext('2d').getImageData(0, 0, 768, 768).data;
      let hash = 0, maxAlpha = 0, coloured = false;
      for (let i = 0; i < pixels.length; i += 4) {
        hash = (Math.imul(hash, 31) + pixels[i + 3]) | 0;
        maxAlpha = Math.max(maxAlpha, pixels[i + 3]);
        if (pixels[i + 3] && (pixels[i] || pixels[i + 1] || pixels[i + 2])) coloured = true;
      }
      return { hash, maxAlpha, coloured, top: pixels[(100 * 768 + 150) * 4 + 3], bottom: pixels[(600 * 768 + 600) * 4 + 3], local: pixels[(300 * 768 + 350) * 4 + 3], middle: pixels[(300 * 768 + 400) * 4 + 3] };
    });
    const fixture = await page.evaluate(() => {
      const canvas = document.createElement('canvas'); canvas.width = canvas.height = 768;
      const ctx = canvas.getContext('2d'); ctx.strokeStyle = '#000'; ctx.lineWidth = 3;
      for (const [x, y, xx, yy] of [[100, 100, 200, 100], [550, 600, 650, 600], [280, 300, 470, 300]]) {
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(xx, yy); ctx.stroke();
      }
      return canvas.toDataURL('image/png');
    });
    const result = (s, serial, candidate) => ({ serial, status: 'ready', revision: s.revision, source_revision: s.revision, preview_revision: s.revision, intent_revision: s.intent_revision, candidate_id: candidate, ghost: fixture, metrics: { total_seconds: 2.5 } });
    await line(280, 300, 320, 300);
    let s = await state(); response = result(s, 1, 'whole-A');
    await page.waitForFunction(() => window.DrawingLab.debugState().candidate_id === 'whole-A');
    const initialInk = (await state()).inkhash, whole = await snapshot();
    assert.equal((await state()).version, 4); assert.ok(whole.top > 0 && whole.bottom > 0 && whole.middle > 0);
    assert.ok(whole.maxAlpha >= 24 && whole.maxAlpha <= 26); assert.equal(whole.coloured, false);
    checks.push('entire reference is visible in remote canvas regions at 10% black alpha');
    await move(700, 700); await page.locator('#prompt').hover();
    assert.equal((await snapshot()).hash, whole.hash); assert.equal((await state()).inkhash, initialInk);
    checks.push('cursor position and leaving canvas do not clip or change the whole reference');
    await page.locator('#reference-opacity').fill('6');
    assert.ok((await snapshot()).maxAlpha <= 16); assert.equal((await state()).inkhash, initialInk);
    await page.locator('#show-reference').uncheck(); assert.equal((await snapshot()).hash, 0);
    await page.locator('#show-reference').check();
    assert.ok((await snapshot()).bottom > 0); assert.equal((await state()).inkhash, initialInk);
    checks.push('opacity and reference visibility modify only the ghost layer');

    await move(320, 300); await page.mouse.down();
    assert.ok((await snapshot()).local > 0, 'trace guidance remains visible at pen-down');
    await move(330, 300); assert.ok((await snapshot()).local > 0, 'following the reference does not hide it');
    await move(360, 330);
    const held = await snapshot(); assert.equal(held.local, 0); assert.ok(held.bottom > 0);
    response = result(s, 2, 'held-obsolete');
    await page.waitForFunction(() => window.DrawingLab.debugState().serial >= 2);
    assert.equal((await state()).candidate_id, 'whole-A'); assert.equal((await snapshot()).hash, held.hash);
    await page.mouse.up();
    assert.equal((await state()).candidate_id, 'whole-A'); assert.equal((await state()).correctionRegions, 1);
    s = await state(); response = result(s, 3, 'whole-B');
    await page.waitForFunction(() => window.DrawingLab.debugState().candidate_id === 'whole-B');
    assert.equal((await state()).correctionRegions, 0); assert.ok((await snapshot()).middle > 0);
    checks.push('drawing masks local outdated guesses; asynchronous stale results cannot replace the candidate');

    await page.locator('#auto-suggest').check();
    const firstRequest = page.waitForRequest(r => r.url().endsWith('/suggest'));
    await line(360, 330, 380, 360); await firstRequest;
    const secondRequest = page.waitForRequest(r => r.url().endsWith('/suggest'));
    await line(380, 360, 400, 380); await secondRequest;
    await page.locator('#auto-suggest').uncheck();
    assert.equal(requests.at(-1).revision, (await state()).revision);
    assert.ok(requests.at(-1).revision > requests.at(-2).revision);
    assert.equal(requests.at(-1).strokes.length, (await state()).strokes);
    assert.ok(requests.at(-1).strokes.every(stroke => stroke.tool !== 'pen' || stroke.size === 3));
    response = { ...result(await state(), 4, null), ghost: null, status: 'running' };
    await page.waitForFunction(() => window.DrawingLab.debugState().serial >= 4);
    assert.equal((await state()).candidate_id, 'whole-B'); assert.ok((await snapshot()).bottom > 0);
    checks.push('each committed change requests the current actual ink; pending inference preserves unaffected context');

    s = await state(); response = result(s, 5, 'whole-C');
    await page.waitForFunction(() => window.DrawingLab.debugState().candidate_id === 'whole-C');
    await page.locator('#pin-reference').click(); response = result(s, 6, 'pinned-obsolete');
    await page.waitForFunction(() => window.DrawingLab.debugState().serial >= 6);
    assert.equal((await state()).candidate_id, 'whole-C');
    await page.locator('#pin-reference').click();
    checks.push('explicit pin holds the whole reference while newer results arrive');

    const beforeErase = await state();
    await page.locator('#tool-eraser').click(); await page.locator('#eraser-size').fill('60');
    await line(300, 300, 300, 300); const afterErase = await state();
    assert.ok(afterErase.alphaPixels < beforeErase.alphaPixels); assert.equal(afterErase.ghostVisible, false);
    response = result(s, 7, 'erased-obsolete');
    await page.waitForFunction(() => window.DrawingLab.debugState().serial >= 7);
    assert.equal((await state()).candidate_id, null);
    await page.locator('#undo').click(); assert.equal((await state()).inkhash, beforeErase.inkhash);
    await page.locator('#redo').click(); assert.equal((await state()).inkhash, afterErase.inkhash);
    checks.push('resizable eraser and exact undo/redo invalidate obsolete results');

    s = await state(); response = result(s, 8, 'before-intent');
    await page.waitForFunction(() => window.DrawingLab.debugState().candidate_id === 'before-intent');
    await page.locator('#prompt').fill('changed pose and proportions');
    response = result(s, 9, 'obsolete-intent');
    await page.waitForFunction(() => window.DrawingLab.debugState().serial >= 9);
    assert.equal((await state()).candidate_id, null); assert.equal((await state()).inkhash, afterErase.inkhash);
    checks.push('new prompt intent blocks earlier image results without modifying real ink');
    response = { ...result(await state(), 10, 'malformed'), ghost: 'data:image/png;base64,not-a-valid-png' };
    await page.waitForFunction(() => document.querySelector('#activity').classList.contains('error'));
    assert.equal((await state()).ghostVisible, false); assert.equal((await state()).inkhash, afterErase.inkhash);
    checks.push('invalid image payload fails safely and keeps real ink');

    response = result(await state(), 11, 'export-visible');
    await page.waitForFunction(() => window.DrawingLab.debugState().candidate_id === 'export-visible');
    const exportRequest = page.waitForRequest(r => r.url().endsWith('/exports'));
    await page.locator('#export-png').click(); await exportRequest;
    const savedRequest = page.waitForRequest(r => r.url().endsWith('/projects') && r.method() === 'POST');
    await page.locator('#save-project').click(); await savedRequest;
    assert.equal(saved.document.strokes.length, (await state()).strokes); assert.equal(saved.document.version, 1);
    const exportSample = await page.evaluate(async source => {
      const image = new Image(); image.src = source; await image.decode();
      const canvas = document.createElement('canvas'); canvas.width = canvas.height = 768;
      const context = canvas.getContext('2d'); context.drawImage(image, 0, 0);
      return [...context.getImageData(150, 100, 1, 1).data];
    }, exported.ink);
    assert.deepEqual(exportSample, [255, 255, 255, 255]);
    checks.push('export excludes visible AI reference and saved project retains editable real strokes');
    assert.equal(errors.length, 0, errors.join('\n'));
    await page.screenshot({ path: path.join(root, 'logs', 'browser-v3-smoke.png'), fullPage: true });
    const report = { passed: checks.length, checks, errors, scope: 'Synthetic whole-reference UI transport and ink preservation; no model quality claim' };
    fs.writeFileSync(path.join(root, 'logs', 'browser-v3-smoke.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
  } finally { await context.close(); }
})().catch(error => { console.error(error); process.exit(1); });
