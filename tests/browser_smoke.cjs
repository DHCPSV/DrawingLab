const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
process.env.TEMP = process.env.TMP = path.join(root, '.tmp');
process.env.PLAYWRIGHT_BROWSERS_PATH = path.join(root, '.cache', 'playwright');

(async () => {
  const context = await chromium.launchPersistentContext(path.join(root, '.cache', 'browser-smoke-profile'), {
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless: true, viewport: { width: 1440, height: 1024 },
    args: ['--disable-crash-reporter', '--no-first-run', '--no-default-browser-check'],
  });
  const page = context.pages()[0];
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const checks = [];
  try {
    await page.goto(process.env.DRAWINGLAB_TEST_URL || 'http://127.0.0.1:8765');
    await page.waitForFunction(() => window.DrawingLab?.debugState().session);
    await page.locator('#auto-suggest').uncheck();
    await page.locator('#prompt').fill('a manga girl, short hair, standing, simple white shirt');
    const box = await page.locator('#ink').boundingBox();
    const point = (x, y) => ({ x: box.x + x * box.width / 768, y: box.y + y * box.height / 768 });
    let p = point(200, 300); await page.mouse.move(p.x, p.y); await page.mouse.down();
    p = point(500, 300); await page.mouse.move(p.x, p.y, { steps: 30 }); await page.mouse.up();
    const drawn = await page.evaluate(() => window.DrawingLab.debugState());
    assert.equal(drawn.strokes, 1); assert.ok(drawn.alphaPixels > 600);
    assert.equal(await page.locator('#tool-description').textContent(), '黑色画笔 · 3 px');
    const colored = await page.evaluate(() => {
      const pixels = document.querySelector('#ink').getContext('2d').getImageData(0,0,768,768).data;
      for (let i=0;i<pixels.length;i+=4) if(pixels[i+3] && (pixels[i] || pixels[i+1] || pixels[i+2])) return true;
      return false;
    });
    assert.equal(colored, false); checks.push('black fixed-width pen / scaled canvas pointer input');
    await page.locator('#tool-eraser').click();
    await page.locator('#eraser-size').fill('60');
    p = point(350,300); await page.mouse.move(p.x,p.y); await page.mouse.down(); await page.mouse.up();
    const erased = await page.evaluate(() => window.DrawingLab.debugState());
    assert.ok(erased.alphaPixels < drawn.alphaPixels); checks.push('resizable eraser removes actual ink');
    await page.locator('#undo').click();
    assert.equal((await page.evaluate(() => window.DrawingLab.debugState())).inkhash, drawn.inkhash);
    await page.locator('#redo').click();
    assert.equal((await page.evaluate(() => window.DrawingLab.debugState())).inkhash, erased.inkhash);
    checks.push('undo and redo reproduce exact ink');
    const saveResponse = page.waitForResponse(r => r.url().endsWith('/api/projects') && r.request().method()==='POST');
    await page.locator('#save-project').click();
    const saved = await (await saveResponse).json();
    assert.ok(saved.path.startsWith(path.join(root,'data','projects'))); assert.ok(fs.existsSync(saved.path));
    const savedDoc = JSON.parse(fs.readFileSync(saved.path, 'utf8')).document;
    assert.equal(savedDoc.strokes.length,2); checks.push('editable project saved only inside project');
    const exportResponse = page.waitForResponse(r => r.url().endsWith('/api/exports'));
    await page.locator('#export-png').click();
    const exported = await (await exportResponse).json();
    assert.ok(exported.path.startsWith(path.join(root,'data','exports'))); assert.ok(fs.existsSync(exported.path));
    checks.push('artwork PNG exported inside project');
    await page.locator('#open-project').click();
    await page.locator('#project-list button').first().click();
    assert.equal((await page.evaluate(() => window.DrawingLab.debugState())).inkhash, erased.inkhash);
    checks.push('saved project opens with exact same ink');
    assert.equal((await page.evaluate(() => window.DrawingLab.debugState())).ghostVisible, false);
    checks.push('no fabricated reference when model is unavailable or unused');
    await page.screenshot({path:path.join(root,'logs','browser-smoke.png'),fullPage:true});
    await page.setViewportSize({width:980,height:820});
    const resized = await page.locator('#ink').boundingBox();
    assert.ok(resized.width > 300 && Math.abs(resized.width - resized.height) < 2);
    assert.equal(errors.length,0,errors.join('\n')); checks.push('responsive square canvas / no browser runtime errors');
    fs.writeFileSync(path.join(root,'logs','browser-smoke.json'), JSON.stringify({passed:checks.length,checks,errors,project:saved.name,export:exported.name},null,2));
    console.log(JSON.stringify({passed:checks.length,checks,errors},null,2));
  } finally { await context.close(); }
})().catch(error => { console.error(error); process.exit(1); });
