(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const SIZE = 768;
  const ink = $('ink'), ctx = ink.getContext('2d', { willReadFrequently: true });
  const ghost = $('ghost'), ghostCtx = ghost.getContext('2d');
  const rawCanvas = $('ai-original'), rawCtx = rawCanvas.getContext('2d');
  const state = {
    session: null, revision: 0, intent: 0, barrier: 0, serial: 0,
    strokes: [], history: [], redo: [], activeStroke: null, pointerId: null,
    proposal: null, amount: 0.35, comparing: false, requestedTask: 'completion',
    region: null, selectionStart: null, lockedIds: new Set(), correctionReady: false,
    rawReference:null, rawRevision:null, rawShown:false, rawOpacity:0.25, rawVisible:false,
    tool: 'pen', eraserSize: 30, focus: { x: 384, y: 384 },
    reference: null, referencePixels: null, candidate: null, pending: null, pinned: false, appliedSerial: -1, receivedSerial: -1,
    shown: true, opacity: 0.10, seed: 42, modelReady: false,
    sourceRevision: null, previewRevision: null, corrections: [], ghostVisible: false,
    debounce: null, polling: false, connecting: false, lastMessage: '', activeRequest: null,
  };
  let networkChain = Promise.resolve();
  function activity(message, error = false) {
    $('activity').textContent = message;
    $('activity').classList.toggle('error', error);
  }
  async function api(url, options = {}) {
    const response = await fetch(url, { ...options, headers: { 'Content-Type': 'application/json', ...(options.headers || {}) } });
    let result;
    try { result = await response.json(); } catch (_) { result = {}; }
    if (!response.ok) {
      let reason = result.message || result.detail || `请求失败 (${response.status})`;
      if (typeof reason !== 'string') reason = JSON.stringify(reason);
      const error = new Error(reason); error.status = response.status; throw error;
    }
    return result;
  }
  const post = (url, body) => api(url, { method: 'POST', body: JSON.stringify(body) });
  function enqueue(operation) {
    const run = networkChain.then(operation);
    networkChain = run.catch(() => {});
    return run;
  }
  function drawStroke(stroke, context = ctx) {
    if (!stroke.points.length) return;
    context.save();
    context.globalCompositeOperation = stroke.tool === 'eraser' ? 'destination-out' : 'source-over';
    context.strokeStyle = '#000'; context.fillStyle = '#000'; context.lineWidth = stroke.size;
    context.lineCap = 'round'; context.lineJoin = 'round';
    const first = stroke.points[0];
    context.beginPath(); context.arc(first.x, first.y, stroke.size / 2, 0, Math.PI * 2); context.fill();
    if (stroke.points.length > 1) {
      context.beginPath(); context.moveTo(first.x, first.y);
      for (let i = 1; i < stroke.points.length; i++) context.lineTo(stroke.points[i].x, stroke.points[i].y);
      context.stroke();
    }
    context.restore();
  }
  function previewStrokes() {
    if (!state.proposal || state.comparing || state.amount === 0) return state.strokes;
    const updates = state.proposal.updates;
    return state.strokes.map(stroke => {
      const update = updates.get(stroke.id);
      if (!update) return stroke;
      return { ...stroke, points: stroke.points.map((p, i) => ({
        x: p.x + state.amount * (update.target_points[i].x - p.x),
        y: p.y + state.amount * (update.target_points[i].y - p.y) })) };
    });
  }
  function realInkCanvas() {
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = SIZE;
    const context = canvas.getContext('2d');
    state.strokes.forEach(stroke => drawStroke(stroke, context));
    return canvas;
  }
  function replay() {
    ctx.clearRect(0, 0, SIZE, SIZE);
    previewStrokes().forEach(stroke => drawStroke(stroke));
    if (state.activeStroke) drawStroke(state.activeStroke);
    updateHistory();
  }
  function updateHistory() {
    $('undo').disabled = !state.history.length || !!state.activeStroke;
    $('redo').disabled = !state.redo.length || !!state.activeStroke;
    $('stroke-count').textContent = `${state.strokes.length} 笔`;
    $('canvas-empty').hidden = state.strokes.length > 0 || !!state.activeStroke;
  }
  function referenceState() {
    if (state.proposal) { $('reference-state').textContent = state.comparing ? '正在对比原稿' : `修稿预览 · ${Math.round(state.amount*100)}% · 尚未应用`; return; }
    $('reference-state').textContent = !state.shown ? '参考线已隐藏' : !state.reference ? '等待整幅参考线' : state.pinned ? '整幅参考已固定' : state.sourceRevision < state.revision ? (state.corrections.length ? '已避开修改处 · 等待更新' : '参考待更新 · 可继续画') : '整幅浅线 · 可自由修改';
  }
  const strokeId = () => crypto.randomUUID();
  function strokeBounds(stroke, margin = 64) {
    if (!stroke?.points.length) return null;
    let left = SIZE, top = SIZE, right = 0, bottom = 0;
    for (const point of stroke.points) {
      left = Math.min(left, point.x); top = Math.min(top, point.y);
      right = Math.max(right, point.x); bottom = Math.max(bottom, point.y);
    }
    const padding = margin + stroke.size / 2;
    return { x: left - padding, y: top - padding, width: right - left + 2 * padding, height: bottom - top + 2 * padding };
  }
  function nearReference(point) {
    if (!state.referencePixels) return false;
    const x = Math.round(point.x), y = Math.round(point.y), radius = 6;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const px = x + dx, py = y + dy;
        if (dx * dx + dy * dy <= radius * radius && px >= 0 && py >= 0 && px < SIZE && py < SIZE && state.referencePixels[py * SIZE + px] > 32) return true;
      }
    }
    return false;
  }
  function renderGhost() {
    ghostCtx.clearRect(0, 0, SIZE, SIZE);
    state.ghostVisible = false;
    if (state.shown && state.reference && !state.proposal && !(state.rawShown && state.rawReference)) {
      ghostCtx.save();
      ghostCtx.globalAlpha = state.opacity;
      ghostCtx.drawImage(state.reference, 0, 0, SIZE, SIZE);
      ghostCtx.globalAlpha = 1;
      // A new mark is a correction to the old hypothesis. Leave other regions
      // visible, but remove its surrounding guess until the model has updated.
      const masks = [...state.corrections];
      if (state.activeStroke && (state.activeStroke.tool === 'eraser' || state.activeStroke.conflict)) masks.push(strokeBounds(state.activeStroke));
      for (const box of masks) if (box) ghostCtx.clearRect(box.x, box.y, box.width, box.height);
      ghostCtx.globalCompositeOperation = 'destination-out';
      ghostCtx.drawImage(ink, 0, 0);
      ghostCtx.restore();
      state.ghostVisible = true;
    }
    renderRawReference(); referenceState(); updateHistory();
  }
  function renderRawReference() {
    rawCtx.clearRect(0,0,SIZE,SIZE);
    state.rawVisible = Boolean(state.rawShown && state.rawReference && !state.comparing);
    if (state.rawVisible) {
      rawCtx.save(); rawCtx.globalAlpha=state.rawOpacity;
      rawCtx.drawImage(state.rawReference,0,0,SIZE,SIZE); rawCtx.restore();
    }
    $('ai-original-status').textContent = !state.rawReference ? '生成 AI 修稿或更新参考后，可查看模型的完整构想。'
      : `${state.rawRevision===state.revision ? '本次' : '上一版'} AI 完整原图，包含尚未采用的部分；仅作叠加参考，不进入保存或导出。`;
  }
  function setRawReference(reference, result) {
    state.rawReference = reference?.canvas || null;
    state.rawRevision = reference ? (result.source_revision ?? result.revision) : null;
  }
  function clearReference(preserveRaw = false) {
    state.reference = null; state.referencePixels = null; state.candidate = null; state.pending = null;
    state.sourceRevision = null; state.previewRevision = null; state.corrections = [];
    if (!preserveRaw) { state.rawReference = null; state.rawRevision = null; }
    $('metrics').textContent = '';
    renderGhost();
  }
  function invalidate(reset = false) {
    if (!state.session) return Promise.resolve();
    const session = state.session, revision = state.revision, intent = state.intent;
    return enqueue(() => post(`/api/sessions/${session}/invalidate`, { revision, intent_revision: intent, reset }))
      .catch((error) => activity(`状态同步失败：${error.message}`, true));
  }
  function invalidateMeaning(reset = true, preserveRaw = false) {
    clearTimeout(state.debounce); state.intent++; state.barrier = state.revision;
    dismissCorrection();
    setPinned(false); clearReference(preserveRaw); return invalidate(reset);
  }
  function dismissCorrection() {
    state.proposal = null; state.comparing = false; state.requestedTask = 'completion';
    $('correction-preview').hidden = true; $('preview-badge').hidden = true;
    replay(); showSelection();
  }
  function showSelection() {
    const box = $('selection-box'), r = state.region;
    box.hidden = !r;
    if (r) {
      box.style.left = `${r.x / SIZE * 100}%`; box.style.top = `${r.y / SIZE * 100}%`;
      box.style.width = `${r.width / SIZE * 100}%`; box.style.height = `${r.height / SIZE * 100}%`;
    }
    $('selection-info').textContent = `${r ? '局部修稿' : '全图轻修'} · ${state.lockedIds.size} 笔锁定`;
    $('lock-region').disabled = !r || !!state.proposal;
  }
  const insideRegion = (p, r) => !r || (p.x >= r.x && p.y >= r.y && p.x <= r.x + r.width && p.y <= r.y + r.height);
  function receiveCorrection(result) {
    if (!compatible(result) || state.activeStroke || result.task !== state.requestedTask) return;
    const proposal = result.correction;
    if (!proposal || proposal.source_revision !== state.revision || proposal.intent_revision !== state.intent || !Array.isArray(proposal.stroke_updates)) throw new Error('修稿版本不匹配，请重新生成。');
    const originals = new Map(state.strokes.map(s => [s.id, s])), updates = new Map();
    for (const update of proposal.stroke_updates) {
      const source = originals.get(update.id);
      if (!source || source.tool !== 'pen' || state.lockedIds.has(update.id) || updates.has(update.id) || !Array.isArray(update.target_points) || !Array.isArray(update.original_points) || source.points.length !== update.target_points.length || source.points.length !== update.original_points.length) throw new Error('修稿笔画与原稿不匹配。');
      for (let i = 0; i < source.points.length; i++) {
        const p = source.points[i], target = update.target_points[i], original = update.original_points[i];
        if (!target || !original || !Number.isFinite(target.x) || !Number.isFinite(target.y) || target.x < 0 || target.y < 0 || target.x > SIZE || target.y > SIZE || Math.abs(p.x-original.x)>1e-6 || Math.abs(p.y-original.y)>1e-6 || (!insideRegion(p, state.region) && (p.x!==target.x || p.y!==target.y))) throw new Error('修稿坐标校验失败，原稿保持不变。');
      }
      updates.set(update.id, update);
    }
    clearTimeout(state.debounce);
    state.proposal = { ...proposal, updates }; state.comparing = false;
    state.appliedSerial = result.serial;
    $('correction-preview').hidden = false; $('preview-badge').hidden = false;
    const m = proposal.metrics || {}, ai = m.ai_matched_strokes || 0;
    $('correction-details').textContent = `${updates.size} 笔可微调 · ${ai ? 'AI 修形与线条整理' : '本次只做线条整理'}。0% 保留原稿，100% 为本轮建议。`;
    $('apply-correction').disabled = !updates.size || state.amount === 0;
    $('metrics').textContent = `本次修稿 ${(result.metrics?.elapsed_seconds || 0).toFixed(2)} 秒`;
    replay(); renderGhost(); showSelection(); activity(result.message); refreshStatus();
  }
  async function requestCorrection(task = 'correction', variation = false) {
    if (sameRequestInFlight(task)) return;
    if (state.activeStroke || state.selectionStart) return;
    if (!state.session) { activity('本地服务尚未连接。', true); return; }
    if (task === 'correction' && (!state.modelReady || !state.correctionReady)) { activity('AI 修稿需要 Klein；可以先使用整理线条。'); return; }
    if (!state.strokes.some(s => s.tool === 'pen')) { activity('先画几笔，再进行修稿。'); return; }
    if (state.strokes.reduce((n, s) => n + s.points.length, 0) > 100000) { activity('修稿点数超过 100,000；请先保存较小的画稿。', true); return; }
    if (variation) state.seed = (state.seed + 104729) % 2147483647;
    invalidateMeaning(); state.requestedTask = task;
    const session = state.session;
    state.activeRequest = {task, revision: state.revision, intent: state.intent};
    const request = { task, revision: state.revision, intent_revision: state.intent,
      prompt: $('prompt').value.trim() || (task === 'cleanup' ? 'line cleanup' : 'Refine the same character sketch'),
      ink: realInkCanvas().toDataURL('image/png'), strokes: state.strokes.map(s => ({ id:s.id,tool:s.tool,size:s.size,points:s.points,provenance:s.provenance || 'human' })),
      region: state.region ? {...state.region} : null, locked_ids: [...state.lockedIds], seed: state.seed,
      focus: {...state.focus} };
    activity(task === 'cleanup' ? '正在整理线条；不改变构图。' : '正在根据原稿提出修形方案；你仍可继续画。');
    enqueue(async () => {
      if (session !== state.session || request.revision !== state.revision || request.intent_revision !== state.intent) return;
      await post(`/api/sessions/${session}/suggest`, request);
    }).catch(error => { if (request.intent_revision === state.intent) { state.activeRequest = null; activity(`修稿未完成：${error.message}`, true); state.requestedTask = 'completion'; } });
  }
  $('correct-now').onclick = () => requestCorrection('correction');
  $('cleanup-now').onclick = () => requestCorrection('cleanup');
  $('correction-strength').oninput = () => {
    state.amount = Number($('correction-strength').value) / 100;
    $('correction-value').textContent = `${Math.round(state.amount*100)}%`;
    $('apply-correction').disabled = !state.proposal || state.amount === 0;
    replay(); renderGhost();
  };
  function compare(value) { state.comparing = value; replay(); renderRawReference(); referenceState(); }
  $('compare-original').onpointerdown = event => { event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId); compare(true); };
  for (const type of ['pointerup','pointercancel','lostpointercapture','blur']) $('compare-original').addEventListener(type, () => compare(false));
  $('compare-original').onkeydown = event => { if ([' ', 'Enter'].includes(event.key)) { event.preventDefault(); compare(true); } };
  $('compare-original').onkeyup = () => compare(false);
  $('cancel-correction').onclick = () => { invalidateMeaning(); showSelection(); activity('已取消修稿预览，原稿保持不变。'); };
  $('apply-correction').onclick = () => {
    if (!state.proposal || state.amount === 0 || state.activeStroke) return;
    state.comparing = false;
    const before = new Map(state.strokes.map(s => [s.id,s])), proposal = state.proposal;
    const changes = [];
    state.strokes = previewStrokes().map(s => {
      if (!proposal.updates.has(s.id)) return s;
      const original = before.get(s.id);
      const after = {...s, original_points: original.original_points || original.points, provenance:'assisted'};
      changes.push({id:s.id,before:original,after}); return after;
    });
    state.history.push({kind:'correction',proposal_id:proposal.id,amount:state.amount,changes}); state.redo = [];
    state.revision++; invalidateMeaning(true,true); replay(); showSelection();
    activity(`已应用 ${changes.length} 笔修稿；一次撤销可恢复原稿。`);
  };
  $('select-region').onclick = () => chooseTool('select');
  $('clear-region').onclick = () => { invalidateMeaning(); state.region = null; showSelection(); chooseTool('pen'); activity('修稿范围已设为全图，锁定笔画仍然保留。'); };
  $('lock-region').onclick = () => {
    if (!state.region || state.activeStroke) return;
    invalidateMeaning();
    for (const s of state.strokes) if (s.tool === 'pen' && s.points.some(p => insideRegion(p,state.region))) state.lockedIds.add(s.id);
    showSelection(); activity('选中笔画已锁定，后续修稿不会移动这些笔画。');
  };
  $('unlock-all').onclick = () => { invalidateMeaning(); state.lockedIds.clear(); showSelection(); activity('已解除全部修稿锁定。'); };
  function coordinate(event) {
    const rect = ink.getBoundingClientRect();
    return { x: Math.max(0, Math.min(SIZE, (event.clientX - rect.left) * SIZE / rect.width)), y: Math.max(0, Math.min(SIZE, (event.clientY - rect.top) * SIZE / rect.height)) };
  }
  function cursor(event) {
    const element = $('brush-cursor');
    element.hidden = state.tool !== 'eraser' || event.pointerType === 'touch';
    if (element.hidden) return;
    const rect = ink.getBoundingClientRect();
    const point = coordinate(event), ratio = rect.width / SIZE;
    element.style.width = element.style.height = `${state.eraserSize * ratio}px`;
    element.style.left = `${point.x * ratio}px`; element.style.top = `${point.y * ratio}px`;
  }
  ink.addEventListener('pointerdown', (event) => {
    if (state.activeStroke || state.selectionStart || (event.pointerType === 'mouse' && event.button !== 0)) return;
    event.preventDefault(); clearTimeout(state.debounce);
    const point = coordinate(event);
    if (state.proposal || state.requestedTask !== 'completion') invalidateMeaning(true,true);
    state.pointerId = event.pointerId;
    if (state.tool === 'select') {
      invalidateMeaning(); state.selectionStart = point; state.region = null;
      ink.setPointerCapture(event.pointerId); showSelection(); return;
    }
    state.activeStroke = { id: strokeId(), tool: state.tool, size: state.tool === 'pen' ? 3 : state.eraserSize, points: [point] };
    ink.setPointerCapture(event.pointerId);
    drawStroke(state.activeStroke); updateHistory(); cursor(event); renderGhost();
  });
  ink.addEventListener('pointermove', (event) => {
    cursor(event);
    if (state.selectionStart && event.pointerId === state.pointerId) {
      const a = state.selectionStart, b = coordinate(event);
      state.region = {x:Math.min(a.x,b.x),y:Math.min(a.y,b.y),width:Math.abs(a.x-b.x),height:Math.abs(a.y-b.y)};
      showSelection(); return;
    }
    if (!state.activeStroke) return;
    if (!state.activeStroke || event.pointerId !== state.pointerId) return;
    event.preventDefault();
    const events = event.getCoalescedEvents ? event.getCoalescedEvents() : [event];
    for (const item of events.length ? events : [event]) {
      const point = coordinate(item), points = state.activeStroke.points, last = points[points.length - 1];
      if (Math.hypot(point.x - last.x, point.y - last.y) < 0.4) continue;
      points.push(point);
      // Keep nearby trace guidance visible. Mask only actual departures from
      // the hypothesis; this never skips the subsequent model request.
      const first = points[0];
      if (state.reference && state.activeStroke.tool === 'pen' && Math.hypot(point.x - first.x, point.y - first.y) > 8 && !nearReference(point)) state.activeStroke.conflict = true;
      drawStroke({ ...state.activeStroke, points: [last, point] });
    }
    renderGhost();
  });
  function finishStroke(event) {
    if (state.selectionStart && event.pointerId === state.pointerId) {
      state.selectionStart = null; state.pointerId = null;
      if (!state.region || state.region.width < 8 || state.region.height < 8) state.region = null;
      if (ink.hasPointerCapture(event.pointerId)) ink.releasePointerCapture(event.pointerId);
      showSelection(); chooseTool('pen'); activity(state.region ? '局部范围已选定。点击 AI 修稿或整理线条。' : '选区过小，已恢复全图范围。'); return;
    }
    if (!state.activeStroke || event.pointerId !== state.pointerId) return;
    const stroke = state.activeStroke;
    if (event.type === 'pointerup') {
      const point = coordinate(event), last = stroke.points[stroke.points.length - 1];
      if (Math.hypot(point.x - last.x, point.y - last.y) > 0.4) stroke.points.push(point);
    }
    const end = stroke.points[stroke.points.length - 1], first = stroke.points[0];
    const conflict = stroke.tool === 'pen' && state.reference && (stroke.conflict || (Math.hypot(end.x - first.x, end.y - first.y) > 8 && !nearReference(end)));
    delete stroke.conflict;
    state.activeStroke = null; state.pointerId = null;
    state.strokes.push(stroke); state.history.push({kind:'draw',stroke}); state.redo = []; state.revision++;
    state.focus = { ...stroke.points[stroke.points.length - 1] };
    if (ink.hasPointerCapture(event.pointerId)) ink.releasePointerCapture(event.pointerId);
    replay();
    if (stroke.tool === 'eraser') {
      state.barrier = state.revision; setPinned(false); clearReference(); invalidate(false);
      activity('已擦除。下一次建议会以当前线稿为准。');
    } else {
      if (conflict) state.corrections.push(strokeBounds(stroke));
      applyPending(); renderGhost();
    }
    scheduleSuggestion();
  }
  ink.addEventListener('pointerup', finishStroke);
  ink.addEventListener('pointercancel', finishStroke);
  ink.addEventListener('lostpointercapture', finishStroke);
  ink.addEventListener('pointerleave', () => { $('brush-cursor').hidden = true; });
  function chooseTool(tool) {
    if (state.activeStroke) return;
    state.tool = tool;
    for (const name of ['pen', 'eraser', 'select']) {
      $(`tool-${name}`).classList.toggle('active', tool === name);
      $(`tool-${name}`).setAttribute('aria-pressed', String(tool === name));
    }
    $('eraser-settings').hidden = tool !== 'eraser';
    $('tool-description').textContent = tool === 'pen' ? '黑色画笔 · 3 px' : tool === 'select' ? '拖动框选修稿范围' : '擦除真实笔迹';
    ink.style.cursor = tool === 'eraser' ? 'none' : 'crosshair';
    $('brush-cursor').hidden = true;
    renderGhost();
  }
  $('tool-pen').onclick = () => chooseTool('pen');
  $('tool-eraser').onclick = () => chooseTool('eraser');
  $('tool-select').onclick = () => chooseTool('select');
  $('eraser-size').oninput = () => { state.eraserSize = Number($('eraser-size').value); $('eraser-value').textContent = `${state.eraserSize} px`; };
  function historyAction(redo) {
    if (state.activeStroke || state.selectionStart || !(redo ? state.redo : state.history).length) return;
    const entry = (redo ? state.redo : state.history).pop();
    if (entry.kind === 'draw') { if (redo) state.strokes.push(entry.stroke); else state.strokes.pop(); }
    else {
      const changes = new Map(entry.changes.map(c => [c.id,redo ? c.after : c.before]));
      state.strokes = state.strokes.map(s => changes.get(s.id) || s);
    }
    (redo ? state.history : state.redo).push(entry);
    state.lockedIds = new Set([...state.lockedIds].filter(id => state.strokes.some(s => s.id === id)));
    state.revision++;
    const last = state.strokes[state.strokes.length - 1];
    state.focus = last ? { ...last.points[last.points.length - 1] } : { x: 384, y: 384 };
    invalidateMeaning(); replay(); showSelection(); activity(redo ? '已重做。' : '已撤销。'); scheduleSuggestion();
  }
  $('undo').onclick = () => historyAction(false);
  $('redo').onclick = () => historyAction(true);
  document.addEventListener('keydown', (event) => {
    if (/INPUT|TEXTAREA|SELECT/.test(event.target.tagName) || event.target.isContentEditable || $('projects-dialog').open) return;
    const key = event.key.toLowerCase();
    if ((event.ctrlKey || event.metaKey) && key === 'z') { event.preventDefault(); historyAction(event.shiftKey); }
    else if ((event.ctrlKey || event.metaKey) && key === 'y') { event.preventDefault(); historyAction(true); }
    else if (!event.ctrlKey && !event.metaKey && !event.altKey && key === 'b') chooseTool('pen');
    else if (!event.ctrlKey && !event.metaKey && !event.altKey && key === 'e') chooseTool('eraser');
    else if (!event.ctrlKey && !event.metaKey && !event.altKey && key === 's') chooseTool('select');
    else if (key === 'escape') { invalidateMeaning(); state.region = null; showSelection(); chooseTool('pen'); }
  });
  function scheduleSuggestion() {
    clearTimeout(state.debounce);
    if (!$('auto-suggest').checked || state.pinned || !state.strokes.length || state.proposal || state.requestedTask !== 'completion' || state.selectionStart) return;
    state.debounce = setTimeout(() => requestSuggestion(false), 450);
  }
  function transportStrokes() {
    // RDP only simplifies a request copy; the saved and displayed pen history stays exact.
    let simplificationBudget = 1000000;
    function simplify(points) {
      if (points.length <= 2) return points;
      const keep = new Uint8Array(points.length); keep[0] = keep[points.length - 1] = 1;
      const stack = [[0, points.length - 1]];
      while (stack.length) {
        const [start, end] = stack.pop(), a = points[start], b = points[end];
        const dx = b.x - a.x, dy = b.y - a.y, squared = dx * dx + dy * dy;
        let largest = 0.35 * 0.35, index = -1;
        for (let i = start + 1; i < end; i++) {
          if (--simplificationBudget < 0) return points;
          const p = points[i], t = squared ? Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / squared)) : 0;
          const error = (p.x - a.x - t * dx) ** 2 + (p.y - a.y - t * dy) ** 2;
          if (error > largest) { largest = error; index = i; }
        }
        if (index !== -1) { keep[index] = 1; stack.push([start, index], [index, end]); }
      }
      return points.filter((_, index) => keep[index]);
    }
    let count = 0;
    const strokes = state.strokes.map((stroke) => {
      const points = simplify(stroke.points); count += points.length;
      return { id: stroke.id, tool: stroke.tool, size: stroke.size, points, provenance:stroke.provenance || 'human' };
    });
    if (strokes.length > 20000 || count > 100000) throw new Error('笔画数据超过本次建议的容量。请保存项目后，使用较小的画稿生成建议。');
    return strokes;
  }
  function sameRequestInFlight(task) {
    const active = state.activeRequest;
    return active && active.task === task && active.revision === state.revision && active.intent === state.intent;
  }
  async function requestSuggestion(force) {
    clearTimeout(state.debounce);
    if (sameRequestInFlight('completion')) return;
    if (state.activeStroke) return;
    if (!state.session) { activity('本地服务尚未连接。你仍可继续绘画，连接恢复后再保存。', true); return; }
    if (!state.modelReady) { activity('推理模型尚未就绪。绘画和导出可用；参考线需要后端模型。', true); return; }
    if (!$('prompt').value.trim()) { activity('先写下角色、动作或画风，再生成参考线。'); $('prompt').focus(); return; }
    if (!state.strokes.length) { activity('先画一条轮廓，让参考线有一个起点。'); return; }
    if (state.pinned && !force) { activity('参考线已固定。取消固定后可继续更新。'); return; }
    if (state.proposal || state.requestedTask !== 'completion') invalidateMeaning();
    if (force) { setPinned(false); state.seed = (state.seed + 104729) % 2147483647; invalidateMeaning(); }
    let strokes;
    try { strokes = transportStrokes(); } catch (error) { activity(error.message, true); return; }
    const request = { task:'completion', revision: state.revision, intent_revision: state.intent, prompt: $('prompt').value.trim(), ink: realInkCanvas().toDataURL('image/png'), strokes, lookahead: 60, focus: { ...state.focus }, eraser: state.strokes[state.strokes.length - 1]?.tool === 'eraser', force, seed: state.seed };
    const session = state.session;
    state.activeRequest = {task:'completion', revision: state.revision, intent: state.intent};
    activity('正在根据你的当前笔迹更新整幅参考线，可继续画。');
    enqueue(async () => {
      if (request.intent_revision !== state.intent || request.revision !== state.revision || request.revision < state.barrier || session !== state.session) return;
      await post(`/api/sessions/${session}/suggest`, request);
    }).catch((error) => { if (request.intent_revision === state.intent) { state.activeRequest = null; activity(`无法生成参考线：${error.message}`, true); } });
  }
  $('suggest-now').onclick = () => requestSuggestion(false);
  $('next-candidate').onclick = () => state.requestedTask === 'completion' ? requestSuggestion(true) : requestCorrection(state.requestedTask,true);
  $('auto-suggest').onchange = () => { if ($('auto-suggest').checked) scheduleSuggestion(); else clearTimeout(state.debounce); };
  $('prompt').oninput = () => { invalidateMeaning(); activity('创作意图已更新。'); scheduleSuggestion(); };
  function setPinned(value) {
    state.pinned = value;
    $('pin-reference').setAttribute('aria-pressed', String(value));
    $('pin-reference').textContent = value ? '取消固定' : '固定参考';
    if (value) { clearTimeout(state.debounce); state.pending = null; }
    referenceState();
  }
  $('pin-reference').onclick = () => {
    if (!state.reference && !state.pinned) { activity('生成整幅参考线后，才能固定。'); return; }
    setPinned(!state.pinned);
    if (!state.pinned) scheduleSuggestion();
  };
  $('show-reference').onchange = () => { state.shown = $('show-reference').checked; renderGhost(); };
  $('show-ai-original').onchange = () => { state.rawShown = $('show-ai-original').checked; $('ai-original-settings').hidden=!state.rawShown; renderGhost(); };
  $('ai-original-opacity').oninput = () => { state.rawOpacity=Number($('ai-original-opacity').value)/100; $('ai-original-value').textContent=`${Math.round(state.rawOpacity*100)}%`; renderRawReference(); };
  $('reference-opacity').oninput = () => { state.opacity = Number($('reference-opacity').value) / 100; $('opacity-value').textContent = `${Math.round(state.opacity * 100)}%`; renderGhost(); };
  function compatible(result) {
    return result.intent_revision === state.intent && Number.isInteger(result.revision) && result.revision >= state.barrier && result.revision === state.revision;
  }
  function applyPending() {
    const item = state.pending;
    if (!item || state.activeStroke || state.pinned) return;
    state.pending = null;
    if (!compatible(item.result) || item.result.serial < state.appliedSerial) return;
    state.appliedSerial = item.result.serial;
    if (item.clear) { clearReference(); setRawReference(item.raw,item.result); renderGhost(); activity(item.result.message || '本次没有新增建议。', item.result.status === 'error'); return; }
    setRawReference(item.raw,item.result);
    state.reference = item.reference.canvas; state.referencePixels = item.reference.coverage; state.candidate = item.result.candidate_id;
    state.sourceRevision = item.result.source_revision ?? item.result.revision;
    state.previewRevision = item.result.preview_revision ?? item.result.revision;
    state.corrections = [];
    const metrics = item.result.metrics || {};
    const promptWarning = metrics.prompt_warning || metrics.metadata?.prompt_warning;
    renderGhost();
    activity([item.result.message || '整幅参考线已更新。可以沿着画，也可以按你的想法修改；停笔后会重新补全。', promptWarning].filter(Boolean).join(' '));
    const seconds = metrics.total_seconds ?? metrics.elapsed_seconds ?? metrics.inference_seconds ?? (Number.isFinite(metrics.total_ms) ? metrics.total_ms / 1000 : undefined);
    $('metrics').textContent = Number.isFinite(seconds) ? `本次更新 ${seconds.toFixed(2)} 秒` : '';
    refreshStatus();
  }
  async function decodeReference(source, original = false) {
    if (typeof source !== 'string' || !source.startsWith('data:image/png;base64,') || source.length > 16 * 1024 * 1024) throw new Error('服务端参考线格式无效，请重新生成。');
    const bitmap = new Image();
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('参考线读取超时，请重新生成。')), 10000);
      bitmap.onload = () => { clearTimeout(timeout); resolve(); };
      bitmap.onerror = () => { clearTimeout(timeout); reject(new Error('参考线图像无法读取，请重新生成。')); };
      bitmap.src = source;
    });
    if (!bitmap.naturalWidth || bitmap.naturalWidth !== bitmap.naturalHeight || bitmap.naturalWidth > 2048) throw new Error('服务端参考线尺寸无效。');
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = SIZE;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(bitmap, 0, 0, SIZE, SIZE);
    // Original-reference mode preserves the full raster, including shading and
    // model additions. Only the display layer desaturates and adjusts opacity.
    if (original) return {canvas};
    // Accept opaque black/white or transparent line art while displaying only
    // black ink with alpha. The model can never introduce colour into the UI.
    const data = context.getImageData(0, 0, SIZE, SIZE), pixels = data.data, coverage = new Uint8Array(SIZE * SIZE);
    for (let i = 0; i < pixels.length; i += 4) {
      const luminance = 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
      pixels[i + 3] = Math.round(pixels[i + 3] * (1 - luminance / 255));
      coverage[i / 4] = pixels[i + 3];
      pixels[i] = pixels[i + 1] = pixels[i + 2] = 0;
    }
    context.putImageData(data, 0, 0);
    return { canvas, coverage };
  }
  async function receiveResult(result, session) {
    if (!compatible(result) || state.pinned || result.serial <= state.appliedSerial || session !== state.session || (state.pending && state.pending.result.serial > result.serial)) return;
    try {
      if (result.task && result.task !== state.requestedTask) return;
      const raw = result.raw_reference ? await decodeReference(result.raw_reference,true) : null;
      if (!compatible(result) || state.pinned || session!==state.session || result.serial<=state.appliedSerial) return;
      if (result.correction) { if (state.activeStroke) return; receiveCorrection(result); setRawReference(raw,result); renderGhost(); return; }
      if (result.status !== 'ready') { state.pending={result,clear:true,raw}; applyPending(); return; }
      const reference = await decodeReference(result.ghost);
      // Decoding is asynchronous: a stroke, erase, new prompt, or reconnect may
      // have superseded this result in the meantime.
      if (!compatible(result) || state.pinned || session !== state.session || result.serial <= state.appliedSerial || (state.pending && state.pending.result.serial > result.serial)) return;
      state.pending = { result, reference, raw }; applyPending();
    } catch (error) {
      if (compatible(result) && session === state.session && !state.pinned) { clearReference(); activity(error.message, true); }
    }
  }
  async function poll() {
    if (!state.session || state.polling) return;
    state.polling = true;
    const session = state.session;
    try {
      const result = await api(`/api/sessions/${session}/result?after=${state.serial}`);
      if (session !== state.session) return;
      state.lastMessage = '';
      if (Number.isFinite(result.serial) && result.serial >= state.serial) state.serial = result.serial;
      if (['ready','error','rejected','noop'].includes(result.status)) {
        if (compatible(result) && result.task === state.activeRequest?.task) state.activeRequest = null;
        if (result.serial > state.receivedSerial) { state.receivedSerial = result.serial; await receiveResult(result, session); }
      }
      else if ((result.status === 'queued' || result.status === 'running') && compatible(result) && !state.pinned) {
        activity(result.status === 'queued' ? '建议已排队，优先处理最新线稿…' : result.message || '正在处理当前画稿。');
      }
    } catch (error) {
      if (error.status === 404) { state.session = null; state.pending = null; connectSession(); }
      if (state.lastMessage !== error.message) { activity(`结果连接中断：${error.message}。绘画已保留在当前页面。`, true); state.lastMessage = error.message; }
    }
    finally { state.polling = false; }
  }
  async function refreshStatus() {
    try {
      const result = await api('/api/status'), ready = Boolean(result.model?.ready);
      state.modelReady = ready;
      state.correctionReady = Boolean(result.features?.correction);
      const model = result.model || {};
      const engineName = model.engine === 'klein' ? 'Klein' : model.engine === 'doodle' ? 'DoodleAssist' : '本地模型';
      const label = !ready ? `${engineName}尚未就绪` : model.phase === 'loading' ? `正在加载 ${engineName}` : model.busy ? `${engineName} 处理中 ${model.step || 0}/${model.total_steps || '…'}` : `${engineName} 已就绪`;
      $('model-status').className = `model-status ${ready ? 'ready' : 'unavailable'}`;
      $('model-status').lastElementChild.textContent = label;
      const promptWarning = result.model?.metadata?.prompt_warning || result.metadata?.prompt_warning || result.model?.prompt_warning;
      $('model-status').title = [result.model?.message || label, promptWarning].filter(Boolean).join('\n');
      if (!ready && !state.proposal) activity([result.model?.message || '后端模型尚未就绪。可以先绘画、保存项目和导出 PNG。', promptWarning].filter(Boolean).join(' '));
      if (!state.session) await connectSession();
    } catch (error) {
      state.modelReady = false; $('model-status').className = 'model-status unavailable';
      $('model-status').lastElementChild.textContent = '本地服务未连接';
      activity('无法连接本地服务。当前仍可绘画和撤销；重新连接后可保存到项目文件夹。', true);
    }
  }
  function documentData() { return { version: 1, app_version:4, width: SIZE, height: SIZE, prompt: $('prompt').value, strokes: state.strokes, edit_history: state.history, redo_history: state.redo, locked_ids:[...state.lockedIds] }; }
  $('export-png').onclick = async () => {
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = SIZE;
    const context = canvas.getContext('2d'); context.fillStyle = '#fff'; context.fillRect(0, 0, SIZE, SIZE); context.drawImage(realInkCanvas(), 0, 0);
    try { const result = await post('/api/exports', { ink: canvas.toDataURL('image/png') }); activity(`PNG 已保存到项目 data/exports：${result.name}`); }
    catch (error) { activity(`PNG 保存失败：${error.message}。请检查本地服务。`, true); }
  };
  async function saveProject() {
    if (state.activeStroke) return;
    try { const result = await post('/api/projects', { document: documentData() }); activity(`项目已保存：${result.name}`); }
    catch (error) { activity(`项目保存失败：${error.message}。请检查本地服务，当前笔迹仍保留在页面。`, true); }
  }
  $('save-project').onclick = saveProject;
  $('download-json').onclick = async () => { await saveProject(); $('projects-dialog').close(); };
  async function openProjects() {
    $('projects-dialog').showModal(); $('project-list').textContent = '正在读取…';
    try {
      const result = await api('/api/projects'); $('project-list').replaceChildren();
      if (!result.projects?.length) $('project-list').textContent = '还没有已保存的项目。';
      for (const project of result.projects || []) {
        const row = document.createElement('div'); row.className = 'project-row';
        const details = document.createElement('div'), name = document.createElement('strong'), date = document.createElement('span'), button = document.createElement('button');
        name.textContent = project.name; date.textContent = project.modified ? String(project.modified) : ''; button.textContent = '打开';
        details.append(name, date); row.append(details, button); $('project-list').append(row);
        button.onclick = async () => {
          try { const data = await api(`/api/projects/${encodeURIComponent(project.name)}`); loadDocument(data.document); $('projects-dialog').close(); }
          catch (error) { activity(`无法打开：${error.message}`, true); }
        };
      }
    } catch (_) { $('project-list').textContent = '暂时无法读取服务端项目。仍可从已有 JSON 文件导入。'; }
  }
  $('open-project').onclick = openProjects;
  $('close-projects').onclick = () => $('projects-dialog').close();
  $('projects-dialog').addEventListener('click', (event) => { if (event.target === $('projects-dialog')) { const rect = event.target.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.target.close(); } });
  $('import-json').onclick = () => $('json-file').click();
  function loadDocument(value) {
    if (!value || value.version !== 1 || value.width !== SIZE || value.height !== SIZE || typeof value.prompt !== 'string' || value.prompt.length > 2000 || !Array.isArray(value.strokes) || value.strokes.length > 20000) throw new Error('项目格式不支持，需要 768×768 的版本 1 项目，描述最多 2000 字。');
    let total = 0; const ids = new Set();
    const strokes = value.strokes.map((stroke) => {
      if (!stroke || !['pen', 'eraser'].includes(stroke.tool) || !Number.isFinite(stroke.size) || stroke.size < 1 || stroke.size > 100 || !Array.isArray(stroke.points) || !stroke.points.length) throw new Error('项目笔迹格式不正确。');
      total += stroke.points.length; if (total > 500000) throw new Error('项目笔迹过多。');
      const points = stroke.points.map((point) => {
        if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) || point.x < 0 || point.y < 0 || point.x > SIZE || point.y > SIZE) throw new Error('项目坐标不正确。');
        return { x: point.x, y: point.y };
      });
      const id = typeof stroke.id === 'string' && /^[A-Za-z0-9_-]{1,100}$/.test(stroke.id) && !ids.has(stroke.id) ? stroke.id : strokeId();
      ids.add(id);
      const result = { id, tool: stroke.tool, size: stroke.tool === 'pen' ? 3 : stroke.size, points };
      if (stroke.provenance === 'assisted') result.provenance = 'assisted';
      if (Array.isArray(stroke.original_points) && stroke.original_points.length === points.length && stroke.original_points.every(p => Number.isFinite(p?.x) && Number.isFinite(p?.y) && p.x>=0 && p.x<=SIZE && p.y>=0 && p.y<=SIZE)) result.original_points = stroke.original_points.map(p => ({x:p.x,y:p.y}));
      return result;
    });
    if (state.strokes.length && !window.confirm('打开项目会替换当前画布。请先保存需要保留的笔迹，是否继续？')) return;
    const restored = restoreHistory(value.edit_history, value.redo_history, strokes);
    state.strokes = strokes; state.history = restored.history; state.redo = restored.redo;
    state.lockedIds = new Set((Array.isArray(value.locked_ids) ? value.locked_ids : []).filter(id => ids.has(id)));
    state.region = null; state.revision++; $('prompt').value = value.prompt;
    const last = strokes[strokes.length - 1]; state.focus = last ? { ...last.points[last.points.length - 1] } : { x: 384, y: 384 };
    setPinned(false); invalidateMeaning(); replay(); showSelection(); activity('项目已打开。可以继续绘画、修稿或撤销已保存的修改。');
  }
  function restoreHistory(history, redo, strokes) {
    const fallback = {history:strokes.map(stroke => ({kind:'draw',stroke})),redo:[]};
    if (!Array.isArray(history) || !Array.isArray(redo) || history.length + redo.length > 20000) return fallback;
    let total = 0;
    function validStroke(s) {
      if (!s || typeof s.id !== 'string' || !['pen','eraser'].includes(s.tool) || !Array.isArray(s.points) || !s.points.length || !Number.isFinite(s.size) || s.size<1 || s.size>100 || (s.tool==='pen' && s.size!==3)) return false;
      total += s.points.length;
      return total <= 1000000 && s.points.every(p => Number.isFinite(p?.x) && Number.isFinite(p?.y) && p.x>=0 && p.y>=0 && p.x<=SIZE && p.y<=SIZE);
    }
    function validEntry(e) {
      if (e?.kind === 'draw') return validStroke(e.stroke);
      return e?.kind === 'correction' && Array.isArray(e.changes) && e.changes.length>0 && Number.isFinite(e.amount) && e.amount>=0 && e.amount<=1 && new Set(e.changes.map(c=>c.id)).size===e.changes.length && e.changes.every(c => c.id===c.before?.id && c.id===c.after?.id && validStroke(c.before) && validStroke(c.after));
    }
    if (![...history,...redo].every(validEntry)) return fallback;
    // Validate the edit journal by replaying it, including redo's reverse stack order.
    const current = [];
    function apply(e) {
      if (e.kind==='draw') { if (current.some(s=>s.id===e.stroke.id)) return false; current.push(e.stroke); return true; }
      for (const c of e.changes) {
        const index = current.findIndex(s=>s.id===c.id);
        if (index<0 || JSON.stringify(current[index].points)!==JSON.stringify(c.before.points)) return false;
        current[index]=c.after;
      }
      return true;
    }
    if (!history.every(apply) || current.length!==strokes.length || current.some((s,i)=>s.id!==strokes[i].id || s.tool!==strokes[i].tool || s.size!==strokes[i].size || JSON.stringify(s.points)!==JSON.stringify(strokes[i].points))) return fallback;
    if (![...redo].reverse().every(apply)) return fallback;
    return {history,redo};
  }
  $('json-file').onchange = async () => {
    const file = $('json-file').files[0]; if (!file) return;
    try { if (file.size > 30 * 1024 * 1024) throw new Error('文件超过 30 MB。'); const parsed = JSON.parse(await file.text()); loadDocument(parsed.document || parsed); $('projects-dialog').close(); }
    catch (error) { activity(`导入失败：${error.message}`, true); }
    finally { $('json-file').value = ''; }
  };
  async function connectSession() {
    if (state.session || state.connecting) return;
    state.connecting = true;
    try {
      const result = await post('/api/sessions', {});
      if (typeof result.id !== 'string' || !result.id) throw new Error('服务没有返回有效会话。');
      state.session = result.id; state.activeRequest = null; state.serial = 0; state.appliedSerial = -1; state.receivedSerial = -1;
      if (state.revision || state.intent) await invalidate(true);
      if (state.modelReady) activity('写下想画的内容，从几笔轮廓开始。停笔后生成整幅浅线，继续画会更新补全。');
    } catch (error) { activity(`会话未连接：${error.message}。当前仍可绘画，稍后自动重试。`, true); }
    finally { state.connecting = false; }
  }
  async function start() {
    await refreshStatus();
    const project = new URLSearchParams(location.search).get('project');
    if (project && /^drawing-[0-9-]+\.json$/.test(project)) {
      try { const data = await api(`/api/projects/${encodeURIComponent(project)}`); loadDocument(data.document); }
      catch (error) { activity(`未能打开指定项目：${error.message}`, true); }
    }
    setInterval(poll, 500); setInterval(refreshStatus, 10000);
  }
  Object.defineProperty(window, 'DrawingLab', { value: Object.freeze({ debugState: () => {
    const pixels = ctx.getImageData(0, 0, SIZE, SIZE).data;
    let hash = 2166136261, alphaPixels = 0;
    for (let i = 3; i < pixels.length; i += 4) { hash = Math.imul(hash ^ pixels[i], 16777619); if (pixels[i]) alphaPixels++; }
    const tip = state.activeStroke?.points.at(-1) || state.strokes.at(-1)?.points.at(-1) || state.focus;
    return Object.freeze({ version: 4, revision: state.revision, intent_revision: state.intent, candidate_id: state.candidate, serial: state.serial, appliedSerial: state.appliedSerial, drawing: !!state.activeStroke, tool: state.tool, strokes: state.strokes.length, strokeIds: Object.freeze(state.strokes.map((stroke) => stroke.id)), redo: state.redo.length, inkhash: (hash >>> 0).toString(16), alphaPixels, pinned: state.pinned, ghostVisible: state.ghostVisible, referenceOpacity: state.opacity, sourceRevision: state.sourceRevision, previewRevision: state.previewRevision, correctionRegions: state.corrections.length, correctionId:state.proposal?.id || null, correctionStrokes:state.proposal?.updates.size || 0, correctionAmount:state.amount, requestedTask:state.requestedTask, history:state.history.length, lockedIds:[...state.lockedIds], region:state.region, modelReady: state.modelReady, session: state.session, focus: Object.freeze({ ...state.focus }), tip: Object.freeze({ ...tip }) });
  } }), writable: false, configurable: false });
  updateHistory(); start();
})();
