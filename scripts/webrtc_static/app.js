// FluxRT WebRTC client — main wiring.
// Input acquisition/compositing is delegated to InputProcessor; this file
// handles tabs, the RTCPeerConnection, generation controls, reference images,
// lip transfer, and the stats readout.

import { InputProcessor } from '/static/input_processor.js';

const $ = (id) => document.getElementById(id);

// ── elements ──────────────────────────────────────────────────────────────
const v = $('v');
const invSlot = $('invSlot');
const stage = $('stage');
const inputToolbar = $('inputToolbar');
const tbBrush = $('tbBrush');
const tbErase = $('tbErase');
const tbColor = $('tbColor');
const tbSize = $('tbSize');
const tbSizeLbl = $('tbSizeLbl');
const tbClear = $('tbClear');
const statusEl = $('status');
const fpsBar = $('fpsBar');
const logEl = $('log');

const startBtn = $('start');
const stopBtn = $('stop');
const showInput = $('showInput');

const promptIn = $('prompt');
const seedIn = $('seed');
const stepsIn = $('steps');

const comfySelect = $('comfySelect');
const comfyPullBtn = $('comfyPull');
const comfyEditBtn = $('comfyEdit');
const comfyStatus = $('comfyStatus');

const drop = $('drop');
const fileIn = $('file');
const preview = $('preview');
const refMeta = $('refMeta');
const clearRefBtn = $('clearRef');
const refRow = $('refRow');

const lipXfer = $('lipXfer');
const lipStatus = $('lipStatus');

const useCam = $('useCam');
const camSelect = $('camSelect');
const flipInput = $('flipInput');
const inputStatus = $('inputStatus');

const inputView = $('inputView');
const drawEnable = $('drawEnable');
const drawColor = $('drawColor');
const drawSize = $('drawSize');
const drawSizeLbl = $('drawSizeLbl');
const drawClear = $('drawClear');

const handMarker = $('handMarker');
const markerLandmark = $('markerLandmark');
const markerColor = $('markerColor');
const markerSize = $('markerSize');
const markerSizeLbl = $('markerSizeLbl');
const trailToggle = $('trailToggle');
const trailLen = $('trailLen');
const trailLenLbl = $('trailLenLbl');
const poseStatus = $('poseStatus');

// ── state ─────────────────────────────────────────────────────────────────
let pc = null;
let ch = null;
let lastSeenRefVersion = 0;

const input = new InputProcessor({
  onStatus: (t) => (poseStatus.textContent = t),
  onLog: (m) => logLine(m),
});

// ── helpers ─────────────────────────────────────────────────────────────────
function logLine(s) {
  const t = new Date().toLocaleTimeString();
  logEl.textContent += `[${t}] ${s}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}
function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = cls || '';
}

// ── tabs ──────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.tab-panel').forEach((p) =>
      p.classList.toggle('active', p.dataset.tab === tab)
    );
    placeInputCanvas();
  });
});

function outputTabActive() {
  const p = document.querySelector('.tab-panel[data-tab="output"]');
  return p && p.classList.contains('active');
}

// The single compositing canvas lives wherever it's visible: the Output-tab
// stage (split view, with the edit toolbar) when "Show input preview" is on
// and the Output tab is active; otherwise parked in the Input tab.
function placeInputCanvas() {
  const canvas = input.canvasEl;
  if (!canvas) {
    stage.classList.remove('split');
    return;
  }
  if (outputTabActive() && showInput.checked) {
    if (canvas.parentElement !== invSlot) invSlot.appendChild(canvas);
    stage.classList.add('split');
  } else {
    if (canvas.parentElement !== inputView) inputView.appendChild(canvas);
    stage.classList.remove('split');
  }
}
showInput.addEventListener('change', placeInputCanvas);

// ── centralized draw state (shared by input-tab controls + stage toolbar) ───
let drawMode = 'off'; // 'off' | 'brush' | 'eraser'

function setDrawMode(mode) {
  drawMode = mode;
  input.setEraser(mode === 'eraser');
  drawEnable.checked = mode !== 'off';
  tbBrush.classList.toggle('active', mode === 'brush');
  tbErase.classList.toggle('active', mode === 'eraser');
}
function setDrawColorAll(c) {
  input.setDrawColor(c);
  drawColor.value = c;
  tbColor.value = c;
}
function setDrawSizeAll(n) {
  input.setDrawSize(n);
  drawSize.value = n;
  drawSizeLbl.textContent = n + 'px';
  tbSize.value = n;
  tbSizeLbl.textContent = String(n);
}

// ── input pipeline (camera + preview + draw), independent of WebRTC ─────────
async function startInputPipeline() {
  const { canvas, label } = await input.start(camSelect.value || null);
  logLine('Camera pipeline started: ' + label);
  inputView.innerHTML = '';
  inputView.appendChild(canvas);
  bindDrawing(canvas);
  drawEnable.disabled = false;
  drawColor.disabled = false;
  drawSize.disabled = false;
  drawClear.disabled = false;
  showInput.disabled = false;
  placeInputCanvas();
}

function stopInputPipeline() {
  input.stop();
  inputView.innerHTML =
    '<div class="dim" style="padding:24px;">Enable your camera to preview &amp; draw on the input.</div>';
  drawEnable.disabled = true;
  drawColor.disabled = true;
  drawSize.disabled = true;
  drawClear.disabled = true;
  showInput.checked = false;
  showInput.disabled = true;
  setDrawMode('off');
  stage.classList.remove('split');
}

function bindDrawing(canvas) {
  canvas.onpointerdown = (e) => {
    if (drawMode === 'off') return;
    canvas.setPointerCapture(e.pointerId);
    input.beginStroke(e.clientX, e.clientY);
  };
  canvas.onpointermove = (e) => {
    if (drawMode === 'off') return;
    input.moveStroke(e.clientX, e.clientY);
  };
  canvas.onpointerup = () => input.endStroke();
  canvas.onpointercancel = () => input.endStroke();
  canvas.onpointerleave = () => input.endStroke();
}

// Input-tab draw controls
drawEnable.addEventListener('change', () => setDrawMode(drawEnable.checked ? 'brush' : 'off'));
drawColor.addEventListener('input', () => setDrawColorAll(drawColor.value));
drawSize.addEventListener('input', () => setDrawSizeAll(drawSize.value));
drawClear.addEventListener('click', () => input.clearDrawing());

// Stage edit toolbar (mirrors the same draw state)
tbBrush.addEventListener('click', () => setDrawMode(drawMode === 'brush' ? 'off' : 'brush'));
tbErase.addEventListener('click', () => setDrawMode(drawMode === 'eraser' ? 'off' : 'eraser'));
tbColor.addEventListener('input', () => setDrawColorAll(tbColor.value));
tbSize.addEventListener('input', () => setDrawSizeAll(tbSize.value));
tbClear.addEventListener('click', () => input.clearDrawing());

// ── WebRTC ──────────────────────────────────────────────────────────────────
async function start() {
  startBtn.disabled = true;
  setStatus('connecting...', '');

  pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
  pc.ontrack = (e) => {
    logLine('Track received');
    v.srcObject = e.streams[0];
  };
  pc.oniceconnectionstatechange = () => {
    logLine('ICE: ' + pc.iceConnectionState);
    if (['connected', 'completed'].includes(pc.iceConnectionState)) {
      setStatus('live', 'live');
      stopBtn.disabled = false;
    } else if (['failed', 'disconnected'].includes(pc.iceConnectionState)) {
      setStatus('disconnected', 'err');
    }
  };

  ch = pc.createDataChannel('ctrl');
  ch.onopen = () => logLine('Control channel open');
  ch.onmessage = (e) => onCtrlMessage(e.data);
  ch.onclose = () => logLine('Control channel closed');

  if (useCam.checked) {
    try {
      if (!input.active) await startInputPipeline();
      const [vt] = input.outputStream.getVideoTracks();
      pc.addTransceiver(vt, { direction: 'sendrecv', streams: [input.outputStream] });
    } catch (e) {
      logLine('Camera access failed: ' + e.message);
      setStatus('camera blocked', 'err');
      startBtn.disabled = false;
      useCam.checked = false;
      pc.addTransceiver('video', { direction: 'recvonly' });
    }
  } else {
    pc.addTransceiver('video', { direction: 'recvonly' });
  }

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') return resolve();
    const check = () => {
      if (pc.iceGatheringState === 'complete') {
        pc.removeEventListener('icegatheringstatechange', check);
        resolve();
      }
    };
    pc.addEventListener('icegatheringstatechange', check);
  });

  const res = await fetch('/offer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
  });
  if (!res.ok) {
    setStatus('offer rejected', 'err');
    startBtn.disabled = false;
    return;
  }
  await pc.setRemoteDescription(await res.json());
  logLine('SDP exchange complete');
}

function stop() {
  if (ch) { try { ch.close(); } catch (_) {} ch = null; }
  if (pc) { try { pc.close(); } catch (_) {} pc = null; }
  // Leave the camera pipeline running — it's bound to the Input-tab toggle,
  // not the connection, so preview + drawing survive a disconnect/reconnect.
  v.srcObject = null;
  placeInputCanvas();
  // Reset recv-fps accumulators so the next session's first sample isn't a
  // bogus delta against the previous connection's framesReceived.
  lastRecvFps = '—';
  lastFrames = null;
  lastT = null;
  setStatus('idle', '');
  startBtn.disabled = false;
  stopBtn.disabled = true;
  logLine('Stopped');
}

function sendCtrl(msg) {
  if (!ch || ch.readyState !== 'open') {
    logLine('Control channel not ready');
    return;
  }
  ch.send(msg);
}

startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);
window.addEventListener('beforeunload', stop);

// ── control-channel inbound ─────────────────────────────────────────────────
function onCtrlMessage(msg) {
  if (typeof msg !== 'string') return;
  if (msg.startsWith('ref:set:')) {
    const ver = parseInt(msg.slice('ref:set:'.length), 10);
    if (!isNaN(ver) && ver > lastSeenRefVersion) {
      lastSeenRefVersion = ver;
      refreshPreview(ver);
      logLine(`Reference updated by another client (v${ver})`);
    }
  } else if (msg.startsWith('ref:clear')) {
    const ver = parseInt(msg.split(':')[2] || '0', 10);
    if (!isNaN(ver) && ver > lastSeenRefVersion) {
      lastSeenRefVersion = ver;
      clearPreview();
      logLine(`Reference cleared by another client (v${ver})`);
    }
  } else if (msg === 'input:peer') {
    inputStatus.textContent = useCam.checked ? 'input: peer (you)' : 'input: peer (other client)';
    logLine('Pipeline input now from a peer');
  } else if (msg === 'input:server') {
    inputStatus.textContent = 'input: server';
    logLine('Pipeline input now from server camera');
  } else if (msg === 'lip:on' || msg === 'lip:off') {
    const on = msg === 'lip:on';
    if (lipXfer.checked !== on) lipXfer.checked = on;
    lipStatus.textContent = on ? 'lipsync: ON' : 'lipsync: OFF';
    logLine('Lip transfer ' + (on ? 'enabled' : 'disabled'));
  } else if (msg.startsWith('state:prompt:')) {
    const val = msg.slice('state:prompt:'.length);
    if (document.activeElement !== promptIn && promptIn.value !== val) {
      promptIn.value = val;
      logLine('Prompt synced from server');
    }
  } else if (msg.startsWith('state:seed:')) {
    const val = msg.slice('state:seed:'.length);
    if (document.activeElement !== seedIn) seedIn.value = val;
  } else if (msg.startsWith('state:steps:')) {
    const val = msg.slice('state:steps:'.length);
    if (document.activeElement !== stepsIn) stepsIn.value = val;
  } else {
    logLine('server: ' + msg);
  }
}

// ── prompt textarea (auto-grow + Enter/Shift+Enter) ─────────────────────────
function autoSizePrompt() {
  promptIn.style.height = 'auto';
  promptIn.style.height = promptIn.scrollHeight + 2 + 'px';
}
promptIn.addEventListener('input', autoSizePrompt);
autoSizePrompt();
const _pd = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
Object.defineProperty(promptIn, 'value', {
  get() { return _pd.get.call(this); },
  set(x) { _pd.set.call(this, x); autoSizePrompt(); },
  configurable: true,
});
promptIn.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
    const x = promptIn.value.trim();
    if (x) {
      e.preventDefault();
      sendCtrl('prompt:' + x);
    }
  }
});
seedIn.addEventListener('change', () => sendCtrl('seed:' + seedIn.value));
stepsIn.addEventListener('change', () => sendCtrl('steps:' + stepsIn.value));

// ── facial-feature builder bar ──────────────────────────────────────────────
// Each dropdown contributes one phrase; selecting any feature recomposes the
// prompt from all active features and sends it.
const FEATURE_ORDER = ['eyes', 'eyebrows', 'nose', 'ears', 'mouth'];
const FEATURES = {
  eyes: {
    emoji: '👁️',
    label: 'eyes',
    opts: [
      'giant googly eyes',
      'glowing neon cyber eyes',
      'heterochromia, one blue eye and one green eye',
      'huge sparkling anime eyes',
      'wise wrinkled squinting eyes',
    ],
  },
  eyebrows: {
    emoji: '🤨',
    label: 'eyebrows',
    opts: [
      'huge bushy caterpillar eyebrows',
      'thin dramatically arched eyebrows',
      'thick connected unibrow',
      'glowing painted neon eyebrows',
      'completely shaved off eyebrows',
    ],
  },
  nose: {
    emoji: '👃',
    label: 'nose',
    opts: [
      'big round red clown nose',
      'long crooked witch nose',
      'tiny upturned button nose',
      'pig snout nose',
      'golden nose ring through a wide nose',
    ],
  },
  ears: {
    emoji: '👂',
    label: 'ears',
    opts: [
      'pointy elf ears',
      'enormous floppy elephant ears',
      'furry pointed wolf ears on top of the head',
      'cybernetic robot ears with antennae',
      'stretched gauged earlobes with big hoops',
    ],
  },
  mouth: {
    emoji: '👄',
    label: 'mouth',
    opts: [
      'wide gold-tooth grin',
      'huge toothy cartoon smile',
      'bushy walrus mustache over the mouth',
      'sharp vampire fangs',
      'glowing neon lips',
    ],
  },
};

const featureState = {};
const featSelects = {};
let serverDefaultPrompt = '';

function applyFeatures() {
  const parts = FEATURE_ORDER.map((k) => featureState[k]).filter(Boolean);
  if (!parts.length) return;
  const text = 'person with ' + parts.join(', ');
  promptIn.value = text;
  if (ch && ch.readyState === 'open') ch.send('prompt:' + text);
}

function randomizeFeatures() {
  FEATURE_ORDER.forEach((k) => {
    const opts = FEATURES[k].opts;
    const pick = opts[Math.floor(Math.random() * opts.length)];
    featureState[k] = pick;
    if (featSelects[k]) featSelects[k].value = pick;
  });
  applyFeatures();
}

function resetFeatures() {
  FEATURE_ORDER.forEach((k) => {
    featureState[k] = '';
    if (featSelects[k]) featSelects[k].value = '';
  });
  const text = serverDefaultPrompt || '';
  promptIn.value = text;
  if (text && ch && ch.readyState === 'open') ch.send('prompt:' + text);
}

(function buildFeatureBar() {
  const bar = $('featBar');
  FEATURE_ORDER.forEach((k) => {
    const f = FEATURES[k];
    const sel = document.createElement('select');
    sel.className = 'feat';
    const def = document.createElement('option');
    def.value = '';
    def.textContent = `${f.emoji} ${f.label}`;
    sel.appendChild(def);
    f.opts.forEach((p) => {
      const o = document.createElement('option');
      o.value = p;
      o.textContent = p;
      sel.appendChild(o);
    });
    sel.addEventListener('change', () => {
      featureState[k] = sel.value;
      applyFeatures();
    });
    featSelects[k] = sel;
    bar.appendChild(sel);
  });

  const rand = document.createElement('button');
  rand.textContent = '🎲 Randomize';
  rand.title = 'Random feature for every slot';
  rand.addEventListener('click', randomizeFeatures);
  bar.appendChild(rand);

  const reset = document.createElement('button');
  reset.textContent = '↺ Reset';
  reset.title = 'Clear features, restore default prompt';
  reset.addEventListener('click', resetFeatures);
  bar.appendChild(reset);
})();

// ── reference image ─────────────────────────────────────────────────────────
function refreshPreview(versionLabel) {
  preview.src = '/reference?t=' + Date.now();
  preview.classList.add('shown');
  refMeta.textContent = versionLabel ? `reference v${versionLabel}` : 'reference active';
}
function clearPreview() {
  preview.classList.remove('shown');
  preview.removeAttribute('src');
  refMeta.textContent = 'no reference';
}
async function uploadReference(file) {
  if (!file || !file.type.startsWith('image/')) return logLine('Not an image file');
  if (file.size > 10 * 1024 * 1024) return logLine('File too large (>10 MB)');
  refMeta.textContent = 'uploading...';
  try {
    const r = await fetch('/reference', {
      method: 'POST',
      headers: { 'Content-Type': file.type || 'application/octet-stream' },
      body: file,
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      logLine('Reference upload failed: ' + (err.detail || r.statusText));
      refMeta.textContent = 'upload failed';
      return;
    }
    const j = await r.json();
    lastSeenRefVersion = j.version || lastSeenRefVersion;
    logLine(`Reference set: ${j.size[0]}x${j.size[1]} (v${j.version})`);
    refMeta.textContent = `active ${j.size[0]}x${j.size[1]} (v${j.version})`;
    preview.src = URL.createObjectURL(file);
    preview.classList.add('shown');
  } catch (e) {
    logLine('Reference upload error: ' + e);
    refMeta.textContent = 'error';
  }
}
drop.addEventListener('click', () => fileIn.click());
fileIn.addEventListener('change', () => { if (fileIn.files.length) uploadReference(fileIn.files[0]); });
['dragenter', 'dragover'].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); drop.classList.add('over'); })
);
['dragleave', 'drop'].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); drop.classList.remove('over'); })
);
drop.addEventListener('drop', (e) => {
  if (e.dataTransfer && e.dataTransfer.files.length) uploadReference(e.dataTransfer.files[0]);
});
window.addEventListener('paste', (e) => {
  if (!e.clipboardData) return;
  for (const item of e.clipboardData.items) {
    if (item.type.startsWith('image/')) {
      const f = item.getAsFile();
      if (f) uploadReference(f);
      break;
    }
  }
});
clearRefBtn.addEventListener('click', async () => {
  try {
    const r = await fetch('/reference', { method: 'DELETE' });
    if (r.ok) {
      const j = await r.json().catch(() => ({}));
      if (j.version) lastSeenRefVersion = j.version;
      clearPreview();
      logLine('Reference cleared' + (j.version ? ` (v${j.version})` : ''));
    }
  } catch (e) {
    logLine('Clear error: ' + e);
  }
});

// ── comfy puller ────────────────────────────────────────────────────────────
async function loadComfyServers() {
  try {
    const r = await fetch('/comfy/servers');
    const j = await r.json();
    comfySelect.innerHTML = '';
    if (!j.servers || j.servers.length === 0) {
      const o = document.createElement('option');
      o.value = ''; o.textContent = '(none configured)';
      comfySelect.appendChild(o);
      comfyPullBtn.disabled = true;
      return;
    }
    j.servers.forEach((s) => {
      const o = document.createElement('option');
      o.value = s.name; o.textContent = `${s.name} (${s.url})`;
      comfySelect.appendChild(o);
    });
  } catch (e) {
    logLine('Comfy server list error: ' + e);
  }
}
comfyEditBtn.addEventListener('click', async () => {
  const name = comfySelect.value;
  if (!name) {
    comfyStatus.textContent = 'pick a comfy server first';
    return;
  }
  if (!input.active || !input.canvasEl) {
    comfyStatus.textContent = 'enable your camera (Input tab) first';
    return;
  }
  comfyEditBtn.disabled = true;
  comfyStatus.textContent = `snapping → Qwen edit on ${name}...`;
  try {
    const blob = await new Promise((res) => input.canvasEl.toBlob(res, 'image/png'));
    const prompt = promptIn.value.trim();
    const r = await fetch(
      '/comfy/edit?server=' + encodeURIComponent(name) + '&prompt=' + encodeURIComponent(prompt),
      { method: 'POST', headers: { 'Content-Type': 'image/png' }, body: blob }
    );
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      comfyStatus.textContent = 'edit error: ' + (err.detail || r.statusText);
      logLine('Qwen edit failed: ' + (err.detail || r.statusText));
      return;
    }
    const j = await r.json();
    lastSeenRefVersion = j.version || lastSeenRefVersion;
    comfyStatus.textContent = `qwen edit → reference (v${j.version})`;
    refreshPreview(j.version);
    logLine(`Qwen edit done: ${j.filename} (v${j.version})`);
  } catch (e) {
    comfyStatus.textContent = 'edit error: ' + e.message;
    logLine('Qwen edit error: ' + e);
  } finally {
    comfyEditBtn.disabled = false;
  }
});

comfyPullBtn.addEventListener('click', async () => {
  const name = comfySelect.value;
  if (!name) return;
  comfyPullBtn.disabled = true;
  comfyStatus.textContent = `pulling from ${name}...`;
  try {
    const r = await fetch('/comfy/pull?server=' + encodeURIComponent(name), { method: 'POST' });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      comfyStatus.textContent = 'error: ' + (err.detail || r.statusText);
      logLine('Comfy pull failed: ' + (err.detail || r.statusText));
      return;
    }
    const j = await r.json();
    lastSeenRefVersion = j.version || lastSeenRefVersion;
    comfyStatus.textContent = `pulled ${j.filename} (v${j.version})`;
    refreshPreview(j.version);
    logLine(`Comfy pulled: ${j.filename} from ${name} (v${j.version})`);
  } catch (e) {
    comfyStatus.textContent = 'error: ' + e.message;
    logLine('Comfy pull error: ' + e);
  } finally {
    comfyPullBtn.disabled = false;
  }
});

// ── lip transfer ────────────────────────────────────────────────────────────
lipXfer.addEventListener('change', async () => {
  const on = lipXfer.checked;
  lipXfer.disabled = true;
  try {
    const r = await fetch('/lip-transfer?on=' + (on ? 'true' : 'false'), { method: 'POST' });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      logLine('Lip transfer toggle failed: ' + (err.detail || r.statusText));
      lipXfer.checked = !on;
      return;
    }
    const j = await r.json();
    lipStatus.textContent = j.lip_active ? 'lipsync: ON' : 'lipsync: OFF';
  } catch (e) {
    logLine('Lip transfer error: ' + e);
    lipXfer.checked = !on;
  } finally {
    lipXfer.disabled = false;
  }
});

// ── input tab: camera + effects → InputProcessor ────────────────────────────
async function refreshCameras() {
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    const cams = devs.filter((d) => d.kind === 'videoinput');
    camSelect.innerHTML = '';
    if (cams.length === 0) {
      const o = document.createElement('option');
      o.value = ''; o.textContent = '(no cameras — grant permission)';
      camSelect.appendChild(o);
      return;
    }
    const def = document.createElement('option');
    def.value = ''; def.textContent = 'Default camera';
    camSelect.appendChild(def);
    cams.forEach((c, i) => {
      const o = document.createElement('option');
      o.value = c.deviceId; o.textContent = c.label || `Camera ${i + 1}`;
      camSelect.appendChild(o);
    });
  } catch (e) {
    logLine('Camera enumeration error: ' + e.message);
  }
}

useCam.addEventListener('change', async () => {
  camSelect.disabled = !useCam.checked;
  flipInput.disabled = !useCam.checked;
  handMarker.disabled = !useCam.checked;
  if (!useCam.checked) {
    flipInput.checked = false;
    handMarker.checked = false;
    poseStatus.textContent = '';
    input.setMirror(false);
    input.setMarkerEnabled(false);
    stopInputPipeline();
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    logLine('getUserMedia unavailable — needs HTTPS or a secure-origin allowlist.');
    useCam.checked = false;
    camSelect.disabled = true;
    flipInput.disabled = true;
    handMarker.disabled = true;
    return;
  }
  try {
    const tmp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    tmp.getTracks().forEach((t) => t.stop());
  } catch (e) {
    logLine('Camera permission denied: ' + e.message);
    useCam.checked = false;
    camSelect.disabled = true;
    flipInput.disabled = true;
    handMarker.disabled = true;
    return;
  }
  // Default to mirrored (selfie) view when the user enables their own camera.
  flipInput.checked = true;
  input.setMirror(true);
  await refreshCameras();
  // Start the live preview/draw pipeline immediately (before WebRTC connect).
  try {
    await startInputPipeline();
  } catch (e) {
    logLine('Camera start failed: ' + e.message);
    useCam.checked = false;
    camSelect.disabled = true;
    flipInput.disabled = true;
    handMarker.disabled = true;
  }
});

// Switching camera mid-session restarts the pipeline on the new device.
camSelect.addEventListener('change', async () => {
  if (!useCam.checked || !input.active) return;
  input.stop();
  try {
    await startInputPipeline();
  } catch (e) {
    logLine('Camera switch failed: ' + e.message);
  }
});

flipInput.addEventListener('change', () => input.setMirror(flipInput.checked));

handMarker.addEventListener('change', async () => {
  if (handMarker.checked) {
    poseStatus.textContent = 'marker: ON';
    await input.setMarkerEnabled(true);
  } else {
    poseStatus.textContent = 'marker: OFF';
    input.setMarkerEnabled(false);
  }
});
markerLandmark.addEventListener('change', () => input.setLandmark(markerLandmark.value));
markerColor.addEventListener('input', () => input.setColor(markerColor.value));
markerSize.addEventListener('input', () => {
  markerSizeLbl.textContent = markerSize.value + 'px';
  input.setSize(markerSize.value);
});
trailToggle.addEventListener('change', () => input.setTrail(trailToggle.checked));
trailLen.addEventListener('input', () => {
  trailLenLbl.textContent = trailLen.value;
  input.setTrailLen(trailLen.value);
});

if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener('devicechange', () => {
    if (useCam.checked) refreshCameras();
  });
}

// ── health probe ────────────────────────────────────────────────────────────
async function probeHealth() {
  try {
    const r = await fetch('/healthz');
    const j = await r.json();
    if (!j.reference_enabled) {
      refRow.classList.add('disabled');
      $('comfyRow').classList.add('disabled');
      refMeta.textContent = 'disabled in config';
      drop.textContent = 'Reference disabled — start server with --config configs/config_with_reference.json';
    } else if (j.reference_set) {
      lastSeenRefVersion = j.reference_version || 0;
      refreshPreview(j.reference_version);
    }
    if (j.prompt && !serverDefaultPrompt) serverDefaultPrompt = j.prompt;
    inputStatus.textContent = j.input_source === 'peer' ? 'input: peer (other client)' : 'input: server';
    if (j.lip_enabled) {
      lipXfer.disabled = false;
      lipXfer.checked = !!j.lip_active;
      lipStatus.textContent = j.lip_active ? 'lipsync: ON' : 'lipsync: OFF';
    } else {
      lipXfer.disabled = true;
      lipXfer.checked = false;
      lipStatus.textContent = 'lipsync: unavailable (add lip_transfer to config)';
    }
  } catch (_) {}
}
probeHealth();
loadComfyServers();

// ── stats readout (navbar) ──────────────────────────────────────────────────
let lastRecvFps = '—';
let lastFrames = null;
let lastT = null;
const perf = { pipe: '—', interp: '—', proc: '—', vram: '—', recv: '—' };

function renderStats() {
  fpsBar.textContent =
    `pipe ${perf.pipe} (×interp ${perf.interp})  ·  recv ${perf.recv}  ·  ` +
    `proc ${perf.proc}  ·  vram ${perf.vram}`;
}
async function pollPerf() {
  try {
    const r = await fetch('/healthz', { cache: 'no-store' });
    if (r.ok) {
      const j = await r.json();
      perf.pipe = j.fps_pipeline ? j.fps_pipeline.toFixed(1) : '—';
      perf.interp = j.fps_interpolated ? j.fps_interpolated.toFixed(1) : '—';
      perf.proc = j.proc_time_ms ? j.proc_time_ms.toFixed(0) + 'ms' : '—';
      perf.vram = j.vram_mb ? (j.vram_mb / 1024).toFixed(1) + 'GB' : '—';
    }
  } catch (_) {}
  if (pc) {
    try {
      const stats = await pc.getStats();
      stats.forEach((rep) => {
        if (rep.type === 'inbound-rtp' && rep.kind === 'video') {
          if (typeof rep.framesPerSecond === 'number') {
            lastRecvFps = rep.framesPerSecond.toFixed(1);
          } else if (typeof rep.framesReceived === 'number' && typeof rep.timestamp === 'number') {
            if (lastFrames !== null && rep.timestamp > lastT) {
              const dt = (rep.timestamp - lastT) / 1000;
              if (dt > 0) lastRecvFps = ((rep.framesReceived - lastFrames) / dt).toFixed(1);
            }
            lastFrames = rep.framesReceived;
            lastT = rep.timestamp;
          }
        }
      });
    } catch (_) {}
    perf.recv = lastRecvFps;
  } else {
    perf.recv = '—';
  }
  renderStats();
}
setInterval(pollPerf, 1000);
pollPerf();
