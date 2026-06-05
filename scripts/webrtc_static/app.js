// FluxRT WebRTC client — main wiring.
// Input acquisition/compositing is delegated to InputProcessor; this file
// handles tabs, the RTCPeerConnection, generation controls, reference images,
// lip transfer, and the stats readout.

import { InputProcessor } from '/static/input_processor.js';

const $ = (id) => document.getElementById(id);

// ── elements ──────────────────────────────────────────────────────────────
const v = $('v');
const inv = $('inv');
const stage = $('stage');
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
  });
});

// ── input preview (side-by-side, output tab) ───────────────────────────────
function applyInputPreview() {
  if (showInput.checked && input.outputStream) {
    inv.srcObject = input.outputStream;
    stage.classList.add('split');
  } else {
    inv.srcObject = null;
    stage.classList.remove('split');
  }
}
showInput.addEventListener('change', applyInputPreview);

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
      const { label } = await input.start(camSelect.value || null);
      logLine('Local camera acquired: ' + label);
      const [vt] = input.outputStream.getVideoTracks();
      pc.addTransceiver(vt, { direction: 'sendrecv', streams: [input.outputStream] });
      showInput.disabled = false;
      applyInputPreview();
    } catch (e) {
      logLine('Camera access failed: ' + e.message);
      setStatus('camera blocked', 'err');
      startBtn.disabled = false;
      useCam.checked = false;
      input.stop();
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
  input.stop();
  inv.srcObject = null;
  stage.classList.remove('split');
  showInput.disabled = true;
  v.srcObject = null;
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
    showInput.checked = false;
    showInput.disabled = true;
    flipInput.checked = false;
    handMarker.checked = false;
    poseStatus.textContent = '';
    input.setMirror(false);
    input.setMarkerEnabled(false);
    applyInputPreview();
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
  await refreshCameras();
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
