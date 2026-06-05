// InputProcessor — owns all client-side input acquisition, editing, and
// compositing before frames are sent to the FluxRT pipeline over WebRTC.
//
// Pipeline:
//   getUserMedia(deviceId) -> hidden <video> -> 2D canvas (effect chain)
//     -> canvas.captureStream() -> consumed by both the RTCPeerConnection
//        sender AND the local "input preview" element.
//
// Effect chain (applied per rAF frame, in order):
//   1. mirror      — horizontal flip (scaleX(-1))
//   2. hand marker — MediaPipe PoseLandmarker; draws a colored circle (+ optional
//                    fading trail) on a chosen body landmark.
//
// Because detection runs against the already-composited canvas, the chosen
// landmark naturally tracks whichever side the user perceives, regardless of
// the mirror toggle, and the marker is baked into the stream the pipeline sees.

const POSE_BUNDLE =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs';
const POSE_WASM =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';
const POSE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/' +
  'pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

function hexToRgb(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec((hex || '').trim());
  if (!m) return '255, 60, 60';
  const n = parseInt(m[1], 16);
  return `${(n >> 16) & 0xff}, ${(n >> 8) & 0xff}, ${n & 0xff}`;
}

export class InputProcessor {
  constructor({ onStatus, onLog } = {}) {
    this.onStatus = onStatus || (() => {});
    this.onLog = onLog || (() => {});

    this.rawStream = null; // from getUserMedia
    this.canvasStream = null; // from canvas.captureStream() — sent to peer + preview
    this.hiddenVideo = null;
    this.canvas = null;
    this.ctx = null;
    this.raf = 0;

    // Effect options (mutated live by the UI; the draw loop reads them).
    this.opts = {
      mirror: false,
      marker: false,
      landmark: 15, // MediaPipe BlazePose: 15=left wrist, 16=right wrist, ...
      color: '#ff3c3c',
      size: 32,
      trail: false,
      trailLen: 20,
    };

    this.trail = []; // rolling buffer of {x, y} canvas-space points
    this.poseLandmarker = null;
    this.poseLoading = false;

    // Persistent freehand drawing layer composited on top of every frame.
    this.drawCanvas = null;
    this.drawCtx = null;
    this.draw = { color: '#ffffff', size: 6 };
    this._drawing = false;
    this._last = null;
  }

  // ── pose model (lazy) ───────────────────────────────────────────────────────
  async ensurePose() {
    if (this.poseLandmarker) return this.poseLandmarker;
    if (this.poseLoading) return null;
    this.poseLoading = true;
    this.onStatus('loading pose model...');
    try {
      const vision = await import(POSE_BUNDLE);
      const resolver = await vision.FilesetResolver.forVisionTasks(POSE_WASM);
      this.poseLandmarker = await vision.PoseLandmarker.createFromOptions(resolver, {
        baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      this.onStatus('marker: ready');
      this.onLog('Pose landmarker loaded');
    } catch (e) {
      this.onStatus('pose load error');
      this.onLog('Pose landmarker error: ' + e);
    } finally {
      this.poseLoading = false;
    }
    return this.poseLandmarker;
  }

  // ── lifecycle ───────────────────────────────────────────────────────────────
  async start(deviceId) {
    const constraints = {
      audio: false,
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      },
    };
    this.rawStream = await navigator.mediaDevices.getUserMedia(constraints);
    const [vt] = this.rawStream.getVideoTracks();
    const s = vt.getSettings();
    const W = s.width || 1280;
    const H = s.height || 720;

    this.hiddenVideo = document.createElement('video');
    this.hiddenVideo.srcObject = this.rawStream;
    this.hiddenVideo.muted = true;
    this.hiddenVideo.playsInline = true;
    await this.hiddenVideo.play();

    this.canvas = document.createElement('canvas');
    this.canvas.width = W;
    this.canvas.height = H;
    this.ctx = this.canvas.getContext('2d');

    // Persistent drawing layer at the same resolution; survives frames.
    this.drawCanvas = document.createElement('canvas');
    this.drawCanvas.width = W;
    this.drawCanvas.height = H;
    this.drawCtx = this.drawCanvas.getContext('2d');

    const draw = () => {
      this._drawFrame(W, H);
      this.raf = requestAnimationFrame(draw);
    };
    this.raf = requestAnimationFrame(draw);

    this.canvasStream = this.canvas.captureStream(30);
    if (this.opts.marker) this.ensurePose();
    return { stream: this.canvasStream, label: vt.label || 'camera', canvas: this.canvas };
  }

  stop() {
    if (this.raf) {
      cancelAnimationFrame(this.raf);
      this.raf = 0;
    }
    if (this.rawStream) {
      this.rawStream.getTracks().forEach((t) => {
        try {
          t.stop();
        } catch (_) {}
      });
      this.rawStream = null;
    }
    if (this.canvasStream) {
      this.canvasStream.getTracks().forEach((t) => {
        try {
          t.stop();
        } catch (_) {}
      });
      this.canvasStream = null;
    }
    if (this.hiddenVideo) {
      try {
        this.hiddenVideo.srcObject = null;
      } catch (_) {}
      this.hiddenVideo = null;
    }
    this.canvas = null;
    this.ctx = null;
    this.drawCanvas = null;
    this.drawCtx = null;
    this._drawing = false;
    this._last = null;
    this.trail.length = 0;
  }

  get outputStream() {
    return this.canvasStream;
  }

  get active() {
    return !!this.canvasStream;
  }

  // ── option setters ──────────────────────────────────────────────────────────
  setMirror(v) {
    this.opts.mirror = !!v;
  }
  async setMarkerEnabled(v) {
    this.opts.marker = !!v;
    if (!v) this.trail.length = 0;
    else await this.ensurePose();
  }
  setLandmark(n) {
    this.opts.landmark = parseInt(n, 10) | 0;
    this.trail.length = 0;
  }
  setColor(c) {
    this.opts.color = c;
  }
  setSize(px) {
    this.opts.size = parseInt(px, 10) || 32;
  }
  setTrail(v) {
    this.opts.trail = !!v;
    if (!v) this.trail.length = 0;
  }
  setTrailLen(n) {
    this.opts.trailLen = parseInt(n, 10) || 20;
  }

  // ── per-frame compositing ─────────────────────────────────────────────────
  _drawFrame(W, H) {
    if (!this.hiddenVideo || !this.ctx) return;
    const ctx = this.ctx;
    const o = this.opts;

    // 1. base frame (optionally mirrored)
    if (o.mirror) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(this.hiddenVideo, -W, 0, W, H);
      ctx.restore();
    } else {
      ctx.drawImage(this.hiddenVideo, 0, 0, W, H);
    }

    // 2. hand marker
    if (o.marker && this.poseLandmarker) {
      let cx = null,
        cy = null;
      try {
        const res = this.poseLandmarker.detectForVideo(this.canvas, performance.now());
        const lms = res && res.landmarks && res.landmarks[0];
        if (lms) {
          const lm = lms[o.landmark];
          if (lm && (lm.visibility === undefined || lm.visibility > 0.5)) {
            cx = lm.x * W;
            cy = lm.y * H;
          }
        }
      } catch (_) {
        // model mid-init / canvas not ready — skip this frame
      }

      const rgb = hexToRgb(o.color);
      const baseR = o.size;
      const maxTrail = o.trail ? o.trailLen : 0;

      if (cx !== null && o.trail) {
        this.trail.push({ x: cx, y: cy });
        while (this.trail.length > maxTrail) this.trail.shift();
      } else if (!o.trail) {
        this.trail.length = 0;
      }

      if (o.trail && this.trail.length > 1) {
        for (let i = 0; i < this.trail.length; i++) {
          const p = this.trail[i];
          const t = (i + 1) / this.trail.length; // newest = 1
          const r = baseR * (0.35 + 0.65 * t);
          ctx.beginPath();
          ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${rgb}, ${0.15 + 0.55 * t})`;
          ctx.fill();
        }
      }

      if (cx !== null) {
        ctx.beginPath();
        ctx.arc(cx, cy, baseR, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${rgb}, 0.9)`;
        ctx.fill();
        ctx.lineWidth = Math.max(2, baseR * 0.1);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
        ctx.stroke();
      }
    } else if (this.trail.length) {
      this.trail.length = 0;
    }

    // 3. freehand drawing layer (always topmost)
    if (this.drawCanvas) ctx.drawImage(this.drawCanvas, 0, 0, W, H);
  }

  // ── freehand drawing ────────────────────────────────────────────────────────
  _toCanvas(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect();
    return {
      x: ((clientX - r.left) / r.width) * this.canvas.width,
      y: ((clientY - r.top) / r.height) * this.canvas.height,
    };
  }
  beginStroke(clientX, clientY) {
    if (!this.drawCtx) return;
    this._drawing = true;
    const p = this._toCanvas(clientX, clientY);
    this._last = p;
    const c = this.drawCtx;
    c.beginPath();
    c.fillStyle = this.draw.color;
    c.arc(p.x, p.y, this.draw.size / 2, 0, Math.PI * 2);
    c.fill();
  }
  moveStroke(clientX, clientY) {
    if (!this._drawing || !this.drawCtx) return;
    const p = this._toCanvas(clientX, clientY);
    const c = this.drawCtx;
    c.strokeStyle = this.draw.color;
    c.lineWidth = this.draw.size;
    c.lineCap = 'round';
    c.lineJoin = 'round';
    c.beginPath();
    c.moveTo(this._last.x, this._last.y);
    c.lineTo(p.x, p.y);
    c.stroke();
    this._last = p;
  }
  endStroke() {
    this._drawing = false;
    this._last = null;
  }
  clearDrawing() {
    if (this.drawCtx) this.drawCtx.clearRect(0, 0, this.drawCanvas.width, this.drawCanvas.height);
  }
  setDrawColor(c) {
    this.draw.color = c;
  }
  setDrawSize(n) {
    this.draw.size = parseInt(n, 10) || 6;
  }
}
