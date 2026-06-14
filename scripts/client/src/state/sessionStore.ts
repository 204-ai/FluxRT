// WebRTC session + ctrl channel + log + server health. The RTCPeerConnection
// and DataChannel are module-level (long-lived, non-reactive); the store
// holds only renderable state.

import { create } from 'zustand'
import { decodeCtrl, encodeCtrl, type CtrlOut } from '../lib/ctrlProtocol'
import { getHealthz, postOffer, type Healthz } from '../lib/api'
import { rail, rtLog, setRuntimeLogger } from './runtime'
import { usePipelineStore } from './pipelineStore'
import { usePromptStore } from './promptStore'
import { useReferenceStore } from './referenceStore'
import { isFocused } from './focusRegistry'

export type SessionStatus =
  | 'idle'
  | 'connecting...'
  | 'live'
  | 'disconnected'
  | 'camera blocked'
  | 'input blocked'
  | 'offer rejected'

export type InputRole = 'you' | 'peer' | 'server'

interface PerfStats {
  pipe: string
  interp: string
  proc: string
  vram: string
  recv: string
}

interface SessionState {
  status: SessionStatus
  statusCls: '' | 'live' | 'err'
  connected: boolean
  starting: boolean
  logText: string
  inputRole: InputRole
  lipEnabled: boolean
  lipActive: boolean
  serverDefaultPrompt: string
  perf: PerfStats

  logLine(msg: string): void
  start(): Promise<void>
  stop(): void
  sendCtrl(msg: CtrlOut): boolean
  boot(): Promise<void>
  pollPerf(): Promise<void>
  setLip(enabled: boolean, active: boolean): void
}

let pc: RTCPeerConnection | null = null
let ch: RTCDataChannel | null = null
let outputSender: RTCRtpSender | null = null
const MAX_LOG_LINES = 500
let logLines: string[] = []

// recv-fps accumulators; reset on stop so the next session's first sample
// isn't a bogus delta against the previous connection's framesReceived.
let lastRecvFps = '—'
let lastFrames: number | null = null
let lastT: number | null = null

type RemoteTrackHandler = (stream: MediaStream) => void
let onRemoteTrack: RemoteTrackHandler = () => {}
export function setRemoteTrackHandler(fn: RemoteTrackHandler): void {
  onRemoteTrack = fn
}

export const useSessionStore = create<SessionState>((set, get) => ({
  status: 'idle',
  statusCls: '',
  connected: false,
  starting: false,
  logText: '',
  inputRole: 'server',
  lipEnabled: false,
  lipActive: false,
  serverDefaultPrompt: '',
  perf: { pipe: '—', interp: '—', proc: '—', vram: '—', recv: '—' },

  logLine(msg) {
    const t = new Date().toLocaleTimeString()
    logLines.push(`[${t}] ${msg}`)
    if (logLines.length > MAX_LOG_LINES) logLines = logLines.slice(-MAX_LOG_LINES)
    set({ logText: logLines.join('\n') + '\n' })
  },

  async start() {
    const { logLine } = get()
    if (pc) return
    set({ starting: true, status: 'connecting...', statusCls: '' })

    pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] })
    pc.ontrack = (e) => {
      logLine('Track received')
      onRemoteTrack(e.streams[0])
    }
    pc.oniceconnectionstatechange = () => {
      if (!pc) return
      logLine('ICE: ' + pc.iceConnectionState)
      if (['connected', 'completed'].includes(pc.iceConnectionState)) {
        set({ status: 'live', statusCls: 'live', connected: true })
      } else if (['failed', 'disconnected'].includes(pc.iceConnectionState)) {
        set({ status: 'disconnected', statusCls: 'err' })
      }
    }

    ch = pc.createDataChannel('ctrl')
    ch.onopen = () => logLine('Control channel open')
    ch.onclose = () => logLine('Control channel closed')
    ch.onmessage = (e) => dispatchCtrl(e.data)

    const pipeline = usePipelineStore.getState()
    if (pipeline.camEnabled || pipeline.videoLoaded) {
      try {
        if (!rail.active) await pipeline.startPipeline()
        const stream = rail.outputStream
        const [vt] = stream?.getVideoTracks() ?? []
        if (!vt || !stream) throw new Error('no output track')
        outputSender = pc.addTransceiver(vt, { direction: 'sendrecv', streams: [stream] }).sender
      } catch (e) {
        logLine('Input source failed: ' + (e instanceof Error ? e.message : e))
        set({ status: 'input blocked', statusCls: 'err', starting: false })
        if (pipeline.camEnabled && !pipeline.videoLoaded) {
          usePipelineStore.setState({ camEnabled: false })
        }
        pc.addTransceiver('video', { direction: 'recvonly' })
      }
    } else {
      pc.addTransceiver('video', { direction: 'recvonly' })
    }

    const offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    // Wait for ICE gathering, capped at 2s: LAN host candidates arrive
    // immediately; a slow STUN server must not stall connect.
    await new Promise<void>((resolve) => {
      if (!pc || pc.iceGatheringState === 'complete') return resolve()
      const finish = () => {
        clearTimeout(timer)
        pc?.removeEventListener('icegatheringstatechange', check)
        resolve()
      }
      const check = () => {
        if (pc?.iceGatheringState === 'complete') finish()
      }
      const timer = setTimeout(finish, 2000)
      pc.addEventListener('icegatheringstatechange', check)
    })

    try {
      const answer = await postOffer(pc.localDescription!)
      await pc.setRemoteDescription(answer)
      logLine('SDP exchange complete')
      set({ starting: false })
    } catch {
      set({ status: 'offer rejected', statusCls: 'err', starting: false })
    }
  },

  stop() {
    const { logLine } = get()
    if (ch) {
      try {
        ch.close()
      } catch {
        /* closing */
      }
      ch = null
    }
    if (pc) {
      try {
        pc.close()
      } catch {
        /* closing */
      }
      pc = null
    }
    outputSender = null
    // Camera pipeline keeps running — bound to the Input-tab toggle, not the
    // connection, so preview + drawing survive a disconnect/reconnect.
    onRemoteTrack(new MediaStream())
    lastRecvFps = '—'
    lastFrames = null
    lastT = null
    set({
      status: 'idle',
      statusCls: '',
      connected: false,
      starting: false,
      inputRole: 'server',
    })
    logLine('Stopped')
  },

  sendCtrl(msg) {
    if (!ch || ch.readyState !== 'open') {
      get().logLine('Control channel not ready')
      return false
    }
    ch.send(encodeCtrl(msg))
    return true
  },

  async boot() {
    setRuntimeLogger((m) => get().logLine(m))
    try {
      const j = await getHealthz()
      applyHealthBoot(j)
    } catch {
      /* server unreachable — UI stays idle */
    }
    void useReferenceStore.getState().loadComfyServers()
    void usePromptStore.getState().loadSavedPrompts()
  },

  async pollPerf() {
    const perf = { ...get().perf }
    try {
      const j = await getHealthz()
      perf.pipe = j.fps_pipeline ? j.fps_pipeline.toFixed(1) : '—'
      perf.interp = j.fps_interpolated ? j.fps_interpolated.toFixed(1) : '—'
      perf.proc = j.proc_time_ms ? j.proc_time_ms.toFixed(0) + 'ms' : '—'
      perf.vram = j.vram_mb ? (j.vram_mb / 1024).toFixed(1) + 'GB' : '—'
    } catch {
      /* keep last values */
    }
    if (pc) {
      try {
        const stats = await pc.getStats()
        stats.forEach((rep) => {
          if (rep.type === 'inbound-rtp' && rep.kind === 'video') {
            if (typeof rep.framesPerSecond === 'number') {
              lastRecvFps = rep.framesPerSecond.toFixed(1)
            } else if (typeof rep.framesReceived === 'number' && typeof rep.timestamp === 'number') {
              if (lastFrames !== null && lastT !== null && rep.timestamp > lastT) {
                const dt = (rep.timestamp - lastT) / 1000
                if (dt > 0) lastRecvFps = ((rep.framesReceived - lastFrames) / dt).toFixed(1)
              }
              lastFrames = rep.framesReceived
              lastT = rep.timestamp
            }
          }
        })
      } catch {
        /* stats unavailable mid-teardown */
      }
      perf.recv = lastRecvFps
    } else {
      perf.recv = '—'
    }
    set({ perf })
  },

  setLip(enabled, active) {
    set({ lipEnabled: enabled, lipActive: active })
  },
}))

function applyHealthBoot(j: Healthz): void {
  const session = useSessionStore.getState()
  const refStore = useReferenceStore.getState()

  if (!j.reference_enabled) {
    refStore.setDisabled()
  } else if (j.reference_set) {
    refStore.syncVersion(j.reference_version || 0)
  }
  if (j.prompt && !session.serverDefaultPrompt) {
    useSessionStore.setState({ serverDefaultPrompt: j.prompt })
    if (!usePromptStore.getState().prompt) usePromptStore.setState({ prompt: j.prompt })
  }
  useSessionStore.setState({ inputRole: j.input_source === 'peer' ? 'peer' : 'server' })

  if (j.lip_enabled) {
    session.setLip(true, !!j.lip_active)
  } else {
    session.setLip(false, false)
  }

  // Someone is already connected — join as a viewer: jump to Output tab and
  // connect right away (recvonly; the <video> is muted+autoplay).
  if ((j.peers ?? 0) > 0 && !pc) {
    session.logLine(`${j.peers} client(s) already connected — auto-starting viewer`)
    session.start().catch((e) => {
      session.logLine('Auto-start failed: ' + (e instanceof Error ? e.message : e))
      useSessionStore.setState({ starting: false })
    })
  }
}

function dispatchCtrl(raw: unknown): void {
  if (typeof raw !== 'string') return
  const session = useSessionStore.getState()
  const m = decodeCtrl(raw)
  switch (m.kind) {
    case 'refSet':
      useReferenceStore.getState().remoteSet(m.version)
      break
    case 'refClear':
      useReferenceStore.getState().remoteClear(m.version)
      break
    case 'inputRole':
      useSessionStore.setState({ inputRole: m.role })
      session.logLine(
        m.role === 'you'
          ? 'You are steering the pipeline input'
          : m.role === 'peer'
            ? 'Pipeline input now from a peer'
            : 'Pipeline input now from server camera',
      )
      break
    case 'lip':
      useSessionStore.setState({ lipActive: m.on })
      session.logLine('Lip transfer ' + (m.on ? 'enabled' : 'disabled'))
      break
    case 'statePrompt':
      if (!isFocused('prompt') && usePromptStore.getState().prompt !== m.text) {
        usePromptStore.setState({ prompt: m.text })
        session.logLine('Prompt synced from server')
      }
      break
    case 'stateSeed':
      if (!isFocused('seed')) usePromptStore.setState({ seed: m.value })
      break
    case 'stateSteps':
      if (!isFocused('steps')) usePromptStore.setState({ steps: m.value })
      break
    case 'promptsChanged':
      void usePromptStore.getState().loadSavedPrompts()
      break
    case 'unknown':
      session.logLine('server: ' + m.raw)
      break
  }
}

export function logToSession(msg: string): void {
  useSessionStore.getState().logLine(msg)
}

// When the rail (re)starts and builds a NEW output track, swap it onto the live
// sender so a pipeline restart (camera/video toggle while connected) doesn't
// freeze the remote on the old, ended track — replaceTrack needs no renegotiation.
rail.setOutputTrackHandler((track) => {
  if (!pc || !outputSender || !track || outputSender.track === track) return
  void outputSender.replaceTrack(track).catch((e) =>
    logToSession('output track replace failed: ' + (e instanceof Error ? e.message : e)),
  )
})

// Make rail logs visible even before boot() runs.
setRuntimeLogger(logToSession)
export { rtLog }
