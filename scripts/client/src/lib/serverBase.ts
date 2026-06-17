// Configurable backend base URL. Lets the client run anywhere (local `vite dev`,
// a static host) and connect to a remote FluxRT backend DIRECTLY — bypassing the
// vite dev proxy. Empty = same origin (dev proxy / server-bundled client, the
// legacy behavior). Set via localStorage 'fluxrt_server' or the Server field.
//
// When set to a cross-origin URL the backend must send CORS headers (run_webrtc.py
// CORSMiddleware) and be WebRTC-reachable (public IP, or STUN/TURN).

const KEY = 'fluxrt_server'

/** Configured backend origin (no trailing slash), or '' for same-origin. */
export function serverBase(): string {
  try {
    return (localStorage.getItem(KEY) || '').trim().replace(/\/+$/, '')
  } catch {
    return ''
  }
}

export function setServerBase(url: string): void {
  try {
    const v = url.trim().replace(/\/+$/, '')
    if (v) localStorage.setItem(KEY, v)
    else localStorage.removeItem(KEY)
  } catch {
    /* localStorage unavailable — ignore */
  }
}

/** Prefix a server path with the configured base (same-origin when unset). */
export function api(path: string): string {
  return serverBase() + path
}
