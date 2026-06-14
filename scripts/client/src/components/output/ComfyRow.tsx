import { useReferenceStore } from '../../state/referenceStore'

export function ComfyRow() {
  const r = useReferenceStore()
  const noServers = r.comfyServers.length === 0
  return (
    <>
      <div className={'ref' + (r.enabled ? '' : ' disabled')}>
        <label className="dim">Comfy server:</label>
        <select
          style={{ minWidth: 120 }}
          value={r.comfyServer}
          onChange={(e) => r.setComfyServer(e.target.value)}
        >
          {noServers ? (
            <option value="">(none configured)</option>
          ) : (
            r.comfyServers.map((s) => (
              <option key={s.name} value={s.name}>
                {`${s.name} (${s.url})`}
              </option>
            ))
          )}
        </select>
        <button
          disabled={r.comfyBusy || noServers}
          title="Snap the input frame, run Qwen-Image-Edit on it, use as reference"
          onClick={() => void r.doComfyEdit()}
        >
          📸 Snap → Qwen edit
        </button>
        <button disabled={r.comfyBusy || noServers} onClick={() => void r.doComfyPull()}>
          Pull latest output → reference
        </button>
        {r.comfyBusy && <span className="spinner" />}
        <span className="dim">{r.comfyStatus}</span>
      </div>
      <div className="controls">
        <input
          type="text"
          className="comfy-prompt"
          placeholder="Qwen edit prompt (sent with Snap → Qwen edit)"
          value={r.comfyPrompt}
          onChange={(e) => r.setComfyPrompt(e.target.value)}
        />
      </div>
    </>
  )
}
