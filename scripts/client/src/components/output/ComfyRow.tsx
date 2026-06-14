import { useReferenceStore } from '../../state/referenceStore'

export function ComfyRow() {
  const r = useReferenceStore()
  const noServers = r.comfyServers.length === 0
  const hasServer = !!r.comfyServer
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
            <>
              <option value="">— select —</option>
              {r.comfyServers.map((s) => (
                <option key={s.name} value={s.name}>
                  {`${s.name} (${s.url})`}
                </option>
              ))}
            </>
          )}
        </select>
        {/* Comfy controls only appear once a server is selected. */}
        {hasServer && (
          <>
            <button
              className="icon-btn"
              disabled={r.comfyBusy}
              title="Snap the input frame, run Qwen-Image-Edit on it, use as reference"
              aria-label="Snap and Qwen-edit"
              onClick={() => void r.doComfyEdit()}
            >
              📸
            </button>
            <button
              className="icon-btn"
              disabled={r.comfyBusy}
              title="Pull latest ComfyUI output as reference"
              aria-label="Pull latest output as reference"
              onClick={() => void r.doComfyPull()}
            >
              ⬇
            </button>
            {r.comfyBusy && <span className="spinner" />}
            <span className="dim">{r.comfyStatus}</span>
          </>
        )}
      </div>
      {hasServer && (
        <div className="controls">
          <input
            type="text"
            className="comfy-prompt"
            placeholder="Qwen edit prompt (sent with Snap → Qwen edit)"
            value={r.comfyPrompt}
            onChange={(e) => r.setComfyPrompt(e.target.value)}
          />
        </div>
      )}
    </>
  )
}
