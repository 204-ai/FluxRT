// Reference image — a single compact box that doubles as the drop target:
// drop / paste / click sets OR replaces the image. Sits inline next to the
// prompt input. The image itself is the dropzone; a 🗑 overlay clears it.

import { useEffect } from 'react'
import { useReferenceStore } from '../../state/referenceStore'
import { useFileDrop } from '../../lib/useFileDrop'

export function ReferencePanel() {
  const r = useReferenceStore()
  const { drag, open, inputProps, dropProps } = useFileDrop((f) => void r.upload(f))
  const hasImg = r.previewShown && !!r.previewUrl

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      if (!e.clipboardData) return
      for (const item of e.clipboardData.items) {
        if (item.type.startsWith('image/')) {
          const f = item.getAsFile()
          if (f) void useReferenceStore.getState().upload(f)
          break
        }
      }
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [])

  return (
    <div className={'ref-box' + (r.enabled ? '' : ' disabled')}>
      <div
        className={'ref-drop' + (drag ? ' drag' : '') + (hasImg ? ' has-img' : '')}
        title={hasImg ? 'Click / drop / paste to replace the reference' : 'Drop / paste an image or click to choose a reference'}
        onClick={open}
        {...dropProps}
      >
        {hasImg ? (
          <img src={r.previewUrl} alt="reference" />
        ) : (
          <span className="ref-hint">{r.dropHint}</span>
        )}
        {hasImg && (
          <button
            className="ref-clear"
            title="Clear reference image"
            aria-label="Clear reference image"
            onClick={(e) => {
              e.stopPropagation()
              void r.clear()
            }}
          >
            🗑
          </button>
        )}
      </div>
      <input {...inputProps} accept="image/*" hidden />
    </div>
  )
}
