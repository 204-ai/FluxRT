// Reference image — minimal: a shared drop/upload zone (click / drag / global
// paste), a small preview, and an icon-only clear button.

import { useEffect } from 'react'
import { useReferenceStore } from '../../state/referenceStore'
import { DropZone } from '../DropZone'

export function ReferencePanel() {
  const r = useReferenceStore()

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
    <div className={'ref ref-inline' + (r.enabled ? '' : ' disabled')}>
      <DropZone
        accept="image/*"
        label={r.dropHint}
        onFile={(f) => void r.upload(f)}
        title="Drop / paste an image or click to choose a reference"
      />
      {r.previewShown && r.previewUrl && <img className="preview shown" src={r.previewUrl} alt="reference preview" />}
      <button className="icon-btn" title="Clear reference image" aria-label="Clear reference image" onClick={() => void r.clear()}>
        🗑
      </button>
    </div>
  )
}
