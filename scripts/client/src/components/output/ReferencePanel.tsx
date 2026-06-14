// Reference image: drop-zone (click / drag / global paste), preview, clear.

import { useEffect, useRef, useState } from 'react'
import { useReferenceStore } from '../../state/referenceStore'

export function ReferencePanel() {
  const r = useReferenceStore()
  const fileRef = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)

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
      <div
        className={'drop' + (over ? ' over' : '')}
        style={{ minHeight: 54 }}
        onClick={() => fileRef.current?.click()}
        onDragEnter={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={(e) => {
          e.preventDefault()
          setOver(false)
        }}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          if (e.dataTransfer.files.length) void r.upload(e.dataTransfer.files[0])
        }}
      >
        {r.dropHint}
      </div>
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          if (e.target.files?.length) void r.upload(e.target.files[0])
          e.target.value = ''
        }}
      />
      {r.previewShown && r.previewUrl && <img className="preview shown" src={r.previewUrl} alt="reference preview" />}
      <div className="meta">{r.meta}</div>
      <button onClick={() => void r.clear()}>Clear</button>
    </div>
  )
}
