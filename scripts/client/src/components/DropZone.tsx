// Minimal drop / upload zone: click to choose or drag-drop a file. Shared by
// the video source and the reference-image panel.

import { useRef, useState } from 'react'

export function DropZone({
  accept,
  label,
  onFile,
  title,
}: {
  accept: string
  label: string
  onFile: (file: File) => void
  title?: string
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onFile(f)
          e.target.value = ''
        }}
      />
      <div
        className={'drop-zone' + (drag ? ' drag' : '')}
        title={title}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          if (!drag) setDrag(true)
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDrag(false)
          const f = e.dataTransfer.files?.[0]
          if (f) onFile(f)
        }}
      >
        {label}
      </div>
    </>
  )
}
