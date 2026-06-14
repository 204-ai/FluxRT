// Shared file drop/click/upload behavior: a hidden <input>, a drag-highlight
// flag, and the drag/drop handlers. Used by the DropZone box and the reference
// panel (where the image itself is the drop target).

import { useRef, useState, type ChangeEvent, type DragEvent } from 'react'

export function useFileDrop(onFile: (file: File) => void) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)

  return {
    drag,
    /** Open the native file picker. */
    open: () => inputRef.current?.click(),
    /** Spread onto the hidden <input type="file">. */
    inputProps: {
      ref: inputRef,
      type: 'file' as const,
      onChange: (e: ChangeEvent<HTMLInputElement>) => {
        const f = e.target.files?.[0]
        if (f) onFile(f)
        e.target.value = ''
      },
    },
    /** Spread onto the drop target element. */
    dropProps: {
      onDragOver: (e: DragEvent) => {
        e.preventDefault()
        if (!drag) setDrag(true)
      },
      onDragLeave: () => setDrag(false),
      onDrop: (e: DragEvent) => {
        e.preventDefault()
        setDrag(false)
        const f = e.dataTransfer.files?.[0]
        if (f) onFile(f)
      },
    },
  }
}
