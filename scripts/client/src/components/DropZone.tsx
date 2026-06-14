// Minimal drop / upload zone: click to choose or drag-drop a file. Shared by
// the video source and the reference-image panel (drag/drop logic in useFileDrop).

import { useFileDrop } from '../lib/useFileDrop'

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
  const { drag, open, inputProps, dropProps } = useFileDrop(onFile)

  return (
    <>
      <input {...inputProps} accept={accept} style={{ display: 'none' }} />
      <div
        className={'drop-zone' + (drag ? ' drag' : '')}
        title={title}
        onClick={open}
        {...dropProps}
      >
        {label}
      </div>
    </>
  )
}
