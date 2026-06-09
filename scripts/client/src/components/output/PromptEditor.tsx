// Auto-growing prompt textarea: Enter sends, Shift+Enter newline. Inbound
// state:prompt sync is focus-guarded via the focus registry.

import { useEffect, useRef } from 'react'
import { usePromptStore } from '../../state/promptStore'
import { registerFocusable } from '../../state/focusRegistry'

export function PromptEditor() {
  const prompt = usePromptStore((s) => s.prompt)
  const ref = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    registerFocusable('prompt', ref.current)
    return () => registerFocusable('prompt', null)
  }, [])

  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = el.scrollHeight + 2 + 'px'
  }, [prompt])

  return (
    <textarea
      ref={ref}
      id="prompt"
      rows={1}
      placeholder="Prompt — Enter to apply, Shift+Enter for newline"
      value={prompt}
      onChange={(e) => usePromptStore.getState().setPromptLocal(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
          const x = prompt.trim()
          if (x) {
            e.preventDefault()
            usePromptStore.getState().sendPrompt(x)
          }
        }
      }}
    />
  )
}
