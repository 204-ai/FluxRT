// Parser for imported prompt files. Lenient by design — a hand-authored
// text file and a list exported from the server should both just work.

import type { SavedPrompt } from './api'

function entry(prompt: string, style = 0, tracking = 0, stability = 0): SavedPrompt {
  return { prompt, style, tracking, stability }
}

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0
}

/**
 * Parse an imported prompts file into SavedPrompt entries. Accepts either:
 *  - JSON: an array of strings, an array of `{prompt, style?, tracking?,
 *    stability?}` objects, or a `{prompts: [...]}` wrapper (the exact shape
 *    GET /prompts returns, so a list exported from the server round-trips).
 *  - Plain text: one prompt per line; blank lines and `#` comments are ignored.
 *
 * Whitespace-only prompts are dropped. Returns [] for empty or unparseable
 * input. JSON that fails to parse falls back to line-based parsing.
 */
export function parsePromptsFile(text: string): SavedPrompt[] {
  const trimmed = text.trim()
  if (!trimmed) return []

  if (trimmed[0] === '[' || trimmed[0] === '{') {
    try {
      const data: unknown = JSON.parse(trimmed)
      const arr = Array.isArray(data)
        ? data
        : Array.isArray((data as { prompts?: unknown }).prompts)
          ? (data as { prompts: unknown[] }).prompts
          : null
      if (arr) {
        const out: SavedPrompt[] = []
        for (const e of arr) {
          if (typeof e === 'string') {
            const p = e.trim()
            if (p) out.push(entry(p))
          } else if (e && typeof e === 'object' && typeof (e as SavedPrompt).prompt === 'string') {
            const o = e as Partial<SavedPrompt> & { prompt: string }
            const p = o.prompt.trim()
            if (p) out.push(entry(p, num(o.style), num(o.tracking), num(o.stability)))
          }
        }
        return out
      }
    } catch {
      /* not valid JSON — treat the whole file as plain text below */
    }
  }

  return trimmed
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('#'))
    .map((p) => entry(p))
}
