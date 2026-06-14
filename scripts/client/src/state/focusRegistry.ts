// Inbound state:* sync must not clobber a field the user is editing
// (otherwise two clients' prompt boxes fight). Components register their
// input elements; the ctrl dispatcher checks focus before applying.

const elements = new Map<string, HTMLElement>()

export function registerFocusable(key: 'prompt' | 'seed' | 'steps', el: HTMLElement | null): void {
  if (el) elements.set(key, el)
  else elements.delete(key)
}

export function isFocused(key: 'prompt' | 'seed' | 'steps'): boolean {
  const el = elements.get(key)
  return !!el && document.activeElement === el
}
