// Typed codec for the `ctrl` DataChannel text protocol.
// Single source of truth for the wire strings — both directions.
// Server side: run_webrtc.py (@channel.on("message") + broadcast_ctrl).

export type CtrlOut =
  | { kind: 'prompt'; text: string }
  | { kind: 'seed'; value: number }
  | { kind: 'steps'; value: number }

export type CtrlIn =
  | { kind: 'refSet'; version: number }
  | { kind: 'refClear'; version: number }
  | { kind: 'inputRole'; role: 'you' | 'peer' | 'server' }
  | { kind: 'lip'; on: boolean }
  | { kind: 'statePrompt'; text: string }
  | { kind: 'stateSeed'; value: string }
  | { kind: 'stateSteps'; value: string }
  | { kind: 'promptsChanged' }
  | { kind: 'unknown'; raw: string }

export function encodeCtrl(msg: CtrlOut): string {
  switch (msg.kind) {
    case 'prompt':
      return 'prompt:' + msg.text
    case 'seed':
      return 'seed:' + msg.value
    case 'steps':
      return 'steps:' + msg.value
  }
}

export function decodeCtrl(raw: string): CtrlIn {
  if (raw.startsWith('ref:set:')) {
    const version = parseInt(raw.slice('ref:set:'.length), 10)
    if (!isNaN(version)) return { kind: 'refSet', version }
  } else if (raw.startsWith('ref:clear')) {
    const version = parseInt(raw.split(':')[2] || '0', 10)
    if (!isNaN(version)) return { kind: 'refClear', version }
  } else if (raw === 'input:you') {
    return { kind: 'inputRole', role: 'you' }
  } else if (raw === 'input:peer') {
    return { kind: 'inputRole', role: 'peer' }
  } else if (raw === 'input:server') {
    return { kind: 'inputRole', role: 'server' }
  } else if (raw === 'lip:on' || raw === 'lip:off') {
    return { kind: 'lip', on: raw === 'lip:on' }
  } else if (raw.startsWith('state:prompt:')) {
    return { kind: 'statePrompt', text: raw.slice('state:prompt:'.length) }
  } else if (raw.startsWith('state:seed:')) {
    return { kind: 'stateSeed', value: raw.slice('state:seed:'.length) }
  } else if (raw.startsWith('state:steps:')) {
    return { kind: 'stateSteps', value: raw.slice('state:steps:'.length) }
  } else if (raw === 'prompts:changed') {
    return { kind: 'promptsChanged' }
  }
  return { kind: 'unknown', raw }
}
