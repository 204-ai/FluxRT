import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// All signaling/control HTTP goes to the FastAPI server; WebRTC media itself
// is peer-to-peer and needs no proxy.
const target = process.env.FLUXRT_SERVER ?? 'http://localhost:8765'
const proxied = [
  '/offer',
  '/reference',
  '/prompts',
  '/prompt',
  '/seed',
  '/steps',
  '/comfy',
  '/healthz',
  '/lip-transfer',
]

export default defineConfig({
  plugins: [react()],
  build: { outDir: 'dist' },
  // ES-format workers so the pipeline worker can code-split its dynamic import()
  // of onnxruntime-web (lazy-loaded only when the depth pass is enabled).
  worker: { format: 'es' },
  server: {
    proxy: Object.fromEntries(
      proxied.map((p) => [p, { target, changeOrigin: true }]),
    ),
  },
  test: {
    environment: 'node',
  },
})
