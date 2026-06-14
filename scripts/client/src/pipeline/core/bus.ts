import type { BusReader } from './types'

/** Minimal latest-value store shared between analyzers and effects. */
export class AnalyzerBus implements BusReader {
  private values = new Map<string, unknown>()

  get<T>(key: string): T | undefined {
    return this.values.get(key) as T | undefined
  }

  set(key: string, value: unknown): void {
    this.values.set(key, value)
  }

  clear(): void {
    this.values.clear()
  }
}
