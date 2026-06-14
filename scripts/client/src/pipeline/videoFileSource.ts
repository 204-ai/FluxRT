// Owns the file-backed <video> element used as an input layer. Lives outside
// the rail so playback position/state survive pipeline restarts (source-set
// changes restart the rail; this element doesn't). Framework-agnostic.

export interface VideoFileMeta {
  width: number
  height: number
  duration: number
}

export class VideoFileSource {
  readonly el: HTMLVideoElement
  private url = ''

  constructor() {
    this.el = document.createElement('video')
    this.el.muted = true
    this.el.playsInline = true
    this.el.preload = 'auto'
  }

  get loaded(): boolean {
    return this.url !== ''
  }

  async load(file: File): Promise<VideoFileMeta> {
    if (!file.type.startsWith('video/')) throw new Error('not a video file')
    this.revoke()
    this.url = URL.createObjectURL(file)
    this.el.src = this.url

    await new Promise<void>((resolve, reject) => {
      const cleanup = () => {
        this.el.removeEventListener('loadedmetadata', onMeta)
        this.el.removeEventListener('error', onErr)
      }
      const onMeta = () => {
        cleanup()
        resolve()
      }
      const onErr = () => {
        cleanup()
        reject(new Error('video decode error — unsupported codec?'))
      }
      this.el.addEventListener('loadedmetadata', onMeta)
      this.el.addEventListener('error', onErr)
    }).catch((e) => {
      this.unload()
      throw e
    })

    if (this.el.videoWidth === 0) {
      this.unload()
      throw new Error('no video track (audio-only or unsupported codec)')
    }
    // Muted + invoked from the file-pick gesture → autoplay-safe.
    void this.el.play().catch(() => {})
    return { width: this.el.videoWidth, height: this.el.videoHeight, duration: this.el.duration }
  }

  unload(): void {
    this.el.pause()
    this.el.removeAttribute('src')
    this.el.load()
    this.revoke()
  }

  private revoke(): void {
    if (this.url) {
      URL.revokeObjectURL(this.url)
      this.url = ''
    }
  }

  async play(): Promise<void> {
    await this.el.play()
  }
  pause(): void {
    this.el.pause()
  }
  seek(t: number): void {
    this.el.currentTime = t
  }
  setRate(r: number): void {
    this.el.playbackRate = r
  }
  setLoop(on: boolean): void {
    this.el.loop = on
  }
}
