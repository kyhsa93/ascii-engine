import { Framebuffer } from '../core/framebuffer.ts'
import { DEFAULT_CELL_ASPECT } from '../core/camera.ts'

export type ColorMode = 'truecolor' | '256' | 'none'

export interface TerminalOptions {
  stream?: NodeJS.WriteStream
  color?: ColorMode
  /** Cell width divided by cell height. Terminals cannot report this. */
  cellAspect?: number
  /** Rows to keep clear at the bottom, e.g. for a status line. */
  reserveRows?: number
}

/** Guesses the richest colour mode the environment admits to supporting. */
export function detectColorMode(env: NodeJS.ProcessEnv = process.env): ColorMode {
  if (env.NO_COLOR) return 'none'
  if (env.COLORTERM === 'truecolor' || env.COLORTERM === '24bit') return 'truecolor'
  if (env.TERM?.includes('256color')) return '256'
  return 'none'
}

/** Packs an 0..1 RGB triple into a 24-bit integer, for cheap frame diffing. */
function pack(r: number, g: number, b: number): number {
  const q = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255)
  return (q(r) << 16) | (q(g) << 8) | q(b)
}

/**
 * Maps a packed colour onto the xterm-256 palette: the 6x6x6 colour cube,
 * or the finer 24-step grey ramp when all three channels agree closely.
 */
export function to256(packed: number): number {
  const r = (packed >> 16) & 0xff
  const g = (packed >> 8) & 0xff
  const b = packed & 0xff
  if (Math.abs(r - g) < 8 && Math.abs(g - b) < 8) {
    if (r < 8) return 16
    if (r > 248) return 231
    return 232 + Math.round(((r - 8) / 247) * 23)
  }
  const c = (v: number) => Math.round((v / 255) * 5)
  return 16 + 36 * c(r) + 6 * c(g) + c(b)
}

export class Terminal {
  readonly stream: NodeJS.WriteStream
  readonly colorMode: ColorMode
  readonly cellAspect: number
  readonly reserveRows: number

  private prevChars = new Uint32Array(0)
  private prevColor = new Int32Array(0)
  private framebufferCache: Framebuffer | null = null
  private entered = false
  private cleanup: (() => void) | null = null

  constructor(options: TerminalOptions = {}) {
    this.stream = options.stream ?? process.stdout
    this.colorMode = options.color ?? detectColorMode()
    this.cellAspect = options.cellAspect ?? DEFAULT_CELL_ASPECT
    this.reserveRows = options.reserveRows ?? 0
  }

  get width(): number {
    return this.stream.columns ?? 80
  }

  get height(): number {
    return Math.max(1, (this.stream.rows ?? 24) - this.reserveRows)
  }

  /**
   * The framebuffer for this terminal's current size, reallocated whenever the
   * window changes shape. The returned object is reused between frames.
   */
  framebuffer(): Framebuffer {
    const w = this.width
    const h = this.height
    const fb = this.framebufferCache
    if (fb && fb.width === w && fb.height === h) return fb

    const next = new Framebuffer(w, h)
    this.framebufferCache = next
    this.prevChars = new Uint32Array(w * h)
    this.prevColor = new Int32Array(w * h).fill(-2)
    if (this.entered) this.stream.write('\x1b[2J')
    return next
  }

  enter(): void {
    if (this.entered) return
    this.entered = true
    // Alternate screen buffer, so quitting restores whatever was on screen.
    this.stream.write('\x1b[?1049h\x1b[?25l\x1b[2J')

    const restore = () => this.exit()
    this.cleanup = restore
    process.on('exit', restore)
    process.on('SIGINT', () => {
      restore()
      process.exit(0)
    })
    process.on('SIGTERM', () => {
      restore()
      process.exit(0)
    })
  }

  exit(): void {
    if (!this.entered) return
    this.entered = false
    this.stream.write('\x1b[0m\x1b[?25h\x1b[?1049l')
    if (process.stdin.isTTY && process.stdin.isRaw) process.stdin.setRawMode(false)
    if (this.cleanup) process.off('exit', this.cleanup)
    this.cleanup = null
  }

  /** Writes only the cells that changed since the previous frame. */
  present(fb: Framebuffer): void {
    const { width, height, chars, color } = fb
    if (this.prevChars.length !== width * height) {
      this.prevChars = new Uint32Array(width * height)
      this.prevColor = new Int32Array(width * height).fill(-2)
    }

    const out: string[] = []
    let penX = -1
    let penY = -1
    let penColor = -2

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x
        const ch = chars[idx] === 0 ? 32 : chars[idx]!
        // A space shows no foreground, so its colour is not worth an escape
        // sequence — and pretending it has none collapses long empty runs.
        const want =
          ch === 32 || this.colorMode === 'none'
            ? -1
            : pack(color[idx * 3]!, color[idx * 3 + 1]!, color[idx * 3 + 2]!)

        if (ch === this.prevChars[idx] && want === this.prevColor[idx]) continue

        if (penY !== y || penX !== x) out.push(`\x1b[${y + 1};${x + 1}H`)
        if (want !== -1 && want !== penColor) {
          out.push(this.colorMode === 'truecolor'
            ? `\x1b[38;2;${(want >> 16) & 0xff};${(want >> 8) & 0xff};${want & 0xff}m`
            : `\x1b[38;5;${to256(want)}m`)
          penColor = want
        }
        out.push(String.fromCharCode(ch))
        penX = x + 1
        penY = y
        this.prevChars[idx] = ch
        this.prevColor[idx] = want
      }
    }

    if (out.length > 0) this.stream.write(out.join(''))
  }

  /** Draws a single line of text below the framebuffer, in the reserved rows. */
  status(text: string, row = 0): void {
    if (row >= this.reserveRows) return
    const y = this.height + row + 1
    this.stream.write(`\x1b[${y};1H\x1b[0m\x1b[K${text.slice(0, this.width)}`)
  }

  /** Turns on raw mode and reports every keypress as a raw escape string. */
  onKey(handler: (key: string) => void): void {
    if (!process.stdin.isTTY) return
    process.stdin.setRawMode(true)
    process.stdin.resume()
    process.stdin.setEncoding('utf8')
    process.stdin.on('data', (data: string) => handler(data))
  }

  onResize(handler: () => void): void {
    this.stream.on('resize', handler)
  }
}

export interface LoopHandle {
  stop(): void
  readonly fps: number
}

/**
 * Calls `frame` at roughly `targetFps`, passing the seconds elapsed since the
 * previous call. Uses `setTimeout` rather than a tight loop so keypresses and
 * resize events still get a turn.
 */
export function runLoop(frame: (dt: number, elapsed: number) => void, targetFps = 60): LoopHandle {
  const interval = 1000 / targetFps
  const start = performance.now()
  let last = start
  let running = true
  let timer: NodeJS.Timeout | null = null

  let windowStart = start
  let windowFrames = 0
  let fps = 0

  const tick = () => {
    if (!running) return
    const now = performance.now()
    const dt = (now - last) / 1000
    last = now

    windowFrames++
    if (now - windowStart >= 500) {
      fps = (windowFrames * 1000) / (now - windowStart)
      windowStart = now
      windowFrames = 0
    }

    frame(dt, (now - start) / 1000)

    const spent = performance.now() - now
    timer = setTimeout(tick, Math.max(0, interval - spent))
  }

  timer = setTimeout(tick, 0)

  return {
    stop() {
      running = false
      if (timer) clearTimeout(timer)
    },
    get fps() {
      return fps
    },
  }
}
