import { Framebuffer } from '../core/framebuffer.ts'

/**
 * Presents a framebuffer inside a `<pre>` element.
 *
 * Unlike a terminal, the browser can be *asked* how wide a character is, so
 * the cell aspect used for the projection is measured rather than assumed.
 */
export class PreSurface {
  readonly element: HTMLElement
  cellWidth = 8
  cellHeight = 16
  cols = 80
  rows = 24

  private framebufferCache: Framebuffer | null = null

  constructor(element: HTMLElement) {
    this.element = element
    this.measure()
  }

  get cellAspect(): number {
    return this.cellWidth / this.cellHeight
  }

  /**
   * Measures one character cell against the element's own font, then works out
   * how many fit. The probe spans two lines so its height gives the real line
   * box rather than the glyph's ink.
   */
  measure(): void {
    const probe = document.createElement('span')
    probe.style.position = 'absolute'
    probe.style.visibility = 'hidden'
    probe.style.whiteSpace = 'pre'
    probe.textContent = `${'M'.repeat(50)}\n${'M'.repeat(50)}`
    this.element.appendChild(probe)
    const rect = probe.getBoundingClientRect()
    probe.remove()

    if (rect.width > 0) this.cellWidth = rect.width / 50
    if (rect.height > 0) this.cellHeight = rect.height / 2

    const box = this.element.getBoundingClientRect()
    this.cols = Math.max(1, Math.floor(box.width / this.cellWidth))
    this.rows = Math.max(1, Math.floor(box.height / this.cellHeight))
  }

  framebuffer(): Framebuffer {
    const fb = this.framebufferCache
    if (fb && fb.width === this.cols && fb.height === this.rows) return fb
    const next = new Framebuffer(this.cols, this.rows)
    this.framebufferCache = next
    return next
  }

  /**
   * Rewrites the element's markup, coalescing runs of equal colour into one
   * `<span>` so a frame costs a few hundred DOM nodes instead of one per cell.
   */
  present(fb: Framebuffer): void {
    const { width, height, chars, color } = fb
    const parts: string[] = []
    let run = ''
    let runColor = ''

    const flush = () => {
      if (run.length === 0) return
      parts.push(runColor === '' ? escapeHtml(run) : `<span style="color:${runColor}">${escapeHtml(run)}</span>`)
      run = ''
    }

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x
        const ch = chars[idx] === 0 ? 32 : chars[idx]!
        const want = ch === 32 ? '' : hex(color[idx * 3]!, color[idx * 3 + 1]!, color[idx * 3 + 2]!)
        if (want !== runColor) {
          flush()
          runColor = want
        }
        run += String.fromCharCode(ch)
      }
      run += '\n'
    }
    flush()

    this.element.innerHTML = parts.join('')
  }
}

function hex(r: number, g: number, b: number): string {
  const q = (v: number) =>
    Math.round(Math.min(1, Math.max(0, v)) * 255)
      .toString(16)
      .padStart(2, '0')
  return `#${q(r)}${q(g)}${q(b)}`
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>]/g, (c) => (c === '&' ? '&amp;' : c === '<' ? '&lt;' : '&gt;'))
}
