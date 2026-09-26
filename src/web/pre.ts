import { Framebuffer } from '../core/framebuffer.ts'

/**
 * Presents a framebuffer inside a `<pre>` element.
 *
 * Unlike a terminal, the browser can be *asked* how wide a character is, so
 * the cell aspect used for the projection is measured rather than assumed.
 *
 * Getting a frame onto the page used to cost more than drawing it. Written as
 * one `innerHTML`, a 163 by 50 grid produced 122 KiB of markup in 3,322 spans;
 * the parse alone measured 2.7 milliseconds, before the browser threw away
 * every node from the frame before and laid the new ones out. Rendering the
 * framebuffer that filled it took 0.36.
 *
 * Three changes fixed that, and the middle one did most of the work.
 *
 * A row owns a node that outlives the frame, and a row whose markup has not
 * changed is not touched. That is worth a lot while the view is still and
 * nothing at all while it is turning, when every row changes anyway.
 *
 * Colour is rounded to `colourBits` a channel. This is a performance control
 * wearing the clothes of a colour control: a run of text ends wherever the
 * colour changes, and distance shading changes it almost every cell, so eight
 * bits meant thirty-six spans in a row of 163 and five bits means fifteen.
 * Turning went from 37.6 frames a second to 60.4 on the same machine, and the
 * two are not distinguishable side by side -- the largest error is four parts
 * in 255, which is smaller than one step of the shading being quantised.
 *
 * And the colour is compared as a packed integer, with hex written from a table
 * only where a run begins. Building the markup was 3.77 milliseconds a frame at
 * 327 by 102 and is 0.64: nearly all of it had been calling `toString(16)`
 * three times per cell to discover the colour had not changed.
 */
export class PreSurface {
  readonly element: HTMLElement
  cellWidth = 8
  cellHeight = 16
  cols = 80
  rows = 24

  /**
   * Bits of each colour channel that reach the page.
   *
   * Five by default. Eight restores exact colour and costs about a third of the
   * frame rate at a typical size; the note above has the measurements.
   */
  colourBits = 5

  private framebufferCache: Framebuffer | null = null
  /** One node per row of the grid, reused frame to frame. */
  private lines: HTMLElement[] = []
  /** What each of those nodes currently holds, so an unchanged row can be skipped. */
  private written: string[] = []
  /** Two hex digits per quantised level, rebuilt only when the depth changes. */
  private digits: string[] = []
  private digitsFor = -1

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
   * Rewrites only the rows that differ from the frame before.
   *
   * The newline stays inside the row's own node so the element remains one run
   * of preformatted text. A block element per row would hand line spacing to
   * layout, and the cell height measured above would stop being a row's height.
   */
  present(fb: Framebuffer): void {
    const { width, height, chars, color } = fb
    if (this.lines.length !== height) this.reset(height)

    const bits = Math.max(1, Math.min(8, Math.floor(this.colourBits)))
    const steps = (1 << bits) - 1
    if (this.digitsFor !== steps) {
      const table: string[] = []
      for (let i = 0; i <= steps; i++) {
        table.push(
          Math.round((i / steps) * 255)
            .toString(16)
            .padStart(2, '0'),
        )
      }
      this.digits = table
      this.digitsFor = steps
    }
    const digits = this.digits

    for (let y = 0; y < height; y++) {
      let markup = ''
      let start = 0
      let packed = -1
      const base = y * width

      // One past the end, so the last run closes through the same branch as
      // every other one.
      for (let x = 0; x <= width; x++) {
        const idx = base + x
        const ch = x === width ? -1 : chars[idx] === 0 ? 32 : chars[idx]!
        // A space carries no colour, which is what lets a blank row be a single
        // run rather than one run per shade of nothing.
        let here = -1
        if (ch >= 0 && ch !== 32) {
          const r = quantise(color[idx * 3]!, steps)
          const g = quantise(color[idx * 3 + 1]!, steps)
          const b = quantise(color[idx * 3 + 2]!, steps)
          here = (r << 16) | (g << 8) | b
        }
        if (here === packed) continue

        if (x > start) {
          const text = escaped(chars, base + start, x - start)
          markup =
            packed < 0
              ? markup + text
              : `${markup}<span style="color:#${digits[(packed >> 16) & 255]}${digits[(packed >> 8) & 255]}${
                  digits[packed & 255]
                }">${text}</span>`
        }
        start = x
        packed = here
      }
      markup += '\n'

      // Compared as markup rather than as cells, because the markup is what the
      // browser would have to redo and two frames can round to the same one.
      if (this.written[y] === markup) continue
      this.written[y] = markup
      this.lines[y]!.innerHTML = markup
    }
  }

  /** Builds a fresh row per line, for the first frame and after a resize. */
  private reset(height: number): void {
    this.element.textContent = ''
    this.lines = []
    this.written = []
    for (let y = 0; y < height; y++) {
      const line = document.createElement('span')
      this.element.appendChild(line)
      this.lines.push(line)
      this.written.push('')
    }
  }
}

/** A channel as one of `steps + 1` levels, clamped. */
function quantise(value: number, steps: number): number {
  const clamped = value < 0 ? 0 : value > 1 ? 1 : value
  return Math.round(clamped * steps)
}

/**
 * A run of cells as text, with the three characters markup cares about escaped.
 *
 * Built a character at a time rather than with a replace: a run is a dozen
 * characters and there are thousands of them a frame, so the regular expression
 * cost more than the comparison it saved.
 */
function escaped(chars: Uint32Array, from: number, count: number): string {
  let out = ''
  for (let i = 0; i < count; i++) {
    const code = chars[from + i] === 0 ? 32 : chars[from + i]!
    out += code === 38 ? '&amp;' : code === 60 ? '&lt;' : code === 62 ? '&gt;' : String.fromCharCode(code)
  }
  return out
}
