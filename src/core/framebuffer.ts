import { RAMPS, luminance, rampChar } from './ramp.ts'

/**
 * A grid of character cells with a colour and a depth per cell.
 *
 * `depth` holds 1/w rather than a view-space distance, because 1/w is what
 * interpolates linearly across a triangle in screen space. Larger means
 * nearer; 0 means nothing has been drawn there.
 */
export class Framebuffer {
  readonly width: number
  readonly height: number
  readonly chars: Uint32Array
  readonly color: Float32Array
  readonly depth: Float32Array

  constructor(width: number, height: number) {
    this.width = Math.max(1, Math.floor(width))
    this.height = Math.max(1, Math.floor(height))
    const cells = this.width * this.height
    this.chars = new Uint32Array(cells)
    this.color = new Float32Array(cells * 3)
    this.depth = new Float32Array(cells)
  }

  clear(r = 0, g = 0, b = 0, char = 32): void {
    this.chars.fill(char)
    this.depth.fill(0)
    for (let i = 0; i < this.color.length; i += 3) {
      this.color[i] = r
      this.color[i + 1] = g
      this.color[i + 2] = b
    }
  }

  /**
   * Fills in the glyph of every cell a shader left on auto (char 0), mapping
   * the cell's luminance onto `ramp`. Call once per frame after drawing and
   * before presenting.
   */
  resolve(ramp: string = RAMPS.short): void {
    for (let i = 0; i < this.chars.length; i++) {
      if (this.chars[i] !== 0) continue
      const c = i * 3
      this.chars[i] = rampChar(ramp, luminance(this.color[c]!, this.color[c + 1]!, this.color[c + 2]!))
    }
  }

  /** The frame as plain text, one line per row. Colour is dropped. */
  toString(): string {
    const lines: string[] = []
    for (let y = 0; y < this.height; y++) {
      let line = ''
      for (let x = 0; x < this.width; x++) {
        const ch = this.chars[y * this.width + x]!
        line += String.fromCharCode(ch === 0 ? 32 : ch)
      }
      lines.push(line)
    }
    return lines.join('\n')
  }
}
