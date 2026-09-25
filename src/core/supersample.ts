/**
 * Antialiasing, by drawing into a finer grid and averaging back down.
 *
 * One sample per character cell means a silhouette can only be in or out, so
 * it comes out as a staircase. Sampling the scene on a grid `factor` times
 * finer in each direction and averaging the colour gives a half-covered cell
 * half the brightness — and since this renderer picks its glyph from
 * luminance *after* shading, a half-lit cell becomes a middling character on
 * its own. The edge stops being a staircase and becomes a gradient, with no
 * special case anywhere in the rasterizer.
 *
 * It works for the marched path too, because it is the framebuffer that
 * changes rather than anything that draws into one.
 */

import { Framebuffer } from './framebuffer.ts'

export class Supersampler {
  /** Sub-samples per cell along each axis; the cost is the square of it. */
  readonly factor: number
  /** Output size, in character cells. */
  readonly width: number
  readonly height: number
  /**
   * The fine grid to draw into.
   *
   * Its cells have the same *shape* as the output's — both axes are scaled by
   * the same factor — so `aspectFor` gives the same answer for either and the
   * camera needs no adjustment.
   */
  readonly buffer: Framebuffer

  constructor(width: number, height: number, factor = 2) {
    this.factor = Math.max(1, Math.floor(factor))
    this.width = Math.max(1, Math.floor(width))
    this.height = Math.max(1, Math.floor(height))
    this.buffer = new Framebuffer(this.width * this.factor, this.height * this.factor)
  }

  clear(r = 0, g = 0, b = 0, char = 32): void {
    this.buffer.clear(r, g, b, char)
  }

  /**
   * Box-filters the fine grid down into `out`.
   *
   * Colour is the mean of the sub-samples, which is what antialiases. Depth is
   * the *nearest* of them rather than the mean, so a partly covered cell still
   * occludes anything drawn behind it afterwards; averaging depth would put
   * the cell somewhere no surface actually is.
   *
   * A glyph a shader forced explicitly cannot be averaged, so the nearest
   * sub-sample's glyph wins outright and that edge stays hard. A cell no
   * sub-sample covered keeps the background's space, exactly as `clear` left
   * it, so an untouched frame resolves identically with or without this.
   */
  resolveInto(out: Framebuffer): void {
    const n = this.factor
    if (out.width !== this.width || out.height !== this.height) {
      throw new Error(`resolveInto: expected a ${this.width}x${this.height} target, got ${out.width}x${out.height}`)
    }

    const src = this.buffer
    const samples = n * n
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        let r = 0
        let g = 0
        let b = 0
        let nearest = 0
        let nearestChar = 32

        for (let sy = 0; sy < n; sy++) {
          const row = (y * n + sy) * src.width
          for (let sx = 0; sx < n; sx++) {
            const i = row + x * n + sx
            const c = i * 3
            r += src.color[c]!
            g += src.color[c + 1]!
            b += src.color[c + 2]!
            const d = src.depth[i]!
            if (d > nearest) {
              nearest = d
              nearestChar = src.chars[i]!
            }
          }
        }

        const o = y * this.width + x
        const c = o * 3
        out.color[c] = r / samples
        out.color[c + 1] = g / samples
        out.color[c + 2] = b / samples
        out.depth[o] = nearest
        out.chars[o] = nearest > 0 ? nearestChar : 32
      }
    }
  }
}
