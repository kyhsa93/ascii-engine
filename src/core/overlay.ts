/**
 * Annotations drawn into the frame: text, lines, and an axis gizmo.
 *
 * Text is the one thing a character renderer gets for nothing. A pixel engine
 * needs a font atlas and a sampler to put a label on screen; here a label is
 * already made of the same stuff as the picture, so writing one is writing
 * glyphs into cells. `Framebuffer.resolve` only fills cells a shader left on
 * auto, so text written here survives it untouched.
 *
 * Two rules for using any of this. Draw overlays last: text ignores the depth
 * buffer entirely, so anything drawn afterwards paints over it. And draw them
 * into the *output* framebuffer rather than a `Supersampler`'s fine grid —
 * averaging a glyph with its neighbours is how you turn text into smudge.
 */

import type { Mat4 } from './mat4.ts'
import { transformPoint } from './mat4.ts'
import type { RenderTarget } from './raster.ts'
import type { Vec3 } from './vec3.ts'
import { vec3 } from './vec3.ts'

/** A point counts as in front of the camera once `z + w` clears this. */
const NEAR_EPSILON = 1e-6

const clip = new Float32Array(4)
const clipB = new Float32Array(4)

export interface TextOptions {
  color?: Vec3
  /** Which part of the string sits at the given column. */
  align?: 'left' | 'center' | 'right'
}

/**
 * Writes a string into the grid at a cell position.
 *
 * Characters outside the grid are dropped rather than wrapped, so a label near
 * an edge loses its tail instead of reappearing on the next row. Spaces are
 * written like any other character — a string with interior spaces punches
 * holes in whatever is behind it, which is occasionally what you want and
 * always what you get.
 */
export function drawText(
  target: RenderTarget,
  x: number,
  y: number,
  text: string,
  options: TextOptions = {},
): void {
  const row = Math.round(y)
  if (row < 0 || row >= target.height) return

  const color = options.color ?? vec3(1, 1, 1)
  const align = options.align ?? 'left'
  const start =
    align === 'left' ? Math.round(x) : align === 'center' ? Math.round(x) - (text.length >> 1) : Math.round(x) - text.length + 1

  for (let i = 0; i < text.length; i++) {
    const col = start + i
    if (col < 0 || col >= target.width) continue
    const idx = row * target.width + col
    target.chars[idx] = text.charCodeAt(i)
    const o = idx * 3
    target.color[o] = color.x
    target.color[o + 1] = color.y
    target.color[o + 2] = color.z
  }
}

export interface LabelOptions extends TextOptions {
  /** Cells to shift the label by, after projection. */
  dx?: number
  dy?: number
  /** Drop the label when something nearer already holds its anchor cell. */
  occlude?: boolean
}

/**
 * Projects a world-space point and writes a label there.
 *
 * Returns whether anything was drawn. A point behind the camera is dropped
 * rather than projected: dividing by a negative w mirrors it to the opposite
 * side of the screen, where the label would look perfectly plausible and be
 * in entirely the wrong place.
 */
export function label3(
  target: RenderTarget,
  position: Vec3,
  viewProjection: Mat4,
  text: string,
  options: LabelOptions = {},
): boolean {
  transformPoint(viewProjection, position.x, position.y, position.z, clip)
  const w = clip[3]!
  if (clip[2]! + w < NEAR_EPSILON) return false

  const invW = 1 / w
  const cx = (clip[0]! * invW * 0.5 + 0.5) * target.width
  const cy = (0.5 - clip[1]! * invW * 0.5) * target.height
  // Cell i spans [i, i + 1) and the rasterizer samples its centre at i + 0.5,
  // so the cell holding a screen coordinate is its floor. Rounding instead
  // puts every overlay half a cell off the geometry it is annotating.
  const col = Math.floor(cx) + (options.dx ?? 0)
  const rowY = Math.floor(cy) + (options.dy ?? 0)

  if (options.occlude) {
    if (col < 0 || col >= target.width || rowY < 0 || rowY >= target.height) return false
    // Depth is 1/w, so a larger value is nearer.
    if (target.depth[rowY * target.width + col]! > invW) return false
  }

  drawText(target, col, rowY, text, options)
  return true
}

export interface LineOptions {
  color?: Vec3
  /** Glyph to force; 0 lets `resolve` pick one from luminance. */
  char?: number
  /** Test and write depth, so the line occludes and is occluded. */
  depthTest?: boolean
}

/**
 * Draws a straight line between two world-space points.
 *
 * Clipped against the near plane for the same reason triangles are: an
 * endpoint behind the camera survives the perspective divide with a negative
 * w and lands mirrored, drawing a line to somewhere it does not go. Depth
 * interpolates as 1/w, which is the quantity that is linear in screen space.
 */
export function drawLine3(
  target: RenderTarget,
  from: Vec3,
  to: Vec3,
  viewProjection: Mat4,
  options: LineOptions = {},
): void {
  transformPoint(viewProjection, from.x, from.y, from.z, clip)
  transformPoint(viewProjection, to.x, to.y, to.z, clipB)

  let ax = clip[0]!, ay = clip[1]!, az = clip[2]!, aw = clip[3]!
  let bx = clipB[0]!, by = clipB[1]!, bz = clipB[2]!, bw = clipB[3]!
  const da = az + aw
  const db = bz + bw
  if (da < NEAR_EPSILON && db < NEAR_EPSILON) return

  if (da < NEAR_EPSILON || db < NEAR_EPSILON) {
    const t = da / (da - db)
    const nx = ax + (bx - ax) * t
    const ny = ay + (by - ay) * t
    const nz = az + (bz - az) * t
    const nw = aw + (bw - aw) * t
    if (da < NEAR_EPSILON) {
      ax = nx
      ay = ny
      az = nz
      aw = nw
    } else {
      bx = nx
      by = ny
      bz = nz
      bw = nw
    }
  }

  const aInv = 1 / aw
  const bInv = 1 / bw
  const sx0 = (ax * aInv * 0.5 + 0.5) * target.width
  const sy0 = (0.5 - ay * aInv * 0.5) * target.height
  const sx1 = (bx * bInv * 0.5 + 0.5) * target.width
  const sy1 = (0.5 - by * bInv * 0.5) * target.height

  const color = options.color ?? vec3(1, 1, 1)
  const char = options.char ?? 0
  const depthTest = options.depthTest ?? true

  const steps = Math.max(1, Math.ceil(Math.max(Math.abs(sx1 - sx0), Math.abs(sy1 - sy0))))
  for (let i = 0; i <= steps; i++) {
    const t = i / steps
    // Floor, not round: a screen coordinate lands in the cell below it,
    // because cell i covers [i, i + 1). Rounding shifts the whole line half a
    // cell away from the surfaces it is supposed to sit on.
    const col = Math.floor(sx0 + (sx1 - sx0) * t)
    const rowY = Math.floor(sy0 + (sy1 - sy0) * t)
    if (col < 0 || col >= target.width || rowY < 0 || rowY >= target.height) continue

    const idx = rowY * target.width + col
    const invW = aInv + (bInv - aInv) * t
    if (depthTest && invW <= target.depth[idx]!) continue

    if (depthTest) target.depth[idx] = invW
    target.chars[idx] = char
    const o = idx * 3
    target.color[o] = color.x
    target.color[o + 1] = color.y
    target.color[o + 2] = color.z
  }
}

export interface AxesOptions {
  origin?: Vec3
  length?: number
  /** x, y and z in that order. */
  colors?: [Vec3, Vec3, Vec3]
  /** Put a letter at the far end of each axis. */
  labels?: boolean
  depthTest?: boolean
}

/** Three lines from a point, one along each axis, optionally lettered. */
export function drawAxes(target: RenderTarget, viewProjection: Mat4, options: AxesOptions = {}): void {
  const origin = options.origin ?? vec3(0, 0, 0)
  const length = options.length ?? 1
  const colors = options.colors ?? [vec3(1, 0.4, 0.4), vec3(0.45, 1, 0.5), vec3(0.5, 0.65, 1)]
  const depthTest = options.depthTest ?? true
  const labels = options.labels ?? true

  const ends: Vec3[] = [
    { x: origin.x + length, y: origin.y, z: origin.z },
    { x: origin.x, y: origin.y + length, z: origin.z },
    { x: origin.x, y: origin.y, z: origin.z + length },
  ]

  for (let i = 0; i < 3; i++) {
    drawLine3(target, origin, ends[i]!, viewProjection, { color: colors[i]!, depthTest })
  }
  if (!labels) return
  // After the lines, so a letter is never painted over by its own axis.
  for (let i = 0; i < 3; i++) {
    label3(target, ends[i]!, viewProjection, 'xyz'[i]!, { color: colors[i]!, dx: 1 })
  }
}
