/**
 * The scanline rasterizer: clip-space triangles in, character cells out.
 *
 * Everything here is allocation-free once warm. Vertices travel as flat runs
 * of floats rather than objects, and the scratch buffers below are reused
 * across every triangle — which is safe because rasterization never reenters
 * itself.
 */

/** Floats per clip-space vertex: `x y z w  wx wy wz  nx ny nz  u v`. */
export const CLIP_STRIDE = 12

/** How many of those are attributes to interpolate: everything after `x y z w`. */
const ATTRIBUTES = CLIP_STRIDE - 4

/** Floats per projected vertex: `sx sy invW` then the attributes, all /w. */
const SCREEN_STRIDE = 3 + ATTRIBUTES

export interface RenderTarget {
  readonly width: number
  readonly height: number
  readonly chars: Uint32Array
  readonly color: Float32Array
  readonly depth: Float32Array
}

export interface Fragment {
  /** World-space position of this cell's sample point. */
  px: number
  py: number
  pz: number
  /** Interpolated surface normal. Not normalized — do that in the shader. */
  nx: number
  ny: number
  nz: number
  /** Texture coordinates. Zero on a mesh that carries none. */
  u: number
  v: number
  /** Cell coordinates within the target. */
  cx: number
  cy: number
  /** Depth as 1/w. Larger is nearer. */
  invW: number
}

export interface Surface {
  r: number
  g: number
  b: number
  /** Leave at 0 to let `Framebuffer.resolve` choose a glyph from luminance. */
  char: number
}

export type Shader = (f: Fragment, out: Surface) => void

export type Cull = 'back' | 'front' | 'none'

/** A vertex counts as in front of the near plane once `z + w` clears this. */
const NEAR_EPSILON = 1e-6

const clipIn = new Float32Array(8 * CLIP_STRIDE)
const clipOut = new Float32Array(8 * CLIP_STRIDE)
const screen = new Float32Array(8 * SCREEN_STRIDE)

const frag: Fragment = { px: 0, py: 0, pz: 0, nx: 0, ny: 0, nz: 0, u: 0, v: 0, cx: 0, cy: 0, invW: 0 }
const surf: Surface = { r: 0, g: 0, b: 0, char: 0 }

/**
 * Clips a convex polygon against the near plane (`z + w >= 0`) with
 * Sutherland-Hodgman. Without this, vertices behind the camera survive the
 * perspective divide with a negative w and land mirrored on screen.
 */
function clipNear(src: Float32Array, count: number, dst: Float32Array): number {
  let n = 0
  for (let i = 0; i < count; i++) {
    const a = i * CLIP_STRIDE
    const b = ((i + 1) % count) * CLIP_STRIDE
    const da = src[a + 2]! + src[a + 3]!
    const db = src[b + 2]! + src[b + 3]!
    const aIn = da >= NEAR_EPSILON
    const bIn = db >= NEAR_EPSILON

    if (aIn) {
      for (let k = 0; k < CLIP_STRIDE; k++) dst[n * CLIP_STRIDE + k] = src[a + k]!
      n++
    }
    if (aIn !== bIn) {
      const t = da / (da - db)
      const o = n * CLIP_STRIDE
      for (let k = 0; k < CLIP_STRIDE; k++) {
        const va = src[a + k]!
        dst[o + k] = va + (src[b + k]! - va) * t
      }
      n++
    }
  }
  return n
}

function rasterize(
  target: RenderTarget,
  a: number,
  b: number,
  c: number,
  shader: Shader,
  cull: Cull,
): void {
  const ax = screen[a]!, ay = screen[a + 1]!
  const bx = screen[b]!, by = screen[b + 1]!
  const cx = screen[c]!, cy = screen[c + 1]!

  const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
  if (area === 0) return

  // A front face is wound counter-clockwise in NDC, and the viewport flips y,
  // so on screen it comes out clockwise — a negative signed area.
  const front = area < 0
  if (cull === 'back' && !front) return
  if (cull === 'front' && front) return

  const width = target.width
  const minX = Math.max(0, Math.floor(Math.min(ax, bx, cx)))
  const maxX = Math.min(width - 1, Math.ceil(Math.max(ax, bx, cx)))
  const minY = Math.max(0, Math.floor(Math.min(ay, by, cy)))
  const maxY = Math.min(target.height - 1, Math.ceil(Math.max(ay, by, cy)))
  if (minX > maxX || minY > maxY) return

  const invArea = 1 / area
  const chars = target.chars
  const color = target.color
  const depth = target.depth

  for (let y = minY; y <= maxY; y++) {
    const sy = y + 0.5
    const row = y * width
    for (let x = minX; x <= maxX; x++) {
      const sx = x + 0.5

      // Edge functions, each opposite the vertex it weights. Dividing by the
      // signed area normalizes them into barycentrics that sum to 1, and
      // flips the sign for back-facing triangles so one test covers both.
      const l0 = ((cx - bx) * (sy - by) - (cy - by) * (sx - bx)) * invArea
      if (l0 < 0) continue
      const l1 = ((ax - cx) * (sy - cy) - (ay - cy) * (sx - cx)) * invArea
      if (l1 < 0) continue
      const l2 = ((bx - ax) * (sy - ay) - (by - ay) * (sx - ax)) * invArea
      if (l2 < 0) continue

      const invW = l0 * screen[a + 2]! + l1 * screen[b + 2]! + l2 * screen[c + 2]!
      const idx = row + x
      if (invW <= depth[idx]!) continue

      // Attributes were divided by w before interpolation; multiplying the
      // result by w again is what makes them perspective-correct.
      const w = 1 / invW
      frag.px = (l0 * screen[a + 3]! + l1 * screen[b + 3]! + l2 * screen[c + 3]!) * w
      frag.py = (l0 * screen[a + 4]! + l1 * screen[b + 4]! + l2 * screen[c + 4]!) * w
      frag.pz = (l0 * screen[a + 5]! + l1 * screen[b + 5]! + l2 * screen[c + 5]!) * w
      frag.nx = (l0 * screen[a + 6]! + l1 * screen[b + 6]! + l2 * screen[c + 6]!) * w
      frag.ny = (l0 * screen[a + 7]! + l1 * screen[b + 7]! + l2 * screen[c + 7]!) * w
      frag.nz = (l0 * screen[a + 8]! + l1 * screen[b + 8]! + l2 * screen[c + 8]!) * w
      frag.u = (l0 * screen[a + 9]! + l1 * screen[b + 9]! + l2 * screen[c + 9]!) * w
      frag.v = (l0 * screen[a + 10]! + l1 * screen[b + 10]! + l2 * screen[c + 10]!) * w
      frag.cx = x
      frag.cy = y
      frag.invW = invW

      surf.r = 0
      surf.g = 0
      surf.b = 0
      surf.char = 0
      shader(frag, surf)

      depth[idx] = invW
      const o = idx * 3
      color[o] = surf.r
      color[o + 1] = surf.g
      color[o + 2] = surf.b
      chars[idx] = surf.char
    }
  }
}

/**
 * Draws one clip-space triangle: `tri` holds three consecutive
 * `CLIP_STRIDE`-float vertices. Clipping may split it into a fan of up to
 * two triangles, each rasterized in turn.
 */
export function submitTriangle(
  target: RenderTarget,
  tri: Float32Array,
  shader: Shader,
  cull: Cull = 'back',
): void {
  clipIn.set(tri.subarray(0, 3 * CLIP_STRIDE))
  const n = clipNear(clipIn, 3, clipOut)
  if (n < 3) return

  for (let i = 0; i < n; i++) {
    const o = i * CLIP_STRIDE
    const invW = 1 / clipOut[o + 3]!
    const s = i * SCREEN_STRIDE
    screen[s] = (clipOut[o]! * invW * 0.5 + 0.5) * target.width
    screen[s + 1] = (0.5 - clipOut[o + 1]! * invW * 0.5) * target.height
    screen[s + 2] = invW
    for (let k = 0; k < ATTRIBUTES; k++) screen[s + 3 + k] = clipOut[o + 4 + k]! * invW
  }

  for (let i = 1; i + 1 < n; i++) {
    rasterize(target, 0, i * SCREEN_STRIDE, (i + 1) * SCREEN_STRIDE, shader, cull)
  }
}
