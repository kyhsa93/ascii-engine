import type { Camera } from './camera.ts'
import type { Fragment, RenderTarget, Shader, Surface } from './raster.ts'
import type { Sdf } from './sdf.ts'
import { cross, normalize, sub } from './vec3.ts'

export interface MarchOptions {
  /** Give up after this many steps along a ray. */
  maxSteps?: number
  /** A hit is declared once the field falls below this. */
  epsilon?: number
  /** Stop looking past this distance. Defaults to the camera's far plane. */
  maxDistance?: number
}

// A distance field has no vertices, so it has neither texture coordinates nor
// triangle edges: every marched fragment reads (0, 0) for uv and zero for the
// edge distance. An untextured shader ignores the first; a wireframe shader
// would paint a marched surface entirely as wire, which is the honest answer
// to asking a field for the edges it does not have.
const frag: Fragment = {
  px: 0, py: 0, pz: 0,
  nx: 0, ny: 0, nz: 0,
  u: 0, v: 0,
  edge: 0,
  cx: 0, cy: 0,
  invW: 0,
}
const surf: Surface = { r: 0, g: 0, b: 0, char: 0 }

/**
 * Renders a distance field into the same framebuffer the rasterizer writes to,
 * so marched and triangulated geometry can occlude each other correctly.
 *
 * The bridge between the two is the depth convention. The rasterizer stores
 * 1/w, where w is the distance along the *view axis* — not along the ray. A
 * ray that leaves the camera at an angle covers `t * cos(a)` of view axis for
 * `t` of travel, and folding that factor in is the whole of what makes the two
 * paths agree; leaving it out tilts the marched surface toward the camera at
 * the edges of the frame, where it is least obvious and most wrong.
 */
export function marchScene(
  target: RenderTarget,
  field: Sdf,
  camera: Camera,
  aspect: number,
  shader: Shader,
  options: MarchOptions = {},
): void {
  const maxSteps = options.maxSteps ?? 96
  const epsilon = options.epsilon ?? 1e-3
  const maxDistance = options.maxDistance ?? camera.far

  const forward = normalize(sub(camera.target, camera.position))
  const right = normalize(cross(forward, camera.up))
  const up = cross(right, forward)
  const tanHalf = Math.tan(camera.fovY / 2)

  const ex = camera.position.x
  const ey = camera.position.y
  const ez = camera.position.z

  const { width, height, chars, color, depth } = target

  for (let y = 0; y < height; y++) {
    const sy = (1 - ((y + 0.5) / height) * 2) * tanHalf
    for (let x = 0; x < width; x++) {
      const sx = (((x + 0.5) / width) * 2 - 1) * tanHalf * aspect

      let dx = forward.x + right.x * sx + up.x * sy
      let dy = forward.y + right.y * sx + up.y * sy
      let dz = forward.z + right.z * sx + up.z * sy
      const len = Math.hypot(dx, dy, dz)
      dx /= len
      dy /= len
      dz /= len

      const cosA = dx * forward.x + dy * forward.y + dz * forward.z
      const idx = y * width + x
      const existing = depth[idx]!
      // Anything already drawn in this cell caps how far the ray is worth
      // following: past that point the march could only find a loser.
      const limit = existing > 0 ? Math.min(maxDistance, 1 / (existing * cosA)) : maxDistance

      let t = camera.near / cosA
      let hit = -1
      for (let step = 0; step < maxSteps && t <= limit; step++) {
        const d = field(ex + dx * t, ey + dy * t, ez + dz * t)
        if (d < epsilon) {
          hit = t
          break
        }
        t += d
      }
      if (hit < 0) continue

      const invW = 1 / (hit * cosA)
      if (invW <= depth[idx]!) continue

      const px = ex + dx * hit
      const py = ey + dy * hit
      const pz = ez + dz * hit

      // The gradient of the field is its normal. Central differences cost six
      // evaluations and are worth it: forward differences bias the normal
      // along the ray, which shows up as a rim of wrong shading on every
      // silhouette.
      const h = epsilon
      let nx = field(px + h, py, pz) - field(px - h, py, pz)
      let ny = field(px, py + h, pz) - field(px, py - h, pz)
      let nz = field(px, py, pz + h) - field(px, py, pz - h)
      const nlen = Math.hypot(nx, ny, nz) || 1
      nx /= nlen
      ny /= nlen
      nz /= nlen

      frag.px = px
      frag.py = py
      frag.pz = pz
      frag.nx = nx
      frag.ny = ny
      frag.nz = nz
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
