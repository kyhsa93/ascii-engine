import type { Shader } from './raster.ts'
import type { Occlusion } from './shadow.ts'
import type { Texture } from './texture.ts'
import { sample } from './texture.ts'
import type { Vec3 } from './vec3.ts'
import { normalize, vec3 } from './vec3.ts'

export interface LambertOptions {
  albedo?: Vec3
  /** Multiplied into the albedo, read at the fragment's texture coordinates. */
  map?: Texture
  /**
   * How much of the light reaches each point. Scales the diffuse and specular
   * terms but never the ambient one — ambient is the light that arrives from
   * everywhere else, and dimming it too turns every shadow into a black hole.
   */
  shadow?: Occlusion
  /** Direction from the surface *toward* the light. Normalized on the way in. */
  light?: Vec3
  lightColor?: Vec3
  /** Fraction of the albedo that survives in shadow. */
  ambient?: number
  /** Blinn-Phong highlight strength; 0 disables the highlight entirely. */
  specular?: number
  shininess?: number
  /** Camera position. Required for a highlight, ignored without one. */
  eye?: Vec3
}

/** Diffuse shading with an optional Blinn-Phong highlight. */
export function lambert(options: LambertOptions = {}): Shader {
  const albedo = options.albedo ?? vec3(1, 1, 1)
  const l = normalize(options.light ?? vec3(0.5, 0.8, 0.6))
  const lightColor = options.lightColor ?? vec3(1, 1, 1)
  const ambient = options.ambient ?? 0.12
  const specular = options.specular ?? 0
  const shininess = options.shininess ?? 32
  const eye = options.eye ?? vec3(0, 0, 0)
  const map = options.map
  const shadow = options.shadow
  const texel = new Float32Array(3)

  return (f, out) => {
    const len = Math.hypot(f.nx, f.ny, f.nz) || 1
    const nx = f.nx / len
    const ny = f.ny / len
    const nz = f.nz / len

    const diffuse = Math.max(0, nx * l.x + ny * l.y + nz * l.z)
    let spec = 0
    if (specular > 0 && diffuse > 0) {
      let vx = eye.x - f.px
      let vy = eye.y - f.py
      let vz = eye.z - f.pz
      const vlen = Math.hypot(vx, vy, vz) || 1
      vx /= vlen
      vy /= vlen
      vz /= vlen
      let hx = l.x + vx
      let hy = l.y + vy
      let hz = l.z + vz
      const hlen = Math.hypot(hx, hy, hz) || 1
      hx /= hlen
      hy /= hlen
      hz /= hlen
      spec = specular * Math.pow(Math.max(0, nx * hx + ny * hy + nz * hz), shininess)
    }

    let ar = albedo.x
    let ag = albedo.y
    let ab = albedo.z
    if (map) {
      sample(map, f.u, f.v, texel)
      ar *= texel[0]!
      ag *= texel[1]!
      ab *= texel[2]!
    }

    // A surface already turned away from the light cannot be shadowed further,
    // and asking would cost a whole second march per cell to learn nothing.
    const lit = shadow && diffuse > 0 ? shadow(f.px, f.py, f.pz) : 1
    const d = diffuse * lit
    spec *= lit

    out.r = Math.min(1, ar * (ambient + d * lightColor.x) + spec)
    out.g = Math.min(1, ag * (ambient + d * lightColor.y) + spec)
    out.b = Math.min(1, ab * (ambient + d * lightColor.z) + spec)
  }
}

/** Paints the raw surface normal as colour. The classic "is my mesh right?" view. */
export function normalColor(): Shader {
  return (f, out) => {
    const len = Math.hypot(f.nx, f.ny, f.nz) || 1
    out.r = f.nx / len * 0.5 + 0.5
    out.g = f.ny / len * 0.5 + 0.5
    out.b = f.nz / len * 0.5 + 0.5
  }
}

export interface WireframeOptions {
  /** Colour of the wire. */
  line?: Vec3
  /** Half-width of the wire, in cells. */
  width?: number
  /**
   * What to draw away from an edge. Given a shader, the interior is that
   * shader's; given nothing, the interior is left blank — which still writes
   * depth, so a solid in front goes on hiding the wires behind it.
   */
  fill?: Shader
  /** Glyph forced on the wire; 0 lets luminance choose one. */
  char?: number
}

/**
 * Draws the edges of each triangle.
 *
 * The width is in cells and stays in cells: the fragment arrives knowing its
 * distance to the nearest edge, so a wire is as thick on a triangle filling
 * the screen as on one a few cells across. Thresholding a barycentric instead
 * — the usual shortcut — makes the wire thin out as the triangle grows.
 *
 * Two things it is honest about. The edges are the *triangle's*, so a quad
 * built from two triangles shows the diagonal between them. And a cell is
 * taller than it is wide, so a wire measured in cells is physically thicker
 * across a horizontal edge than across a vertical one.
 */
export function wireframe(options: WireframeOptions = {}): Shader {
  const line = options.line ?? vec3(0.9, 0.95, 1)
  const width = options.width ?? 0.6
  const fill = options.fill
  const char = options.char ?? 0

  return (f, out) => {
    if (f.edge <= width) {
      out.r = line.x
      out.g = line.y
      out.b = line.z
      out.char = char
      return
    }
    if (fill) {
      fill(f, out)
      return
    }
    // Blank, but still depth-tested: this is what removes hidden lines.
    out.r = 0
    out.g = 0
    out.b = 0
    out.char = 32
  }
}

/** A flat colour, ignoring every light in the scene. */
export function unlit(color: Vec3, char = 0): Shader {
  return (_f, out) => {
    out.r = color.x
    out.g = color.y
    out.b = color.z
    out.char = char
  }
}
