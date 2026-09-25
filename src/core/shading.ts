import type { Shader } from './raster.ts'
import type { Occlusion } from './shadow.ts'
import type { Texture } from './texture.ts'
import { sample } from './texture.ts'
import type { Vec3 } from './vec3.ts'
import { normalize, vec3 } from './vec3.ts'

export interface PointLight {
  /** Where the light is, in world space. */
  position: Vec3
  color?: Vec3
  /** Brightness at one unit of distance, before the range window. */
  intensity?: number
  /**
   * Distance at which the light reaches exactly zero.
   *
   * A plain inverse square never quite ends, so every light would cost every
   * fragment in the scene forever. The falloff is windowed to reach zero *at*
   * the range rather than merely near it, which is what makes the range a
   * bound worth testing against.
   */
  range?: number
  /** Occlusion for this light alone; build it with `shadowFromPoint`. */
  shadow?: Occlusion
}

export interface LambertOptions {
  albedo?: Vec3
  /** Multiplied into the albedo, read at the fragment's texture coordinates. */
  map?: Texture
  /**
   * Point lights, added on top of the directional one.
   *
   * Two things separate these from `light`: the direction is recomputed per
   * fragment, so two surfaces either side of a lamp are lit from opposite
   * sides, and the contribution falls off with distance.
   */
  points?: PointLight[]
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
  const points = options.points ?? []
  const texel = new Float32Array(3)

  return (f, out) => {
    const len = Math.hypot(f.nx, f.ny, f.nz) || 1
    const nx = f.nx / len
    const ny = f.ny / len
    const nz = f.nz / len

    let vx = eye.x - f.px
    let vy = eye.y - f.py
    let vz = eye.z - f.pz
    const vlen = Math.hypot(vx, vy, vz) || 1
    vx /= vlen
    vy /= vlen
    vz /= vlen

    /** Blinn-Phong for one light direction, already normalized. */
    const highlight = (lx: number, ly: number, lz: number): number => {
      if (specular <= 0) return 0
      let hx = lx + vx
      let hy = ly + vy
      let hz = lz + vz
      const hlen = Math.hypot(hx, hy, hz) || 1
      hx /= hlen
      hy /= hlen
      hz /= hlen
      return specular * Math.pow(Math.max(0, nx * hx + ny * hy + nz * hz), shininess)
    }

    // The directional light, unchanged: one direction for the whole scene.
    const facing = Math.max(0, nx * l.x + ny * l.y + nz * l.z)
    // A surface already turned away from the light cannot be shadowed further,
    // and asking would cost a whole second march per cell to learn nothing.
    const litBy = shadow && facing > 0 ? shadow(f.px, f.py, f.pz) : 1
    const reach = facing * litBy

    let dr = reach * lightColor.x
    let dg = reach * lightColor.y
    let db = reach * lightColor.z
    let spec = facing > 0 ? highlight(l.x, l.y, l.z) * litBy : 0

    for (const p of points) {
      let px = p.position.x - f.px
      let py = p.position.y - f.py
      let pz = p.position.z - f.pz
      const d2 = px * px + py * py + pz * pz
      const range = p.range ?? 10
      if (d2 >= range * range) continue

      const dist = Math.sqrt(d2)
      px /= dist || 1
      py /= dist || 1
      pz /= dist || 1
      const towards = Math.max(0, nx * px + ny * py + nz * pz)
      if (towards <= 0) continue

      // Inverse square, windowed so it is exactly zero at the range rather
      // than merely small there. The guard on the denominator keeps a light
      // sitting on a surface from dividing by nothing.
      const s = dist / range
      const window = 1 - s * s * s * s
      const fall = ((p.intensity ?? 1) * window * window) / Math.max(d2, 1e-4)

      const visible = p.shadow ? p.shadow(f.px, f.py, f.pz) : 1
      if (visible <= 0) continue

      const gain = towards * fall * visible
      // White by default, not the directional light's colour. A lamp is its
      // own light: inheriting `lightColor` means dimming the sun silently
      // switches off every lamp in the scene, which is how this was found.
      const color = p.color ?? vec3(1, 1, 1)
      dr += gain * color.x
      dg += gain * color.y
      db += gain * color.z
      spec += highlight(px, py, pz) * fall * visible
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

    out.r = Math.min(1, ar * (ambient + dr) + spec)
    out.g = Math.min(1, ag * (ambient + dg) + spec)
    out.b = Math.min(1, ab * (ambient + db) + spec)
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
