import type { Vec3 } from './vec3.ts'

export interface Mesh {
  /** 3 floats per vertex. */
  positions: Float32Array
  /** 3 floats per vertex, parallel to `positions`. */
  normals: Float32Array
  /** 2 floats per vertex, parallel to `positions`. Absent on an untextured mesh. */
  uvs?: Float32Array
  /** 3 indices per triangle, wound counter-clockwise when seen from outside. */
  indices: Uint32Array
}

interface Builder {
  positions: number[]
  normals: number[]
  uvs: number[]
  indices: number[]
}

function finish(b: Builder): Mesh {
  const mesh: Mesh = {
    positions: new Float32Array(b.positions),
    normals: new Float32Array(b.normals),
    indices: new Uint32Array(b.indices),
  }
  if (b.uvs.length > 0) mesh.uvs = new Float32Array(b.uvs)
  return mesh
}

export function cube(size = 1): Mesh {
  const s = size / 2
  // Each face is listed counter-clockwise as seen from outside, so the
  // rasterizer's back-face test keeps exactly the three faces pointing at
  // the camera.
  const faces: [number[][], number[]][] = [
    [[[-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]], [0, 0, 1]],
    [[[s, -s, -s], [-s, -s, -s], [-s, s, -s], [s, s, -s]], [0, 0, -1]],
    [[[s, -s, s], [s, -s, -s], [s, s, -s], [s, s, s]], [1, 0, 0]],
    [[[-s, -s, -s], [-s, -s, s], [-s, s, s], [-s, s, -s]], [-1, 0, 0]],
    [[[-s, s, s], [s, s, s], [s, s, -s], [-s, s, -s]], [0, 1, 0]],
    [[[-s, -s, -s], [s, -s, -s], [s, -s, s], [-s, -s, s]], [0, -1, 0]],
  ]

  // Corners are listed bottom-left, bottom-right, top-right, top-left, and
  // `v = 0` is the first row of a texture, so the bottom of a face takes v = 1.
  const cornerUv = [
    [0, 1],
    [1, 1],
    [1, 0],
    [0, 0],
  ]

  const b: Builder = { positions: [], normals: [], uvs: [], indices: [] }
  for (const [corners, n] of faces) {
    const base = b.positions.length / 3
    for (let k = 0; k < corners.length; k++) {
      const c = corners[k]!
      b.positions.push(c[0]!, c[1]!, c[2]!)
      b.normals.push(n[0]!, n[1]!, n[2]!)
      b.uvs.push(cornerUv[k]![0]!, cornerUv[k]![1]!)
    }
    b.indices.push(base, base + 1, base + 2, base, base + 2, base + 3)
  }
  return finish(b)
}

export function sphere(radius = 1, segments = 32, rings = 20): Mesh {
  const b: Builder = { positions: [], normals: [], uvs: [], indices: [] }
  for (let i = 0; i <= rings; i++) {
    const phi = (i / rings) * Math.PI
    const sp = Math.sin(phi)
    const cp = Math.cos(phi)
    for (let j = 0; j <= segments; j++) {
      const theta = (j / segments) * Math.PI * 2
      const nx = sp * Math.cos(theta)
      const ny = cp
      const nz = sp * Math.sin(theta)
      b.normals.push(nx, ny, nz)
      b.positions.push(nx * radius, ny * radius, nz * radius)
      // Rings run from the +y pole downward, which is also the direction a
      // texture's rows run, so v needs no flip here.
      b.uvs.push(j / segments, i / rings)
    }
  }
  const stride = segments + 1
  for (let i = 0; i < rings; i++) {
    for (let j = 0; j < segments; j++) {
      const a = i * stride + j
      b.indices.push(a, a + 1, a + stride + 1, a, a + stride + 1, a + stride)
    }
  }
  return finish(b)
}

export function torus(majorRadius = 1, minorRadius = 0.36, majorSegments = 48, minorSegments = 24): Mesh {
  const b: Builder = { positions: [], normals: [], uvs: [], indices: [] }
  for (let i = 0; i <= majorSegments; i++) {
    const u = (i / majorSegments) * Math.PI * 2
    const cu = Math.cos(u)
    const su = Math.sin(u)
    for (let j = 0; j <= minorSegments; j++) {
      const v = (j / minorSegments) * Math.PI * 2
      const cv = Math.cos(v)
      const sv = Math.sin(v)
      const nx = cv * cu
      const ny = sv
      const nz = cv * su
      b.normals.push(nx, ny, nz)
      b.positions.push(majorRadius * cu + minorRadius * nx, minorRadius * ny, majorRadius * su + minorRadius * nz)
      b.uvs.push(i / majorSegments, j / minorSegments)
    }
  }
  const stride = minorSegments + 1
  for (let i = 0; i < majorSegments; i++) {
    for (let j = 0; j < minorSegments; j++) {
      const a = i * stride + j
      b.indices.push(a, a + 1, a + stride, a + 1, a + stride + 1, a + stride)
    }
  }
  return finish(b)
}

/** A flat grid on the xz plane facing +y, centred on the origin. */
export function plane(size = 1, divisions = 1): Mesh {
  const b: Builder = { positions: [], normals: [], uvs: [], indices: [] }
  const step = size / divisions
  const half = size / 2
  for (let i = 0; i <= divisions; i++) {
    for (let j = 0; j <= divisions; j++) {
      b.positions.push(-half + i * step, 0, -half + j * step)
      b.normals.push(0, 1, 0)
      b.uvs.push(i / divisions, j / divisions)
    }
  }
  const stride = divisions + 1
  for (let i = 0; i < divisions; i++) {
    for (let j = 0; j < divisions; j++) {
      const a = i * stride + j
      b.indices.push(a, a + 1, a + stride + 1, a, a + stride + 1, a + stride)
    }
  }
  return finish(b)
}

/** Area-weighted smooth normals, for meshes that arrive without any. */
export function computeNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
  const normals = new Float32Array(positions.length)
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i]! * 3
    const b = indices[i + 1]! * 3
    const c = indices[i + 2]! * 3
    const abx = positions[b]! - positions[a]!
    const aby = positions[b + 1]! - positions[a + 1]!
    const abz = positions[b + 2]! - positions[a + 2]!
    const acx = positions[c]! - positions[a]!
    const acy = positions[c + 1]! - positions[a + 1]!
    const acz = positions[c + 2]! - positions[a + 2]!
    // The un-normalized cross product is twice the triangle's area, so
    // simply accumulating it weights each face by how much surface it owns.
    const nx = aby * acz - abz * acy
    const ny = abz * acx - abx * acz
    const nz = abx * acy - aby * acx
    for (const v of [a, b, c]) {
      normals[v] = normals[v]! + nx
      normals[v + 1] = normals[v + 1]! + ny
      normals[v + 2] = normals[v + 2]! + nz
    }
  }
  for (let i = 0; i < normals.length; i += 3) {
    const x = normals[i]!
    const y = normals[i + 1]!
    const z = normals[i + 2]!
    const len = Math.hypot(x, y, z)
    if (len > 0) {
      normals[i] = x / len
      normals[i + 1] = y / len
      normals[i + 2] = z / len
    }
  }
  return normals
}

/**
 * Reads the subset of Wavefront OBJ that matters here: `v`, `vt`, `vn` and
 * `f`. Faces with more than three corners are fanned, and a face that names a
 * position/texture/normal triple the file has not used before gets its own
 * vertex, so hard edges and texture seams both stay sharp.
 */
export function parseObj(text: string): Mesh {
  const v: number[] = []
  const vt: number[] = []
  const vn: number[] = []
  const positions: number[] = []
  const normals: number[] = []
  const uvs: number[] = []
  const indices: number[] = []
  const seen = new Map<string, number>()
  const hasNormal: boolean[] = []
  let anyUv = false

  const resolve = (raw: number, count: number): number => (raw < 0 ? count + raw : raw - 1)

  const vertexFor = (token: string): number => {
    const parts = token.split('/')
    const pi = resolve(Number(parts[0]), v.length / 3)
    const ti = parts[1] ? resolve(Number(parts[1]), vt.length / 2) : -1
    const ni = parts[2] ? resolve(Number(parts[2]), vn.length / 3) : -1

    // Keyed on the resolved triple rather than the spelling: `1//1` and
    // `1/1/1` may name the same vertex, and caching the text would make two
    // of it, while two faces that differ only in `vt` must stay two.
    const key = `${pi}/${ti}/${ni}`
    const cached = seen.get(key)
    if (cached !== undefined) return cached

    const p = pi * 3
    positions.push(v[p] ?? 0, v[p + 1] ?? 0, v[p + 2] ?? 0)
    if (ni >= 0) {
      const n = ni * 3
      normals.push(vn[n] ?? 0, vn[n + 1] ?? 0, vn[n + 2] ?? 0)
    } else {
      normals.push(0, 0, 0)
    }
    if (ti >= 0) {
      const t = ti * 2
      // OBJ measures v upward from the bottom edge; this engine's textures
      // are stored with v = 0 as the first row, so the axis flips here.
      uvs.push(vt[t] ?? 0, 1 - (vt[t + 1] ?? 0))
      anyUv = true
    } else {
      uvs.push(0, 0)
    }
    hasNormal.push(ni >= 0)

    const index = positions.length / 3 - 1
    seen.set(key, index)
    return index
  }

  for (const line of text.split('\n')) {
    const t = line.trim()
    if (t.length === 0 || t.startsWith('#')) continue
    const parts = t.split(/\s+/)
    const kind = parts[0]
    if (kind === 'v') v.push(Number(parts[1]), Number(parts[2]), Number(parts[3]))
    else if (kind === 'vt') vt.push(Number(parts[1]), Number(parts[2]))
    else if (kind === 'vn') vn.push(Number(parts[1]), Number(parts[2]), Number(parts[3]))
    else if (kind === 'f') {
      const corners = parts.slice(1).map(vertexFor)
      for (let i = 1; i + 1 < corners.length; i++) {
        indices.push(corners[0]!, corners[i]!, corners[i + 1]!)
      }
    }
  }

  const pos = new Float32Array(positions)
  const idx = new Uint32Array(indices)
  const normal = new Float32Array(normals)

  // A file may declare normals and still leave some faces without them. Those
  // vertices would otherwise carry a zero normal, which shades as unlit black
  // rather than as anything obviously broken — so fill in the gaps only, and
  // leave every normal the file did give exactly as written.
  if (hasNormal.includes(false)) {
    const derived = computeNormals(pos, idx)
    for (let i = 0; i < hasNormal.length; i++) {
      if (hasNormal[i]) continue
      normal[i * 3] = derived[i * 3]!
      normal[i * 3 + 1] = derived[i * 3 + 1]!
      normal[i * 3 + 2] = derived[i * 3 + 2]!
    }
  }

  const mesh: Mesh = { positions: pos, indices: idx, normals: normal }
  if (anyUv) mesh.uvs = new Float32Array(uvs)
  return mesh
}

/**
 * Serializes a mesh back to Wavefront OBJ with explicit normals, and texture
 * coordinates when the mesh has them.
 *
 * Every attribute array here is parallel to `positions`, so each vertex emits
 * one `v`, one `vn` and possibly one `vt` at the same index and a face can
 * name them all with the same number.
 */
export function writeObj(mesh: Mesh, name = 'mesh', precision = 6): string {
  const round = (n: number) => {
    const r = Number(n.toFixed(precision))
    return Object.is(r, -0) ? '0' : String(r)
  }
  const uvs = mesh.uvs
  const lines = [`# ${name}`]
  for (let i = 0; i < mesh.positions.length; i += 3) {
    lines.push(`v ${round(mesh.positions[i]!)} ${round(mesh.positions[i + 1]!)} ${round(mesh.positions[i + 2]!)}`)
  }
  if (uvs) {
    // Back to the OBJ convention, measured upward from the bottom edge.
    for (let i = 0; i < uvs.length; i += 2) lines.push(`vt ${round(uvs[i]!)} ${round(1 - uvs[i + 1]!)}`)
  }
  for (let i = 0; i < mesh.normals.length; i += 3) {
    lines.push(`vn ${round(mesh.normals[i]!)} ${round(mesh.normals[i + 1]!)} ${round(mesh.normals[i + 2]!)}`)
  }
  const corner = uvs ? (i: number) => `${i}/${i}/${i}` : (i: number) => `${i}//${i}`
  for (let i = 0; i < mesh.indices.length; i += 3) {
    lines.push(
      `f ${corner(mesh.indices[i]! + 1)} ${corner(mesh.indices[i + 1]! + 1)} ${corner(mesh.indices[i + 2]! + 1)}`,
    )
  }
  return lines.join('\n') + '\n'
}

export interface Bounds {
  min: Vec3
  max: Vec3
  center: Vec3
  /** Distance from `center` to the farthest vertex. */
  radius: number
}

/**
 * The box a mesh occupies, and a sphere around its centre.
 *
 * `boundingRadius` measures from the origin, which is only useful for a mesh
 * that was built there. A file off disk may sit anywhere, so framing one means
 * pointing the camera at its centre and fitting to a radius measured from that
 * centre, not from wherever the origin happens to be.
 */
export function boundingBox(mesh: Mesh): Bounds {
  const p = mesh.positions
  if (p.length === 0) {
    const zero = { x: 0, y: 0, z: 0 }
    return { min: zero, max: zero, center: zero, radius: 0 }
  }

  const min = { x: Infinity, y: Infinity, z: Infinity }
  const max = { x: -Infinity, y: -Infinity, z: -Infinity }
  for (let i = 0; i < p.length; i += 3) {
    min.x = Math.min(min.x, p[i]!)
    min.y = Math.min(min.y, p[i + 1]!)
    min.z = Math.min(min.z, p[i + 2]!)
    max.x = Math.max(max.x, p[i]!)
    max.y = Math.max(max.y, p[i + 1]!)
    max.z = Math.max(max.z, p[i + 2]!)
  }

  const center = { x: (min.x + max.x) / 2, y: (min.y + max.y) / 2, z: (min.z + max.z) / 2 }
  let radius = 0
  for (let i = 0; i < p.length; i += 3) {
    radius = Math.max(radius, Math.hypot(p[i]! - center.x, p[i + 1]! - center.y, p[i + 2]! - center.z))
  }
  return { min, max, center, radius }
}

/** Distance from the origin to the farthest vertex — handy for framing a shot. */
export function boundingRadius(mesh: Mesh): number {
  let max = 0
  for (let i = 0; i < mesh.positions.length; i += 3) {
    max = Math.max(max, Math.hypot(mesh.positions[i]!, mesh.positions[i + 1]!, mesh.positions[i + 2]!))
  }
  return max
}
