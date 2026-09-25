export interface Mesh {
  /** 3 floats per vertex. */
  positions: Float32Array
  /** 3 floats per vertex, parallel to `positions`. */
  normals: Float32Array
  /** 3 indices per triangle, wound counter-clockwise when seen from outside. */
  indices: Uint32Array
}

interface Builder {
  positions: number[]
  normals: number[]
  indices: number[]
}

function finish(b: Builder): Mesh {
  return {
    positions: new Float32Array(b.positions),
    normals: new Float32Array(b.normals),
    indices: new Uint32Array(b.indices),
  }
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

  const b: Builder = { positions: [], normals: [], indices: [] }
  for (const [corners, n] of faces) {
    const base = b.positions.length / 3
    for (const c of corners) {
      b.positions.push(c[0]!, c[1]!, c[2]!)
      b.normals.push(n[0]!, n[1]!, n[2]!)
    }
    b.indices.push(base, base + 1, base + 2, base, base + 2, base + 3)
  }
  return finish(b)
}

export function sphere(radius = 1, segments = 32, rings = 20): Mesh {
  const b: Builder = { positions: [], normals: [], indices: [] }
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
  const b: Builder = { positions: [], normals: [], indices: [] }
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
  const b: Builder = { positions: [], normals: [], indices: [] }
  const step = size / divisions
  const half = size / 2
  for (let i = 0; i <= divisions; i++) {
    for (let j = 0; j <= divisions; j++) {
      b.positions.push(-half + i * step, 0, -half + j * step)
      b.normals.push(0, 1, 0)
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
 * Reads the subset of Wavefront OBJ that matters here: `v`, `vn` and `f`.
 * Faces with more than three corners are fanned, and a face that names a
 * position/normal pair the file has not used before gets its own vertex, so
 * hard edges stay hard.
 */
export function parseObj(text: string): Mesh {
  const v: number[] = []
  const vn: number[] = []
  const positions: number[] = []
  const normals: number[] = []
  const indices: number[] = []
  const seen = new Map<string, number>()
  let sawNormals = false

  const resolve = (raw: number, count: number): number => (raw < 0 ? count + raw : raw - 1)

  const vertexFor = (token: string): number => {
    const cached = seen.get(token)
    if (cached !== undefined) return cached
    const parts = token.split('/')
    const pi = resolve(Number(parts[0]), v.length / 3) * 3
    positions.push(v[pi] ?? 0, v[pi + 1] ?? 0, v[pi + 2] ?? 0)
    if (parts[2]) {
      const ni = resolve(Number(parts[2]), vn.length / 3) * 3
      normals.push(vn[ni] ?? 0, vn[ni + 1] ?? 0, vn[ni + 2] ?? 0)
      sawNormals = true
    } else {
      normals.push(0, 0, 0)
    }
    const index = positions.length / 3 - 1
    seen.set(token, index)
    return index
  }

  for (const line of text.split('\n')) {
    const t = line.trim()
    if (t.length === 0 || t.startsWith('#')) continue
    const parts = t.split(/\s+/)
    const kind = parts[0]
    if (kind === 'v') v.push(Number(parts[1]), Number(parts[2]), Number(parts[3]))
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
  return { positions: pos, indices: idx, normals: sawNormals ? new Float32Array(normals) : computeNormals(pos, idx) }
}

/** Distance from the origin to the farthest vertex — handy for framing a shot. */
export function boundingRadius(mesh: Mesh): number {
  let max = 0
  for (let i = 0; i < mesh.positions.length; i += 3) {
    max = Math.max(max, Math.hypot(mesh.positions[i]!, mesh.positions[i + 1]!, mesh.positions[i + 2]!))
  }
  return max
}
