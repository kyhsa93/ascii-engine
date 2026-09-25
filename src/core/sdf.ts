/**
 * Signed distance fields: shapes described by a function rather than by
 * triangles.
 *
 * A field returns how far the given point is from the surface — negative
 * inside, positive outside — and that single number is enough to render with,
 * because it also says how far a ray may safely travel before it could
 * possibly hit anything. What it buys over a mesh is combination: two fields
 * can be blended into one surface that neither of them contains, which is not
 * an operation triangles have.
 */
export type Sdf = (x: number, y: number, z: number) => number

export function sdSphere(radius: number): Sdf {
  return (x, y, z) => Math.hypot(x, y, z) - radius
}

/** A box with the given half-extents, centred on the origin. */
export function sdBox(hx: number, hy: number, hz = hy): Sdf {
  return (x, y, z) => {
    const qx = Math.abs(x) - hx
    const qy = Math.abs(y) - hy
    const qz = Math.abs(z) - hz
    // Outside, the distance is to the nearest corner region; inside, every
    // component is negative and the largest of them is the distance to the
    // nearest face.
    const outside = Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0))
    return outside + Math.min(Math.max(qx, qy, qz), 0)
  }
}

/** A torus lying in the xz plane, its hole along y. */
export function sdTorus(major: number, minor: number): Sdf {
  return (x, y, z) => Math.hypot(Math.hypot(x, z) - major, y) - minor
}

/** A half-space: the plane through `offset * normal`, solid below it. */
export function sdPlane(nx: number, ny: number, nz: number, offset = 0): Sdf {
  const len = Math.hypot(nx, ny, nz) || 1
  const ux = nx / len
  const uy = ny / len
  const uz = nz / len
  return (x, y, z) => x * ux + y * uy + z * uz - offset
}

export function union(a: Sdf, b: Sdf): Sdf {
  return (x, y, z) => Math.min(a(x, y, z), b(x, y, z))
}

export function intersect(a: Sdf, b: Sdf): Sdf {
  return (x, y, z) => Math.max(a(x, y, z), b(x, y, z))
}

/** `a` with `b` carved out of it. */
export function subtract(a: Sdf, b: Sdf): Sdf {
  return (x, y, z) => Math.max(a(x, y, z), -b(x, y, z))
}

/**
 * A union with a fillet of radius `k` where the two surfaces meet.
 *
 * This is the operation a mesh cannot do: near the seam the result is closer
 * to the camera than either input, so the blend is a surface in its own right
 * rather than two shapes drawn over each other.
 */
export function smoothUnion(a: Sdf, b: Sdf, k: number): Sdf {
  return (x, y, z) => {
    const da = a(x, y, z)
    const db = b(x, y, z)
    const h = Math.min(1, Math.max(0, 0.5 + (0.5 * (db - da)) / k))
    return db + (da - db) * h - k * h * (1 - h)
  }
}

export function translate(shape: Sdf, x: number, y: number, z: number): Sdf {
  return (px, py, pz) => shape(px - x, py - y, pz - z)
}

export function scale(shape: Sdf, factor: number): Sdf {
  // Sampling a shrunken point and scaling the result back keeps the field a
  // true distance; skipping the multiply would make every ray overstep.
  return (x, y, z) => shape(x / factor, y / factor, z / factor) * factor
}

export function rotateY(shape: Sdf, radians: number): Sdf {
  // Rotating a *shape* means sampling the field at the inverse rotation of
  // the point, which is why the signs here are the transpose of rotationY.
  const c = Math.cos(radians)
  const s = Math.sin(radians)
  return (x, y, z) => shape(c * x - s * z, y, s * x + c * z)
}

export function rotateX(shape: Sdf, radians: number): Sdf {
  const c = Math.cos(radians)
  const s = Math.sin(radians)
  return (x, y, z) => shape(x, c * y + s * z, -s * y + c * z)
}

/** Tiles the shape through space on the given periods; 0 leaves an axis alone. */
export function repeat(shape: Sdf, px: number, py = 0, pz = 0): Sdf {
  const fold = (v: number, p: number) => (p > 0 ? v - p * Math.round(v / p) : v)
  return (x, y, z) => shape(fold(x, px), fold(y, py), fold(z, pz))
}
