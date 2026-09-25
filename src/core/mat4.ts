import type { Vec3 } from './vec3.ts'
import { cross, dot, normalize, sub } from './vec3.ts'

/**
 * A 4x4 matrix in **row-major** order: `m[row * 4 + col]`.
 *
 * Points are treated as column vectors, so `transformPoint` computes `m * v`
 * and composition reads right-to-left: `multiply(projection, view)` applies
 * view first.
 */
export type Mat4 = Float32Array

export function identity(): Mat4 {
  const m = new Float32Array(16)
  m[0] = m[5] = m[10] = m[15] = 1
  return m
}

export function multiply(a: Mat4, b: Mat4, out: Mat4 = new Float32Array(16)): Mat4 {
  for (let r = 0; r < 4; r++) {
    const a0 = a[r * 4 + 0]!
    const a1 = a[r * 4 + 1]!
    const a2 = a[r * 4 + 2]!
    const a3 = a[r * 4 + 3]!
    for (let c = 0; c < 4; c++) {
      out[r * 4 + c] =
        a0 * b[0 * 4 + c]! + a1 * b[1 * 4 + c]! + a2 * b[2 * 4 + c]! + a3 * b[3 * 4 + c]!
    }
  }
  return out
}

export function multiplyAll(...ms: Mat4[]): Mat4 {
  let acc = identity()
  for (const m of ms) acc = multiply(acc, m)
  return acc
}

export function translation(x: number, y: number, z: number): Mat4 {
  const m = identity()
  m[3] = x
  m[7] = y
  m[11] = z
  return m
}

export function scaling(x: number, y = x, z = x): Mat4 {
  const m = identity()
  m[0] = x
  m[5] = y
  m[10] = z
  return m
}

export function rotationX(rad: number): Mat4 {
  const m = identity()
  const c = Math.cos(rad)
  const s = Math.sin(rad)
  m[5] = c
  m[6] = -s
  m[9] = s
  m[10] = c
  return m
}

export function rotationY(rad: number): Mat4 {
  const m = identity()
  const c = Math.cos(rad)
  const s = Math.sin(rad)
  m[0] = c
  m[2] = s
  m[8] = -s
  m[10] = c
  return m
}

export function rotationZ(rad: number): Mat4 {
  const m = identity()
  const c = Math.cos(rad)
  const s = Math.sin(rad)
  m[0] = c
  m[1] = -s
  m[4] = s
  m[5] = c
  return m
}

/**
 * Right-handed perspective projection. The camera looks down -z in view space,
 * so points in front of it land with clip `w > 0`, and clip `z` maps to
 * [-1, 1] between the near and far planes.
 */
export function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2)
  const m = new Float32Array(16)
  m[0] = f / aspect
  m[5] = f
  m[10] = (far + near) / (near - far)
  m[11] = (2 * far * near) / (near - far)
  m[14] = -1
  return m
}

export function lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const z = normalize(sub(eye, target))
  const x = normalize(cross(up, z))
  const y = cross(z, x)
  const m = identity()
  m[0] = x.x
  m[1] = x.y
  m[2] = x.z
  m[3] = -dot(x, eye)
  m[4] = y.x
  m[5] = y.y
  m[6] = y.z
  m[7] = -dot(y, eye)
  m[8] = z.x
  m[9] = z.y
  m[10] = z.z
  m[11] = -dot(z, eye)
  return m
}

export function transpose(m: Mat4, out: Mat4 = new Float32Array(16)): Mat4 {
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) out[c * 4 + r] = m[r * 4 + c]!
  }
  return out
}

export function invert(m: Mat4, out: Mat4 = new Float32Array(16)): Mat4 {
  const m00 = m[0]!, m01 = m[1]!, m02 = m[2]!, m03 = m[3]!
  const m10 = m[4]!, m11 = m[5]!, m12 = m[6]!, m13 = m[7]!
  const m20 = m[8]!, m21 = m[9]!, m22 = m[10]!, m23 = m[11]!
  const m30 = m[12]!, m31 = m[13]!, m32 = m[14]!, m33 = m[15]!

  const s0 = m00 * m11 - m10 * m01
  const s1 = m00 * m12 - m10 * m02
  const s2 = m00 * m13 - m10 * m03
  const s3 = m01 * m12 - m11 * m02
  const s4 = m01 * m13 - m11 * m03
  const s5 = m02 * m13 - m12 * m03

  const c5 = m22 * m33 - m32 * m23
  const c4 = m21 * m33 - m31 * m23
  const c3 = m21 * m32 - m31 * m22
  const c2 = m20 * m33 - m30 * m23
  const c1 = m20 * m32 - m30 * m22
  const c0 = m20 * m31 - m30 * m21

  const det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0
  if (det === 0) throw new Error('mat4.invert: matrix is singular')
  const d = 1 / det

  out[0] = (m11 * c5 - m12 * c4 + m13 * c3) * d
  out[1] = (-m01 * c5 + m02 * c4 - m03 * c3) * d
  out[2] = (m31 * s5 - m32 * s4 + m33 * s3) * d
  out[3] = (-m21 * s5 + m22 * s4 - m23 * s3) * d

  out[4] = (-m10 * c5 + m12 * c2 - m13 * c1) * d
  out[5] = (m00 * c5 - m02 * c2 + m03 * c1) * d
  out[6] = (-m30 * s5 + m32 * s2 - m33 * s1) * d
  out[7] = (m20 * s5 - m22 * s2 + m23 * s1) * d

  out[8] = (m10 * c4 - m11 * c2 + m13 * c0) * d
  out[9] = (-m00 * c4 + m01 * c2 - m03 * c0) * d
  out[10] = (m30 * s4 - m31 * s2 + m33 * s0) * d
  out[11] = (-m20 * s4 + m21 * s2 - m23 * s0) * d

  out[12] = (-m10 * c3 + m11 * c1 - m12 * c0) * d
  out[13] = (m00 * c3 - m01 * c1 + m02 * c0) * d
  out[14] = (-m30 * s3 + m31 * s1 - m32 * s0) * d
  out[15] = (m20 * s3 - m21 * s1 + m22 * s0) * d

  return out
}

/**
 * The matrix that transforms normals for a given model matrix: the inverse
 * transpose of its upper-left 3x3, so non-uniform scale does not tilt normals
 * off the surface. Returned as a Mat4 with an identity fourth row and column.
 */
export function normalMatrix(model: Mat4): Mat4 {
  const a = model[0]!, b = model[1]!, c = model[2]!
  const d = model[4]!, e = model[5]!, f = model[6]!
  const g = model[8]!, h = model[9]!, i = model[10]!

  const A = e * i - f * h
  const B = -(d * i - f * g)
  const C = d * h - e * g
  const det = a * A + b * B + c * C
  if (det === 0) throw new Error('mat4.normalMatrix: model matrix has no scale')
  const s = 1 / det

  // inverse of the 3x3 is adj/det; transposing it turns the adjugate's
  // columns back into rows, which is why the cofactors land untransposed.
  const out = identity()
  out[0] = A * s
  out[1] = B * s
  out[2] = C * s
  out[4] = -(b * i - c * h) * s
  out[5] = (a * i - c * g) * s
  out[6] = -(a * h - b * g) * s
  out[8] = (b * f - c * e) * s
  out[9] = -(a * f - c * d) * s
  out[10] = (a * e - b * d) * s
  return out
}

/** Transforms a point (w = 1) and writes `[x, y, z, w]` into `out`. */
export function transformPoint(m: Mat4, x: number, y: number, z: number, out: Float32Array, at = 0): void {
  out[at + 0] = m[0]! * x + m[1]! * y + m[2]! * z + m[3]!
  out[at + 1] = m[4]! * x + m[5]! * y + m[6]! * z + m[7]!
  out[at + 2] = m[8]! * x + m[9]! * y + m[10]! * z + m[11]!
  out[at + 3] = m[12]! * x + m[13]! * y + m[14]! * z + m[15]!
}

/** Transforms a direction (w = 0) and writes `[x, y, z]` into `out`. */
export function transformDirection(m: Mat4, x: number, y: number, z: number, out: Float32Array, at = 0): void {
  out[at + 0] = m[0]! * x + m[1]! * y + m[2]! * z
  out[at + 1] = m[4]! * x + m[5]! * y + m[6]! * z
  out[at + 2] = m[8]! * x + m[9]! * y + m[10]! * z
}
