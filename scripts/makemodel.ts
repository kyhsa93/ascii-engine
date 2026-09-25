/**
 * Writes `models/knot.obj`: a (2, 3) torus knot swept into a tube.
 *
 * The OBJ demo needs something that is plainly a *loaded* model rather than
 * one of the built-in primitives, and a knot is that — a few thousand
 * triangles, shared vertices, and a shape no builder in `mesh.ts` produces.
 *
 *   npm run model
 */

import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import type { Mesh } from '../src/core/mesh.ts'
import { writeObj } from '../src/core/mesh.ts'

const P = 2
const Q = 3
const TUBE = 0.55
const ALONG = 128
const AROUND = 12

/** The knot's centre line: a (P, Q) curve on the surface of a torus. */
function curve(t: number): [number, number, number] {
  const r = 2 + Math.cos(Q * t)
  return [r * Math.cos(P * t), r * Math.sin(P * t), Math.sin(Q * t)]
}

function knot(): Mesh {
  const positions: number[] = []
  const normals: number[] = []
  const indices: number[] = []
  const h = 1e-4

  for (let i = 0; i <= ALONG; i++) {
    const t = (i / ALONG) * Math.PI * 2
    const c = curve(t)
    const ahead = curve(t + h)
    const behind = curve(t - h)

    // A Frenet frame, read straight off the curve: the tangent from the first
    // derivative and the normal from the part of the second derivative that
    // is perpendicular to it. This knot's curvature never vanishes, so the
    // frame is defined everywhere and comes back to itself at the seam —
    // which a parallel-transported frame would not do without a correction.
    let tx = ahead[0] - behind[0]
    let ty = ahead[1] - behind[1]
    let tz = ahead[2] - behind[2]
    const tl = Math.hypot(tx, ty, tz)
    tx /= tl
    ty /= tl
    tz /= tl

    const ax = ahead[0] - 2 * c[0] + behind[0]
    const ay = ahead[1] - 2 * c[1] + behind[1]
    const az = ahead[2] - 2 * c[2] + behind[2]
    const along = ax * tx + ay * ty + az * tz
    let nx = ax - along * tx
    let ny = ay - along * ty
    let nz = az - along * tz
    const nl = Math.hypot(nx, ny, nz)
    nx /= nl
    ny /= nl
    nz /= nl

    const bx = ty * nz - tz * ny
    const by = tz * nx - tx * nz
    const bz = tx * ny - ty * nx

    for (let j = 0; j <= AROUND; j++) {
      const v = (j / AROUND) * Math.PI * 2
      const cv = Math.cos(v)
      const sv = Math.sin(v)
      const ox = cv * nx + sv * bx
      const oy = cv * ny + sv * by
      const oz = cv * nz + sv * bz
      normals.push(ox, oy, oz)
      positions.push(c[0] + TUBE * ox, c[1] + TUBE * oy, c[2] + TUBE * oz)
    }
  }

  const stride = AROUND + 1
  for (let i = 0; i < ALONG; i++) {
    for (let j = 0; j < AROUND; j++) {
      const a = i * stride + j
      indices.push(a, a + 1, a + stride, a + 1, a + stride + 1, a + stride)
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices),
  }
}

const mesh = knot()
const out = resolve(process.cwd(), 'models/knot.obj')
mkdirSync(dirname(out), { recursive: true })
writeFileSync(out, writeObj(mesh, `(${P}, ${Q}) torus knot, ${ALONG}x${AROUND} tube`, 4))

console.log(
  `${out}\n  ${mesh.positions.length / 3} vertices, ${mesh.indices.length / 3} triangles`,
)
