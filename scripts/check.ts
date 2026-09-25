/**
 * The engine's test suite. Every assertion here is about an invariant that is
 * easy to break silently while tuning a renderer: winding, depth ordering,
 * clipping, and the non-square cell that makes ASCII output different from a
 * pixel framebuffer.
 *
 * Run with `npm run check`.
 */

import { Camera, aspectFor, fitDistance } from '../src/core/camera.ts'
import { Framebuffer } from '../src/core/framebuffer.ts'
import {
  identity,
  invert,
  lookAt,
  multiply,
  multiplyAll,
  normalMatrix,
  perspective,
  rotationX,
  rotationY,
  scaling,
  transformPoint,
  translation,
} from '../src/core/mat4.ts'
import { existsSync, readFileSync } from 'node:fs'
import {
  boundingBox,
  boundingRadius,
  computeNormals,
  cube,
  parseObj,
  plane,
  sphere,
  torus,
  writeObj,
  type Mesh,
} from '../src/core/mesh.ts'
import { marchScene, type MarchOptions } from '../src/core/march.ts'
import { drawAxes, drawLine3, drawText, label3 } from '../src/core/overlay.ts'
import { RAMPS } from '../src/core/ramp.ts'
import type { Shader } from '../src/core/raster.ts'
import { drawMesh } from '../src/core/renderer.ts'
import { sdBox, sdSphere, sdTorus, smoothUnion, translate, type Sdf } from '../src/core/sdf.ts'
import { shadowFrom, shadowFromPoint } from '../src/core/shadow.ts'
import { lambert, unlit, wireframe } from '../src/core/shading.ts'
import { Supersampler } from '../src/core/supersample.ts'
import { checker, fromAscii, parsePpm, sample, texture, writePpm } from '../src/core/texture.ts'
import { Terminal, to256 } from '../src/term/ansi.ts'
import { cross, normalize, sub, vec3, type Vec3 } from '../src/core/vec3.ts'

let failed = 0

function test(name: string, fn: () => void): void {
  try {
    fn()
    console.log(`  ok    ${name}`)
  } catch (error) {
    failed++
    console.log(`  FAIL  ${name}`)
    console.log(`        ${(error as Error).message}`)
  }
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message)
}

function close(actual: number, expected: number, epsilon: number, message: string): void {
  if (!(Math.abs(actual - expected) <= epsilon)) {
    throw new Error(`${message}: got ${actual}, expected ${expected} (+/- ${epsilon})`)
  }
}

/** A quad facing +z, wound counter-clockwise as seen from +z. */
function quad(size: number, z: number): Mesh {
  const s = size / 2
  return {
    positions: new Float32Array([-s, -s, z, s, -s, z, s, s, z, -s, s, z]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  }
}

interface Sample {
  px: number
  py: number
  pz: number
  nx: number
  ny: number
  nz: number
  u: number
  v: number
  edge: number
  cx: number
  cy: number
  invW: number
}

/** A shader that keeps a copy of every fragment it is handed. */
function recorder(into: Sample[]): Shader {
  return (f, out) => {
    into.push({
      px: f.px,
      py: f.py,
      pz: f.pz,
      nx: f.nx,
      ny: f.ny,
      nz: f.nz,
      u: f.u,
      v: f.v,
      edge: f.edge,
      cx: f.cx,
      cy: f.cy,
      invW: f.invW,
    })
    out.r = 1
    out.g = 1
    out.b = 1
  }
}

function coverage(fb: Framebuffer): number {
  let n = 0
  for (let i = 0; i < fb.depth.length; i++) if (fb.depth[i]! > 0) n++
  return n
}

function bounds(fb: Framebuffer): { w: number; h: number } {
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (let y = 0; y < fb.height; y++) {
    for (let x = 0; x < fb.width; x++) {
      if (fb.depth[y * fb.width + x]! <= 0) continue
      minX = Math.min(minX, x)
      maxX = Math.max(maxX, x)
      minY = Math.min(minY, y)
      maxY = Math.max(maxY, y)
    }
  }
  return { w: maxX - minX + 1, h: maxY - minY + 1 }
}

function colorAt(fb: Framebuffer, x: number, y: number): [number, number, number] {
  const i = (y * fb.width + x) * 3
  return [fb.color[i]!, fb.color[i + 1]!, fb.color[i + 2]!]
}

interface Facing {
  fragments: number
  /** Fragments whose interpolated normal tips past the terminator. */
  past90: number
  /** The most it tipped, as a cosine against the direction to the camera. */
  worstCos: number
}

/**
 * Renders a mesh with back faces culled and measures how its interpolated
 * normals face the camera.
 *
 * "No visible fragment faces away" is the obvious invariant and it is not
 * true. Smooth shading interpolates vertex normals, and at a silhouette they
 * tip a little past ninety degrees: measured here, 0 of a coarse sphere's
 * fragments do, 1 of 1263 on a fine one (90.7 degrees), and 5 of 601 on the
 * torus knot (96.5). The coarse sphere was passing the strict test on luck.
 *
 * A flipped winding is not subtle in the same way — it keeps the far side of
 * the mesh, whose normals point away across the whole frame — so a fraction
 * and a worst case together separate the artifact from the bug where a single
 * unnormalized dot product cannot.
 */
function facingSurvey(mesh: Mesh, camera: Camera, width = 60, height = 30): Facing {
  const samples: Sample[] = []
  const fb = new Framebuffer(width, height)
  fb.clear()
  drawMesh(fb, mesh, identity(), camera.viewProjection(aspectFor(width, height, 0.5)), recorder(samples), 'back')

  let past90 = 0
  let worstCos = 1
  for (const s of samples) {
    const vx = camera.position.x - s.px
    const vy = camera.position.y - s.py
    const vz = camera.position.z - s.pz
    const cos =
      (s.nx * vx + s.ny * vy + s.nz * vz) / (Math.hypot(s.nx, s.ny, s.nz) * Math.hypot(vx, vy, vz) || 1)
    if (cos < 0) past90++
    worstCos = Math.min(worstCos, cos)
  }
  return { fragments: samples.length, past90, worstCos }
}

function assertFacesCamera(label: string, f: Facing): void {
  assert(f.fragments > 200, `${label}: expected a solid silhouette, got ${f.fragments} fragments`)
  assert(
    f.past90 / f.fragments < 0.02,
    `${label}: ${f.past90} of ${f.fragments} fragments face away (${((f.past90 / f.fragments) * 100).toFixed(1)}%)`,
  )
  assert(
    f.worstCos > -0.25,
    `${label}: a fragment faces ${((Math.acos(Math.max(-1, f.worstCos)) * 180) / Math.PI).toFixed(1)} degrees away`,
  )
}

/** A write stream that keeps what was written instead of showing it. */
function fakeStream(columns: number, rows: number): { stream: NodeJS.WriteStream; chunks: string[] } {
  const chunks: string[] = []
  const fake = {
    columns,
    rows,
    write(s: string) {
      chunks.push(s)
      return true
    },
    on() {
      return fake
    },
  }
  return { stream: fake as unknown as NodeJS.WriteStream, chunks }
}

/**
 * Replays an ANSI stream into a character grid the way a terminal would.
 *
 * The presenter only writes the cells that changed, so the only honest way to
 * check its output is to reconstruct the screen it would produce and compare
 * that against the framebuffer — counting escape sequences proves nothing.
 */
function replay(stream: string, width: number, height: number): string {
  const grid = Array.from({ length: height }, () => Array<string>(width).fill(' '))
  let cx = 0
  let cy = 0
  let i = 0
  while (i < stream.length) {
    if (stream[i] === '\x1b' && stream[i + 1] === '[') {
      let j = i + 2
      while (j < stream.length && !/[A-Za-z]/.test(stream[j]!)) j++
      const params = stream.slice(i + 2, j)
      if (stream[j] === 'H') {
        const [r, c] = params.split(';').map(Number)
        cy = (r || 1) - 1
        cx = (c || 1) - 1
      } else if (stream[j] === 'J') {
        for (const row of grid) row.fill(' ')
      }
      i = j + 1
      continue
    }
    if (cy < height && cx < width) grid[cy]![cx] = stream[i]!
    cx++
    i++
  }
  return grid.map((r) => r.join('')).join('\n')
}

console.log('\nmatrices')

test('invert round-trips a composed transform', () => {
  const m = multiplyAll(translation(1, -2, 3), rotationY(0.7), rotationX(-0.4), scaling(2, 0.5, 1.5))
  const round = multiply(invert(m), m)
  const id = identity()
  for (let i = 0; i < 16; i++) close(round[i]!, id[i]!, 1e-4, `element ${i}`)
})

test('perspective maps the near and far planes onto the clip range', () => {
  const near = 0.5
  const far = 50
  const p = perspective(Math.PI / 4, 1.5, near, far)
  const out = new Float32Array(4)

  transformPoint(p, 0, 0, -near, out)
  close(out[3]!, near, 1e-5, 'w at the near plane should equal the view distance')
  close(out[2]! / out[3]!, -1, 1e-4, 'ndc z at the near plane')

  transformPoint(p, 0, 0, -far, out)
  close(out[2]! / out[3]!, 1, 1e-3, 'ndc z at the far plane')
})

test('lookAt places the target straight down -z', () => {
  const eye = vec3(3, 4, 5)
  const v = lookAt(eye, vec3(0, 0, 0), vec3(0, 1, 0))
  const out = new Float32Array(4)
  transformPoint(v, 0, 0, 0, out)
  close(out[0]!, 0, 1e-4, 'view-space x')
  close(out[1]!, 0, 1e-4, 'view-space y')
  close(out[2]!, -Math.hypot(3, 4, 5), 1e-3, 'view-space z')
})

test('normalMatrix keeps normals perpendicular under non-uniform scale', () => {
  // A point on the unit circle and a tangent there. Scaling y by 3 tilts the
  // tangent; naively scaling the normal the same way would not keep the two
  // at right angles.
  const n = vec3(0.6, 0.8, 0)
  const t = vec3(-0.8, 0.6, 0)
  const model = scaling(1, 3, 1)
  const nm = normalMatrix(model)
  const out = new Float32Array(4)

  transformPoint(model, t.x, t.y, t.z, out)
  const tx = out[0]!, ty = out[1]!, tz = out[2]!
  transformPoint(nm, n.x, n.y, n.z, out)
  close(out[0]! * tx + out[1]! * ty + out[2]! * tz, 0, 1e-5, 'normal . tangent after transform')
})

console.log('\nrasterizer')

test('a screen-filling quad covers every cell', () => {
  const fb = new Framebuffer(40, 20)
  fb.clear()
  const camera = new Camera({ position: vec3(0, 0, 1) })
  drawMesh(fb, quad(100, 0), identity(), camera.viewProjection(aspectFor(40, 20)), unlit(vec3(1, 1, 1)))
  assert(coverage(fb) === 800, `expected all 800 cells covered, got ${coverage(fb)}`)
})

test('the depth buffer keeps the nearer surface whatever the draw order', () => {
  const camera = new Camera({ position: vec3(0, 0, 3) })
  const vp = camera.viewProjection(aspectFor(20, 10))
  const near = quad(4, 0)
  const far = quad(4, -1)
  const red = unlit(vec3(1, 0, 0))
  const blue = unlit(vec3(0, 0, 1))

  for (const farFirst of [true, false]) {
    const fb = new Framebuffer(20, 10)
    fb.clear()
    if (farFirst) {
      drawMesh(fb, far, identity(), vp, blue)
      drawMesh(fb, near, identity(), vp, red)
    } else {
      drawMesh(fb, near, identity(), vp, red)
      drawMesh(fb, far, identity(), vp, blue)
    }
    const [r, , b] = colorAt(fb, 10, 5)
    assert(r === 1 && b === 0, `far-first=${farFirst}: expected the near red quad, got rgb(${r}, _, ${b})`)
  }
})

test('near-plane clipping never emits a fragment from behind the camera', () => {
  const fb = new Framebuffer(60, 30)
  fb.clear()
  const camera = new Camera({ position: vec3(0, 0, 0), target: vec3(0, 0, -1), near: 0.1, fovY: Math.PI / 4 })
  // The near edge is in front of the camera and the far corner is behind it,
  // so the triangle has to be cut. Note the corners are deliberately *not*
  // collinear in projection — a sliver would pass this test for free.
  const straddling: Mesh = {
    positions: new Float32Array([-2, -0.4, -3, 2, -0.4, -3, 0, 0.6, 2]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2]),
  }
  const samples: Sample[] = []
  drawMesh(fb, straddling, identity(), camera.viewProjection(aspectFor(60, 30)), recorder(samples), 'none')

  assert(samples.length > 0, 'the visible part of the triangle should still draw')
  for (const s of samples) {
    assert(Number.isFinite(s.invW) && s.invW > 0, `fragment depth should be finite and positive, got ${s.invW}`)
    assert(s.invW <= 1 / camera.near + 1e-3, `fragment nearer than the near plane: 1/w = ${s.invW}`)
    assert(s.pz <= -camera.near + 1e-3, `fragment behind the near plane: world z = ${s.pz}`)
  }

  // Where the fragments land is the part that a missing clip would get wrong.
  // Clipped, the triangle opens upward from its near edge and fills the top of
  // the screen. Unclipped, the behind-the-camera corner projects mirrored and
  // the whole thing collapses into a band across the middle instead.
  assert(fb.depth[0]! > 0, 'the clipped triangle should reach the top-left cell')
  assert(fb.depth[29 * 60]! === 0, 'nothing should reach the bottom row, below the near edge')
  assert(coverage(fb) > 900, `expected the upper screen filled, got ${coverage(fb)} cells`)
})

test('geometry entirely behind the camera draws nothing', () => {
  const fb = new Framebuffer(30, 15)
  fb.clear()
  const camera = new Camera({ position: vec3(0, 0, 0), target: vec3(0, 0, -1) })
  drawMesh(fb, quad(4, 3), identity(), camera.viewProjection(aspectFor(30, 15)), unlit(vec3(1, 1, 1)), 'none')
  assert(coverage(fb) === 0, `expected no coverage, got ${coverage(fb)} cells`)
})

console.log('\nmeshes and winding')

const facingCases: { name: string; mesh: Mesh; eye: ReturnType<typeof vec3>; up?: ReturnType<typeof vec3> }[] = [
  { name: 'cube', mesh: cube(2), eye: vec3(2.5, 2, 3.5) },
  { name: 'sphere', mesh: sphere(1.2), eye: vec3(0, 1.5, 3.5) },
  { name: 'torus', mesh: torus(1.1, 0.4), eye: vec3(1.5, 2.2, 3) },
  { name: 'plane', mesh: plane(3, 4), eye: vec3(0, 3, 0), up: vec3(0, 0, -1) },
]

for (const c of facingCases) {
  test(`${c.name} shows only faces that point at the camera`, () => {
    const camera = new Camera({ position: c.eye, ...(c.up ? { up: c.up } : {}) })
    assertFacesCamera(c.name, facingSurvey(c.mesh, camera))
  })
}

test('a mesh wound inside out fails that same measure', () => {
  // The tolerance above has to be loose enough for smooth shading and tight
  // enough to still be worth having. This is the case it exists to catch: with
  // the winding reversed, culling keeps the far side of the cube, whose
  // normals point away from the camera across the whole silhouette.
  const source = cube(2)
  const flipped: Mesh = {
    positions: source.positions,
    normals: source.normals,
    indices: Uint32Array.from(source.indices),
  }
  for (let i = 0; i < flipped.indices.length; i += 3) {
    const swap = flipped.indices[i]!
    flipped.indices[i] = flipped.indices[i + 2]!
    flipped.indices[i + 2] = swap
  }

  const f = facingSurvey(flipped, new Camera({ position: vec3(2.5, 2, 3.5) }))
  assert(f.fragments > 200, `the flipped cube should still draw something, got ${f.fragments}`)
  assert(
    f.past90 / f.fragments > 0.9,
    `expected nearly every fragment to face away, got ${((f.past90 / f.fragments) * 100).toFixed(1)}%`,
  )
  assert(f.worstCos < -0.5, `expected a decisive failure, worst cosine was ${f.worstCos.toFixed(3)}`)
})

test('back-face culling is redundant for a closed convex mesh', () => {
  // If the winding and the depth test are both right, dropping back faces can
  // only save work — it cannot change the picture.
  const camera = new Camera({ position: vec3(2.5, 2, 3.5) })
  const vp = camera.viewProjection(aspectFor(60, 30))
  const shader = lambert({ albedo: vec3(1, 0.8, 0.5), specular: 0.4, eye: camera.position })
  const frames = (['back', 'none'] as const).map((cull) => {
    const fb = new Framebuffer(60, 30)
    fb.clear()
    drawMesh(fb, cube(2), identity(), vp, shader, cull)
    fb.resolve(RAMPS.long)
    return fb.toString()
  })
  assert(frames[0] === frames[1], 'culled and unculled renders of a convex mesh should match')
})

test('cell aspect correction makes a sphere twice as wide as tall in cells', () => {
  // The projection is told the grid is physically square, so the silhouette of
  // a sphere has to span twice as many cells across as down.
  const fb = new Framebuffer(80, 40)
  fb.clear()
  const camera = new Camera({ position: vec3(0, 0, 4), fovY: Math.PI / 4 })
  drawMesh(fb, sphere(1, 48, 32), identity(), camera.viewProjection(aspectFor(80, 40, 0.5)), unlit(vec3(1, 1, 1)))
  const { w, h } = bounds(fb)
  const ratio = w / h
  assert(ratio > 1.85 && ratio < 2.2, `silhouette ${w}x${h} cells has ratio ${ratio.toFixed(3)}, expected about 2`)
})

test('fitDistance is bound by whichever half angle is tighter', () => {
  const wide = fitDistance(1, Math.PI / 4, 2)
  const square = fitDistance(1, Math.PI / 4, 1)
  const tall = fitDistance(1, Math.PI / 4, 0.4)
  // Past aspect 1 the horizontal half angle is the wider of the two, so the
  // vertical field of view binds and widening the grid further buys nothing.
  // Below it the horizontal angle takes over and the camera has to back off.
  close(wide, square, 1e-9, 'a grid wider than it is tall is bound by the vertical fov')
  assert(tall > square * 1.5, `a portrait grid should need far more room, got ${tall} against ${square}`)
  close(square, 1.1 / Math.sin(Math.PI / 8), 1e-9, 'the square case should reduce to the vertical fit')
})

test('the default framing keeps the subject off the edges, landscape or portrait', () => {
  // The bug this holds shut: a distance chosen for a wide grid slices the
  // subject off at the sides of a portrait one, where the horizontal half
  // angle — not the vertical field of view — is the tighter constraint.
  const mesh = cube(2)
  const radius = boundingRadius(mesh)

  for (const [w, h] of [
    [80, 24],
    [49, 57],
  ] as const) {
    const fb = new Framebuffer(w, h)
    fb.clear()
    const aspect = aspectFor(w, h, 0.574)
    const camera = new Camera({ fovY: Math.PI / 3.2 })
    camera.orbit(0.6, 0.35, fitDistance(radius, camera.fovY, aspect))
    drawMesh(fb, mesh, rotationY(0.8), camera.viewProjection(aspect), unlit(vec3(1, 1, 1)))

    assert(coverage(fb) > 0, `nothing drawn on a ${w}x${h} grid`)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (fb.depth[y * w + x]! <= 0) continue
        assert(
          x > 0 && x < w - 1 && y > 0 && y < h - 1,
          `the subject reaches cell ${x},${y} on a ${w}x${h} grid — it is being cut off`,
        )
      }
    }
  }
})

test('parseObj reads positions, faces and explicit normals', () => {
  const mesh = parseObj(`
    # a quad facing +z
    v -1 -1 0
    v  1 -1 0
    v  1  1 0
    v -1  1 0
    vn 0 0 1
    f 1//1 2//1 3//1 4//1
  `)
  assert(mesh.positions.length === 12, `expected 4 vertices, got ${mesh.positions.length / 3}`)
  assert(mesh.indices.length === 6, `expected the quad fanned into 2 triangles, got ${mesh.indices.length / 3}`)
  close(mesh.normals[2]!, 1, 1e-6, 'the declared normal should survive')
})

test('parseObj derives normals when the file has none', () => {
  const mesh = parseObj('v -1 -1 0\nv 1 -1 0\nv 0 1 0\nf 1 2 3\n')
  close(mesh.normals[2]!, 1, 1e-6, 'a counter-clockwise face seen from +z should get a +z normal')
})

test('computeNormals weights each face by its area', () => {
  // Two triangles meeting at a shared vertex, one ten times the other. The
  // shared normal should lean toward the larger face's orientation.
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 10])
  const indices = new Uint32Array([0, 1, 2, 0, 3, 1])
  const normals = computeNormals(positions, indices)
  close(Math.hypot(normals[0]!, normals[1]!, normals[2]!), 1, 1e-5, 'normals should come out unit length')
  assert(Math.abs(normals[1]!) > Math.abs(normals[2]!), 'the larger face should dominate the shared normal')
})

console.log('\nwavefront obj')

test('parseObj fills in only the normals a file leaves out', () => {
  // A file can declare normals and still have faces that do not use them.
  // Those vertices used to ship a zero normal, which shades as unlit black --
  // a failure that looks like a lighting choice rather than like a bug.
  const mesh = parseObj(`
    v 0 0 0
    v 1 0 0
    v 1 1 0
    v 0 1 0
    vn 0 0 1
    f 1//1 2//1 3//1
    f 1 3 4
  `)
  for (let i = 0; i < mesh.normals.length; i += 3) {
    const len = Math.hypot(mesh.normals[i]!, mesh.normals[i + 1]!, mesh.normals[i + 2]!)
    close(len, 1, 1e-5, `vertex ${i / 3} has no usable normal`)
  }
  // The declared normal must survive untouched, not be averaged away.
  close(mesh.normals[2]!, 1, 1e-6, 'the declared normal was overwritten')
})

test('parseObj treats two spellings of one vertex as one vertex', () => {
  // The same triple written two completely different ways: once counting from
  // the front and once from the back. Keying the vertex cache on the text of
  // an index rather than on what it resolves to would make six vertices here.
  const mesh = parseObj(`
    v 0 0 0
    v 1 0 0
    v 1 1 0
    vn 0 0 1
    f 1//1 2//1 3//1
    f -3//-1 -2//-1 -1//-1
  `)
  assert(mesh.positions.length / 3 === 3, `expected 3 vertices, got ${mesh.positions.length / 3}`)
  assert(mesh.indices.length / 3 === 2, `expected 2 triangles, got ${mesh.indices.length / 3}`)
})

test('parseObj splits a vertex that two faces give different texture coordinates', () => {
  // The other side of the same rule, and the reason the key is a triple: a
  // texture seam is exactly one position appearing with two different uvs, and
  // merging those would drag the seam across the face.
  const mesh = parseObj(`
    v 0 0 0
    v 1 0 0
    v 1 1 0
    vt 0 0
    vt 1 0
    vn 0 0 1
    f 1/1/1 2/1/1 3/1/1
    f 1/2/1 2/1/1 3/1/1
  `)
  assert(mesh.positions.length / 3 === 4, `expected the seam vertex to split, got ${mesh.positions.length / 3}`)
  assert(mesh.uvs !== undefined, 'texture coordinates should have been read')
})

test('writeObj and parseObj round-trip a mesh into the same picture', () => {
  // The strongest statement available about a serializer: the frame drawn
  // from the reloaded mesh is the frame drawn from the original, cell for
  // cell. A cube is the case that matters, because its vertices share
  // positions but not normals -- collapsing those would round the edges off.
  const original = cube(2)
  const reloaded = parseObj(writeObj(original, 'cube'))

  assert(
    reloaded.positions.length === original.positions.length,
    `vertex count changed: ${original.positions.length / 3} -> ${reloaded.positions.length / 3}`,
  )

  const camera = new Camera({ position: vec3(2.6, 2.1, 3.6), fovY: Math.PI / 3.2 })
  const aspect = aspectFor(70, 30, 0.5)
  const shader = lambert({ albedo: vec3(1, 0.85, 0.6), specular: 0.4, eye: camera.position })
  const frames = [original, reloaded].map((m) => {
    const fb = new Framebuffer(70, 30)
    fb.clear(0, 0, 0)
    drawMesh(fb, m, rotationY(0.8), camera.viewProjection(aspect), shader)
    fb.resolve(RAMPS.long)
    return fb.toString()
  })
  assert(frames[0] === frames[1], `the reloaded mesh renders differently:\n${frames[1]}`)
})

test('boundingBox follows a mesh that is not at the origin', () => {
  const shifted = cube(2)
  for (let i = 0; i < shifted.positions.length; i += 3) shifted.positions[i] = shifted.positions[i]! + 5

  const b = boundingBox(shifted)
  close(b.center.x, 5, 1e-6, 'centre x')
  close(b.center.y, 0, 1e-6, 'centre y')
  close(b.radius, Math.sqrt(3), 1e-5, 'radius should be measured from the centre, not the origin')
  // The origin-relative measure is the one that would mis-frame this mesh.
  assert(boundingRadius(shifted) > b.radius * 2, 'boundingRadius should be much larger here')
})

test('the shipped model loads and renders facing the camera', () => {
  const path = 'models/knot.obj'
  assert(existsSync(path), `${path} is missing -- run "npm run model"`)
  const mesh = parseObj(readFileSync(path, 'utf8'))

  assert(mesh.indices.length / 3 > 1000, `expected a few thousand triangles, got ${mesh.indices.length / 3}`)
  for (let i = 0; i < mesh.normals.length; i += 3) {
    const len = Math.hypot(mesh.normals[i]!, mesh.normals[i + 1]!, mesh.normals[i + 2]!)
    assert(len > 0.9, `vertex ${i / 3} of the model has a degenerate normal`)
  }

  // The same facing measure the built-in meshes are held to, which is what
  // would catch a winding reversed by the round-trip through the file.
  const b = boundingBox(mesh)
  const camera = new Camera({ target: b.center, fovY: Math.PI / 3.2 })
  const aspect = aspectFor(74, 32, 0.5)
  camera.orbit(0.7, 0.4, fitDistance(b.radius, camera.fovY, aspect))
  assertFacesCamera('the model', facingSurvey(mesh, camera, 74, 32))

  const fb = new Framebuffer(74, 32)
  fb.clear(0, 0, 0)
  drawMesh(
    fb,
    mesh,
    identity(),
    camera.viewProjection(aspect),
    lambert({ albedo: vec3(0.9, 0.78, 0.55), specular: 0.4, shininess: 28, eye: camera.position }),
  )
  fb.resolve(RAMPS.long)
  console.log('\n' + fb.toString() + '\n')
})

console.log('\ntextures')

test('texture coordinates are perspective-correct across a receding plane', () => {
  // The test that separates a correct interpolator from an affine one, and the
  // reason its ground truth is a ray-plane intersection rather than the
  // fragment's own world position: position and uv ride the same interpolator,
  // so comparing them to each other would pass with both of them wrong.
  const width = 70
  const height = 30
  const camera = new Camera({ position: vec3(0, 1.5, 9), target: vec3(0, 0, -8), fovY: Math.PI / 3 })
  const aspect = aspectFor(width, height, 0.5)

  const samples: Sample[] = []
  const fb = new Framebuffer(width, height)
  fb.clear()
  drawMesh(fb, plane(20, 1), identity(), camera.viewProjection(aspect), recorder(samples))
  assert(samples.length > 400, `expected the plane to fill much of the frame, got ${samples.length}`)

  // Affine and perspective-correct interpolation agree on a frustum that is
  // nearly orthographic, so without a wide spread of depths this would pass
  // for free.
  let nearest = 0
  let farthest = Infinity
  for (const s of samples) {
    nearest = Math.max(nearest, s.invW)
    farthest = Math.min(farthest, s.invW)
  }
  assert(nearest / farthest > 4, `too little perspective to be a real test: ratio ${(nearest / farthest).toFixed(2)}`)

  const forward = normalize(sub(camera.target, camera.position))
  const right = normalize(cross(forward, camera.up))
  const up = cross(right, forward)
  const tanHalf = Math.tan(camera.fovY / 2)

  let worst = 0
  for (const s of samples) {
    const sx = (((s.cx + 0.5) / width) * 2 - 1) * tanHalf * aspect
    const sy = (1 - ((s.cy + 0.5) / height) * 2) * tanHalf
    const dx = forward.x + right.x * sx + up.x * sy
    const dy = forward.y + right.y * sx + up.y * sy
    const dz = forward.z + right.z * sx + up.z * sy
    // The ray need not be normalized: t scales out of the intersection.
    const t = -camera.position.y / dy
    const x = camera.position.x + dx * t
    const z = camera.position.z + dz * t
    worst = Math.max(worst, Math.abs(s.u - (x + 10) / 20), Math.abs(s.v - (z + 10) / 20))
  }
  assert(worst < 2e-3, `uv drifts from the ray-traced truth by ${worst.toFixed(5)}`)
})

test('nearest sampling picks the texel the coordinate lands in', () => {
  const tex = texture(
    2,
    2,
    new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]),
    { filter: 'nearest', wrap: 'clamp' },
  )
  const out = new Float32Array(3)
  const at = (u: number, v: number) => {
    sample(tex, u, v, out)
    return [out[0]!, out[1]!, out[2]!].join(',')
  }
  assert(at(0.25, 0.25) === '1,0,0', `top-left should be red, got ${at(0.25, 0.25)}`)
  assert(at(0.75, 0.25) === '0,1,0', `top-right should be green, got ${at(0.75, 0.25)}`)
  assert(at(0.25, 0.75) === '0,0,1', `bottom-left should be blue, got ${at(0.25, 0.75)}`)
  assert(at(0.75, 0.75) === '1,1,1', `bottom-right should be white, got ${at(0.75, 0.75)}`)
})

test('bilinear sampling is centred on the texel, not on its corner', () => {
  // Texel centres sit at half-integer coordinates. Skip that half-texel shift
  // and a texture slides by half a texel whenever it is magnified -- which is
  // invisible on a photograph and obvious on a checkerboard.
  const tex = texture(
    2,
    2,
    new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]),
    { filter: 'bilinear', wrap: 'clamp' },
  )
  const out = new Float32Array(3)

  sample(tex, 0.25, 0.25, out)
  close(out[0]!, 1, 1e-6, 'a texel centre should return that texel exactly')
  close(out[1]!, 0, 1e-6, 'a texel centre should return that texel exactly')

  sample(tex, 0.5, 0.5, out)
  close(out[0]!, 0.5, 1e-6, 'the middle of four texels averages them')
  close(out[1]!, 0.5, 1e-6, 'the middle of four texels averages them')
  close(out[2]!, 0.5, 1e-6, 'the middle of four texels averages them')
})

test('the three wrap modes disagree exactly where they should', () => {
  const data = new Float32Array([0, 0, 0, 1, 1, 1])
  const out = new Float32Array(3)
  const read = (wrap: 'repeat' | 'clamp' | 'mirror', u: number) => {
    sample(texture(2, 1, data, { filter: 'nearest', wrap }), u, 0.5, out)
    return out[0]!
  }

  // One period along: repeat starts over, clamp holds the last texel, and
  // mirror walks back the way it came.
  close(read('repeat', 1.25), 0, 1e-6, 'repeat should return to the first texel')
  close(read('clamp', 1.25), 1, 1e-6, 'clamp should hold the last texel')
  close(read('mirror', 1.25), 1, 1e-6, 'mirror should be walking back down')
  // And below zero, where clamp and repeat part company again.
  close(read('repeat', -0.25), 1, 1e-6, 'repeat should wrap to the last texel')
  close(read('clamp', -0.25), 0, 1e-6, 'clamp should hold the first texel')
})

test('checker alternates and stays crisp', () => {
  const tex = checker(4)
  assert(tex.filter === 'nearest', 'a checker blurred by bilinear filtering is not a checker')
  const out = new Float32Array(3)
  sample(tex, 0.125, 0.125, out)
  const light = out[0]!
  sample(tex, 0.375, 0.125, out)
  const dark = out[0]!
  assert(light > 0.5 && dark < 0.5, `neighbouring squares should differ, got ${light} and ${dark}`)
})

test('fromAscii reads characters back to the brightness they stand for', () => {
  // The inverse of what resolve does: a picture drawn in characters becomes a
  // texture that can be wrapped around a solid drawn in characters.
  const tex = fromAscii('@ .', RAMPS.short)
  assert(tex.width === 3 && tex.height === 1, `expected a 3x1 texture, got ${tex.width}x${tex.height}`)
  close(tex.data[0]!, 1, 1e-6, 'the brightest ramp character is full brightness')
  close(tex.data[3]!, 0, 1e-6, 'a space is black')
  close(tex.data[6]!, 1 / (RAMPS.short.length - 1), 1e-6, 'the second ramp character is one step up')
})

test('ppm round-trips through the writer and the reader', () => {
  const source = texture(3, 2, Float32Array.from({ length: 18 }, (_, i) => i / 17))
  const back = parsePpm(writePpm(source))
  assert(back.width === 3 && back.height === 2, `dimensions changed: ${back.width}x${back.height}`)
  for (let i = 0; i < source.data.length; i++) {
    close(back.data[i]!, source.data[i]!, 1 / 255, `sample ${i} after a round trip`)
  }
})

test('the ppm reader handles comments and the ascii variant', () => {
  const text = 'P3\n# written by hand\n2 1\n255\n255 0 0  0 128 255\n'
  const tex = parsePpm(Uint8Array.from(text, (c) => c.charCodeAt(0)))
  close(tex.data[0]!, 1, 1e-6, 'first pixel red')
  close(tex.data[1]!, 0, 1e-6, 'first pixel green')
  close(tex.data[5]!, 1, 1e-6, 'second pixel blue')
})

test('writeObj and parseObj round-trip texture coordinates', () => {
  // OBJ measures v upward from the bottom and this engine measures it downward
  // from the first row, so the axis is flipped on the way out and back. A flip
  // applied once would survive every other test in here and show up only as an
  // upside-down picture.
  const original = cube(2)
  assert(original.uvs !== undefined, 'the cube should carry texture coordinates')
  const reloaded = parseObj(writeObj(original, 'cube'))
  assert(reloaded.uvs !== undefined, 'texture coordinates did not survive the round trip')
  assert(
    reloaded.uvs!.length === original.uvs!.length,
    `uv count changed: ${original.uvs!.length / 2} -> ${reloaded.uvs!.length / 2}`,
  )
  for (let i = 0; i < original.uvs!.length; i++) {
    close(reloaded.uvs![i]!, original.uvs![i]!, 1e-6, `uv ${i}`)
  }
})

test('a checkered cube shows one shade per face per square colour', () => {
  // Flat shading gives a cube three shades, one per visible face. A checker
  // with two colours must give exactly six -- more would mean the sampler is
  // bleeding between squares, fewer that the map is not being read per
  // fragment at all.
  const fb = new Framebuffer(80, 36)
  fb.clear(0, 0, 0)
  const camera = new Camera({ position: vec3(2.6, 2.1, 3.6), fovY: Math.PI / 3.2 })
  drawMesh(
    fb,
    cube(2),
    identity(),
    camera.viewProjection(aspectFor(80, 36, 0.5)),
    lambert({ albedo: vec3(1, 0.9, 0.7), light: vec3(0.5, 0.8, 0.6), ambient: 0.12, map: checker(4) }),
  )

  const shades = new Set<string>()
  for (let i = 0; i < fb.depth.length; i++) {
    if (fb.depth[i]! <= 0) continue
    shades.add(fb.color.slice(i * 3, i * 3 + 3).join(','))
  }
  assert(shades.size === 6, `expected 3 faces x 2 square colours, got ${shades.size} shades`)

  fb.resolve(RAMPS.long)
  console.log('\n' + fb.toString() + '\n')
})

console.log('\nraymarching')

test('a marched sphere lands where the rasterized one does', () => {
  // The two paths share nothing but the camera and the depth convention, so
  // agreeing on a silhouette means the ray generation, the aspect handling
  // and the 1/w mapping all match the rasterizer's.
  const camera = new Camera({ position: vec3(0, 0, 4), fovY: Math.PI / 4 })
  const aspect = aspectFor(80, 40, 0.5)
  const white = unlit(vec3(1, 1, 1))

  const rastered = new Framebuffer(80, 40)
  rastered.clear()
  drawMesh(rastered, sphere(1, 64, 48), identity(), camera.viewProjection(aspect), white)

  const marched = new Framebuffer(80, 40)
  marched.clear()
  marchScene(marched, sdSphere(1), camera, aspect, white)

  const a = bounds(rastered)
  const b = bounds(marched)
  assert(Math.abs(a.w - b.w) <= 1 && Math.abs(a.h - b.h) <= 1, `silhouettes differ: ${a.w}x${a.h} vs ${b.w}x${b.h}`)

  const ca = coverage(rastered)
  const cb = coverage(marched)
  assert(Math.abs(ca - cb) / ca < 0.05, `coverage differs by more than 5%: ${ca} vs ${cb}`)
})

test('marched depth agrees with rasterized depth to within a step', () => {
  // Not just the outline: the surface has to sit at the same distance, or the
  // two paths will fight over which is in front.
  const camera = new Camera({ position: vec3(0, 0, 4), fovY: Math.PI / 4 })
  const aspect = aspectFor(60, 30, 0.5)
  const white = unlit(vec3(1, 1, 1))

  const rastered = new Framebuffer(60, 30)
  rastered.clear()
  drawMesh(rastered, sphere(1, 96, 64), identity(), camera.viewProjection(aspect), white)

  const marched = new Framebuffer(60, 30)
  marched.clear()
  marchScene(marched, sdSphere(1), camera, aspect, white)

  let compared = 0
  let worst = 0
  for (let i = 0; i < rastered.depth.length; i++) {
    if (rastered.depth[i]! <= 0 || marched.depth[i]! <= 0) continue
    compared++
    worst = Math.max(worst, Math.abs(1 / rastered.depth[i]! - 1 / marched.depth[i]!))
  }
  assert(compared > 200, `too few shared cells to judge: ${compared}`)
  assert(worst < 0.05, `view-space depth differs by up to ${worst.toFixed(4)} units`)
})

test('a marched box lands where the rasterized cube does', () => {
  // Flat faces render as broad areas of one shade, which is exactly what a
  // wrong box field would also look like. Holding it against a cube mesh is
  // what separates "that is what a box looks like" from a bug.
  const camera = new Camera({ position: vec3(2.5, 2, 3.5), fovY: Math.PI / 3.2 })
  const aspect = aspectFor(70, 34, 0.5)
  const white = unlit(vec3(1, 1, 1))

  const rastered = new Framebuffer(70, 34)
  rastered.clear()
  drawMesh(rastered, cube(1.4), identity(), camera.viewProjection(aspect), white)

  const marched = new Framebuffer(70, 34)
  marched.clear()
  marchScene(marched, sdBox(0.7, 0.7, 0.7), camera, aspect, white)

  const a = bounds(rastered)
  const b = bounds(marched)
  assert(Math.abs(a.w - b.w) <= 1 && Math.abs(a.h - b.h) <= 1, `silhouettes differ: ${a.w}x${a.h} vs ${b.w}x${b.h}`)

  const ca = coverage(rastered)
  const cb = coverage(marched)
  assert(Math.abs(ca - cb) / ca < 0.05, `coverage differs by more than 5%: ${ca} vs ${cb}`)

  let worst = 0
  let compared = 0
  for (let i = 0; i < rastered.depth.length; i++) {
    if (rastered.depth[i]! <= 0 || marched.depth[i]! <= 0) continue
    compared++
    worst = Math.max(worst, Math.abs(1 / rastered.depth[i]! - 1 / marched.depth[i]!))
  }
  assert(compared > 200, `too few shared cells to judge: ${compared}`)
  assert(worst < 0.05, `view-space depth differs by up to ${worst.toFixed(4)} units`)
})

test('marched and rasterized geometry occlude each other both ways', () => {
  const camera = new Camera({ position: vec3(0, 0, 4), fovY: Math.PI / 4 })
  const aspect = aspectFor(40, 20, 0.5)
  const red = unlit(vec3(1, 0, 0))
  const blue = unlit(vec3(0, 0, 1))

  for (const marchFirst of [true, false]) {
    // The quad sits at z = 0; the sphere is pushed behind it, so the quad
    // must win no matter which path draws first.
    const fb = new Framebuffer(40, 20)
    fb.clear()
    const drawQuad = () => drawMesh(fb, quad(4, 0), identity(), camera.viewProjection(aspect), red)
    const drawBall = () => marchScene(fb, translate(sdSphere(1), 0, 0, -2), camera, aspect, blue)
    if (marchFirst) {
      drawBall()
      drawQuad()
    } else {
      drawQuad()
      drawBall()
    }
    const [r, , b] = colorAt(fb, 20, 10)
    assert(r === 1 && b === 0, `march-first=${marchFirst}: the far sphere covered the near quad`)
  }

  // And the other way round: in front, the sphere has to win.
  const fb = new Framebuffer(40, 20)
  fb.clear()
  drawMesh(fb, quad(4, 0), identity(), camera.viewProjection(aspect), red)
  marchScene(fb, translate(sdSphere(0.6), 0, 0, 1.5), camera, aspect, blue)
  const [r, , b] = colorAt(fb, 20, 10)
  assert(r === 0 && b === 1, 'the near sphere did not cover the quad behind it')
})

test('gradient normals come out unit length and pointing outward', () => {
  const camera = new Camera({ position: vec3(0, 0, 4), fovY: Math.PI / 4 })
  const samples: Sample[] = []
  const fb = new Framebuffer(50, 25)
  fb.clear()
  marchScene(fb, sdSphere(1), camera, aspectFor(50, 25, 0.5), recorder(samples))

  assert(samples.length > 100, `expected a solid silhouette, got ${samples.length} hits`)
  for (const s of samples) {
    close(Math.hypot(s.nx, s.ny, s.nz), 1, 1e-3, 'normal length')
    // On a sphere at the origin the outward normal is the position itself.
    const dot = (s.nx * s.px + s.ny * s.py + s.nz * s.pz) / Math.hypot(s.px, s.py, s.pz)
    assert(dot > 0.99, `normal is not radial: cos = ${dot.toFixed(4)}`)
  }
})

test('an empty field draws nothing rather than a wall at max distance', () => {
  // A march that runs out of steps has to report a miss. Reporting the last
  // position instead paints a flat sheet across the whole frame.
  const camera = new Camera({ position: vec3(0, 0, 4), fovY: Math.PI / 4 })
  const fb = new Framebuffer(40, 20)
  fb.clear()
  marchScene(fb, translate(sdSphere(1), 0, 0, -400), camera, aspectFor(40, 20, 0.5), unlit(vec3(1, 1, 1)))
  assert(coverage(fb) === 0, `expected an empty frame, got ${coverage(fb)} cells`)
})

test('smoothUnion pulls the seam in by exactly a quarter of k', () => {
  // Two spheres with a gap between them. Halfway along, both fields are
  // positive -- empty space -- and the blend has to reach into it, or the
  // operation is just a union with extra arithmetic.
  const left = translate(sdSphere(1), -1.4, 0, 0)
  const right = translate(sdSphere(1), 1.4, 0, 0)
  const gap = Math.min(left(0, 0, 0), right(0, 0, 0))
  assert(gap > 0, `the spheres should not touch on their own, got ${gap}`)

  // Where the two fields are equal the blend is at its strongest, and the
  // polynomial's reach there is exactly k/4. That is also its budget: a k
  // smaller than four times the gap cannot close the gap at all.
  for (const k of [0.4, 0.9, 1.6]) {
    close(smoothUnion(left, right, k)(0, 0, 0), gap - k / 4, 1e-9, `reach at k=${k}`)
  }

  // Given enough of it, the neck goes solid -- a surface neither sphere has.
  assert(smoothUnion(left, right, 2)(0, 0, 0) < 0, `k = 2 should close a gap of ${gap.toFixed(2)}`)

  // Far from the seam the blend must not disturb the original surfaces.
  close(smoothUnion(left, right, 0.9)(-1.4, 0, 3), left(-1.4, 0, 3), 1e-9, 'the blend leaked away from the seam')
})

test('sdBox reports true distances inside, on and off the surface', () => {
  // The box is the one primitive whose field is easy to get subtly wrong --
  // too large a value makes rays overstep and punch through corners, too
  // small makes them crawl.
  const box = sdBox(0.7, 0.7, 0.7)
  close(box(0, 0, 0), -0.7, 1e-9, 'the centre is half an extent from the nearest face')
  close(box(0.7, 0, 0), 0, 1e-9, 'a face should sit exactly on the surface')
  close(box(1.7, 0, 0), 1, 1e-9, 'straight out from a face')
  close(box(1.7, 1.7, 0), Math.SQRT2, 1e-9, 'diagonally out from an edge')
  close(box(1.7, 1.7, 1.7), Math.sqrt(3), 1e-9, 'diagonally out from a corner')
  close(box(0, 0.3, 0), -0.4, 1e-9, 'inside, the nearest face is the answer')
})

test('a marched scene renders a gradient over its blend', () => {
  const camera = new Camera({ position: vec3(2.4, 1.8, 3.4), fovY: Math.PI / 3.2 })
  const fb = new Framebuffer(70, 30)
  fb.clear(0, 0, 0)
  const field = smoothUnion(sdSphere(1.05), translate(sdBox(0.7, 0.7, 0.7), 0.9, 0.7, 0.4), 0.55)
  marchScene(
    fb,
    field,
    camera,
    aspectFor(70, 30, 0.5),
    lambert({ albedo: vec3(0.95, 0.8, 0.55), specular: 0.4, shininess: 20, eye: camera.position }),
  )
  fb.resolve(RAMPS.long)

  const glyphs = new Set<number>()
  for (let i = 0; i < fb.chars.length; i++) if (fb.depth[i]! > 0) glyphs.add(fb.chars[i]!)
  assert(glyphs.size >= 8, `expected a smooth gradient, got ${glyphs.size} distinct glyphs`)
  console.log('\n' + fb.toString() + '\n')
})

console.log('\nmarch bounds')

/**
 * Every shape in these checks, with a sphere about the origin that holds it.
 *
 * The radii are measured, not guessed. The first version of this list guessed
 * 2.2 for the blend, which is 0.164 short of the 2.364 it actually needs -- the
 * smoothed box sits off-centre and the blend reaches further than either of its
 * parts. The march then began *inside* the surface and reported a hit 0.24
 * deep, which is exactly the silent loss the option's own documentation warns
 * about, arriving in the check that was supposed to catch it.
 */
const BOUNDED: [string, number, Sdf][] = [
  // Each radius clears the surface rather than touching it. A bound equal to
  // the shape's own radius puts the ray's first sample exactly on the surface,
  // which is a different question from the one these checks ask; it gets its
  // own check below.
  ['sphere', 1.1, sdSphere(1)],
  ['box', 1.74, sdBox(1, 1, 1)],
  ['blend', 2.4, smoothUnion(sdSphere(1.05), translate(sdBox(0.7, 0.7, 0.7), 0.9, 0.7, 0.4), 0.55)],
]

/** Renders `field` twice and reports how the two frames and their costs differ. */
function marchTwice(
  field: Sdf,
  options: MarchOptions,
  radius: number,
  width = 70,
  height = 34,
): { cells: number; silhouette: number; worstDepth: number; evalsWithout: number; evalsWith: number } {
  // Framed the way the demos frame a subject, because the saving depends on
  // how much of the grid the subject covers and a camera pushed up against it
  // measures a case nothing actually renders. The same blend saves 36% from
  // two units away and 76% from where `fitDistance` puts the camera.
  const aspect = aspectFor(width, height, 0.5)
  const camera = new Camera({ position: vec3(0, 0, 5), fovY: Math.PI / 3.2 })
  camera.orbit(-0.6, 0.35, fitDistance(radius, camera.fovY, aspect))
  const white = unlit(vec3(1, 1, 1))

  const run = (opts: MarchOptions) => {
    let evals = 0
    const counted: Sdf = (x, y, z) => {
      evals++
      return field(x, y, z)
    }
    const fb = new Framebuffer(width, height)
    fb.clear()
    marchScene(fb, counted, camera, aspect, white, opts)
    return { fb, evals }
  }

  // `exactOptionalPropertyTypes` is on, so an unbounded render is one with the
  // key absent rather than set to undefined.
  const { bounds: _bounds, ...unbounded } = options
  const plain = run(unbounded)
  const bounded = run(options)

  let cells = 0
  let silhouette = 0
  let worstDepth = 0
  for (let i = 0; i < plain.fb.depth.length; i++) {
    const a = plain.fb.depth[i]!
    const b = bounded.fb.depth[i]!
    if (a > 0 !== b > 0) silhouette++
    else if (a > 0 && a !== b) worstDepth = Math.max(worstDepth, Math.abs(1 / a - 1 / b))

    const sameChar = plain.fb.chars[i] === bounded.fb.chars[i]
    const o = i * 3
    const sameColor =
      plain.fb.color[o] === bounded.fb.color[o] &&
      plain.fb.color[o + 1] === bounded.fb.color[o + 1] &&
      plain.fb.color[o + 2] === bounded.fb.color[o + 2]
    if (!(a === b && sameChar && sameColor)) cells++
  }
  return { cells, silhouette, worstDepth, evalsWithout: plain.evals, evalsWith: bounded.evals }
}

test('a bound that holds the surface draws the same silhouette', () => {
  // Not "the same frame": a bounded ray starts at the sphere's entry point
  // rather than the near plane, so sphere tracing stops at a slightly
  // different place along the same ray and the depth moves by up to epsilon.
  // Measured, the gap is 8.7e-4 against an epsilon of 1e-3 -- termination
  // noise, not geometry. What must not move at all is which cells are hit,
  // because losing or gaining one means the clipping is wrong.
  //
  // Every shape is marched before anything is asserted. Asserting inside the
  // loop stops at the first failure and leaves the rest looking like passes,
  // which is how the blend's bad radius hid behind the sphere's for a round.
  const results = BOUNDED.map(
    ([name, radius, field]) => [name, marchTwice(field, { bounds: { radius } }, radius)] as const,
  )

  const lost = results.filter(([, r]) => r.silhouette > 0)
  assert(
    lost.length === 0,
    `silhouette changed: ${lost.map(([n, r]) => `${n} by ${r.silhouette} cells`).join(', ')}`,
  )

  const EPSILON = 1e-3 // marchScene's default
  const drifted = results.filter(([, r]) => r.worstDepth > EPSILON)
  assert(
    drifted.length === 0,
    `depth moved further than one epsilon: ${drifted
      .map(([n, r]) => `${n} by ${r.worstDepth.toExponential(2)}`)
      .join(', ')}`,
  )
})

test('a bound centred away from the origin follows the shape it holds', () => {
  // The centre has to be used, not assumed. An offset shape with a matching
  // offset bound is identical; with the bound left at the origin it is not,
  // which is what shows the centre is read at all.
  const offset = translate(sdSphere(0.8), 1.5, -0.4, 0.6)
  const moved = marchTwice(offset, { bounds: { radius: 0.9, center: vec3(1.5, -0.4, 0.6) } }, 2.4)
  assert(moved.silhouette === 0, `${moved.silhouette} cells gained or lost with a correctly placed bound`)
  assert(moved.worstDepth <= 1e-3, `depth moved ${moved.worstDepth.toExponential(2)} with a correctly placed bound`)

  const stayed = marchTwice(offset, { bounds: { radius: 0.9 } }, 2.4)
  assert(stayed.silhouette > 0, 'a bound left at the origin still drew the offset sphere: the centre is ignored')
})

test('a bound too small to hold the surface loses part of it', () => {
  // The falsification for the check above. If a wrong bound cost nothing
  // visible, "same silhouette" would be proving nothing about the clipping.
  //
  // This is not hypothetical: the blend's radius in this very list was 0.164
  // too small to begin with, and the damage was not a missing edge but a march
  // that began underneath the surface and reported a hit 0.24 too deep. So a
  // bad bound counts as caught if it moves the silhouette *or* drags the depth
  // well past the epsilon that honest termination noise lives in.
  const kept = BOUNDED.map(
    ([name, radius, field]) => [name, marchTwice(field, { bounds: { radius: radius * 0.5 } }, radius)] as const,
  )
  const unharmed = kept.filter(([, r]) => r.silhouette === 0 && r.worstDepth <= 1e-3)
  assert(
    unharmed.length === 0,
    `halving the bound went unnoticed for ${unharmed.map(([n]) => n).join(', ')}, so the bound is not being applied`,
  )
})

test('the bound is what most rays cost, not the step budget', () => {
  // The measurement that motivated this: on the browser grid, 56% to 69% of
  // every field evaluation in a frame belonged to rays that never come near
  // the subject. Those rays now cost a dot product and no samples at all, so
  // the saving is asserted here rather than written into a comment -- three
  // comments in this repo have already turned out wrong the moment they were
  // measured.
  const savings = BOUNDED.map(([name, radius, field]) => {
    const { evalsWithout, evalsWith } = marchTwice(field, { bounds: { radius } }, radius)
    return [name, 1 - evalsWith / evalsWithout] as const
  })
  const weak = savings.filter(([, saved]) => saved <= 0.4)
  assert(
    weak.length === 0,
    `the bound saved too little: ${weak.map(([n, s]) => `${n} ${(s * 100).toFixed(0)}%`).join(', ')}`,
  )
})

test('an exact radius is already too small a bound, by one epsilon', () => {
  // The reason `bounds` asks for clearance rather than the true radius, and
  // the check that stops the next person from "tidying" the margin away.
  //
  // A march calls anything within epsilon of the surface a hit, so the shape
  // it draws is inflated by that much; the ray/sphere test is exact and knows
  // nothing about the margin. Bounded at exactly 1, a unit sphere loses four
  // rim cells, and every one of them is a ray passing 1.000454 from the centre
  // -- genuinely outside the sphere and genuinely inside the epsilon. This is
  // not a defect in the clipping; it is what epsilon means.
  const exact = marchTwice(sdSphere(1), { bounds: { radius: 1 } }, 1)
  assert(
    exact.silhouette > 0,
    'an exact bound lost no cells, so either epsilon changed or the bound is not being applied',
  )

  // A hair of clearance -- less than epsilon itself -- and the rim comes back.
  const cleared = marchTwice(sdSphere(1), { bounds: { radius: 1.001 } }, 1)
  assert(cleared.silhouette === 0, `epsilon of clearance still lost ${cleared.silhouette} cells`)
})

test('a bound the camera cannot see costs no samples whatsoever', () => {
  // Every ray misses, so the honest cost of the frame is zero evaluations --
  // not "few". A loop that still sampled once per ray would pass a "much
  // cheaper" assertion and fail this one.
  const camera = new Camera({ position: vec3(0, 0, 4), fovY: Math.PI / 4 })
  const aspect = aspectFor(40, 20, 0.5)
  let evals = 0
  const counted: Sdf = (x, y, z) => {
    evals++
    return sdSphere(1)(x, y, z)
  }
  const fb = new Framebuffer(40, 20)
  fb.clear()
  // Behind the camera, and bounded there.
  marchScene(fb, translate(counted, 0, 0, 20), camera, aspect, unlit(vec3(1, 1, 1)), {
    bounds: { radius: 1, center: vec3(0, 0, 20) },
  })
  assert(evals === 0, `expected no field evaluations for a subject behind the camera, got ${evals}`)
  assert(coverage(fb) === 0, 'something was drawn for a subject behind the camera')
})

console.log('\nnormal taps')

/**
 * A light for this section alone.
 *
 * Not the shadow section's `SHADOW_LIGHT`: that is declared below these checks
 * and a `const` read before its declaration throws rather than reading as
 * undefined. Borrowing a constant across sections couples their order for no
 * benefit, so this one is local.
 */
const TAP_LIGHT = vec3(0.55, 0.75, 0.6)

/** The angle between two unit vectors, in degrees. */
function angleBetween(a: Vec3, b: Vec3): number {
  const dot = Math.min(1, Math.max(-1, a.x * b.x + a.y * b.y + a.z * b.z))
  return (Math.acos(dot) * 180) / Math.PI
}

/** Marches a field and hands back every normal the renderer produced. */
function normalsOf(field: Sdf, radius: number, taps: 4 | 6): Map<number, Vec3> {
  const out = new Map<number, Vec3>()
  const aspect = aspectFor(70, 34, 0.5)
  const camera = new Camera({ position: vec3(0, 0, 5), fovY: Math.PI / 3.2 })
  camera.orbit(-0.6, 0.35, fitDistance(radius, camera.fovY, aspect))
  const fb = new Framebuffer(70, 34)
  fb.clear()
  const record: Shader = (f, o) => {
    out.set(f.cy * 70 + f.cx, vec3(f.nx, f.ny, f.nz))
    o.r = 1
    o.g = 1
    o.b = 1
  }
  marchScene(fb, field, camera, aspect, record, {
    maxSteps: 64,
    epsilon: 3e-3,
    bounds: { radius: radius + 0.05 },
    normalTaps: taps,
  })
  return out
}

const SMOOTH: [string, number, Sdf][] = [
  ['sphere', 1.3, sdSphere(1.3)],
  ['torus', 1.52, sdTorus(1.1, 0.42)],
]
const CREASED: [string, number, Sdf][] = [['cube', 1.74, sdBox(1, 1, 1)]]

test('four taps give a unit normal pointing out of the surface', () => {
  // The same bar the six-tap normal is held to. A cheaper gradient is still a
  // gradient or it is not usable at all, whatever it costs.
  const bad: string[] = []
  for (const [name, radius, field] of [...SMOOTH, ...CREASED]) {
    for (const [, n] of normalsOf(field, radius, 4)) {
      const len = Math.hypot(n.x, n.y, n.z)
      if (Math.abs(len - 1) > 1e-6) {
        bad.push(`${name} normal of length ${len.toFixed(6)}`)
        break
      }
    }
  }
  assert(bad.length === 0, bad.join(', '))
})

test('on a smooth surface four taps and six agree to under a degree', () => {
  // Where the cheaper normal is meant to be used, it has to be indistinguishable
  // rather than merely close: a degree of tilt is a whole shade on a ten-level
  // ramp at a grazing angle.
  const drift: string[] = []
  for (const [name, radius, field] of SMOOTH) {
    const six = normalsOf(field, radius, 6)
    const four = normalsOf(field, radius, 4)
    let worst = 0
    for (const [cell, a] of six) {
      const b = four.get(cell)
      if (b) worst = Math.max(worst, angleBetween(a, b))
    }
    if (worst >= 1) drift.push(`${name} by ${worst.toFixed(3)} degrees`)
  }
  assert(drift.length === 0, `four taps drifted: ${drift.join(', ')}`)
})

test('on a creased surface they do not, and that is the cost of the option', () => {
  // The falsification for the check above, and the reason six stays the
  // default. The tetrahedron's taps are not axis-aligned, so at an edge the
  // four of them straddle different faces. Measured over surface points, a
  // cube's worst sample is 36 degrees out -- if this ever stops being true,
  // either the estimator changed or the check is no longer reaching an edge.
  let worst = 0
  for (const [, radius, field] of CREASED) {
    const six = normalsOf(field, radius, 6)
    const four = normalsOf(field, radius, 4)
    for (const [cell, a] of six) {
      const b = four.get(cell)
      if (b) worst = Math.max(worst, angleBetween(a, b))
    }
  }
  assert(worst > 1, `a cube's four-tap normals stayed within ${worst.toFixed(3)} degrees of the six-tap ones`)
})

test('the cheaper normal moves under one cell in a hundred', () => {
  // Degrees are not what anyone sees; glyphs are. The creased case is the one
  // that matters here, because its angular error is the large one -- and it
  // lands almost entirely on points directly over an edge, which is thinner
  // than a cell. Measured, 0.0% to 0.8% of drawn cells change.
  const loud: string[] = []
  for (const [name, radius, field] of [...SMOOTH, ...CREASED]) {
    const aspect = aspectFor(70, 34, 0.5)
    const camera = new Camera({ position: vec3(0, 0, 5), fovY: Math.PI / 3.2 })
    camera.orbit(-0.6, 0.35, fitDistance(radius, camera.fovY, aspect))
    const shade = lambert({ albedo: vec3(0.95, 0.75, 0.45), light: TAP_LIGHT, ambient: 0.1 })

    const render = (taps: 4 | 6) => {
      const fb = new Framebuffer(70, 34)
      fb.clear()
      marchScene(fb, field, camera, aspect, shade, {
        maxSteps: 64,
        epsilon: 3e-3,
        bounds: { radius: radius + 0.05 },
        normalTaps: taps,
      })
      fb.resolve(RAMPS.short)
      return fb
    }
    const a = render(6)
    const b = render(4)
    let drawn = 0
    let differ = 0
    for (let i = 0; i < a.depth.length; i++) {
      if (a.depth[i]! <= 0) continue
      drawn++
      if (a.chars[i] !== b.chars[i]) differ++
    }
    if (differ / drawn > 0.01) loud.push(`${name} ${((differ / drawn) * 100).toFixed(1)}%`)
  }
  assert(loud.length === 0, `four taps changed too much of the picture: ${loud.join(', ')}`)
})

console.log('\nshadows')

const SHADOW_LIGHT = vec3(0.55, 0.75, 0.6)

test('the marched shadow agrees with the exact one, fragment by fragment', () => {
  // A sphere is one of the few casters whose shadow has a closed form: the ray
  // from a floor point toward the light is blocked exactly when it passes
  // within the radius. So this is a comparison against arithmetic the marcher
  // takes no part in — not against a picture, and not against a centroid,
  // which a shadow of the right size in the wrong shape would also satisfy.
  const radius = 1.3
  const l = normalize(SHADOW_LIGHT)
  const blocked = (px: number, py: number, pz: number): boolean => {
    const b = px * l.x + py * l.y + pz * l.z
    const c = px * px + py * py + pz * pz - radius * radius
    if (c <= 0) return true
    if (b >= 0) return false
    return b * b - c >= 0
  }

  const hard = shadowFrom(sdSphere(radius), {
    light: SHADOW_LIGHT,
    bias: 0.03,
    epsilon: 2e-3,
    maxDistance: 12,
  })

  // The rasterizer is here only to hand over a spread of floor positions.
  const samples: Sample[] = []
  const fb = new Framebuffer(120, 40)
  fb.clear()
  const camera = new Camera({ position: vec3(-1.6, 1.04, 2.35), fovY: Math.PI / 3.2 })
  drawMesh(
    fb,
    plane(40, 1),
    translation(0, -radius - 0.2, 0),
    camera.viewProjection(aspectFor(120, 40, 0.574)),
    recorder(samples),
  )
  assert(samples.length > 1500, `expected a lot of floor, got ${samples.length} fragments`)

  let inShadow = 0
  let disagree = 0
  let worst = 0
  for (const s of samples) {
    const exact = blocked(s.px, s.py, s.pz)
    if (exact) inShadow++
    if ((hard(s.px, s.py, s.pz) === 0) === exact) continue
    disagree++
    // A fragment that disagrees should be sitting on the silhouette, where a
    // whisker either way decides it. One sitting anywhere else is a bug.
    const b = s.px * l.x + s.py * l.y + s.pz * l.z
    const perpendicular = Math.sqrt(Math.max(0, s.px * s.px + s.py * s.py + s.pz * s.pz - b * b))
    worst = Math.max(worst, Math.abs(perpendicular - radius))
  }

  assert(inShadow > 200, `the shot holds almost no shadow to compare: ${inShadow} fragments`)
  assert(
    disagree / samples.length < 0.005,
    `${disagree} of ${samples.length} fragments disagree with the exact answer`,
  )
  assert(worst < 0.01, `a disagreement sits ${worst.toFixed(4)} units off the silhouette rather than on it`)
})

test('an occluder out of the way leaves the floor exactly as it was', () => {
  // No false shadows: a shadow function that dims a little everywhere would
  // pass a "the shadow is roughly here" test and fail this one.
  const camera = new Camera({ position: vec3(0, 6, 7), fovY: Math.PI / 3 })
  const vp = camera.viewProjection(aspectFor(50, 25, 0.5))
  const far = shadowFrom(translate(sdSphere(1), 0, 1.5, -400), { light: SHADOW_LIGHT })

  const render = (shadow?: ReturnType<typeof shadowFrom>) => {
    const fb = new Framebuffer(50, 25)
    fb.clear()
    drawMesh(
      fb,
      plane(20, 1),
      identity(),
      vp,
      lambert({ albedo: vec3(0.5, 0.5, 0.6), light: SHADOW_LIGHT, ambient: 0.15, ...(shadow ? { shadow } : {}) }),
    )
    return fb
  }

  const plain = render()
  const shadowed = render(far)
  assert(coverage(plain) > 400, `expected a floor to compare, got ${coverage(plain)} cells`)
  assert(
    plain.color.every((v, i) => v === shadowed.color[i]),
    'an occluder four hundred units away changed the picture',
  )
})

test('a lit surface does not shadow itself, and its far side does', () => {
  // Without the starting bias the first sample sits on the caster, where the
  // field is zero, and every lit surface reports itself as blocked.
  const occlusion = shadowFrom(sdSphere(1), { light: SHADOW_LIGHT })
  const l = normalize(SHADOW_LIGHT)
  assert(occlusion(l.x, l.y, l.z) > 0.9, 'the point facing the light is shadowing itself')
  assert(occlusion(-l.x, -l.y, -l.z) === 0, 'the far side of the sphere should be in its own shadow')
})

test('softness turns a hard edge into a penumbra', () => {
  // Walking across the shadow boundary on the floor. A hard shadow answers
  // only 0 or 1 anywhere along that line; a soft one has to produce values in
  // between, or the softness parameter is decoration.
  const field = translate(sdSphere(1), 0, 1.5, 0)
  const hard = shadowFrom(field, { light: SHADOW_LIGHT })
  const soft = shadowFrom(field, { light: SHADOW_LIGHT, softness: 8 })

  const hardValues = new Set<number>()
  const softValues: number[] = []
  for (let i = 0; i <= 40; i++) {
    const x = -2.6 + (i / 40) * 3
    hardValues.add(hard(x, 0, -1.2))
    softValues.push(soft(x, 0, -1.2))
  }

  assert(hardValues.size === 2, `a hard shadow should be all or nothing, got ${[...hardValues].join(', ')}`)
  for (const v of softValues) assert(v >= 0 && v <= 1, `visibility outside 0..1: ${v}`)
  const partial = softValues.filter((v) => v > 0.02 && v < 0.98)
  assert(partial.length >= 3, `expected a penumbra, got ${partial.length} partial values`)
})

test('a shadow dims the light but never the ambient', () => {
  // The rule that keeps a shadow from being a hole in the picture, checked by
  // driving the shader directly at a point known to be fully occluded.
  const albedo = vec3(0.6, 0.5, 0.4)
  const ambient = 0.2
  const shader = lambert({
    albedo,
    light: SHADOW_LIGHT,
    ambient,
    shadow: shadowFrom(translate(sdSphere(1), 0, 1.5, 0), { light: SHADOW_LIGHT }),
  })

  const frag = { px: -1.1, py: 0, pz: -1.2, nx: 0, ny: 1, nz: 0, u: 0, v: 0, edge: 0, cx: 0, cy: 0, invW: 1 }
  const out = { r: 0, g: 0, b: 0, char: 0 }
  shader(frag, out)

  close(out.r, albedo.x * ambient, 1e-6, 'a fully shadowed point should keep exactly its ambient')
  close(out.g, albedo.y * ambient, 1e-6, 'a fully shadowed point should keep exactly its ambient')
  close(out.b, albedo.z * ambient, 1e-6, 'a fully shadowed point should keep exactly its ambient')
  assert(out.r > 0, 'a shadow that reaches zero is a hole, not a shadow')
})

test('a floor of triangles takes a shadow from a field', () => {
  // The whole point of shadowing by position: the floor is rasterized, the
  // caster is a distance field, and the two paths share nothing else.
  const camera = new Camera({ position: vec3(0, 5, 8), fovY: Math.PI / 3 })
  const aspect = aspectFor(78, 30, 0.5)
  const field = translate(sdSphere(1.1), 0, 1.6, 0)
  const occlusion = shadowFrom(field, { light: SHADOW_LIGHT, softness: 10 })

  const fb = new Framebuffer(78, 30)
  fb.clear(0, 0, 0)
  drawMesh(
    fb,
    plane(24, 1),
    identity(),
    camera.viewProjection(aspect),
    lambert({ albedo: vec3(0.55, 0.57, 0.62), light: SHADOW_LIGHT, ambient: 0.18, shadow: occlusion }),
  )
  marchScene(
    fb,
    field,
    camera,
    aspect,
    lambert({ albedo: vec3(0.95, 0.8, 0.55), light: SHADOW_LIGHT, ambient: 0.12, specular: 0.4, eye: camera.position }),
  )
  fb.resolve(RAMPS.long)

  const glyphs = new Set<number>()
  for (let i = 0; i < fb.chars.length; i++) if (fb.depth[i]! > 0) glyphs.add(fb.chars[i]!)
  assert(glyphs.size >= 6, `expected floor, shadow and ball to separate, got ${glyphs.size} glyphs`)
  console.log('\n' + fb.toString() + '\n')
})

console.log('\nshadow bounds')

/** Floor points to ask an occlusion about, spread wide enough to include misses. */
function floorGrid(y: number, span = 6, step = 0.1): [number, number, number][] {
  const points: [number, number, number][] = []
  for (let z = -span; z <= span; z += step) for (let x = -span; x <= span; x += step) points.push([x, y, z])
  return points
}

const SHADOW_BOUND_FLOOR = floorGrid(-1.5)

test('rejecting rays that cannot reach the caster changes no visibility at all', () => {
  // Exactly identical, unlike the camera marcher's bound: that one moves where
  // a surviving ray starts and so moves where sphere tracing stops, while this
  // one only declines to march rays that could not have dimmed anything. There
  // is no reason for a single value to move, so any movement is a defect.
  //
  // Checked across softness because the margin a penumbra needs is derived
  // from it, and a margin that is right at one softness and wrong at another
  // would otherwise pass on whichever one the test happened to pick.
  const field = sdSphere(1.3)
  const drift: string[] = []

  for (const softness of [0, 4, 12, 48]) {
    const plain = shadowFrom(field, { light: SHADOW_LIGHT, softness, maxDistance: 12 })
    const bounded = shadowFrom(field, { light: SHADOW_LIGHT, softness, maxDistance: 12, casterRadius: 1.3 })
    let worst = 0
    for (const [x, y, z] of SHADOW_BOUND_FLOOR) worst = Math.max(worst, Math.abs(plain(x, y, z) - bounded(x, y, z)))
    if (worst !== 0) drift.push(`directional softness ${softness} by ${worst.toExponential(2)}`)

    const lampPlain = shadowFromPoint(field, vec3(1.6, 1.5, 1.6), { softness })
    const lampBounded = shadowFromPoint(field, vec3(1.6, 1.5, 1.6), { softness, casterRadius: 1.3 })
    let lampWorst = 0
    for (const [x, y, z] of SHADOW_BOUND_FLOOR) {
      lampWorst = Math.max(lampWorst, Math.abs(lampPlain(x, y, z) - lampBounded(x, y, z)))
    }
    if (lampWorst !== 0) drift.push(`lamp softness ${softness} by ${lampWorst.toExponential(2)}`)
  }

  assert(drift.length === 0, `visibility moved: ${drift.join(', ')}`)
})

test('a penumbra is made by the rays that pass wide, so the margin is not optional', () => {
  // The falsification, and a warning to anyone who notices that the camera
  // marcher starts its rays at the bounding sphere and wonders why this one
  // does not. Rejecting at the caster's own radius -- no margin for softness --
  // must visibly break the soft band, or the margin is decoration and the
  // check above is proving nothing.
  //
  // Measured while writing this: a sphere of radius 1.3 with softness 12,
  // rejected at 1.35, takes a floor point from 0.1667 to fully lit.
  const field = sdSphere(1.3)
  const soft = shadowFrom(field, { light: SHADOW_LIGHT, softness: 12, maxDistance: 12 })

  // The same rejection the engine does, but with the margin left out.
  const l = normalize(SHADOW_LIGHT)
  const naive = (x: number, y: number, z: number) => {
    const tca = -x * l.x + -y * l.y + -z * l.z
    if (tca < 0) return 1
    const perp2 = x * x + y * y + z * z - tca * tca
    return perp2 > 1.35 * 1.35 ? 1 : soft(x, y, z)
  }

  let worst = 0
  for (const [x, y, z] of SHADOW_BOUND_FLOOR) worst = Math.max(worst, Math.abs(soft(x, y, z) - naive(x, y, z)))
  assert(worst > 0.5, `a margin-free rejection changed visibility by only ${worst.toExponential(2)}`)

  // A hard shadow has no penumbra, so it needs no softness margin -- but it
  // still needs an epsilon one, and that is the half I got wrong first. A
  // march counts a ray as blocked once the field drops below `epsilon`, so the
  // shadow it casts is the caster inflated by that much, while this rejection
  // is exact geometry. Rejecting a hard shadow at exactly the caster's radius
  // loses the rim of the umbra, the same way the camera marcher's `bounds`
  // loses four cells to rays passing 1.000454 from a unit sphere.
  const hard = shadowFrom(field, { light: SHADOW_LIGHT, maxDistance: 12 })
  const hardNaive = (x: number, y: number, z: number) => {
    const tca = -x * l.x + -y * l.y + -z * l.z
    if (tca < 0) return 1
    return x * x + y * y + z * z - tca * tca > 1.3 * 1.3 ? 1 : hard(x, y, z)
  }
  let hardWorst = 0
  for (const [x, y, z] of SHADOW_BOUND_FLOOR) hardWorst = Math.max(hardWorst, Math.abs(hard(x, y, z) - hardNaive(x, y, z)))
  assert(
    hardWorst > 0,
    'a hard shadow survived a rejection at exactly the caster radius, so either epsilon changed or the rim is not being drawn',
  )
})

test('the bound is most of a shadow pass, not a trim', () => {
  // The measurement that motivated this: on the demo's floor, 91% of every
  // field evaluation in a directional shadow pass belonged to rays marching
  // away from the only caster in the scene. Asserted rather than commented.
  //
  // The two lights get different floors, because their ray geometry differs
  // and one threshold would be a number copied rather than derived. A
  // directional light's rays are parallel, so whether one passes near the
  // caster depends only on where the floor point is; a lamp sits inside the
  // scene, so every ray on the floor fans toward one point a couple of units
  // up and far more of them pass close. Measured over the same floor: the
  // directional bound rejects 12234 of 14641 rays, the lamp 8553. It is not
  // the margin -- giving the directional bound the lamp's widest margin still
  // rejects 12369.
  //
  // The floors sit below the measured savings rather than on them. Measured
  // here: directional 64.9% (22330 evaluations against 63561), lamp 33.1%
  // (50645 against 75669). A floor pressed up against the measurement is a
  // check that fails one day for no reason at all, which is a worse outcome
  // than one that never fires.
  const FLOORS: Record<string, number> = { directional: 0.55, lamp: 0.25 }
  const counts: [string, number][] = []

  for (const [name, make] of [
    [
      'directional',
      (counted: Sdf, casterRadius?: number) =>
        shadowFrom(counted, {
          light: SHADOW_LIGHT,
          softness: 12,
          maxDistance: 12,
          ...(casterRadius === undefined ? {} : { casterRadius }),
        }),
    ],
    [
      'lamp',
      (counted: Sdf, casterRadius?: number) =>
        shadowFromPoint(counted, vec3(1.6, 1.5, 1.6), {
          softness: 12,
          ...(casterRadius === undefined ? {} : { casterRadius }),
        }),
    ],
  ] as const) {
    const spend = (casterRadius?: number) => {
      let evals = 0
      const counted: Sdf = (x, y, z) => {
        evals++
        return sdSphere(1.3)(x, y, z)
      }
      const occl = make(counted, casterRadius)
      for (const [x, y, z] of SHADOW_BOUND_FLOOR) occl(x, y, z)
      return evals
    }
    counts.push([name, 1 - spend(1.3) / spend()])
  }

  const weak = counts.filter(([name, saved]) => saved <= FLOORS[name]!)
  assert(
    weak.length === 0,
    `the bound saved too little: ${weak
      .map(([n, s]) => `${n} ${(s * 100).toFixed(0)}% against a floor of ${(FLOORS[n]! * 100).toFixed(0)}%`)
      .join(', ')}`,
  )
})

console.log('\nsupersampling')

/** A white half-plane whose right edge sits at `edgeX`, facing +z. */
function halfPlane(edgeX: number): Mesh {
  return {
    positions: new Float32Array([-50, -50, 0, edgeX, -50, 0, edgeX, 50, 0, -50, 50, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  }
}

test('an edge cell comes out at exactly its coverage fraction', () => {
  // The projection is simple enough to invert by hand, so where the edge lands
  // on the grid is known arithmetic rather than something read off the render.
  // The cell it falls in covers a whole number of sub-columns out of `factor`,
  // and the averaged brightness has to be that fraction exactly -- not close
  // to it, and not merely darker than its neighbour.
  const width = 40
  const height = 20
  const factor = 4
  const edgeX = 0.2
  const fovY = Math.PI / 4
  const aspect = aspectFor(width, height, 0.5)

  const f = 1 / Math.tan(fovY / 2)
  // The quad sits at z = 0 and the camera two units back, so clip w is 2.
  const edgeNdc = ((f / aspect) * edgeX) / 2
  const edgeCell = (edgeNdc * 0.5 + 0.5) * width
  const cell = Math.floor(edgeCell)
  let covered = 0
  for (let i = 0; i < factor; i++) if (cell + (i + 0.5) / factor < edgeCell) covered++
  assert(covered > 0 && covered < factor, `the edge should split this cell, ${covered} of ${factor} covered`)

  const camera = new Camera({ position: vec3(0, 0, 2), fovY })
  const vp = camera.viewProjection(aspect)
  const row = height >> 1

  const aa = new Supersampler(width, height, factor)
  aa.clear()
  drawMesh(aa.buffer, halfPlane(edgeX), identity(), vp, unlit(vec3(1, 1, 1)))
  const fb = new Framebuffer(width, height)
  aa.resolveInto(fb)

  close(colorAt(fb, cell, row)[0], covered / factor, 1e-6, `the edge cell should be ${covered}/${factor} lit`)
  close(colorAt(fb, cell - 1, row)[0], 1, 1e-6, 'the cell inside the edge should be fully lit')
  close(colorAt(fb, cell + 1, row)[0], 0, 1e-6, 'the cell outside the edge should be untouched')

  // And the staircase this replaces: one sample per cell can only answer in or
  // out, so the same cell comes back fully lit.
  const plain = new Framebuffer(width, height)
  plain.clear()
  drawMesh(plain, halfPlane(edgeX), identity(), vp, unlit(vec3(1, 1, 1)))
  close(colorAt(plain, cell, row)[0], 1, 1e-6, 'without supersampling the edge cell should be all or nothing')
})

test('supersampling changes nothing where there is no edge', () => {
  // Averaging identical samples has to be the identity, or this is a blur
  // rather than an antialiaser.
  const camera = new Camera({ position: vec3(0, 0, 1) })
  const vp = camera.viewProjection(aspectFor(40, 20))
  const white = unlit(vec3(1, 1, 1))

  const plain = new Framebuffer(40, 20)
  plain.clear()
  drawMesh(plain, quad(100, 0), identity(), vp, white)

  const aa = new Supersampler(40, 20, 3)
  aa.clear()
  drawMesh(aa.buffer, quad(100, 0), identity(), vp, white)
  const resolved = new Framebuffer(40, 20)
  aa.resolveInto(resolved)

  assert(coverage(plain) === 800 && coverage(resolved) === 800, 'both renders should cover the whole grid')
  assert(
    plain.color.every((v, i) => Math.abs(v - resolved.color[i]!) < 1e-6),
    'a fully covered frame came out different after resampling',
  )
})

test('a resolved cell keeps the nearest depth, not the average one', () => {
  // A partly covered cell has to go on occluding what is behind it. An
  // averaged depth would put it somewhere between the surface and the
  // background, which is where nothing is.
  const camera = new Camera({ position: vec3(0, 0, 3) })
  const vp = camera.viewProjection(aspectFor(40, 20))

  const aa = new Supersampler(40, 20, 4)
  aa.clear()
  drawMesh(aa.buffer, halfPlane(0.2), identity(), vp, unlit(vec3(1, 0, 0)))
  const fb = new Framebuffer(40, 20)
  aa.resolveInto(fb)

  const edge = fb.color.slice(0, fb.color.length)
  // Now a farther quad across the whole grid: it must lose everywhere the
  // near half-plane left any coverage at all, edge cells included.
  drawMesh(fb, quad(100, -1), identity(), vp, unlit(vec3(0, 0, 1)))

  let partial = 0
  for (let i = 0; i < fb.depth.length; i++) {
    const before = edge[i * 3]!
    if (before <= 0 || before >= 1) continue
    partial++
    close(fb.color[i * 3]!, before, 1e-6, `a partly covered cell at ${i % 40},${(i / 40) | 0} was overdrawn`)
  }
  assert(partial > 5, `expected a column of partly covered cells, found ${partial}`)
})

test('a glyph a shader forced survives the averaging', () => {
  // Colour can be averaged and a character cannot, so the nearest sub-sample's
  // choice wins outright and that edge stays hard.
  const camera = new Camera({ position: vec3(0, 0, 1) })
  const aa = new Supersampler(8, 2, 3)
  aa.clear()
  drawMesh(aa.buffer, quad(100, 0), identity(), camera.viewProjection(aspectFor(8, 2)), unlit(vec3(1, 1, 1), 64))
  const fb = new Framebuffer(8, 2)
  aa.resolveInto(fb)
  fb.resolve(RAMPS.short)
  assert(fb.toString() === '@@@@@@@@\n@@@@@@@@', `forced glyphs did not survive:\n${fb.toString()}`)
})

test('a supersampled silhouette gains shades a single sample cannot have', () => {
  const camera = new Camera({ position: vec3(0, 0, 4), fovY: Math.PI / 4 })
  const vp = camera.viewProjection(aspectFor(60, 28, 0.5))
  const white = unlit(vec3(1, 1, 1))

  const plain = new Framebuffer(60, 28)
  plain.clear()
  drawMesh(plain, sphere(1.2, 64, 48), identity(), vp, white)

  const aa = new Supersampler(60, 28, 3)
  aa.clear()
  drawMesh(aa.buffer, sphere(1.2, 64, 48), identity(), vp, white)
  const smoothed = new Framebuffer(60, 28)
  aa.resolveInto(smoothed)

  const shades = (fb: Framebuffer) => {
    const seen = new Set<number>()
    for (let i = 0; i < fb.depth.length; i++) seen.add(Math.round(fb.color[i * 3]! * 1000))
    return seen.size
  }
  assert(shades(plain) === 2, `one sample per cell can only be in or out, got ${shades(plain)} shades`)
  assert(shades(smoothed) >= 5, `expected a range of partial coverage, got ${shades(smoothed)} shades`)

  smoothed.resolve(RAMPS.long)
  console.log('\n' + smoothed.toString() + '\n')
})

console.log('\nwireframe')

test('the edge distance is the real perpendicular distance, in cells', () => {
  // Ground truth is plane geometry on the projected triangle, worked out from
  // the camera by hand. The rasterizer's own barycentrics take no part in it.
  const width = 50
  const height = 26
  const fovY = Math.PI / 4
  const aspect = aspectFor(width, height, 0.5)
  const f = 1 / Math.tan(fovY / 2)
  const camera = new Camera({ position: vec3(0, 0, 3), fovY })

  // A triangle at z = 0, so every vertex divides by the same w of 3.
  const corners: [number, number][] = [
    [-1.1, -0.9],
    [1.3, -0.7],
    [0.1, 1.2],
  ]
  const mesh: Mesh = {
    positions: new Float32Array(corners.flatMap(([x, y]) => [x, y, 0])),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2]),
  }

  const screen = corners.map(([x, y]) => [
    (((f / aspect) * x) / 3 / 2 + 0.5) * width,
    (0.5 - (f * y) / 3 / 2) * height,
  ])

  const distanceToSegmentLine = (px: number, py: number, p: number[], q: number[]): number => {
    const ex = q[0]! - p[0]!
    const ey = q[1]! - p[1]!
    // The line through p and q, not the segment: a triangle's edge extends to
    // its corners, and that is where the three distances meet.
    return Math.abs(ex * (py - p[1]!) - ey * (px - p[0]!)) / Math.hypot(ex, ey)
  }

  const samples: Sample[] = []
  const fb = new Framebuffer(width, height)
  fb.clear()
  drawMesh(fb, mesh, identity(), camera.viewProjection(aspect), recorder(samples), 'none')
  assert(samples.length > 100, `expected a solid triangle, got ${samples.length} fragments`)

  let worst = 0
  for (const s of samples) {
    const px = s.cx + 0.5
    const py = s.cy + 0.5
    const expected = Math.min(
      distanceToSegmentLine(px, py, screen[1]!, screen[2]!),
      distanceToSegmentLine(px, py, screen[2]!, screen[0]!),
      distanceToSegmentLine(px, py, screen[0]!, screen[1]!),
    )
    worst = Math.max(worst, Math.abs(s.edge - expected))
  }
  assert(worst < 1e-3, `the edge distance is off by up to ${worst.toFixed(5)} cells`)
})

test('the wire keeps its width however big the triangle gets', () => {
  // The reason the fragment carries a distance and not a barycentric. A
  // barycentric threshold is a fraction of the triangle, so the same setting
  // would draw a wire several times thicker on the near quad than the far one.
  const camera = new Camera({ position: vec3(0, 0, 4), fovY: Math.PI / 4 })
  const vp = camera.viewProjection(aspectFor(90, 44, 0.5))
  const line = vec3(1, 1, 1)

  const runAt = (z: number) => {
    const fb = new Framebuffer(90, 44)
    fb.clear()
    drawMesh(fb, quad(3, z), identity(), vp, wireframe({ line, width: 2.5 }))

    let minX = Infinity
    let maxX = -1
    for (let x = 0; x < 90; x++) {
      for (let y = 0; y < 44; y++) {
        if (fb.depth[y * 90 + x]! <= 0) continue
        minX = Math.min(minX, x)
        maxX = Math.max(maxX, x)
      }
    }
    // Count the wire along the row through the middle, starting at the left
    // edge of the quad: an axis-aligned quad puts a vertical edge there.
    const row = 22
    let wire = 0
    for (let x = minX; x <= maxX; x++) {
      if (colorAt(fb, x, row)[0] < 0.5) break
      wire++
    }
    return { span: maxX - minX + 1, wire }
  }

  const near = runAt(0)
  const far = runAt(-8)
  assert(near.span >= far.span * 2.5, `the two quads should differ a lot in size: ${near.span} against ${far.span}`)
  assert(near.wire > 1 && far.wire > 1, `expected a wire on both, got ${near.wire} and ${far.wire}`)
  assert(
    Math.abs(near.wire - far.wire) <= 1,
    `the wire changed width with the triangle: ${near.wire} cells against ${far.wire}`,
  )
})

test('a blank interior still hides what is behind it', () => {
  // Hidden-line removal is not a feature of the shader, it is the depth buffer
  // doing its usual job -- but only because the blank interior is written
  // rather than skipped.
  const camera = new Camera({ position: vec3(0, 0, 4), fovY: Math.PI / 4 })
  const vp = camera.viewProjection(aspectFor(50, 24, 0.5))

  const fb = new Framebuffer(50, 24)
  fb.clear()
  drawMesh(fb, quad(2, 0), identity(), vp, wireframe({ line: vec3(1, 1, 1), width: 0.6 }))
  const insideBefore = colorAt(fb, 25, 12)[0]
  assert(insideBefore < 0.5, 'the middle of the quad should be blank, not wire')

  drawMesh(fb, quad(100, -2), identity(), vp, unlit(vec3(1, 0, 0)))
  assert(colorAt(fb, 25, 12)[0] < 0.5, 'a farther surface showed through the blank interior')
  assert(colorAt(fb, 1, 12)[0] === 1, 'the farther surface should still be visible outside the quad')
})

test('a fill shader takes the interior and the wire keeps the edges', () => {
  const camera = new Camera({ position: vec3(0, 0, 4), fovY: Math.PI / 4 })
  const vp = camera.viewProjection(aspectFor(50, 24, 0.5))
  const fb = new Framebuffer(50, 24)
  fb.clear()
  drawMesh(
    fb,
    quad(2, 0),
    identity(),
    vp,
    wireframe({ line: vec3(0, 0, 1), width: 0.6, fill: unlit(vec3(1, 0, 0)) }),
  )

  let wire = 0
  let filled = 0
  for (let i = 0; i < fb.depth.length; i++) {
    if (fb.depth[i]! <= 0) continue
    if (fb.color[i * 3 + 2]! === 1) wire++
    else if (fb.color[i * 3]! === 1) filled++
  }
  assert(wire > 20, `expected a wire around the quad, got ${wire} cells`)
  assert(filled > wire, `the fill should cover more than the wire: ${filled} against ${wire}`)
})

test('a marched surface reports no edges and comes out all wire', () => {
  // A field has no triangles, so it has no edges to be near: every marched
  // fragment reads zero. Pinned here so it cannot change quietly.
  const camera = new Camera({ position: vec3(0, 0, 4), fovY: Math.PI / 4 })
  const fb = new Framebuffer(40, 20)
  fb.clear()
  marchScene(fb, sdSphere(1), camera, aspectFor(40, 20, 0.5), wireframe({ line: vec3(1, 1, 1), width: 0.6 }))

  let covered = 0
  let wire = 0
  for (let i = 0; i < fb.depth.length; i++) {
    if (fb.depth[i]! <= 0) continue
    covered++
    if (fb.color[i * 3]! === 1) wire++
  }
  assert(covered > 100, `expected a marched sphere, got ${covered} cells`)
  assert(wire === covered, `${covered - wire} marched cells were not treated as edges`)
})

test('a wireframe cube shows its front edges and not its back ones', () => {
  const camera = new Camera({ position: vec3(2.6, 2.1, 3.6), fovY: Math.PI / 3.2 })
  const fb = new Framebuffer(74, 32)
  fb.clear(0, 0, 0)
  drawMesh(
    fb,
    cube(2),
    identity(),
    camera.viewProjection(aspectFor(74, 32, 0.5)),
    wireframe({ line: vec3(1, 0.95, 0.85), width: 0.55 }),
  )
  fb.resolve(RAMPS.short)
  console.log('\n' + fb.toString() + '\n')

  let wire = 0
  for (let i = 0; i < fb.depth.length; i++) if (fb.color[i * 3]! > 0.5) wire++
  assert(wire > 60, `expected the cube's visible edges, got ${wire} cells`)
})

console.log('\noverlay')

/** The glyph at a cell, as a string, for readable assertions. */
function glyphAt(fb: Framebuffer, x: number, y: number): string {
  return String.fromCharCode(fb.chars[y * fb.width + x]!)
}

test('drawText puts the glyphs where it says and clips rather than wraps', () => {
  const fb = new Framebuffer(10, 3)
  fb.clear()
  drawText(fb, 2, 1, 'abc')
  assert(glyphAt(fb, 2, 1) === 'a' && glyphAt(fb, 3, 1) === 'b' && glyphAt(fb, 4, 1) === 'c', 'abc is not at 2,1')
  assert(glyphAt(fb, 1, 1) === ' ' && glyphAt(fb, 5, 1) === ' ', 'drawText spilled into its neighbours')

  // Off the left: the characters that fall outside are dropped, and the rest
  // stay in their own columns rather than shifting in.
  drawText(fb, -1, 0, 'xyz')
  assert(glyphAt(fb, 0, 0) === 'y' && glyphAt(fb, 1, 0) === 'z', `clipped left wrong: ${glyphAt(fb, 0, 0)}`)

  // Off the right: no wrap onto the next row.
  drawText(fb, 9, 2, 'pq')
  assert(glyphAt(fb, 9, 2) === 'p', 'the last column should still take a character')
  assert(glyphAt(fb, 0, 0) === 'y', 'a character wrapped onto another row')

  const before = fb.chars.join(',')
  drawText(fb, 0, 5, 'zz')
  assert(fb.chars.join(',') === before, 'a row outside the grid still wrote something')
})

test('drawText aligns on the column it is given', () => {
  const fb = new Framebuffer(12, 2)
  fb.clear()
  drawText(fb, 5, 0, 'abcd', { align: 'center' })
  assert(glyphAt(fb, 3, 0) === 'a', `centred text starts at ${glyphAt(fb, 3, 0)}`)
  drawText(fb, 5, 1, 'abcd', { align: 'right' })
  assert(glyphAt(fb, 5, 1) === 'd', `right-aligned text ends at ${glyphAt(fb, 5, 1)}`)
})

test('resolve leaves overlay text alone', () => {
  // The claim that makes text cheap here: `resolve` only fills cells a shader
  // left on auto, so a glyph written directly survives it.
  const fb = new Framebuffer(8, 1)
  fb.clear(0, 0, 0)
  drawText(fb, 1, 0, 'hello', { color: vec3(1, 1, 1) })
  fb.resolve(RAMPS.long)
  assert(fb.toString().includes('hello'), `text did not survive resolve: ${fb.toString()}`)
})

test('label3 lands on the cell the projection puts it in', () => {
  const width = 60
  const height = 30
  const fovY = Math.PI / 4
  const aspect = aspectFor(width, height, 0.5)
  const f = 1 / Math.tan(fovY / 2)
  const camera = new Camera({ position: vec3(0, 0, 3), fovY })

  // At z = 0 the clip w is the camera distance, so the projection is one
  // division and can be written out here without touching the renderer.
  const point = vec3(0.62, 0.25, 0)
  const sx = (((f / aspect) * point.x) / 3 / 2 + 0.5) * width
  const sy = (0.5 - (f * point.y) / 3 / 2) * height

  // Cell i spans [i, i + 1) and the rasterizer samples its centre at i + 0.5,
  // so the cell holding a screen coordinate is its floor. The first version of
  // this test rounded -- the same mistake the implementation was making -- and
  // so agreed with it about a label sitting half a cell off.
  const col = Math.floor(sx)
  const row = Math.floor(sy)
  assert(
    sx - col >= 0.5 && sy - row >= 0.5,
    `this point cannot tell floor from round: ${sx.toFixed(3)}, ${sy.toFixed(3)}`,
  )

  const fb = new Framebuffer(width, height)
  fb.clear()
  assert(label3(fb, point, camera.viewProjection(aspect), 'A'), 'the label should have been drawn')
  assert(glyphAt(fb, col, row) === 'A', `expected A at ${col},${row}, found "${glyphAt(fb, col, row)}"`)
})

test('a label behind the camera is dropped, not mirrored', () => {
  // Dividing by a negative w puts the point on the opposite side of the
  // screen, where a label looks perfectly plausible and is in entirely the
  // wrong place. This is the assertion that keeps that from happening.
  const fb = new Framebuffer(40, 20)
  fb.clear()
  const camera = new Camera({ position: vec3(0, 0, 3), target: vec3(0, 0, 0), fovY: Math.PI / 4 })
  const before = fb.chars.join(',')

  const drew = label3(fb, vec3(0.4, 0.2, 9), camera.viewProjection(aspectFor(40, 20, 0.5)), 'B')
  assert(!drew, 'a point behind the camera reported that it drew')
  assert(fb.chars.join(',') === before, 'a point behind the camera still wrote to the grid')
})

test('an occluded label is dropped only when asked', () => {
  const camera = new Camera({ position: vec3(0, 0, 3), fovY: Math.PI / 4 })
  const vp = camera.viewProjection(aspectFor(40, 20, 0.5))
  const behind = vec3(0, 0, -2)

  const fb = new Framebuffer(40, 20)
  fb.clear()
  drawMesh(fb, quad(100, 0), identity(), vp, unlit(vec3(0.2, 0.2, 0.2)))

  assert(!label3(fb, behind, vp, 'C', { occlude: true }), 'a label behind a wall was drawn anyway')
  assert(label3(fb, behind, vp, 'C'), 'without occlude the label should be drawn regardless')
  assert(fb.toString().includes('C'), 'the unoccluded label is missing from the frame')
})

test('a line stays on the segment between its ends', () => {
  const width = 60
  const height = 30
  const fovY = Math.PI / 4
  const aspect = aspectFor(width, height, 0.5)
  const f = 1 / Math.tan(fovY / 2)
  const camera = new Camera({ position: vec3(0, 0, 3), fovY })

  const a = vec3(-1.2, -0.7, 0)
  const b = vec3(1.1, 0.8, 0)
  const project = (p: typeof a) => [
    (((f / aspect) * p.x) / 3 / 2 + 0.5) * width,
    (0.5 - (f * p.y) / 3 / 2) * height,
  ]
  const [ax, ay] = project(a)
  const [bx, by] = project(b)

  const fb = new Framebuffer(width, height)
  fb.clear(0, 0, 0)
  drawLine3(fb, a, b, camera.viewProjection(aspect), { color: vec3(1, 1, 1) })

  let drawn = 0
  let worst = 0
  const ex = bx! - ax!
  const ey = by! - ay!
  const len = Math.hypot(ex, ey)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (fb.color[(y * width + x) * 3]! < 0.5) continue
      drawn++
      worst = Math.max(worst, Math.abs(ex * (y + 0.5 - ay!) - ey * (x + 0.5 - ax!)) / len)
    }
  }
  assert(drawn > 20, `expected a line of cells, got ${drawn}`)
  // Every sample sits exactly on the line before it is quantised, and
  // quantising can move it half a cell on each axis, so the furthest a
  // correctly drawn cell can end up is hypot(0.5, 0.5). Anything beyond that
  // is a systematic offset rather than rounding -- which is how the half-cell
  // one this caught showed up, at 1.186.
  assert(worst < 0.75, `a cell sits ${worst.toFixed(3)} cells off the segment`)

  // One name for the quantiser, used everywhere this test turns a screen
  // coordinate into a cell. Spelling it out at each site is how the half-cell
  // mistake got into three places at once -- the implementation, the label
  // test, and these two lookups, which kept rounding after the rest had
  // stopped.
  const cellOf = (s: number) => Math.floor(s)
  const litAt = (sx: number, sy: number) => fb.color[(cellOf(sy) * width + cellOf(sx)) * 3]! > 0.5
  assert(litAt(ax!, ay!), 'the first endpoint was not drawn')
  assert(litAt(bx!, by!), 'the second endpoint was not drawn')
})

test('a line half behind the camera is cut, not folded over', () => {
  // With the near plane respected this runs from the middle of the frame up
  // and off the top. Without it, the far end divides by a negative w, lands
  // below the frame instead, and the visible part is the bottom few rows --
  // so where the cells are is what tells the two apart.
  const width = 40
  const height = 20
  const camera = new Camera({ position: vec3(0, 0, 0), target: vec3(0, 0, -1), fovY: Math.PI / 4 })
  const fb = new Framebuffer(width, height)
  fb.clear(0, 0, 0)
  drawLine3(fb, vec3(0, -0.5, -2), vec3(0, 1, 1), camera.viewProjection(aspectFor(width, height, 0.5)), {
    color: vec3(1, 1, 1),
  })

  let minRow = Infinity
  let maxRow = -1
  const columns = new Set<number>()
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (fb.color[(y * width + x) * 3]! < 0.5) continue
      minRow = Math.min(minRow, y)
      maxRow = Math.max(maxRow, y)
      columns.add(x)
    }
  }
  assert(maxRow >= 0, 'the visible half of the line should still be drawn')
  assert(minRow === 0, `the clipped line should reach the top row, it starts at ${minRow}`)
  assert(maxRow <= 17, `nothing should be drawn below the near end, but a cell sits at row ${maxRow}`)
  assert(columns.size <= 2, `a vertical line should stay in one column, it used ${columns.size}`)
})

test('a line is hidden behind nearer geometry', () => {
  const camera = new Camera({ position: vec3(0, 0, 3), fovY: Math.PI / 4 })
  const vp = camera.viewProjection(aspectFor(40, 20, 0.5))
  const fb = new Framebuffer(40, 20)
  fb.clear(0, 0, 0)
  drawMesh(fb, quad(100, 0), identity(), vp, unlit(vec3(0.3, 0.3, 0.3)))
  drawLine3(fb, vec3(-1, 0, -2), vec3(1, 0, -2), vp, { color: vec3(1, 1, 1) })

  let bright = 0
  for (let i = 0; i < fb.depth.length; i++) if (fb.color[i * 3]! > 0.9) bright++
  assert(bright === 0, `${bright} cells of a line behind a wall came through`)
})

test('drawAxes draws three lettered axes', () => {
  const camera = new Camera({ position: vec3(2.4, 1.8, 3.2), fovY: Math.PI / 3.2 })
  const fb = new Framebuffer(70, 28)
  fb.clear(0, 0, 0)
  const vp = camera.viewProjection(aspectFor(70, 28, 0.5))
  drawMesh(fb, cube(1.2), identity(), vp, lambert({ albedo: vec3(0.5, 0.42, 0.3), ambient: 0.15 }))
  drawAxes(fb, vp, { length: 1.6 })
  fb.resolve(RAMPS.short)

  const frame = fb.toString()
  for (const letter of ['x', 'y', 'z']) {
    assert(frame.includes(letter), `the ${letter} axis has no label`)
  }
  // Each axis has its own colour, so counting distinct strong hues finds them.
  const hues = new Set<string>()
  for (let i = 0; i < fb.depth.length; i++) {
    const r = fb.color[i * 3]!
    const g = fb.color[i * 3 + 1]!
    const b = fb.color[i * 3 + 2]!
    if (Math.max(r, g, b) < 0.4) continue
    hues.add(`${r > 0.9 ? 1 : 0}${g > 0.9 ? 1 : 0}${b > 0.9 ? 1 : 0}`)
  }
  assert(hues.size >= 3, `expected three axis colours, found ${hues.size}`)
  console.log('\n' + frame + '\n')
})

console.log('\npoint lights')

/** Drives a shader at one fragment, with a flat surface facing +y at the origin. */
function shadeAt(shader: Shader, px: number, py: number, pz: number, n = vec3(0, 1, 0)) {
  const frag = { px, py, pz, nx: n.x, ny: n.y, nz: n.z, u: 0, v: 0, edge: 9, cx: 0, cy: 0, invW: 1 }
  const out = { r: 0, g: 0, b: 0, char: 0 }
  shader(frag, out)
  return out
}

test('a point light falls off as the inverse square', () => {
  // With the range far away its window is barely doing anything, so what is
  // left is the physics: twice as far is a quarter as bright.
  const shader = lambert({
    albedo: vec3(1, 1, 1),
    light: vec3(0, 1, 0),
    lightColor: vec3(0, 0, 0),
    ambient: 0,
    points: [{ position: vec3(0, 2, 0), intensity: 1, range: 1000 }],
  })

  const near = shadeAt(shader, 0, 0, 0).r
  const far = shadeAt(shader, 0, -2, 0).r
  assert(near > 0 && far > 0, `both points should be lit, got ${near} and ${far}`)
  close(near / far, 4, 0.01, 'doubling the distance should quarter the light')
})

test('a light brighter than the ramp clamps rather than wrapping', () => {
  // Worth its own assertion because it is also the trap in the test below:
  // the shader's output is clamped, so comparing it against an unclamped
  // formula reads "exactly 1" and looks like the light failing.
  const shader = lambert({
    albedo: vec3(1, 1, 1),
    light: vec3(0, 1, 0),
    lightColor: vec3(0, 0, 0),
    ambient: 0,
    points: [{ position: vec3(0, 1, 0), intensity: 50, range: 20 }],
  })
  close(shadeAt(shader, 0, 0, 0).r, 1, 0, 'an overbright light should saturate at one')
})

test('the falloff is exactly the formula, window and all', () => {
  const position = vec3(0, 3, 0)
  // Dim enough that every distance below stays under the shader's clamp --
  // otherwise this compares a clamped output against an unclamped formula.
  const intensity = 0.5
  const range = 5
  const shader = lambert({
    albedo: vec3(1, 1, 1),
    light: vec3(0, 1, 0),
    lightColor: vec3(0, 0, 0),
    ambient: 0,
    points: [{ position, intensity, range }],
  })

  for (const d of [1, 2, 3, 4.5]) {
    const s = d / range
    const w = 1 - s * s * s * s
    // The surface faces straight up at the light, so the cosine is one.
    close(shadeAt(shader, 0, 3 - d, 0).r, (intensity * w * w) / (d * d), 1e-9, `falloff at distance ${d}`)
  }
})

test('the range is a bound, not a suggestion', () => {
  // A falloff that only gets small near its range costs every fragment in the
  // scene forever. This one has to reach zero exactly at it.
  const shader = lambert({
    albedo: vec3(1, 1, 1),
    light: vec3(0, 1, 0),
    lightColor: vec3(0, 0, 0),
    ambient: 0,
    points: [{ position: vec3(0, 4, 0), intensity: 5, range: 4 }],
  })

  assert(shadeAt(shader, 0, 1, 0).r > 0, 'inside the range the light should reach')
  close(shadeAt(shader, 0, 0, 0).r, 0, 0, 'at exactly the range the light must be zero')
  close(shadeAt(shader, 0, -1, 0).r, 0, 0, 'past the range the light must stay zero')
})

test('a point light lights two surfaces from opposite sides', () => {
  // The claim that separates a lamp from a direction: a directional light
  // cannot do this, because its direction is the same everywhere.
  const shader = lambert({
    albedo: vec3(1, 1, 1),
    light: vec3(0, 1, 0),
    lightColor: vec3(0, 0, 0),
    ambient: 0,
    points: [{ position: vec3(0, 0, 0), intensity: 1, range: 10 }],
  })

  // Two surfaces either side of the lamp, each facing it: normals point in
  // opposite directions and both must be lit.
  const above = shadeAt(shader, 0, 2, 0, vec3(0, -1, 0)).r
  const below = shadeAt(shader, 0, -2, 0, vec3(0, 1, 0)).r
  close(above, below, 1e-9, 'symmetric surfaces should be lit equally')
  assert(above > 0, 'a surface facing the lamp should be lit')

  // And turning one away puts it out, without touching the other.
  close(shadeAt(shader, 0, 2, 0, vec3(0, 1, 0)).r, 0, 1e-9, 'a surface facing away should be dark')
})

test('an occluder beyond the lamp casts no shadow', () => {
  // The bug a bounded ray exists to prevent: marching past the light finds
  // whatever is behind it and darkens a surface the light does reach.
  const lamp = vec3(0, 2, 0)
  const past = shadowFromPoint(translate(sdSphere(0.5), 0, 4, 0), lamp)
  const between = shadowFromPoint(translate(sdSphere(0.5), 0, 1, 0), lamp)

  close(past(0, 0, 0), 1, 1e-9, 'a sphere behind the lamp blocked nothing and must not darken this')
  close(between(0, 0, 0), 0, 1e-9, 'a sphere between the surface and the lamp must block it')
})

test('a lamp shadow follows the lamp, not a fixed direction', () => {
  // An occluder directly between surface and lamp blocks it; move the lamp
  // sideways and the same occluder stops mattering.
  const field = translate(sdSphere(0.4), 0, 1, 0)
  close(shadowFromPoint(field, vec3(0, 2, 0))(0, 0, 0), 0, 1e-9, 'straight overhead the sphere is in the way')
  close(shadowFromPoint(field, vec3(4, 2, 0))(0, 0, 0), 1, 1e-9, 'off to the side it is not')
})

test('a lamp draws a bright pool that fades with distance', () => {
  const camera = new Camera({ position: vec3(0, 4.5, 6), target: vec3(0, 0, 0), fovY: Math.PI / 3 })
  const aspect = aspectFor(78, 30, 0.5)
  const fb = new Framebuffer(78, 30)
  fb.clear(0, 0, 0)
  const lamp = vec3(0, 1.5, 0)
  drawMesh(
    fb,
    plane(16, 1),
    identity(),
    camera.viewProjection(aspect),
    lambert({
      albedo: vec3(0.85, 0.82, 0.75),
      lightColor: vec3(0, 0, 0),
      light: vec3(0, 1, 0),
      ambient: 0.06,
      points: [{ position: lamp, intensity: 3.5, range: 9 }],
    }),
  )
  fb.resolve(RAMPS.long)

  // Brightest under the lamp, dimmer further out along the floor.
  const centre = fb.color[(Math.floor(fb.height * 0.62) * fb.width + 39) * 3]!
  const edge = fb.color[(Math.floor(fb.height * 0.62) * fb.width + 4) * 3]!
  assert(centre > edge * 2, `the pool should fall off across the floor: ${centre} against ${edge}`)
  console.log('\n' + fb.toString() + '\n')
})

console.log('\ncharacter output')

test('resolve walks the ramp from dark to bright', () => {
  const fb = new Framebuffer(3, 1)
  fb.clear()
  fb.chars.fill(0)
  for (const [i, lum] of [0, 0.5, 1].entries()) {
    fb.color[i * 3] = lum
    fb.color[i * 3 + 1] = lum
    fb.color[i * 3 + 2] = lum
  }
  fb.resolve(RAMPS.short)
  const a = RAMPS.short.indexOf(String.fromCharCode(fb.chars[0]!))
  const b = RAMPS.short.indexOf(String.fromCharCode(fb.chars[1]!))
  const c = RAMPS.short.indexOf(String.fromCharCode(fb.chars[2]!))
  assert(a === 0 && c === RAMPS.short.length - 1, `ramp endpoints wrong: ${a} and ${c}`)
  assert(a < b && b < c, `ramp should be monotonic, got ${a} < ${b} < ${c}`)
})

test('resolve leaves a glyph a shader chose explicitly alone', () => {
  const fb = new Framebuffer(4, 1)
  fb.clear()
  const camera = new Camera({ position: vec3(0, 0, 1) })
  drawMesh(fb, quad(100, 0), identity(), camera.viewProjection(aspectFor(4, 1)), unlit(vec3(1, 1, 1), 64))
  fb.resolve(RAMPS.short)
  assert(fb.toString() === '@@@@', `char code 64 should survive resolve, got ${fb.toString()}`)
})

test('a flat-shaded cube gets exactly one shade per visible face', () => {
  // Three faces of an axis-aligned cube face a corner-on camera, each with a
  // single constant normal. Any extra shade means normals are leaking across
  // an edge; any fewer means two faces are being lit identically.
  const fb = new Framebuffer(70, 32)
  fb.clear(0, 0, 0)
  const camera = new Camera({ position: vec3(2.6, 2.1, 3.6), fovY: Math.PI / 3.2 })
  drawMesh(
    fb,
    cube(2),
    identity(),
    camera.viewProjection(aspectFor(70, 32, 0.5)),
    lambert({ albedo: vec3(0.7, 0.6, 0.45), light: vec3(0.5, 0.8, 0.6), ambient: 0.1 }),
  )

  const shades = new Set<string>()
  for (let i = 0; i < fb.depth.length; i++) {
    if (fb.depth[i]! <= 0) continue
    shades.add(fb.color.slice(i * 3, i * 3 + 3).join(','))
  }
  assert(shades.size === 3, `expected 3 flat shades, got ${shades.size}`)
})

test('a shaded sphere renders a smooth gradient', () => {
  const fb = new Framebuffer(70, 32)
  fb.clear(0, 0, 0)
  const camera = new Camera({ position: vec3(1.6, 1.4, 3.6), fovY: Math.PI / 3.2 })
  drawMesh(
    fb,
    sphere(1.35),
    multiply(rotationY(0.4), rotationX(0.2)),
    camera.viewProjection(aspectFor(70, 32, 0.5)),
    lambert({ albedo: vec3(1, 0.85, 0.6), specular: 0.45, shininess: 24, eye: camera.position }),
  )
  fb.resolve(RAMPS.long)

  const glyphs = new Set<number>()
  for (let i = 0; i < fb.chars.length; i++) if (fb.depth[i]! > 0) glyphs.add(fb.chars[i]!)
  assert(glyphs.size >= 8, `expected a smooth gradient, got ${glyphs.size} distinct glyphs`)
  console.log('\n' + fb.toString() + '\n')
})

test('rendering the same frame twice gives identical buffers', () => {
  const camera = new Camera({ position: vec3(2.5, 2, 3.5) })
  const vp = camera.viewProjection(aspectFor(50, 25))
  const shader = lambert({ albedo: vec3(1, 1, 1), specular: 0.3, eye: camera.position })
  const render = () => {
    const fb = new Framebuffer(50, 25)
    fb.clear(0.01, 0.01, 0.02)
    drawMesh(fb, torus(1, 0.35), rotationY(0.9), vp, shader)
    fb.resolve(RAMPS.long)
    return fb
  }
  const a = render()
  const b = render()
  assert(a.chars.every((v, i) => v === b.chars[i]), 'glyphs differ between two identical renders')
  assert(a.color.every((v, i) => v === b.color[i]), 'colours differ between two identical renders')
})

console.log('\nterminal presenter')

test('present writes nothing when the frame has not changed', () => {
  const { stream, chunks } = fakeStream(10, 4)
  const term = new Terminal({ stream, color: 'none' })
  const fb = term.framebuffer()
  fb.clear()
  fb.chars[0] = 65

  term.present(fb)
  assert(chunks.join('').includes('A'), 'the first frame should be written')

  chunks.length = 0
  term.present(fb)
  assert(chunks.length === 0, `an unchanged frame should write nothing, wrote ${JSON.stringify(chunks)}`)

  fb.chars[0] = 66
  term.present(fb)
  assert(chunks.join('').includes('B'), 'a changed cell should be written')
})

test('the terminal shows exactly the framebuffer, frame after frame', () => {
  // The second frame is where a diffed presenter goes wrong: it writes only
  // the cells that changed, so a stale comparison buffer leaves debris from
  // the previous frame on screen. Replaying the escape stream catches that;
  // checking that bytes were written would not.
  const { stream, chunks } = fakeStream(80, 24)
  const term = new Terminal({ stream, color: 'truecolor' })
  const camera = new Camera({ position: vec3(2.6, 2.1, 3.6), fovY: Math.PI / 3.2 })
  const vp = camera.viewProjection(aspectFor(80, 24, 0.5))
  const shader = lambert({
    albedo: vec3(0.95, 0.75, 0.45),
    specular: 0.45,
    shininess: 24,
    eye: camera.position,
  })

  let expected = ''
  for (const angle of [0.4, 0.9]) {
    const fb = term.framebuffer()
    fb.clear(0, 0, 0)
    drawMesh(fb, cube(2), multiply(rotationY(angle), rotationX(angle * 0.6)), vp, shader)
    fb.resolve(RAMPS.short)
    term.present(fb)
    expected = fb.toString()
  }

  const shown = replay(chunks.join(''), 80, 24)
  assert(shown === expected, `the terminal shows something other than the framebuffer:\n${shown}`)
  console.log('\n' + expected + '\n')
})

test('to256 pins the palette extremes and keeps greys on the grey ramp', () => {
  assert(to256(0xffffff) === 231, `white should be 231, got ${to256(0xffffff)}`)
  assert(to256(0x000000) === 16, `black should be 16, got ${to256(0x000000)}`)
  const grey = to256(0x808080)
  assert(grey >= 232 && grey <= 255, `mid grey should land on the grey ramp, got ${grey}`)
  const orange = to256(0xff8000)
  assert(orange >= 16 && orange <= 231, `a saturated colour should land in the cube, got ${orange}`)
})

console.log(failed === 0 ? '\nall checks passed\n' : `\n${failed} check(s) failed\n`)
process.exit(failed === 0 ? 0 : 1)
