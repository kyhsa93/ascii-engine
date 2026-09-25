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
import { marchScene } from '../src/core/march.ts'
import { RAMPS } from '../src/core/ramp.ts'
import type { Shader } from '../src/core/raster.ts'
import { drawMesh } from '../src/core/renderer.ts'
import { sdBox, sdSphere, smoothUnion, translate } from '../src/core/sdf.ts'
import { lambert, unlit } from '../src/core/shading.ts'
import { Terminal, to256 } from '../src/term/ansi.ts'
import { vec3 } from '../src/core/vec3.ts'

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
  invW: number
}

/** A shader that keeps a copy of every fragment it is handed. */
function recorder(into: Sample[]): Shader {
  return (f, out) => {
    into.push({ px: f.px, py: f.py, pz: f.pz, nx: f.nx, ny: f.ny, nz: f.nz, invW: f.invW })
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
  const mesh = parseObj(`
    v 0 0 0
    v 1 0 0
    v 1 1 0
    vt 0 0
    vn 0 0 1
    f 1//1 2//1 3//1
    f 1/1/1 2//1 3//1
  `)
  assert(mesh.positions.length / 3 === 3, `expected 3 vertices, got ${mesh.positions.length / 3}`)
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
