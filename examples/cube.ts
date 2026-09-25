import { Camera, aspectFor, fitDistance } from '../src/core/camera.ts'
import { multiply, rotationX, rotationY, translation } from '../src/core/mat4.ts'
import { marchScene } from '../src/core/march.ts'
import { boundingRadius, cube, plane, sphere, torus, type Mesh } from '../src/core/mesh.ts'
import { RAMPS, type RampName } from '../src/core/ramp.ts'
import { drawMesh } from '../src/core/renderer.ts'
import { rotateX, rotateY, sdBox, sdSphere, sdTorus, smoothUnion, translate, type Sdf } from '../src/core/sdf.ts'
import { shadowFrom } from '../src/core/shadow.ts'
import { lambert, normalColor } from '../src/core/shading.ts'
import { Supersampler } from '../src/core/supersample.ts'
import { checker } from '../src/core/texture.ts'
import { vec3 } from '../src/core/vec3.ts'
import { Terminal, runLoop } from '../src/term/ansi.ts'

/**
 * Something to look at. `mesh` is what gets drawn when there is one; `field`
 * is the same shape written as a distance function.
 *
 * Every subject carries a field whether or not it is drawn as one, because a
 * rasterized triangle cannot cast a shadow — the occluder always has to be
 * something a ray can be marched through.
 */
interface Subject {
  name: string
  radius: number
  mesh?: Mesh
  field: (spin: number) => Sdf
}

/**
 * The spin, as a field.
 *
 * A mesh turns by a model matrix and a field turns by being sampled in a
 * rotated frame, and the two have to agree or the shadow drifts away from the
 * thing casting it. `Ry(s) * Rx(0.6s)` inverts to sampling at
 * `Rx(-0.6s) * Ry(-s)`, which is what nesting these the other way round does.
 */
const turned = (base: Sdf, spin: number): Sdf => rotateY(rotateX(base, spin * 0.6), spin)

/** One light for both the shading and the shadow; two would disagree. */
const LIGHT = vec3(0.55, 0.75, 0.6)

/** Something for the shadow to fall on. Only drawn when shadows are on. */
const FLOOR = plane(40, 1)

/**
 * A shorter step budget than the default, shared with the browser demo so the
 * two render the same frame.
 *
 * Measured, because the obvious justification turned out to be wrong: at this
 * size it buys nothing at all (2.2 ms a frame against 2.1 for 80x23 cells).
 * It earns its keep on the browser's much larger grid, where the same change
 * is 6.7 ms against 15.8 and costs 41 glyphs out of 8150.
 */
const MARCH = { maxSteps: 64, epsilon: 3e-3 }

/**
 * Sub-samples per cell along each axis when antialiasing is on: four in all.
 *
 * The obvious guess is that four samples cost four times as much, and they do
 * not — only the cells a triangle covers pay that, while clearing and
 * resolving scale with the grid and most of a frame is background. Measured on
 * the sphere: 80x23 cells go from 0.64 ms a frame to 0.84 at 2x and 1.35 at
 * 3x; 163x50 from 0.88 to 2.51 and 5.22. Two is where the staircase goes and
 * the browser still has room to spare.
 */
const SS_FACTOR = 2

const shapes: Subject[] = [
  { name: 'cube', radius: boundingRadius(cube(2)), mesh: cube(2), field: (s) => turned(sdBox(1, 1, 1), s) },
  { name: 'sphere', radius: 1.3, mesh: sphere(1.3), field: () => sdSphere(1.3) },
  { name: 'torus', radius: 1.52, mesh: torus(1.1, 0.42), field: (s) => turned(sdTorus(1.1, 0.42), s) },
  {
    // A sphere and a box welded by a fillet that belongs to neither of them.
    // There is no mesh for this shape: it exists only as a function.
    name: 'blend',
    radius: 2.4,
    field: (spin) =>
      rotateY(smoothUnion(sdSphere(1.05), translate(sdBox(0.7, 0.7, 0.7), 0.9, 0.7, 0.4), 0.55), spin),
  },
]
/** What the texture toggle applies. Six squares reads clearly at terminal size. */
const MAP = checker(6)

const rampNames = Object.keys(RAMPS) as RampName[]

const term = new Terminal({ reserveRows: 1 })
const camera = new Camera({ position: vec3(0, 0, 5), fovY: Math.PI / 3.2 })

let shape = 0
let rampIndex = 0
// The light comes from +x +y +z, so a camera on that same side puts it behind
// the viewer and every shadow hides behind the thing casting it. Measured on
// the sphere: nought shadowed floor cells visible from yaw 0.6, 381 from here.
let yaw = -0.6
let pitch = 0.35
// A multiplier on the distance that frames the subject, not a distance: the
// grid can be any shape, and what "close enough" means depends on its shape.
let zoom = 1
let spinning = true
let showNormals = false
let textured = false
let shadows = false
let antialias = false
let sampler: Supersampler | null = null
let spin = 0

term.enter()
term.onKey((key) => {
  switch (key) {
    case 'q':
    case '\x03':
      loop.stop()
      term.exit()
      process.exit(0)
      break
    case ' ':
      shape = (shape + 1) % shapes.length
      break
    case 'r':
      rampIndex = (rampIndex + 1) % rampNames.length
      break
    case 't':
      textured = !textured
      break
    case 's':
      shadows = !shadows
      break
    case 'a':
      antialias = !antialias
      break
    case 'n':
      showNormals = !showNormals
      break
    case 'p':
      spinning = !spinning
      break
    case '\x1b[A':
      pitch = Math.min(1.4, pitch + 0.1)
      break
    case '\x1b[B':
      pitch = Math.max(-1.4, pitch - 0.1)
      break
    case '\x1b[D':
      yaw -= 0.15
      break
    case '\x1b[C':
      yaw += 0.15
      break
    case '+':
    case '=':
      zoom = Math.max(0.6, zoom - 0.1)
      break
    case '-':
      zoom = Math.min(4, zoom + 0.1)
      break
  }
})

const loop = runLoop((dt) => {
  if (spinning) spin += dt * 0.7

  const fb = term.framebuffer()
  fb.clear(0.02, 0.02, 0.05)

  // Everything below draws into `target`. With antialiasing on that is a finer
  // grid which gets averaged back down at the end; nothing that draws has to
  // know which one it got.
  if (antialias && (!sampler || sampler.width !== fb.width || sampler.height !== fb.height)) {
    sampler = new Supersampler(fb.width, fb.height, SS_FACTOR)
  }
  const target = antialias && sampler ? sampler.buffer : fb
  if (target !== fb) target.clear(0.02, 0.02, 0.05)

  // The fine grid is scaled by the same factor on both axes, so its cells are
  // the same shape as the output's and the aspect below is right for either.
  const aspect = aspectFor(fb.width, fb.height, term.cellAspect)
  const subject = shapes[shape]!
  camera.orbit(yaw, pitch, fitDistance(subject.radius, camera.fovY, aspect) * zoom)
  const vp = camera.viewProjection(aspect)
  const model = multiply(rotationY(spin), rotationX(spin * 0.6))

  const occlusion = shadows
    ? shadowFrom(subject.field(spin), {
        light: LIGHT,
        softness: 12,
        bias: 0.03,
        epsilon: 2e-3,
        maxSteps: 32,
        maxDistance: 12,
      })
    : undefined

  const shader = showNormals
    ? normalColor()
    : lambert({
        albedo: vec3(0.95, 0.75, 0.45),
        light: LIGHT,
        ambient: 0.1,
        specular: 0.45,
        shininess: 24,
        eye: camera.position,
        // A distance field has no vertices and so no texture coordinates:
        // every fragment of one reads (0, 0). The map is offered to meshes only.
        ...(textured && subject.mesh ? { map: MAP } : {}),
        ...(occlusion ? { shadow: occlusion } : {}),
      })

  if (occlusion) {
    // The floor is triangles and what darkens it is a field: the occluder
    // never has to be the thing being drawn.
    drawMesh(
      target,
      FLOOR,
      translation(0, -subject.radius - 0.2, 0),
      vp,
      // Deliberately not a dark floor. Darker, and the shadow's core lands on
      // the ramp's first glyph, which is a space -- and a shadow made of
      // spaces stops reading as a dark patch and starts reading as a hole
      // where the floor ran out.
      lambert({ albedo: vec3(0.5, 0.52, 0.58), light: LIGHT, ambient: 0.22, shadow: occlusion }),
    )
  }

  // Two paths into one framebuffer. The mesh turns by a model matrix; the
  // field has no vertices to move, so it turns by being sampled in a rotated
  // frame instead.
  if (subject.mesh) drawMesh(target, subject.mesh, model, vp, shader)
  else marchScene(target, subject.field(spin), camera, aspect, shader, MARCH)

  if (target !== fb && sampler) sampler.resolveInto(fb)
  fb.resolve(RAMPS[rampNames[rampIndex]!])
  term.present(fb)

  term.status(
    `${shapes[shape]!.name} · ramp ${rampNames[rampIndex]} · ${fb.width}x${fb.height}` +
      `${antialias ? ` · ${SS_FACTOR}x aa` : ''} · ${loop.fps.toFixed(0)} fps` +
      '   [space] shape  [r] ramp  [t] texture  [s] shadow  [a] aa  [n] normals  [p] pause  [q] quit',
  )
}, 60)
