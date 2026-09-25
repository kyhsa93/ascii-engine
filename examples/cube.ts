import { Camera, aspectFor, fitDistance } from '../src/core/camera.ts'
import { multiply, rotationX, rotationY } from '../src/core/mat4.ts'
import { marchScene } from '../src/core/march.ts'
import { boundingRadius, cube, sphere, torus, type Mesh } from '../src/core/mesh.ts'
import { RAMPS, type RampName } from '../src/core/ramp.ts'
import { drawMesh } from '../src/core/renderer.ts'
import { rotateY, sdBox, sdSphere, smoothUnion, translate, type Sdf } from '../src/core/sdf.ts'
import { lambert, normalColor } from '../src/core/shading.ts'
import { vec3 } from '../src/core/vec3.ts'
import { Terminal, runLoop } from '../src/term/ansi.ts'

/** Something to look at: either triangles or a distance field. */
type Subject =
  | { name: string; radius: number; mesh: Mesh }
  | { name: string; radius: number; field: (spin: number) => Sdf }

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

const shapes: Subject[] = [
  { name: 'cube', radius: boundingRadius(cube(2)), mesh: cube(2) },
  { name: 'sphere', radius: 1.3, mesh: sphere(1.3) },
  { name: 'torus', radius: 1.52, mesh: torus(1.1, 0.42) },
  {
    // A sphere and a box welded by a fillet that belongs to neither of them.
    // There is no mesh for this shape: it exists only as a function.
    name: 'blend',
    radius: 2.4,
    field: (spin) =>
      rotateY(smoothUnion(sdSphere(1.05), translate(sdBox(0.7, 0.7, 0.7), 0.9, 0.7, 0.4), 0.55), spin),
  },
]
const rampNames = Object.keys(RAMPS) as RampName[]

const term = new Terminal({ reserveRows: 1 })
const camera = new Camera({ position: vec3(0, 0, 5), fovY: Math.PI / 3.2 })

let shape = 0
let rampIndex = 0
let yaw = 0.6
let pitch = 0.35
// A multiplier on the distance that frames the subject, not a distance: the
// grid can be any shape, and what "close enough" means depends on its shape.
let zoom = 1
let spinning = true
let showNormals = false
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

  const aspect = aspectFor(fb.width, fb.height, term.cellAspect)
  const subject = shapes[shape]!
  camera.orbit(yaw, pitch, fitDistance(subject.radius, camera.fovY, aspect) * zoom)
  const vp = camera.viewProjection(aspect)
  const model = multiply(rotationY(spin), rotationX(spin * 0.6))

  const shader = showNormals
    ? normalColor()
    : lambert({
        albedo: vec3(0.95, 0.75, 0.45),
        light: vec3(0.55, 0.75, 0.6),
        ambient: 0.1,
        specular: 0.45,
        shininess: 24,
        eye: camera.position,
      })

  // Two paths into one framebuffer. The mesh turns by a model matrix; the
  // field has no vertices to move, so it turns by being sampled in a rotated
  // frame instead.
  if ('mesh' in subject) drawMesh(fb, subject.mesh, model, vp, shader)
  else marchScene(fb, subject.field(spin), camera, aspect, shader, MARCH)

  fb.resolve(RAMPS[rampNames[rampIndex]!])
  term.present(fb)

  term.status(
    `${shapes[shape]!.name} · ramp ${rampNames[rampIndex]} · ${fb.width}x${fb.height} · ${loop.fps.toFixed(0)} fps` +
      '   [space] shape  [r] ramp  [n] normals  [p] pause  [arrows] orbit  [+/-] zoom  [q] quit',
  )
}, 60)
