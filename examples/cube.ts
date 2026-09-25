import { Camera, aspectFor, fitDistance } from '../src/core/camera.ts'
import { multiply, rotationX, rotationY } from '../src/core/mat4.ts'
import { boundingRadius, cube, sphere, torus } from '../src/core/mesh.ts'
import { RAMPS, type RampName } from '../src/core/ramp.ts'
import { drawMesh } from '../src/core/renderer.ts'
import { lambert, normalColor } from '../src/core/shading.ts'
import { vec3 } from '../src/core/vec3.ts'
import { Terminal, runLoop } from '../src/term/ansi.ts'

const shapes = [
  { name: 'cube', mesh: cube(2) },
  { name: 'sphere', mesh: sphere(1.3) },
  { name: 'torus', mesh: torus(1.1, 0.42) },
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
  camera.orbit(yaw, pitch, fitDistance(boundingRadius(shapes[shape]!.mesh), camera.fovY, aspect) * zoom)
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

  drawMesh(fb, shapes[shape]!.mesh, model, vp, shader)
  fb.resolve(RAMPS[rampNames[rampIndex]!])
  term.present(fb)

  term.status(
    `${shapes[shape]!.name} · ramp ${rampNames[rampIndex]} · ${fb.width}x${fb.height} · ${loop.fps.toFixed(0)} fps` +
      '   [space] shape  [r] ramp  [n] normals  [p] pause  [arrows] orbit  [+/-] zoom  [q] quit',
  )
}, 60)
