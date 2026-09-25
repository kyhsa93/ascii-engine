/**
 * Loads a Wavefront OBJ and turns it over in the terminal.
 *
 *   npm run obj                 # models/knot.obj
 *   node examples/obj.ts a.obj  # anything else
 *
 * Unlike the shape demo, nothing here knows how big the subject is or where
 * it sits: both come from the file, so the camera is aimed at the mesh's own
 * centre and fitted to a radius measured from there.
 */

import { readFileSync } from 'node:fs'
import { basename, resolve } from 'node:path'
import { Camera, aspectFor, fitDistance } from '../src/core/camera.ts'
import { multiply, rotationX, rotationY, translation } from '../src/core/mat4.ts'
import { boundingBox, parseObj } from '../src/core/mesh.ts'
import { RAMPS, type RampName } from '../src/core/ramp.ts'
import { drawMesh } from '../src/core/renderer.ts'
import { lambert, normalColor } from '../src/core/shading.ts'
import { vec3 } from '../src/core/vec3.ts'
import { Terminal, runLoop } from '../src/term/ansi.ts'

const path = resolve(process.cwd(), process.argv[2] ?? 'models/knot.obj')
let mesh
try {
  mesh = parseObj(readFileSync(path, 'utf8'))
} catch (error) {
  console.error(`could not read ${path}: ${(error as Error).message}`)
  process.exit(1)
}

const bounds = boundingBox(mesh)
if (mesh.indices.length === 0) {
  console.error(`${path} holds no faces`)
  process.exit(1)
}

// Spin the mesh about its own centre rather than the origin, which is where
// the origin happens to be and not where the model is.
const recentre = translation(-bounds.center.x, -bounds.center.y, -bounds.center.z)

const rampNames = Object.keys(RAMPS) as RampName[]
const term = new Terminal({ reserveRows: 1 })
const camera = new Camera({ fovY: Math.PI / 3.2 })

let rampIndex = 0
let yaw = 0.6
let pitch = 0.35
let zoom = 1
let spinning = true
let showNormals = false
let spin = 0

term.enter()

const loop = runLoop((dt) => {
  if (spinning) spin += dt * 0.5

  const fb = term.framebuffer()
  fb.clear(0.02, 0.02, 0.05)

  const aspect = aspectFor(fb.width, fb.height, term.cellAspect)
  camera.orbit(yaw, pitch, fitDistance(bounds.radius, camera.fovY, aspect) * zoom)
  const vp = camera.viewProjection(aspect)
  const model = multiply(multiply(rotationY(spin), rotationX(spin * 0.45)), recentre)

  const shader = showNormals
    ? normalColor()
    : lambert({
        albedo: vec3(0.9, 0.78, 0.55),
        light: vec3(0.55, 0.75, 0.6),
        ambient: 0.1,
        specular: 0.4,
        shininess: 28,
        eye: camera.position,
      })

  drawMesh(fb, mesh, model, vp, shader)
  fb.resolve(RAMPS[rampNames[rampIndex]!])
  term.present(fb)

  term.status(
    `${basename(path)} · ${mesh.positions.length / 3} verts · ${mesh.indices.length / 3} tris · ` +
      `ramp ${rampNames[rampIndex]} · ${loop.fps.toFixed(0)} fps` +
      '   [r] ramp  [n] normals  [p] pause  [arrows] orbit  [+/-] zoom  [q] quit',
  )
}, 60)

term.onKey((key) => {
  switch (key) {
    case 'q':
    case '\x03':
      loop.stop()
      term.exit()
      process.exit(0)
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
