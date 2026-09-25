import { Camera, aspectFor, fitDistance } from '../src/core/camera.ts'
import { multiply, rotationX, rotationY } from '../src/core/mat4.ts'
import { boundingRadius, cube, sphere, torus } from '../src/core/mesh.ts'
import { RAMPS, type RampName } from '../src/core/ramp.ts'
import { drawMesh } from '../src/core/renderer.ts'
import { lambert, normalColor } from '../src/core/shading.ts'
import { vec3 } from '../src/core/vec3.ts'
import { PreSurface } from '../src/web/pre.ts'

const shapes = [
  { name: 'cube', mesh: cube(2) },
  { name: 'sphere', mesh: sphere(1.3) },
  { name: 'torus', mesh: torus(1.1, 0.42) },
]
const rampNames = Object.keys(RAMPS) as RampName[]

const screen = document.getElementById('screen')!
const stats = document.getElementById('stats')!
const surface = new PreSurface(screen)
const camera = new Camera({ fovY: Math.PI / 3.2 })

let shape = 0
let rampIndex = 0
let yaw = 0.6
let pitch = 0.35
// A multiplier on the distance that frames the subject, not a distance: a
// phone in portrait is a far narrower frustum than a desktop window, and a
// fixed distance that suits one cuts the subject off in the other.
let zoom = 1
let spinning = true
let showNormals = false
let spin = 0

let frames = 0
let fps = 0
let fpsWindow = performance.now()
let last = performance.now()
let frameCount = 0

// A read-only probe for scripts/viewcheck.ts. The cell aspect is the one
// number that cannot be checked outside a browser -- it comes from measuring
// the page's own font -- so the check has to read the value actually in use
// rather than measure a second one of its own.
;(window as unknown as { __engine: unknown }).__engine = {
  get cellAspect() {
    return surface.cellAspect
  },
  get cols() {
    return surface.cols
  },
  get rows() {
    return surface.rows
  },
  get frames() {
    return frameCount
  },
  get shape() {
    return shapes[shape]!.name
  },
}

addEventListener('resize', () => surface.measure())

let dragging = false
screen.addEventListener('pointerdown', (e) => {
  dragging = true
  screen.setPointerCapture(e.pointerId)
})
screen.addEventListener('pointerup', () => {
  dragging = false
})
screen.addEventListener('pointermove', (e) => {
  if (!dragging) return
  yaw += e.movementX * 0.01
  pitch = Math.max(-1.4, Math.min(1.4, pitch + e.movementY * 0.01))
})
screen.addEventListener(
  'wheel',
  (e) => {
    e.preventDefault()
    zoom = Math.max(0.6, Math.min(4, zoom + Math.sign(e.deltaY) * 0.1))
  },
  { passive: false },
)

document.querySelectorAll<HTMLButtonElement>('button[data-action]').forEach((button) => {
  button.addEventListener('click', () => {
    if (button.dataset.action === 'shape') shape = (shape + 1) % shapes.length
    if (button.dataset.action === 'ramp') rampIndex = (rampIndex + 1) % rampNames.length
    if (button.dataset.action === 'normals') showNormals = !showNormals
    if (button.dataset.action === 'pause') spinning = !spinning
  })
})

function frame(now: number): void {
  const dt = (now - last) / 1000
  last = now
  if (spinning) spin += dt * 0.7

  frameCount++
  frames++
  if (now - fpsWindow >= 500) {
    fps = (frames * 1000) / (now - fpsWindow)
    fpsWindow = now
    frames = 0
  }

  const fb = surface.framebuffer()
  fb.clear(0.02, 0.02, 0.05)

  const aspect = aspectFor(fb.width, fb.height, surface.cellAspect)
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
  surface.present(fb)

  stats.textContent = `${shapes[shape]!.name} · ramp ${rampNames[rampIndex]} · ${fb.width}x${fb.height} cells · ${fps.toFixed(0)} fps`
  requestAnimationFrame(frame)
}

requestAnimationFrame(frame)
