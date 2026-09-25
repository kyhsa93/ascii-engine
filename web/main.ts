import { Camera, aspectFor } from '../src/core/camera.ts'
import { multiply, rotationX, rotationY } from '../src/core/mat4.ts'
import { cube, sphere, torus } from '../src/core/mesh.ts'
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
let distance = 5
let spinning = true
let showNormals = false
let spin = 0

let frames = 0
let fps = 0
let fpsWindow = performance.now()
let last = performance.now()

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
    distance = Math.max(2.2, Math.min(20, distance + Math.sign(e.deltaY) * 0.4))
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

  frames++
  if (now - fpsWindow >= 500) {
    fps = (frames * 1000) / (now - fpsWindow)
    fpsWindow = now
    frames = 0
  }

  const fb = surface.framebuffer()
  fb.clear(0.02, 0.02, 0.05)

  camera.orbit(yaw, pitch, distance)
  const vp = camera.viewProjection(aspectFor(fb.width, fb.height, surface.cellAspect))
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
