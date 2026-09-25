import { Camera, aspectFor, fitDistance } from '../src/core/camera.ts'
import { multiply, rotationX, rotationY, translation } from '../src/core/mat4.ts'
import { marchScene } from '../src/core/march.ts'
import { boundingRadius, cube, plane, sphere, torus, type Mesh } from '../src/core/mesh.ts'
import { drawAxes, drawText } from '../src/core/overlay.ts'
import { RAMPS, type RampName } from '../src/core/ramp.ts'
import { drawMesh } from '../src/core/renderer.ts'
import { rotateX, rotateY, sdBox, sdSphere, sdTorus, smoothUnion, translate, type Sdf } from '../src/core/sdf.ts'
import { shadowFrom, shadowFromPoint } from '../src/core/shadow.ts'
import { lambert, normalColor, wireframe } from '../src/core/shading.ts'
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

/** Where the lamp sits when the point light is on, and how far it carries. */
const LAMP = vec3(1.6, 1.5, 1.6)
const LAMP_RANGE = 6

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
 * Clearance added to a subject's radius before it is used as a march bound.
 *
 * A march calls anything within `epsilon` of the surface a hit, so the shape it
 * draws is slightly larger than the shape itself, and a bound of exactly the
 * right radius clips that rim away. Comfortably more than `MARCH.epsilon`,
 * because the radii here are rounded up by as little as 0.001 -- the sphere's
 * declared 1.3 against a surface that reaches 1.299.
 */
const BOUND_SLACK = 0.05

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
let wired = false
let axes = false
let lamp = false
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
    case 'w':
      wired = !wired
      break
    case 'o':
      axes = !axes
      break
    case 'l':
      lamp = !lamp
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
        // The subject's own radius, not a padded one: unlike the marcher's
        // `bounds`, the margin a penumbra needs is derived from `softness` and
        // `maxDistance` inside the occluder. Padding it here would only widen
        // an already generous margin and give back the saving.
        casterRadius: subject.radius,
      })
    : undefined

  // One lamp, shared by everything it touches -- the same reason `LIGHT` is a
  // single constant. Two copies drift, and a pool of light that does not sit
  // under the bright side of the subject reads as a stain rather than a lamp.
  //
  // The lamp gets its own occlusion, because a shadow belongs to the light
  // that casts it: the directional light and this one disagree about which
  // way is toward the light at every point.
  const lamps = lamp
    ? [
        {
          position: LAMP,
          intensity: 6,
          range: LAMP_RANGE,
          ...(shadows
            ? { shadow: shadowFromPoint(subject.field(spin), LAMP, { softness: 12, casterRadius: subject.radius }) }
            : {}),
        },
      ]
    : []

  const lit = showNormals
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
        ...(lamps.length ? { points: lamps } : {}),
      })

  // No fill, so the interior comes out blank -- and blank still writes depth,
  // which is what leaves the far edges hidden behind the near faces.
  //
  // A whole cell wide, because anything thinner comes apart. A cube's edges
  // are a connected graph, so a correct drawing of them is one 8-connected
  // blob; measured, 0.6 leaves the terminal's in ten pieces and the browser's
  // in twenty-seven, 0.85 joins the terminal's up but not the browser's, and
  // 1.0 joins both. Supersampling does not rescue a thinner one -- averaging
  // only dims it further.
  const shader = wired ? wireframe({ width: 1 }) : lit

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
      // The lamp reaches the floor too, and this is the one surface that shows
      // what a point light *is*: on the subject it could be any highlight, but
      // a pool on flat ground is the falloff drawn as a shape. Measured over
      // the floor plane, 170 of 625 world samples change glyph, the ramp goes
      // `=` to `#` under the lamp, and the far corners are untouched -- the
      // range is a visible edge, not a number in a comment.
      //
      // It is not free, and the toggle is deliberate for that reason. Measured
      // in the browser over four-second windows: 39 fps down to 22 with the
      // lamp on, and 23 down to 11 with antialiasing and the axes on as well.
      lambert({
        albedo: vec3(0.5, 0.52, 0.58),
        light: LIGHT,
        ambient: 0.22,
        shadow: occlusion,
        ...(lamps.length ? { points: lamps } : {}),
      }),
    )
  }

  // Two paths into one framebuffer. The mesh turns by a model matrix; the
  // field has no vertices to move, so it turns by being sampled in a rotated
  // frame instead.
  if (subject.mesh) drawMesh(target, subject.mesh, model, vp, shader)
  else
    marchScene(target, subject.field(spin), camera, aspect, shader, {
      ...MARCH,
      // Every subject is modelled about the origin and already declares the
      // radius that frames it, so the bound costs nothing to supply. Measured,
      // it drops field evaluations by 70% to 84% depending on the shape --
      // most of a marched frame was rays discovering empty space.
      bounds: { radius: subject.radius + BOUND_SLACK },
      // Four taps rather than six. Safe here because the only subject without
      // a mesh -- and so the only one this branch ever draws -- is the blend,
      // which is smooth: measured, its normals move at most 28 degrees at the
      // worst single sample and 0.7% of drawn cells change glyph. Give the
      // cube a field-only variant one day and this needs revisiting, because a
      // tetrahedron's taps straddle faces at a crease.
      normalTaps: 4,
    })

  if (target !== fb && sampler) sampler.resolveInto(fb)

  // Overlays go into the output grid and never into the supersampler's fine
  // one: averaging a glyph with its neighbours is how text becomes smudge.
  // After the downsample and before `resolve`, so an axis left on auto still
  // gets its glyph picked from its own brightness.
  if (axes) {
    // Two settings, each measured rather than guessed.
    //
    // Short, because the camera fits the subject's radius with a little
    // margin and the *vertical* half angle is the tight one: a tip at 1.4
    // radii puts its letter on row -9 and 1.2 on row -3, while 1.0 lands on
    // row 2 and 0.9 on row 5. Only the y label is ever lost, which is why the
    // frame looked fine with an x and a z in it.
    //
    // And no depth test, because short enough to keep the labels also means
    // buried: the sphere's radius *is* the fitted radius, so its axes sit
    // entirely inside it. Depth-tested, the three axes draw 1, 1 and 1 cells
    // on the sphere and 1, 9 and 12 on the cube -- uneven even within one
    // subject. Untested they draw 23, 21 and 27 on both. A gizmo is an
    // annotation, like the letters that label it.
    drawAxes(fb, vp, { length: subject.radius * 0.9, depthTest: false })
    // A caption rather than a label on the subject: a projected point can
    // leave the frame, and the name is worth more than the exactness.
    drawText(fb, 1, 1, subject.name, { color: vec3(0.85, 0.9, 1) })
  }

  fb.resolve(RAMPS[rampNames[rampIndex]!])
  term.present(fb)

  term.status(
    `${shapes[shape]!.name} · ramp ${rampNames[rampIndex]} · ${fb.width}x${fb.height}` +
      `${antialias ? ` · ${SS_FACTOR}x aa` : ''}${wired ? ' · wire' : ''}${axes ? ' · axes' : ''}` +
      `${lamp ? ' · lamp' : ''} · ${loop.fps.toFixed(0)} fps` +
      '   [space] shape [r] ramp [t] tex [s] shadow [a] aa [w] wire [o] axes [l] lamp [n] normals [p] pause [q] quit',
  )
}, 60)
