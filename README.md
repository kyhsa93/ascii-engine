# ascii-engine

A 3D software renderer whose framebuffer is a grid of characters.

There is no GPU, no canvas and no `<canvas>` pixel buffer. Triangles go through
a real pipeline — model, view, projection, near-plane clipping, perspective
divide, edge-function rasterization, a depth buffer — and the fragment stage
ends by picking a glyph off a character ramp instead of writing a pixel.

The engine core knows nothing about where the frame goes. It fills a
`Framebuffer` (glyph + colour + depth per cell); a presenter takes it from
there. Two ship with the repo: ANSI escape sequences for a terminal, and a
`<pre>` element for a browser.

```
npm install
npm run demo      # terminal: spinning cube, sphere and torus
npm run web       # browser: the same scene in a <pre>
npm run check     # typecheck + test suite
npm run viewcheck # build, then drive the browser demo in Chromium
```

## Layout

```
src/core/     the renderer — no terminal, no DOM
  vec3.ts       small vector helpers
  mat4.ts       row-major 4x4 matrices, projection, lookAt, normal matrix
  mesh.ts       cube / sphere / torus / plane builders, OBJ parser
  camera.ts     view and projection matrices, cell-aspect correction
  raster.ts     near-plane clipping, perspective divide, scanline fill
  renderer.ts   mesh -> triangles -> rasterizer
  shading.ts    Lambert + Blinn-Phong, normal debug view
  framebuffer.ts  the character grid, and luminance -> glyph resolution
  ramp.ts       character ramps
src/term/     ANSI presenter (diffed writes, 256-colour and truecolour)
src/web/      <pre> presenter (measures its own cell size)
examples/     terminal demo
web/          browser demo
scripts/      test suite
```

## The three things that make this different from a pixel renderer

**A cell is not square.** A terminal cell is roughly twice as tall as it is
wide, so a projection told "the viewport is 80 by 40" renders everything
squashed. `aspectFor(width, height, cellAspect)` scales the aspect ratio by the
cell's own shape; the browser presenter measures that ratio instead of
assuming it. A sphere is the test case — get this wrong and it is an ellipse.

**Depth is stored as 1/w, not z.** Only 1/w interpolates linearly across a
triangle in screen space. Storing it that way is also what makes
perspective-correct attributes cheap: normals and world positions are divided
by w before interpolation and multiplied back after.

**The glyph is chosen last, from luminance.** Shaders write colour, not
characters. `Framebuffer.resolve(ramp)` maps each cell's Rec. 709 luminance
onto a ramp such as `" .:-=+*#%@"`. A shader that wants a specific glyph —
a wireframe, a marker — can set `out.char` and `resolve` leaves it alone.

## Writing a shader

A shader is a plain function. It is handed the interpolated fragment and a
scratch surface to write into; both objects are reused, so it allocates
nothing per cell.

```ts
import type { Shader } from './src/core/raster.ts'

const banded: Shader = (f, out) => {
  const len = Math.hypot(f.nx, f.ny, f.nz) || 1
  const light = Math.max(0, (f.nx * 0.5 + f.ny * 0.8 + f.nz * 0.6) / len)
  const step = Math.round(light * 4) / 4
  out.r = step
  out.g = step * 0.8
  out.b = step * 0.5
}
```

## Drawing a frame

```ts
import { Camera, aspectFor } from './src/core/camera.ts'
import { Framebuffer } from './src/core/framebuffer.ts'
import { rotationY } from './src/core/mat4.ts'
import { cube } from './src/core/mesh.ts'
import { RAMPS } from './src/core/ramp.ts'
import { drawMesh } from './src/core/renderer.ts'
import { lambert } from './src/core/shading.ts'
import { vec3 } from './src/core/vec3.ts'

const fb = new Framebuffer(100, 40)
fb.clear()

const camera = new Camera({ position: vec3(0, 0, 5) })
const vp = camera.viewProjection(aspectFor(fb.width, fb.height, 0.5))

drawMesh(fb, cube(2), rotationY(0.6), vp, lambert({ albedo: vec3(1, 0.8, 0.5) }))
fb.resolve(RAMPS.long)
console.log(fb.toString())
```

## Demo controls

Terminal and browser share the same scene.

| key | what it does |
| --- | --- |
| `space` | cycle cube / sphere / torus |
| `r` | cycle character ramp |
| `n` | toggle the normal debug view |
| `p` | pause the spin |
| arrows | orbit the camera |
| `+` / `-` | zoom |
| `q` | quit |

In the browser, drag to orbit and scroll to zoom.

## Notes

- Source is TypeScript with explicit `.ts` import extensions, which Node 24
  runs directly — the terminal demo has no build step. Vite handles the
  browser build.
- `npm run check` covers the invariants that break quietly while tuning a
  renderer: matrix round-trips, near-plane clipping, depth ordering, mesh
  winding, cell-aspect correction, and the terminal presenter's frame diff.
