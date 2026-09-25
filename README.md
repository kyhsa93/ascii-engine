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
npm run demo      # terminal: spinning cube, sphere, torus and a raymarched blend
npm run obj       # terminal: a Wavefront OBJ off disk
npm run web       # browser: the same scene in a <pre>
npm run check     # typecheck + test suite
npm run viewcheck # build, then drive the browser demo in Chromium
```

`npm run obj` takes a path: `node examples/obj.ts your-model.obj`. With none it
loads `models/knot.obj`, which `npm run model` regenerates.

## Layout

```
src/core/     the renderer — no terminal, no DOM
  vec3.ts       small vector helpers
  mat4.ts       row-major 4x4 matrices, projection, lookAt, normal matrix
  mesh.ts       cube / sphere / torus / plane builders, OBJ reader and writer
  camera.ts     view and projection matrices, cell-aspect correction
  raster.ts     near-plane clipping, perspective divide, scanline fill
  renderer.ts   mesh -> triangles -> rasterizer
  sdf.ts        signed distance primitives and the operations that combine them
  march.ts      distance field -> rays -> the same framebuffer
  texture.ts    sampling with wrap and filter modes, checker, ASCII art, PPM
  shadow.ts     a second march, toward the light
  supersample.ts  draw on a finer grid, average back down
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

## A second way in: distance fields

Triangles are not the only way to describe a surface. A signed distance field
is a function that answers "how far is this point from the surface" — negative
inside, positive outside — and that one number is enough to render with,
because it also says how far a ray may travel before it could hit anything.

`marchScene` walks those rays into the *same* `Framebuffer`, through the same
`Shader` type, so marched and triangulated geometry occlude each other in one
frame:

```ts
import { marchScene } from './src/core/march.ts'
import { sdBox, sdSphere, smoothUnion, translate } from './src/core/sdf.ts'

drawMesh(fb, floor, identity(), vp, lambert({ albedo: vec3(0.3, 0.3, 0.35) }))
marchScene(
  fb,
  smoothUnion(sdSphere(1.05), translate(sdBox(0.7, 0.7, 0.7), 0.9, 0.7, 0.4), 0.55),
  camera,
  aspect,
  lambert({ albedo: vec3(0.95, 0.8, 0.55), specular: 0.4, eye: camera.position }),
)
```

What makes the two paths agree is the depth convention. The rasterizer stores
1/w, where w is distance along the *view axis* — not along the ray. A ray
leaving the camera at an angle covers `t * cos(a)` of view axis per `t` of
travel, and folding that factor in is the whole of the bridge. Leave it out
and the marched surface tilts toward the camera at the edges of the frame,
where it is least obvious and most wrong. `npm run check` pins it down by
rendering a sphere both ways and comparing silhouette, coverage and depth.

What the field buys over a mesh is combination. `smoothUnion` welds two shapes
with a fillet that belongs to neither of them — near the seam the result is
closer to the camera than either input, so the blend is a surface in its own
right rather than two shapes drawn over each other. There is no mesh for the
demo's `blend` subject; it exists only as a function.

It is the expensive path: every cell walks its own ray, where the rasterizer
touches a cell only if a triangle covers it. At 80x23 cells the demo's blend
costs 2.1 ms a frame against 0.17 ms for a rasterized cube. Both demos give
the marcher a shorter step budget than the default — which, measured, buys
nothing at terminal size and cuts a 163x50 browser frame from 15.8 ms to
6.7 ms, for a difference of 41 glyphs out of 8150.

## Loading a model

`parseObj` reads the part of Wavefront OBJ that describes geometry — `v`, `vt`,
`vn`, and `f` with any of its index spellings, negative indices included — and
fans polygons into triangles. `writeObj` goes the other way.

```ts
import { readFileSync } from 'node:fs'
import { boundingBox, parseObj } from './src/core/mesh.ts'

const mesh = parseObj(readFileSync('models/knot.obj', 'utf8'))
const bounds = boundingBox(mesh)
```

A mesh off disk is not the same problem as one from a builder: nothing knows
how big it is or where it sits. `boundingBox` answers both, and the OBJ demo
aims the camera at the mesh's own centre and fits to a radius measured from
there — `boundingRadius`, which measures from the origin, would frame a model
built somewhere else far too small.

Two details the round-trip test pins down. A cube's vertices share positions
but not normals, so a reader that merges by position alone rounds its edges
off; the reader keys on the position/normal pair instead. And a file may
declare normals and still leave some faces without them — those vertices get
derived normals, but only those, because a zero normal shades as unlit black
and reads as a lighting choice rather than as a bug.

## Textures

Texture coordinates travel through the pipeline as two more interpolated
attributes and arrive on the fragment as `u` and `v`. `lambert` takes a `map`
and multiplies it into the albedo; any shader can call `sample` itself.

```ts
import { checker, fromAscii, parsePpm, sample } from './src/core/texture.ts'

drawMesh(fb, cube(2), model, vp, lambert({ albedo: vec3(1, 0.9, 0.7), map: checker(6) }))
```

Nothing here decodes PNG or JPEG. Adding an image library to a renderer that
draws with characters would be a strange trade, so what it reads instead is
PPM — a real image format a few dozen lines can parse — alongside two sources
that need no file at all: a procedural `checker`, and `fromAscii`, which reads
a block of ASCII art back through a ramp into brightness. That last one is the
inverse of what `Framebuffer.resolve` does, and it means a picture drawn in
characters can be wrapped around a solid drawn in characters.

Three details the tests hold shut:

- **Interpolation is perspective-correct.** Affine texture coordinates are the
  classic wobble of a software rasterizer, and they are invisible until the
  surface is steep. The check renders a plane receding to the horizon and
  compares every fragment's `u` and `v` against a ray-plane intersection
  computed outside the rasterizer — not against the fragment's own world
  position, which rides the same interpolator and would be wrong in step.
- **Bilinear sampling is centred on the texel, not its corner.** Skip the
  half-texel shift and the picture slides by half a texel whenever it is
  magnified: invisible on a photograph, obvious on a checkerboard.
- **The v axis flips at the OBJ boundary.** OBJ measures v upward from the
  bottom edge; a texture here is stored with `v = 0` as its first row. The
  reader and writer each flip, and a flip applied once rather than twice
  survives every other test and shows up only as an upside-down picture.

## Shadows

A shadow is the same march again, from the surface toward the light: if
anything is in the way, the point is dark. `shadowFrom` returns a function of
position, and `lambert` takes it as `shadow`.

```ts
import { shadowFrom } from './src/core/shadow.ts'

const occlusion = shadowFrom(field, { light, softness: 12 })
drawMesh(fb, floor, identity(), vp, lambert({ light, shadow: occlusion }))
```

Because the answer needs only a *position*, this reaches across the two render
paths: the floor in the demo is triangles, and what darkens it is a distance
field it has nothing else to do with. The limit is the other direction — an
occluder has to be a field, since casting from a mesh would need a
ray-triangle structure this renderer does not have. So every demo subject
carries a field as well as a mesh, and the two have to describe the same
shape in the same pose or the shadow drifts away from the thing casting it.

The marching itself is held against arithmetic rather than against a picture.
A sphere's shadow has a closed form — the ray from a floor point toward the
light is blocked exactly when it passes within the radius — so `npm run check`
asks both and requires every floor fragment in the shot to agree, bar the
handful sitting on the silhouette where a whisker either way decides it.

Three things that are easy to get wrong here:

- **Ambient is not shadowed.** It stands for light arriving from everywhere
  else; scaling it too turns every shadow into a black hole.
- **The ray starts at a bias, not at zero.** A point on the caster is by
  definition at distance zero from the field, so an unbiased ray reports a hit
  immediately and the whole object goes black.
- **Softness comes free from the march.** Tracking the closest the ray came to
  a surface, measured against how far it had travelled to get there, gives a
  penumbra without a second pass: graze an edge from far away and you are
  barely dimmed, graze it from close up and you are nearly blocked.
- **The umbra has to clear the ramp's first glyph.** This one only bites a
  renderer made of characters. A shadow whose luminance rounds to index zero
  is drawn in spaces, and a shadow made of spaces does not read as a dark
  patch — it reads as a hole where the floor ran out. Measured on the demo's
  first floor: 334 of its 336 umbra cells came out blank. Raising the floor's
  albedo and ambient put every one of them on a glyph instead.
- **A light behind the camera hides every shadow it casts.** Not a bug, just
  geometry: the shadow falls exactly where the caster already covers the
  screen. The demo opens from the other side for that reason — measured on the
  sphere, no shadowed floor cell at all is visible from the old default view
  and 381 are from the new one, and raising the camera does not rescue it.

## Antialiasing

One sample per cell means a silhouette is either in or out, so it comes out as
a staircase. `Supersampler` draws the scene on a grid `factor` times finer in
each direction and averages the colour back down.

```ts
import { Supersampler } from './src/core/supersample.ts'

const aa = new Supersampler(fb.width, fb.height, 2)
aa.clear()
drawMesh(aa.buffer, mesh, model, vp, shader)   // or marchScene(aa.buffer, ...)
aa.resolveInto(fb)
fb.resolve(RAMPS.long)
```

This suits a character renderer better than it suits a pixel one. Because the
glyph is picked from luminance *after* shading, a half-covered cell comes out
half as bright and therefore lands on a middling character by itself — the
edge becomes a gradient with no special case anywhere in the rasterizer. And
because it is the framebuffer that changes rather than anything that draws
into one, the marched path gets it for free.

Four things worth knowing:

- **The fine grid has the same shape as the output.** Both axes scale by the
  same factor, so `aspectFor` gives the same answer for either and the camera
  needs no adjustment.
- **Colour is averaged; depth is not.** A cell takes the *nearest* of its
  sub-samples, so a partly covered cell still occludes what is drawn behind it
  afterwards. Averaging depth would place the cell where no surface is.
- **A forced glyph cannot be averaged.** A shader that sets `out.char` wins
  outright at the nearest sub-sample, and that edge stays hard.
- **It costs less than the square of the factor.** Only the cells a triangle
  actually covers pay that; clearing and resolving scale with the grid, and
  most of a frame is usually background. Measured on a sphere: 80x23 cells go
  from 0.64 ms a frame to 0.84 at 2x and 1.35 at 3x, and 163x50 from 0.88 to
  2.51 and 5.22.
- **A cell no sub-sample touched keeps the background's space**, so a frame
  with nothing in it resolves identically either way.

The check does not compare pictures. A quad's edge projects to a screen
position that can be worked out from the projection by hand, so the cell it
lands in has an exact coverage — some whole number of sub-columns out of
`factor` — and the resolved brightness has to be that fraction exactly.

## Wireframe

The fragment arrives knowing how far it is from the nearest edge of its
triangle, in cells, so drawing the edges is a shader rather than a second
drawing path.

```ts
import { wireframe } from './src/core/shading.ts'

drawMesh(fb, mesh, model, vp, wireframe({ width: 0.6 }))
drawMesh(fb, mesh, model, vp, wireframe({ width: 0.6, fill: lambert({ ... }) }))
```

Measuring that distance in *cells* rather than in barycentrics is the whole
point. The usual shortcut — colour the fragment when some barycentric drops
below a threshold — makes the wire thin out as its triangle grows on screen,
because a barycentric is a fraction of the triangle rather than a length. An
edge function is already the distance to that edge times the edge's length, so
dividing the triangle's area by each edge length turns the three barycentrics
back into distances; all three factors are constant across the triangle, which
leaves three multiplies per fragment.

Given no `fill`, the interior is drawn blank — and blank still writes depth, so
a solid nearer the camera goes on hiding the wires behind it. Hidden-line
removal is the depth buffer doing its usual job.

Two things it is honest about rather than hiding. The edges belong to the
*triangle*, so a quad built from two of them shows the diagonal, and a
triangle cut by the near plane shows the cut. And a cell is taller than it is
wide, so a wire measured in cells is physically thicker across a horizontal
edge than across a vertical one.

There is also a floor under the width. At one sample per cell a wire much
thinner than a whole cell comes apart, because a diagonal edge keeps missing
the cell centres it passes between. A cube's edges are a connected graph, so
a correct drawing of them is a single 8-connected blob — measured, a width of
0.6 leaves 80x23 cells showing it in ten pieces and 163x50 in twenty-seven,
0.85 joins the smaller grid up but not the larger, and 1.0 joins both. That
is why the demos ask for a whole cell. Supersampling does not rescue a
thinner wire either: averaging spreads the same ink over more cells and dims
it, which reads as more gaps rather than fewer.

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
| `space` | cycle cube / sphere / torus / blend (the raymarched one) |
| `r` | cycle character ramp |
| `t` | toggle the checker map (meshes only — a field has no uv) |
| `s` | toggle the floor and its shadow |
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
