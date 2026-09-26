/**
 * The browser demo, in a real browser, looked at.
 *
 * `npm run check` covers the renderer, but it cannot cover the one number the
 * browser presenter gets from the page itself: the cell aspect, measured off
 * the real font. Get that wrong and every sphere is an ellipse -- on a grid
 * whose shape nothing in Node can see. So this boots the built page, reads
 * the aspect actually in use, and asserts the silhouette is physically round
 * at that aspect.
 *
 * It also catches the failure mode every module-level check misses: a page
 * that throws on boot. Each assertion here would pass in Node.
 *
 *   npm run viewcheck
 */

import { createReadStream, existsSync, mkdirSync, statSync } from 'node:fs'
import { createServer } from 'node:http'
import type { AddressInfo } from 'node:net'
import { extname, join, normalize, resolve } from 'node:path'
import { chromium, type Page } from 'playwright'

const DIST = resolve(process.cwd(), 'dist')
const SHOTS = resolve(process.cwd(), 'shots')

const TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
}

let failed = 0

function check(name: string, fn: () => void): void {
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

interface Probe {
  cellAspect: number
  cols: number
  rows: number
  frames: number
  shape: string
  lines: number
  glyphs: number
  bboxW: number
  bboxH: number
  /** A cheap hash of the whole frame, for "did this change at all" questions. */
  digest: number
  /**
   * Whether the frame holds an x, a y and a z.
   *
   * A safe signal that a letter came from an overlay rather than from shading:
   * the demo's ramp is `short`, which is ` .:-=+*#%@` and contains none of
   * them, and no ramp in the engine contains a lowercase y at all.
   */
  letters: boolean
  /** The distinct characters in the frame, for a readable failure message. */
  sample: string
  minX: number
  maxX: number
  minY: number
  maxY: number
  overflow: boolean
}

/** True when the drawn shape runs into any edge of the grid, i.e. is cut off. */
function cutOff(p: Probe): boolean {
  return p.minX <= 0 || p.maxX >= p.cols - 1 || p.minY <= 0 || p.maxY >= p.rows - 1
}

/** Reads the frame out of the `<pre>` and measures the shape drawn in it. */
function readScreen(page: Page): Promise<Probe> {
  return page.evaluate(() => {
    const probe = (window as unknown as { __engine: Record<string, number | string> }).__engine
    const text = document.getElementById('screen')!.textContent ?? ''
    const lines = text.split('\n')

    let minX = Infinity
    let maxX = -1
    let minY = Infinity
    let maxY = -1
    const glyphs = new Set<string>()
    lines.forEach((line, y) => {
      for (let x = 0; x < line.length; x++) {
        if (line[x] === ' ') continue
        glyphs.add(line[x]!)
        minX = Math.min(minX, x)
        maxX = Math.max(maxX, x)
        minY = Math.min(minY, y)
        maxY = Math.max(maxY, y)
      }
    })

    let digest = 0
    for (let i = 0; i < text.length; i++) digest = (digest * 31 + text.charCodeAt(i)) | 0

    const root = document.documentElement
    return {
      digest,
      letters: text.includes('x') && text.includes('y') && text.includes('z'),
      sample: [...glyphs].sort().join(''),
      cellAspect: probe.cellAspect as number,
      cols: probe.cols as number,
      rows: probe.rows as number,
      frames: probe.frames as number,
      shape: probe.shape as string,
      lines: lines.length,
      glyphs: glyphs.size,
      bboxW: maxX - minX + 1,
      bboxH: maxY - minY + 1,
      minX,
      maxX,
      minY,
      maxY,
      overflow: root.scrollWidth > root.clientWidth,
    }
  })
}

const server = createServer((req, res) => {
  const url = (req.url ?? '/').split('?')[0]!
  const candidate = join(DIST, normalize(url).replace(/^(\.\.[/\\])+/, ''))
  const file = existsSync(candidate) && statSync(candidate).isFile() ? candidate : join(DIST, 'index.html')
  res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream' })
  createReadStream(file).pipe(res)
})

if (!existsSync(join(DIST, 'index.html'))) {
  console.error(`no build at ${DIST} -- run "npm run build" first`)
  process.exit(1)
}
mkdirSync(SHOTS, { recursive: true })

await new Promise<void>((done) => server.listen(0, '127.0.0.1', done))
const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/`

const browser = await chromium.launch()
const problems: string[] = []

try {
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 } })
  const page = await context.newPage()
  page.on('console', (m) => {
    if (m.type() === 'error') problems.push(m.text())
  })
  page.on('pageerror', (e) => problems.push(e.message))

  await page.goto(base, { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(600)

  console.log('\nbrowser demo')

  const first = await readScreen(page)

  check('the page boots without throwing', () => {
    assert(problems.length === 0, `console and page errors: ${problems.join(' | ')}`)
  })

  check('the pre is filled with a grid, not left empty', () => {
    assert(first.cols > 60, `expected a wide grid at 1280px, got ${first.cols} columns`)
    assert(first.rows > 20, `expected a tall grid at 720px, got ${first.rows} rows`)
    assert(first.lines >= first.rows, `the pre holds ${first.lines} lines for ${first.rows} rows`)
    assert(first.glyphs >= 2, `nothing was drawn: ${first.glyphs} distinct glyphs`)
  })

  check('the measured cell aspect is plausible for a monospace font', () => {
    assert(
      first.cellAspect > 0.4 && first.cellAspect < 0.8,
      `measured cell aspect ${first.cellAspect.toFixed(3)} is outside the range a monospace cell can have`,
    )
  })

  await page.waitForTimeout(500)
  const second = await readScreen(page)
  check('frames keep being drawn', () => {
    assert(second.frames > first.frames, `frame count stuck at ${first.frames}`)
  })

  // The sphere is the shape that tells the truth about cell aspect: it is
  // rotation-invariant, so whatever spin it is on, a correct projection draws
  // it physically round.
  await page.click('button[data-action="shape"]')
  await page.waitForTimeout(300)
  const round = await readScreen(page)

  check('the shape button reaches the sphere', () => {
    assert(round.shape === 'sphere', `expected the sphere, got ${round.shape}`)
  })

  check('the sphere is physically round on a non-square grid', () => {
    // Cells are taller than they are wide, so a round sphere must span more
    // of them across than down -- by exactly the cell's own aspect.
    const physical = (round.bboxW * round.cellAspect) / round.bboxH
    assert(
      physical > 0.8 && physical < 1.25,
      `silhouette ${round.bboxW}x${round.bboxH} cells at aspect ${round.cellAspect.toFixed(3)} ` +
        `is ${physical.toFixed(3)} wide per unit tall, expected about 1`,
    )
  })

  check('the sphere renders a gradient, not a flat blob', () => {
    assert(round.glyphs >= 5, `only ${round.glyphs} distinct glyphs across the sphere`)
  })

  // Pause first. The subject spins, so a frame taken before and after any
  // button press differs no matter what the button does — the comparison
  // would pass on a texture toggle wired to nothing at all.
  await page.click('button[data-action="pause"]')
  await page.waitForTimeout(250)
  const plain = await readScreen(page)
  await page.click('button[data-action="texture"]')
  await page.waitForTimeout(250)
  const mapped = await readScreen(page)

  check('the texture button changes what is drawn', () => {
    assert(
      mapped.digest !== plain.digest,
      'the frame is identical with and without the checker map, so the map never reached the shader',
    )
    assert(mapped.shape === plain.shape, `the subject changed underneath the test: ${mapped.shape}`)
    assert(mapped.glyphs >= 5, `the mapped sphere lost its gradient: ${mapped.glyphs} glyphs`)
  })

  // Still paused, so once again the button is the only thing that can have
  // changed the frame. The floor is what makes this worth checking in a
  // browser at all: it is triangles, and what darkens it is a distance field,
  // so this is the one assertion that watches a shadow cross between the two
  // render paths in the page that actually ships.
  await page.click('button[data-action="shadow"]')
  await page.waitForTimeout(350)
  const grounded = await readScreen(page)

  await page.waitForTimeout(400)
  const stillRunning = await readScreen(page)

  check('the shadow button puts a floor under the subject', () => {
    // What this can honestly claim is that the page took the extra work: a
    // floor appeared, nothing threw, and frames kept coming. Whether the
    // shadow lands in the right place is settled in `npm run check`, against
    // a line computed outside the renderer — a browser can only show that the
    // picture changed, which is a much weaker thing to know.
    assert(grounded.digest !== mapped.digest, 'the frame did not change when shadows were switched on')
    assert(
      grounded.maxY >= grounded.rows - 1,
      `expected a floor reaching the bottom row, drawn content stops at row ${grounded.maxY} of ${grounded.rows}`,
    )
    assert(grounded.glyphs >= 5, `the floor flattened the frame to ${grounded.glyphs} glyphs`)
    assert(problems.length === 0, `the shadow path threw: ${problems.join(' | ')}`)
    assert(
      stillRunning.frames > grounded.frames,
      `frames stopped after switching shadows on, stuck at ${grounded.frames}`,
    )
  })

  // Still paused. Antialiasing changes no geometry at all — it draws the same
  // scene on a finer grid and averages back down — so the frame changing is
  // the whole of what a browser can show here. What the averaging actually
  // computes is settled in `npm run check`, against a coverage fraction worked
  // out from the projection by hand.
  await page.click('button[data-action="aa"]')
  await page.waitForTimeout(400)
  const smoothed = await readScreen(page)

  check('the aa button resamples without breaking the frame', () => {
    // This used to also require the glyph count not to fall, and that was
    // wrong twice over. Averaging pulls extremes toward the middle, so on a
    // ten-level ramp two neighbouring shades can land on one -- and which
    // shades are on screen depends on the spin angle the pause happened to
    // catch, so the assertion came and went between runs. It reproduces at
    // 2ea5d46, before any of the lighting work, which is how it was ruled out
    // as a regression. What the averaging actually computes is settled in
    // `npm run check`: a flat silhouette has exactly 2 shades there and a
    // supersampled one at least 5.
    assert(smoothed.digest !== grounded.digest, 'the frame is identical with and without supersampling')
    assert(smoothed.shape === grounded.shape, `the subject changed underneath the test: ${smoothed.shape}`)
    assert(smoothed.glyphs >= 2, `the frame collapsed to ${smoothed.glyphs} glyphs`)
    assert(problems.length === 0, `the supersampling path threw: ${problems.join(' | ')}`)
    assert(smoothed.frames > grounded.frames, `frames stopped after switching aa on, stuck at ${grounded.frames}`)
  })

  // Still paused. Wireframe replaces the interior with blanks and keeps the
  // edges, so the silhouette should not move while what fills it does. How
  // wide the wire comes out, and that it stays that wide on triangles of very
  // different sizes, is settled in `npm run check` against plane geometry --
  // a page can only show that the frame changed and the outline did not.
  await page.click('button[data-action="wire"]')
  await page.waitForTimeout(400)
  const wired = await readScreen(page)

  check('the wire button redraws the subject as its edges', () => {
    assert(wired.digest !== smoothed.digest, 'the frame is identical with and without the wireframe')
    assert(
      Math.abs(wired.bboxW - smoothed.bboxW) <= 2 && Math.abs(wired.bboxH - smoothed.bboxH) <= 2,
      `the silhouette moved: ${smoothed.bboxW}x${smoothed.bboxH} became ${wired.bboxW}x${wired.bboxH}`,
    )
    assert(problems.length === 0, `the wireframe path threw: ${problems.join(' | ')}`)
    assert(wired.frames > smoothed.frames, `frames stopped after switching the wire on, stuck at ${smoothed.frames}`)
  })

  // Still paused. The overlay is the one thing drawn after the downsample and
  // before the ramp is applied, so this is where a page can catch it being
  // wired into the wrong buffer -- text averaged with its neighbours comes
  // back as smudge rather than letters. Where a label lands is settled in
  // `npm run check` against the projection written out by hand.
  await page.click('button[data-action="axes"]')
  await page.waitForTimeout(400)
  const annotated = await readScreen(page)

  check('the axes button writes letters over the frame', () => {
    assert(annotated.digest !== wired.digest, 'the frame is identical with and without the axes')
    assert(annotated.letters, `expected the x, y and z labels, found "${annotated.sample}"`)
    assert(problems.length === 0, `the overlay path threw: ${problems.join(' | ')}`)
    assert(annotated.frames > wired.frames, `frames stopped after switching axes on, stuck at ${wired.frames}`)
  })

  // Still paused. A lamp adds light rather than replacing it, so the frame
  // has to get brighter somewhere and cannot get darker anywhere -- that is
  // the one claim a page can make about it. Whether the falloff is an inverse
  // square, and whether the range really reaches zero, is arithmetic and is
  // settled in `npm run check`.
  // The wireframe shader never calls `lambert` -- with no fill it writes a
  // line colour or a blank -- so a lamp switched on underneath it changes
  // nothing at all. Measured: 667 of 8150 glyphs move when the subject is
  // shaded, and 0 when it is wire. So put the shading back first, or this
  // check asks a question the state cannot answer.
  await page.click('button[data-action="wire"]')
  await page.waitForTimeout(250)
  const reshaded = await readScreen(page)

  await page.click('button[data-action="lamp"]')
  await page.waitForTimeout(400)
  const lamplit = await readScreen(page)

  check('the lamp button adds light without taking any away', () => {
    // Compare against `reshaded`, the frame immediately before the lamp -- not
    // against `annotated`, which still had the wireframe on. Comparing across
    // the wire-off click would pass on that click alone and say nothing about
    // the lamp, which is how this check read before the state was fixed.
    assert(reshaded.digest !== annotated.digest, 'switching the wireframe off changed nothing')
    assert(lamplit.digest !== reshaded.digest, 'the frame is identical with and without the lamp')
    assert(problems.length === 0, `the point light path threw: ${problems.join(' | ')}`)
    assert(lamplit.frames > reshaded.frames, `frames stopped after the lamp, stuck at ${reshaded.frames}`)
  })

  await page.screenshot({ path: join(SHOTS, 'desktop.png') })

  const phone = await context.newPage()
  await phone.goto(base, { waitUntil: 'domcontentloaded' })
  await phone.setViewportSize({ width: 390, height: 844 })
  await phone.waitForTimeout(600)
  const small = await readScreen(phone)

  check('a phone-width viewport neither overflows nor collapses', () => {
    assert(!small.overflow, 'the page scrolls sideways at 390px')
    assert(small.cols > 20 && small.rows > 20, `grid collapsed to ${small.cols}x${small.rows}`)
    assert(small.glyphs >= 2, 'nothing was drawn at phone width')
  })

  check('the subject is framed, not cut off, at both viewports', () => {
    // A portrait grid is a far narrower frustum than a landscape one at the
    // same vertical field of view. This is the assertion the earlier phone
    // check was missing: the page had no overflow, no collapse and plenty of
    // glyphs while the cube ran clean off both sides of the screen.
    assert(!cutOff(round), `desktop: shape spans ${round.minX}..${round.maxX} of ${round.cols} columns`)
    assert(
      !cutOff(small),
      `phone: shape spans ${small.minX}..${small.maxX} of ${small.cols} columns and ` +
        `${small.minY}..${small.maxY} of ${small.rows} rows`,
    )
  })

  await phone.screenshot({ path: join(SHOTS, 'phone.png') })

  // The presenter writes a node per row and rounds colour so that runs of one
  // colour are long. Both are performance decisions with visible consequences,
  // so both are held to something here.
  const presented = await page.evaluate(() => {
    const screen = document.getElementById('screen')!
    const levels = new Set<number>()
    for (let i = 0; i <= 31; i++) levels.add(Math.round((i / 31) * 255))
    const offGrid: string[] = []
    let spans = 0
    for (const span of Array.from(screen.querySelectorAll('span'))) {
      const colour = (span as HTMLElement).style.color
      const parts = /rgb\((\d+),\s*(\d+),\s*(\d+)\)/.exec(colour)
      if (!parts) continue
      spans++
      for (let c = 1; c <= 3; c++) {
        if (!levels.has(Number(parts[c]))) offGrid.push(colour)
      }
    }
    return { rows: screen.children.length, spans, offGrid: offGrid.slice(0, 3), offCount: offGrid.length }
  })

  check('the presenter keeps a node per row of the grid', () => {
    assert(
      presented.rows === first.rows,
      `${presented.rows} row nodes against a grid ${first.rows} rows tall`,
    )
  })

  check('colour reaches the page rounded to the depth asked for', () => {
    // Five bits a channel, which is what keeps a row to about fifteen spans
    // instead of thirty-six. A channel off that grid means the rounding was
    // skipped, and the frame rate goes with it: measured, eight bits cost a
    // third of it.
    assert(presented.spans > 0, 'no coloured spans on screen to check at all')
    assert(
      presented.offCount === 0,
      `${presented.offCount} channels are not on the five-bit grid, e.g. ${presented.offGrid.join(', ')}`,
    )
  })

  console.log(
    `\n  grid ${first.cols}x${first.rows} cells, cell aspect ${first.cellAspect.toFixed(3)}` +
      `, sphere ${round.bboxW}x${round.bboxH}\n  shots in ${SHOTS}\n`,
  )
} finally {
  await browser.close()
  server.close()
}

console.log(failed === 0 ? 'all browser checks passed\n' : `${failed} browser check(s) failed\n`)
process.exit(failed === 0 ? 0 : 1)
