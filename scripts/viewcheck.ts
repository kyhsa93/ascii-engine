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
