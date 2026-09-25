/**
 * Textures: a grid of colours, and the rules for reading between its texels.
 *
 * Nothing here decodes PNG or JPEG — there is no image library in this repo
 * and adding one to draw with characters would be a strange trade. What it
 * reads instead is PPM, which is a real image format that a few dozen lines
 * can parse, plus two sources that need no file at all: a procedural checker,
 * and ASCII art read back as brightness.
 */

import { RAMPS } from './ramp.ts'
import type { Vec3 } from './vec3.ts'

export type Wrap = 'repeat' | 'clamp' | 'mirror'
export type Filter = 'nearest' | 'bilinear'

export interface Texture {
  readonly width: number
  readonly height: number
  /** RGB in 0..1, three floats per texel, row-major with `v = 0` the first row. */
  readonly data: Float32Array
  wrap: Wrap
  filter: Filter
}

export interface TextureOptions {
  wrap?: Wrap
  filter?: Filter
}

export function texture(width: number, height: number, data: Float32Array, options: TextureOptions = {}): Texture {
  if (data.length !== width * height * 3) {
    throw new Error(`texture: expected ${width * height * 3} floats, got ${data.length}`)
  }
  return { width, height, data, wrap: options.wrap ?? 'repeat', filter: options.filter ?? 'bilinear' }
}

/** Folds a texel coordinate back inside the image according to the wrap mode. */
function fold(i: number, n: number, mode: Wrap): number {
  if (mode === 'clamp') return i < 0 ? 0 : i >= n ? n - 1 : i
  if (mode === 'repeat') return ((i % n) + n) % n
  // Mirror repeats over a period of 2n, walking back down the second half.
  const period = 2 * n
  const k = ((i % period) + period) % period
  return k < n ? k : period - 1 - k
}

function texel(tex: Texture, x: number, y: number, out: Float32Array, weight: number): void {
  const i = (fold(y, tex.height, tex.wrap) * tex.width + fold(x, tex.width, tex.wrap)) * 3
  out[0] = out[0]! + tex.data[i]! * weight
  out[1] = out[1]! + tex.data[i + 1]! * weight
  out[2] = out[2]! + tex.data[i + 2]! * weight
}

/** Reads the texture at `(u, v)` and writes RGB into `out`. Allocation-free. */
export function sample(tex: Texture, u: number, v: number, out: Float32Array): void {
  out[0] = 0
  out[1] = 0
  out[2] = 0

  if (tex.filter === 'nearest') {
    texel(tex, Math.floor(u * tex.width), Math.floor(v * tex.height), out, 1)
    return
  }

  // Texel centres sit at half-integer coordinates, so the sample point has to
  // be shifted by half a texel before the floor — without it the filter is a
  // half-texel off, which reads as a picture that slides when it is scaled.
  const fx = u * tex.width - 0.5
  const fy = v * tex.height - 0.5
  const x0 = Math.floor(fx)
  const y0 = Math.floor(fy)
  const tx = fx - x0
  const ty = fy - y0

  texel(tex, x0, y0, out, (1 - tx) * (1 - ty))
  texel(tex, x0 + 1, y0, out, tx * (1 - ty))
  texel(tex, x0, y0 + 1, out, (1 - tx) * ty)
  texel(tex, x0 + 1, y0 + 1, out, tx * ty)
}

/**
 * A checkerboard, one texel per square.
 *
 * Generated at exactly the resolution of the pattern and filtered with
 * `nearest`, so the squares stay square at any magnification. Scaling a
 * higher-resolution checker with `bilinear` would blur every edge instead.
 */
export function checker(
  squares = 8,
  light: Vec3 = { x: 0.95, y: 0.93, z: 0.88 },
  dark: Vec3 = { x: 0.14, y: 0.13, z: 0.17 },
): Texture {
  const data = new Float32Array(squares * squares * 3)
  for (let y = 0; y < squares; y++) {
    for (let x = 0; x < squares; x++) {
      const c = (x + y) % 2 === 0 ? light : dark
      const i = (y * squares + x) * 3
      data[i] = c.x
      data[i + 1] = c.y
      data[i + 2] = c.z
    }
  }
  return texture(squares, squares, data, { filter: 'nearest', wrap: 'repeat' })
}

/**
 * Reads a block of ASCII art as a greyscale texture, mapping each character
 * back through a ramp to the brightness it stands for.
 *
 * The inverse of what `Framebuffer.resolve` does, and the reason it is here:
 * a picture drawn in characters can be wrapped around a solid that is itself
 * drawn in characters. Rows are padded to the longest line, and a character
 * the ramp does not contain counts as full brightness unless it is a space.
 */
export function fromAscii(art: string, ramp: string = RAMPS.long, options: TextureOptions = {}): Texture {
  const rows = art.replace(/\n+$/, '').split('\n')
  const width = Math.max(1, ...rows.map((r) => r.length))
  const height = Math.max(1, rows.length)
  const data = new Float32Array(width * height * 3)

  for (let y = 0; y < height; y++) {
    const row = rows[y] ?? ''
    for (let x = 0; x < width; x++) {
      const ch = row[x] ?? ' '
      const at = ramp.indexOf(ch)
      const level = at >= 0 ? at / (ramp.length - 1) : ch === ' ' ? 0 : 1
      const i = (y * width + x) * 3
      data[i] = level
      data[i + 1] = level
      data[i + 2] = level
    }
  }
  return texture(width, height, data, { filter: options.filter ?? 'nearest', ...options })
}

function isSpace(b: number): boolean {
  return b === 0x20 || b === 0x09 || b === 0x0a || b === 0x0d
}

/**
 * Reads a binary (P6) or ASCII (P3) Netpbm pixmap.
 *
 * The header is whitespace-separated tokens with `#` comments, and for P6
 * exactly one whitespace byte separates the header from the pixel block — so
 * the reader advances by one rather than skipping whitespace, or the first
 * pixel of an image that happens to start with a 0x20 byte would be eaten.
 */
export function parsePpm(bytes: Uint8Array, options: TextureOptions = {}): Texture {
  let at = 0
  const token = (): string => {
    for (;;) {
      while (at < bytes.length && isSpace(bytes[at]!)) at++
      if (bytes[at] === 0x23) while (at < bytes.length && bytes[at] !== 0x0a) at++
      else break
    }
    const start = at
    while (at < bytes.length && !isSpace(bytes[at]!)) at++
    let s = ''
    for (let i = start; i < at; i++) s += String.fromCharCode(bytes[i]!)
    return s
  }

  const magic = token()
  if (magic !== 'P6' && magic !== 'P3') throw new Error(`parsePpm: not a P3 or P6 file (got "${magic}")`)
  const width = Number(token())
  const height = Number(token())
  const max = Number(token())
  if (!(width > 0 && height > 0)) throw new Error(`parsePpm: bad dimensions ${width}x${height}`)
  if (!(max > 0 && max <= 255)) throw new Error(`parsePpm: only 8-bit samples are supported, maxval was ${max}`)

  const data = new Float32Array(width * height * 3)
  if (magic === 'P6') {
    at++
    if (bytes.length - at < data.length) throw new Error('parsePpm: pixel data is short')
    for (let i = 0; i < data.length; i++) data[i] = bytes[at + i]! / max
  } else {
    for (let i = 0; i < data.length; i++) data[i] = Number(token()) / max
  }
  return texture(width, height, data, options)
}

/** Writes a texture as a binary (P6) pixmap. */
export function writePpm(tex: Texture): Uint8Array {
  const header = `P6\n${tex.width} ${tex.height}\n255\n`
  const out = new Uint8Array(header.length + tex.data.length)
  for (let i = 0; i < header.length; i++) out[i] = header.charCodeAt(i)
  for (let i = 0; i < tex.data.length; i++) {
    out[header.length + i] = Math.round(Math.min(1, Math.max(0, tex.data[i]!)) * 255)
  }
  return out
}
