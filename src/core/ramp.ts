/**
 * Character ramps, ordered from darkest to brightest. Which one reads best
 * depends on the font: `short` is safe everywhere, `long` has the finest
 * gradient on a classic monospace face, and `blocks` ignores glyph shape
 * entirely and only varies coverage.
 */
export const RAMPS = {
  short: ' .:-=+*#%@',
  long: ' .\'`^",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$',
  blocks: ' ░▒▓█',
  binary: ' #',
} as const

export type RampName = keyof typeof RAMPS

/** Rec. 709 relative luminance of a linear RGB triple. */
export function luminance(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

/** Picks the glyph of `ramp` that stands for brightness `lum` in [0, 1]. */
export function rampChar(ramp: string, lum: number): number {
  const i = Math.round(Math.min(1, Math.max(0, lum)) * (ramp.length - 1))
  return ramp.charCodeAt(i)
}
