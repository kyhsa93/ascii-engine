/**
 * Shadows, by marching a second ray toward the light.
 *
 * The raymarcher already walks rays, so asking "is anything between this point
 * and the light" costs the same machinery again. What makes it useful beyond
 * the marched path is that the answer only needs a *position*: a rasterized
 * triangle knows where its fragment is, so a mesh can be shadowed by a
 * distance field it has nothing else to do with.
 *
 * The occluders have to be a field, though. Nothing here can cast a shadow
 * from a mesh — that would need a ray-triangle structure this renderer does
 * not have.
 */

import type { Sdf } from './sdf.ts'
import type { Vec3 } from './vec3.ts'
import { normalize, vec3 } from './vec3.ts'

/** How much of the light reaches a point: 0 fully shadowed, 1 fully lit. */
export type Occlusion = (x: number, y: number, z: number) => number

export interface ShadowOptions {
  /** Direction from the surface *toward* the light. Normalized on the way in. */
  light?: Vec3
  /**
   * Penumbra sharpness. Leave at 0 for a hard edge; higher values narrow the
   * soft band, so 8 is a wide penumbra and 64 is nearly hard.
   */
  softness?: number
  /** Stop looking for occluders past this distance. */
  maxDistance?: number
  maxSteps?: number
  epsilon?: number
  /**
   * How far along the ray to start.
   *
   * Without it every lit surface shadows itself: a point on the caster is by
   * definition at distance zero from the field, so the first sample reports a
   * hit and the whole object goes black.
   */
  bias?: number
  /**
   * Radius of a sphere about the origin holding every caster in `field`.
   *
   * Given one, a ray that cannot reach the casters is answered as fully lit
   * without sampling the field at all. On the demo's floor that is most of
   * them: 91% of a directional shadow pass was rays marching away from the
   * only thing in the scene.
   *
   * This is the caster's own radius, not the radius to reject at -- the margin
   * a penumbra needs is worked out from `softness` and `maxDistance`, because
   * getting it wrong is not a rounding error. Softness is a running minimum of
   * `softness * d / t`, so the rays that produce a penumbra are exactly the
   * ones that pass *wide* of the caster; reject those and the soft band
   * snaps to fully lit. Measured on a sphere of radius 1.3 with softness 12,
   * rejecting at 1.35 moves a floor point from 0.1667 to 1.0000.
   *
   * The one thing this must not do is move where a surviving ray starts.
   * Clipping the start to the sphere's entry point is what the camera marcher
   * does and it is pure damage here: it drops the early samples the running
   * minimum is made of, and the penumbra was still wrong at four times the
   * caster's radius. Rays are rejected or left alone, never shortened.
   */
  casterRadius?: number
}

/**
 * How wide of the casters a ray can pass and still matter.
 *
 * Two separate margins, and leaving either out loses geometry.
 *
 * A soft ray dims something wherever `softness * d / t < 1`, so the furthest
 * one that still counts clears the casters by `reach / softness`. Reject
 * inside that and the soft band snaps to fully lit: measured on a sphere of
 * radius 1.3 at softness 12, rejecting at 1.35 takes a floor point from 0.1667
 * to 1.0000.
 *
 * And every ray, hard or soft, counts as blocked once the field falls below
 * `epsilon`, so the shadow a march casts is the caster inflated by that much
 * while this test is exact geometry. A hard shadow needs no penumbra margin
 * but still needs this one -- the same trap as the camera marcher's `bounds`,
 * where a unit sphere bounded at exactly 1 lost four rim cells to rays passing
 * 1.000454 from the centre.
 *
 * Deliberately generous on the softness term: against the smallest radius that
 * reproduces the unbounded answer exactly, it over-estimates by 1.1x to 1.6x
 * across softness 4 to 48, for both a direction and a lamp. Over-estimating
 * costs a few rays that would have been skipped; under-estimating puts a hard
 * edge where a soft one belongs.
 */
function rejectionRadius(casterRadius: number, softness: number, reach: number, epsilon: number): number {
  return casterRadius + epsilon + (softness > 0 ? reach / softness : 0)
}

export interface PointShadowOptions extends Omit<ShadowOptions, 'light' | 'maxDistance'> {
  /**
   * How far short of the light to stop.
   *
   * The ray ends *at* the lamp, so anything past it is not between the surface
   * and the light and must not darken it. Stopping a whisker early also keeps
   * geometry the lamp is sitting on from shadowing everything it lights.
   */
  endBias?: number
}

/**
 * Occlusion for a light at a position rather than a direction.
 *
 * Two things change against `shadowFrom`. The ray's direction is recomputed
 * for every surface point, because a lamp is somewhere rather than somewhere
 * *over there*. And its length is bounded by the distance to the lamp: an
 * occluder beyond the light is behind it, not in front of it, and a march that
 * keeps going puts shadows under things the light never reaches past.
 */
export function shadowFromPoint(field: Sdf, position: Vec3, options: PointShadowOptions = {}): Occlusion {
  const softness = options.softness ?? 0
  const maxSteps = options.maxSteps ?? 48
  const epsilon = options.epsilon ?? 1e-3
  const bias = options.bias ?? 0.02
  const endBias = options.endBias ?? 0.02

  const casterRadius = options.casterRadius

  return (x, y, z) => {
    let dx = position.x - x
    let dy = position.y - y
    let dz = position.z - z
    const distance = Math.hypot(dx, dy, dz)
    if (distance <= bias + endBias) return 1
    dx /= distance
    dy /= distance
    dz /= distance

    const limit = distance - endBias

    // A lamp ray is only as long as the lamp is far, so that distance is the
    // reach the margin is worked out from.
    if (casterRadius !== undefined) {
      const tca = -x * dx + -y * dy + -z * dz
      if (tca < 0) return 1
      const perp2 = x * x + y * y + z * z - tca * tca
      const wide = rejectionRadius(casterRadius, softness, limit, epsilon)
      if (perp2 > wide * wide) return 1
    }
    let t = bias
    let visibility = 1

    for (let step = 0; step < maxSteps && t < limit; step++) {
      const d = field(x + dx * t, y + dy * t, z + dz * t)
      if (d < epsilon) return 0
      if (softness > 0) visibility = Math.min(visibility, (softness * d) / t)
      t += d
    }

    return softness > 0 ? Math.min(1, Math.max(0, visibility)) : 1
  }
}

export function shadowFrom(field: Sdf, options: ShadowOptions = {}): Occlusion {
  const l = normalize(options.light ?? vec3(0.5, 0.8, 0.6))
  const softness = options.softness ?? 0
  const maxDistance = options.maxDistance ?? 20
  const maxSteps = options.maxSteps ?? 48
  const epsilon = options.epsilon ?? 1e-3
  const bias = options.bias ?? 0.02

  const casterRadius = options.casterRadius
  const wide = casterRadius === undefined ? 0 : rejectionRadius(casterRadius, softness, maxDistance, epsilon)

  return (x, y, z) => {
    if (casterRadius !== undefined) {
      // How far the ray passes from the casters, and whether it heads toward
      // them at all. Both are one dot product away; neither samples the field.
      const tca = -x * l.x + -y * l.y + -z * l.z
      if (tca < 0) return 1
      if (x * x + y * y + z * z - tca * tca > wide * wide) return 1
    }

    let t = bias
    let visibility = 1

    for (let step = 0; step < maxSteps && t < maxDistance; step++) {
      const d = field(x + l.x * t, y + l.y * t, z + l.z * t)
      if (d < epsilon) return 0

      // The closest the ray came to a surface, measured against how far it had
      // travelled to get there. A ray that grazes an edge from far away is
      // barely dimmed; one that grazes it from close up is nearly blocked.
      // That ratio is the penumbra, and the march was being walked anyway.
      if (softness > 0) visibility = Math.min(visibility, (softness * d) / t)

      t += d
    }

    return softness > 0 ? Math.min(1, Math.max(0, visibility)) : 1
  }
}
