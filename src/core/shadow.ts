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
}

export function shadowFrom(field: Sdf, options: ShadowOptions = {}): Occlusion {
  const l = normalize(options.light ?? vec3(0.5, 0.8, 0.6))
  const softness = options.softness ?? 0
  const maxDistance = options.maxDistance ?? 20
  const maxSteps = options.maxSteps ?? 48
  const epsilon = options.epsilon ?? 1e-3
  const bias = options.bias ?? 0.02

  return (x, y, z) => {
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
