import type { Mat4 } from './mat4.ts'
import { lookAt, multiply, perspective } from './mat4.ts'
import type { Vec3 } from './vec3.ts'
import { vec3 } from './vec3.ts'

/**
 * Terminal cells are taller than they are wide. Roughly 0.5 (a cell twice as
 * tall as it is wide) matches most terminal fonts; measure the real ratio in
 * the browser instead of guessing.
 */
export const DEFAULT_CELL_ASPECT = 0.5

/**
 * The aspect ratio to hand `Camera.projection`. Counting cells alone would
 * squash the picture, because a cell is not square — the physical width of
 * the grid is `width * cellAspect` cell-heights.
 */
export function aspectFor(width: number, height: number, cellAspect = DEFAULT_CELL_ASPECT): number {
  return (width * cellAspect) / height
}

export class Camera {
  position: Vec3
  target: Vec3
  up: Vec3
  /** Vertical field of view, in radians. */
  fovY: number
  near: number
  far: number

  constructor(options: Partial<Pick<Camera, 'position' | 'target' | 'up' | 'fovY' | 'near' | 'far'>> = {}) {
    this.position = options.position ?? vec3(0, 0, 5)
    this.target = options.target ?? vec3(0, 0, 0)
    this.up = options.up ?? vec3(0, 1, 0)
    this.fovY = options.fovY ?? Math.PI / 4
    this.near = options.near ?? 0.1
    this.far = options.far ?? 100
  }

  view(): Mat4 {
    return lookAt(this.position, this.target, this.up)
  }

  projection(aspect: number): Mat4 {
    return perspective(this.fovY, aspect, this.near, this.far)
  }

  viewProjection(aspect: number): Mat4 {
    return multiply(this.projection(aspect), this.view())
  }

  /** Places the camera on a sphere around its target. */
  orbit(yaw: number, pitch: number, distance: number): void {
    const cp = Math.cos(pitch)
    this.position = {
      x: this.target.x + distance * cp * Math.sin(yaw),
      y: this.target.y + distance * Math.sin(pitch),
      z: this.target.z + distance * cp * Math.cos(yaw),
    }
  }
}
