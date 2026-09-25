import type { Mat4 } from './mat4.ts'
import { multiply, normalMatrix, transformDirection, transformPoint } from './mat4.ts'
import type { Mesh } from './mesh.ts'
import type { Cull, RenderTarget, Shader } from './raster.ts'
import { CLIP_STRIDE, submitTriangle } from './raster.ts'

export interface SceneObject {
  mesh: Mesh
  model: Mat4
  shader: Shader
  cull?: Cull
}

const tri = new Float32Array(3 * CLIP_STRIDE)
const world = new Float32Array(4)
let cache = new Float32Array(0)

/**
 * Transforms a mesh once, then feeds its triangles to the rasterizer.
 *
 * Vertices are transformed per *vertex*, not per triangle corner, so a shared
 * vertex in an indexed mesh costs one matrix multiply no matter how many
 * faces use it.
 */
export function drawMesh(
  target: RenderTarget,
  mesh: Mesh,
  model: Mat4,
  viewProjection: Mat4,
  shader: Shader,
  cull: Cull = 'back',
): void {
  const mvp = multiply(viewProjection, model)
  const nm = normalMatrix(model)
  const vertexCount = mesh.positions.length / 3
  const uvs = mesh.uvs

  if (cache.length < vertexCount * CLIP_STRIDE) cache = new Float32Array(vertexCount * CLIP_STRIDE)

  for (let i = 0; i < vertexCount; i++) {
    const p = i * 3
    const o = i * CLIP_STRIDE
    const x = mesh.positions[p]!
    const y = mesh.positions[p + 1]!
    const z = mesh.positions[p + 2]!
    transformPoint(mvp, x, y, z, cache, o)
    transformPoint(model, x, y, z, world, 0)
    cache[o + 4] = world[0]!
    cache[o + 5] = world[1]!
    cache[o + 6] = world[2]!
    transformDirection(nm, mesh.normals[p]!, mesh.normals[p + 1]!, mesh.normals[p + 2]!, cache, o + 7)
    // Texture coordinates ride along as two more interpolated attributes; a
    // mesh without them reads as (0, 0) everywhere, which is what an untextured
    // shader ignores anyway.
    cache[o + 10] = uvs ? uvs[i * 2]! : 0
    cache[o + 11] = uvs ? uvs[i * 2 + 1]! : 0
  }

  const indices = mesh.indices
  for (let i = 0; i < indices.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const src = indices[i + k]! * CLIP_STRIDE
      for (let j = 0; j < CLIP_STRIDE; j++) tri[k * CLIP_STRIDE + j] = cache[src + j]!
    }
    submitTriangle(target, tri, shader, cull)
  }
}

export function renderScene(target: RenderTarget, objects: readonly SceneObject[], viewProjection: Mat4): void {
  for (const o of objects) drawMesh(target, o.mesh, o.model, viewProjection, o.shader, o.cull ?? 'back')
}
