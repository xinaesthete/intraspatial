// A 3D density field from points, left on the GPU: each voxel gathers the kernel weight of every
// point within reach of its centre. Metaballs are this field with a surface drawn where it crosses
// a level; a 3D density layer is the same thing over measured points.
//
// Two forms of the same sum, with the same output (`docs/gpu-spatial-index-3d.md`, "splatVolume"):
//   - direct: a voxel tests every point. Work is voxels × points; right for a few thousand points.
//   - indexed: the points are first bucketed into a 3D grid index with cells one radius wide
//     (`gridIndex.ts`, built on the device in the same submit), and a voxel tests only the points in
//     its cell and the 26 around it. Because the kernel is exactly 0 beyond the radius, those 27
//     cells hold every point that can contribute: the result is the same sum, not an approximation.
//     Work is voxels × points nearby, which is what a measured cloud of millions needs.
// `index: "auto"` (the default) picks by point count.
//
// It gathers, so no running total is ever stored in the field (see `docs/field-precision.md`).
//
// Kernel: `(1 - d²/R²)²` inside the radius R, 0 outside. It is 1 at a point, falls smoothly to 0
// with zero slope at R, and has no tail to cut off, so the field is exact, not truncated.
//
// Layout: one f32 per voxel, x fastest: `field[x + nx * (y + ny * z)]`. Voxel (x, y, z) is centred
// at `origin + (x + 0.5, y + 0.5, z + 0.5) * voxel`.
import type { GridLattice } from "#src/spatial/bucketGrid";
import { checkBindingSize, compileShader, dispatchGrid, getDevice, sized, submitDispatch } from "../device";
import { encodeGridIndex, type GridIndexCtx, getGridIndexCtx } from "./gridIndex";

const WG = 64;

const SHADER = /* wgsl */ `
struct Uni {
  dims: vec4u,     // nx, ny, nz, number of points
  origin: vec4f,   // xyz of the field's low corner, w the voxel size
  reach: vec4f,    // x the kernel radius, y a weight on every point
  packing: vec4u,  // x f32 per point, y the workgroup grid's width, z 1 when each point has its own weight
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> points: array<f32>;
@group(0) @binding(2) var<storage, read_write> field: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;

@compute @workgroup_size(${WG})
fn splatVolume(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  let v = (wid.x + wid.y * U.packing.y) * ${WG}u + li;
  let voxels = U.dims.x * U.dims.y * U.dims.z;
  if (v >= voxels) {
    return;
  }
  let cell = vec3f(f32(v % U.dims.x), f32((v / U.dims.x) % U.dims.y), f32(v / (U.dims.x * U.dims.y)));
  let centre = U.origin.xyz + (cell + vec3f(0.5)) * U.origin.w;
  let invR2 = 1.0 / (U.reach.x * U.reach.x);
  var sum = 0.0;
  for (var i = 0u; i < U.dims.w; i = i + 1u) {
    let p = i * U.packing.x;
    let d = vec3f(points[p], points[p + 1u], points[p + 2u]) - centre;
    let q = max(1.0 - dot(d, d) * invR2, 0.0);
    sum = sum + q * q * select(1.0, weights[i * U.packing.z], U.packing.z == 1u);
  }
  field[v] = sum * U.reach.y;
}
`;

const INDEXED_SHADER = /* wgsl */ `
struct Uni {
  dims: vec4u,     // nx, ny, nz, number of points
  origin: vec4f,   // xyz of the field's low corner, w the voxel size
  reach: vec4f,    // x the kernel radius, y a weight on every point
  packing: vec4u,  // x f32 per point, y the workgroup grid's width, z 1 when each point has its own weight
  cells: vec4u,    // the index's cells along x, y and z
  low: vec4f,      // xyz of the index's low corner, w its cell size
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> points: array<f32>;
@group(0) @binding(2) var<storage, read_write> field: array<f32>;
// Cell b owns ids[starts[b] .. starts[b + 1]).
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read> starts: array<u32>;
@group(0) @binding(5) var<storage, read> ids: array<u32>;

@compute @workgroup_size(${WG})
fn splatVolumeIndexed(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  let v = (wid.x + wid.y * U.packing.y) * ${WG}u + li;
  let voxels = U.dims.x * U.dims.y * U.dims.z;
  if (v >= voxels) {
    return;
  }
  let cell = vec3f(f32(v % U.dims.x), f32((v / U.dims.x) % U.dims.y), f32(v / (U.dims.x * U.dims.y)));
  let centre = U.origin.xyz + (cell + vec3f(0.5)) * U.origin.w;
  let invR2 = 1.0 / (U.reach.x * U.reach.x);
  // The division and the clamp are gridIndex.ts's, so a voxel centre lands in the cell a point there would.
  let n = vec3i(U.cells.xyz);
  let home = clamp(vec3i(floor((centre - U.low.xyz) / U.low.w)), vec3i(0), n - vec3i(1));
  var sum = 0.0;
  for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
      for (var dx = -1; dx <= 1; dx = dx + 1) {
        let c = home + vec3i(dx, dy, dz);
        if (all(c >= vec3i(0)) && all(c < n)) {
          let b = u32(c.x) + U.cells.x * (u32(c.y) + U.cells.y * u32(c.z));
          let last = starts[b + 1u];
          for (var j = starts[b]; j < last; j = j + 1u) {
            let id = ids[j];
            let p = id * U.packing.x;
            let d = vec3f(points[p], points[p + 1u], points[p + 2u]) - centre;
            let q = max(1.0 - dot(d, d) * invR2, 0.0);
            sum = sum + q * q * select(1.0, weights[id * U.packing.z], U.packing.z == 1u);
          }
        }
      }
    }
  }
  field[v] = sum * U.reach.y;
}
`;

/** Above this many points `index: "auto"` builds the index. Below it the build costs more than it saves. */
export const SPLAT_VOLUME_INDEX_FROM = 2048;

export interface SplatVolumeOptions {
  /** Voxels along x, y and z. */
  readonly dims: readonly [number, number, number];
  /** The field's low corner, in the points' units. */
  readonly origin: readonly [number, number, number];
  /** Edge of a voxel, in the points' units. */
  readonly voxel: number;
  /** How far a point reaches. */
  readonly radius: number;
  /** f32 per point in `points`, x y z first. */
  readonly stride?: number;
  /** Multiplies every point's contribution. */
  readonly weight?: number;
  /** One f32 per point: that point's own weight (a cell's expression of a gene, or 0 to leave a
   *  point out). Absent: every point counts 1. Multiplied by `weight`. */
  readonly weights?: GPUBuffer;
  /** Bucket the points first (see the top of this file). Default "auto": by point count. */
  readonly index?: boolean | "auto";
  /** The device the buffers live on; default the library's own. */
  readonly device?: GPUDevice;
}

const UNI_BYTES = 96;

interface Ctx {
  device: GPUDevice;
  pipeline: GPUComputePipeline;
  indexed: GPUComputePipeline;
  grid: GridIndexCtx;
  uni: GPUBuffer;
  /** Bound in place of `weights` when the caller gives none; never read. */
  noWeights: GPUBuffer;
}

/** Per device, held weakly — see `splatDensity.ts`'s `Gpu` for why. */
const perDevice = new WeakMap<GPUDevice, Promise<Ctx>>();

async function getCtx(device?: GPUDevice): Promise<Ctx> {
  const dev = device ?? (await getDevice());
  let ctx = perDevice.get(dev);
  if (!ctx) {
    ctx = (async () => {
      const module = await compileShader(dev, SHADER, "splatVolume");
      const pipeline = dev.createComputePipeline({ label: "splatVolume", layout: "auto", compute: { module, entryPoint: "splatVolume" } });
      const indexedModule = await compileShader(dev, INDEXED_SHADER, "splatVolume:indexed");
      const indexed = dev.createComputePipeline({
        label: "splatVolume:indexed",
        layout: "auto",
        compute: { module: indexedModule, entryPoint: "splatVolumeIndexed" },
      });
      const grid = await getGridIndexCtx(dev);
      const uni = dev.createBuffer({
        label: "splatVolume:uniforms",
        size: UNI_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const noWeights = dev.createBuffer({ label: "splatVolume:no-weights", size: 4, usage: GPUBufferUsage.STORAGE });
      return { device: dev, pipeline, indexed, grid, uni, noWeights };
    })();
    perDevice.set(dev, ctx);
  }
  return ctx;
}

/** Bytes a field of these dimensions needs. */
export function volumeBytes(dims: readonly [number, number, number]): number {
  return dims[0] * dims[1] * dims[2] * 4;
}

/**
 * Fill `dst` with the density of the first `n` points of `points`. Both stay on the device; `dst`
 * needs STORAGE usage and `volumeBytes(dims)` bytes, and is overwritten whole.
 *
 * Returns a function that does the work when called, with nothing to await: a simulation calls it
 * once per frame between its own steps. Building the pipeline is the only asynchronous part.
 */
export async function createSplatVolume(
  device?: GPUDevice,
): Promise<(points: GPUBuffer, n: number, dst: GPUBuffer, o: SplatVolumeOptions) => void> {
  const { device: dev, pipeline, indexed, grid: gridCtx, uni, noWeights } = await getCtx(device);
  return (points, n, dst, o) => {
    const [nx, ny, nz] = o.dims;
    const voxels = nx * ny * nz;
    const stride = o.stride ?? 4;
    checkBindingSize(dev, "splatVolume", voxels * 4);
    const grid = dispatchGrid(Math.ceil(voxels / WG));
    const useIndex = o.index === true || (o.index !== false && n >= SPLAT_VOLUME_INDEX_FROM);
    // Cells one radius wide over the field's box. A point outside the box is clamped into an edge
    // cell by the index, which keeps it within one cell of every voxel it can reach.
    const cells = (voxelCount: number) => Math.max(1, Math.ceil((voxelCount * o.voxel) / o.radius) + 1);
    const lattice: GridLattice = {
      cols: cells(nx),
      rows: cells(ny),
      depth: cells(nz),
      cell: o.radius,
      minX: o.origin[0],
      minY: o.origin[1],
      minZ: o.origin[2],
    };
    const data = new ArrayBuffer(UNI_BYTES);
    new Uint32Array(data, 0, 4).set([nx, ny, nz, n]);
    new Float32Array(data, 16, 8).set([o.origin[0], o.origin[1], o.origin[2], o.voxel, o.radius, o.weight ?? 1, 0, 0]);
    new Uint32Array(data, 48, 8).set([stride, grid.x, o.weights ? 1 : 0, 0, lattice.cols, lattice.rows, lattice.depth ?? 1, 0]);
    new Float32Array(data, 80, 4).set([o.origin[0], o.origin[1], o.origin[2], o.radius]);
    // Written and consumed in one submit, so two callers cannot mix their uniforms.
    dev.queue.writeBuffer(uni, 0, data);
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: uni } },
      { binding: 1, resource: { buffer: points, size: Math.max(4, n * stride * 4) } },
      { binding: 2, resource: { buffer: dst, size: voxels * 4 } },
      { binding: 3, resource: o.weights ? { buffer: o.weights, size: Math.max(4, n * 4) } : { buffer: noWeights } },
    ];
    if (!useIndex || n === 0) {
      const bindGroup = dev.createBindGroup({ label: "splatVolume", layout: pipeline.getBindGroupLayout(0), entries });
      submitDispatch(dev, "splatVolume", pipeline, bindGroup, grid.x, grid.y);
      return;
    }
    // The index build and the gather that reads it, in one command buffer.
    const enc = dev.createCommandEncoder({ label: "splatVolume:indexed" });
    const index = encodeGridIndex(gridCtx, points, n, lattice, enc, { stride, keyPrefix: "splatVolume:index" });
    const bindGroup = dev.createBindGroup({
      label: "splatVolume:indexed",
      layout: indexed.getBindGroupLayout(0),
      entries: [
        ...entries,
        { binding: 4, resource: sized(index.cellOffsets, (index.M + 1) * 4) },
        { binding: 5, resource: sized(index.pointIds, Math.max(n, 1) * 4) },
      ],
    });
    const pass = enc.beginComputePass({ label: "splatVolume:indexed" });
    pass.setPipeline(indexed);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(grid.x, grid.y);
    pass.end();
    dev.queue.submit([enc.finish()]);
  };
}

/** One call, for a caller that is not in a frame loop: see `createSplatVolume`. */
export async function splatVolumeResident(points: GPUBuffer, n: number, dst: GPUBuffer, o: SplatVolumeOptions): Promise<void> {
  (await createSplatVolume(o.device))(points, n, dst, o);
}
