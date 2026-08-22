// GPU uniform-grid spatial index — `src/spatial/bucketGrid.ts` built on the device.
//
// Same structure, same contract: an offset list (`cellOffsets[M+1]` / `pointIds[n]`) over a lattice of
// square cells whose side is the query radius, so a 3×3 stencil around a query's cell holds
// every point within that radius. `crossPcf.ts` walks it from WGSL exactly as it walked the host
// build; this file only moves the build so the points never round-trip through the host.
// Design: `docs/gpu-spatial-index-and-scan.md` §3.
//
// Three passes in ONE submit — a counting sort with the prefix sum done by `encodeScan`:
//   histogram  cellOf[i] = cell(p_i); atomicAdd(counts[cellOf[i]], 1)
//   scan       start = exclusive scan of counts (M+1 entries: the trailing zero makes start[M] == n)
//   scatter    items[atomicAdd(cursor[cellOf[i]], 1)] = i, with cursor a copy of start
// Order within a cell is whatever the atomics produced and is NOT part of the contract; the
// golden compares per-cell SETS (`gridIndex.gpu.test.ts`). A stable variant can replace the
// scatter later without changing the layout.
//
// WGSL template, not `"use gpu"` TGSL: both data passes are built on `atomicAdd`, which
// ADR-0003 routes to templates. The host-visible lattice arithmetic (`latticeFor`) is shared
// with the CPU build by construction — `latticeFor` IS the CPU build's, and the kernel repeats its
// floor-and-clamp — so the two builds agree on which cell a boundary point lands in.
//
// Third axis: a lattice with `depth`/`minZ` (from `latticeFor(…, zs)`) makes the kernel take z as
// well, which needs `stride: 3`. `stride: 3` on a 2D lattice still indexes xy only.
import { type Bounds2, type Bounds3, type BucketGrid, type GridLattice, latticeFor, numCells } from "../../spatial/bucketGrid";
import { checkBindingSize, compileShader, dispatchGrid, getDevice, MAX_WORKGROUPS_PER_DIM, sized } from "../device";
import { encodeScan, ensureBuf, getScanCtx, readBack, type ScanCtx } from "../scan/prefixSum";

/** One thread per point in the histogram and scatter passes. */
export const GRID_INDEX_WG = 256;
const WG = GRID_INDEX_WG;
/** `Uni` below: 10 words, padded to 16-byte alignment. */
const UNI_BYTES = 48;

export interface GridIndexOptions {
  /** Cell side in world units — the query radius the index is built for. */
  readonly cell: number;
  /** Floats per point in the buffer (`[x, y]` or `[x, y, z]`). With `dims: 2` the first two
   *  are indexed; `dims: 3` needs `stride: 3`. */
  readonly stride?: 2 | 3;
  /** Lattice axes (default 2). `3` indexes z too and takes 6-number bounds. */
  readonly dims?: 2 | 3;
  /** `[minX, minY, maxX, maxY]` (`dims: 2`) or `[minX, minY, minZ, maxX, maxY, maxZ]`
   *  (`dims: 3`). Without it the points' own bounds are used (a host pass). */
  readonly bounds?: Bounds2 | Bounds3;
  /** Test seam, as in `ScanOptions`: exercise the 2-D dispatch fold at small n. */
  readonly maxWorkgroupsPerDim?: number;
  /** Pool namespace. Two indexes recorded into one command buffer need distinct prefixes or
   *  the second overwrites the first's buffers — see `encodeScan`. */
  readonly keyPrefix?: string;
}

/** The index as it lives on the device. The buffers are pooled (grow-only, never destroyed)
 *  under the build's `keyPrefix` and stay valid until the next build under that prefix. */
export interface GridIndexResident {
  /** `M + 1` u32 offsets, `cellOffsets[M] == n`. */
  readonly cellOffsets: GPUBuffer;
  /** `max(n, 1)` u32 point ids grouped by cell. */
  readonly pointIds: GPUBuffer;
  /** Cell count `cols * rows * (depth ?? 1)`. */
  readonly M: number;
  readonly n: number;
  readonly lattice: GridLattice;
}

const SHADER = /* wgsl */ `
struct Uni {
  n: u32,
  gridX: u32,
  cols: u32,
  rows: u32,
  minX: f32,
  minY: f32,
  cell: f32,
  stride: u32,
  depth: u32,   // 0 on a 2D lattice: z is not read
  minZ: f32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> pts: array<f32>;
@group(0) @binding(2) var<storage, read_write> cellOf: array<u32>;
// 'counts' during the histogram, 'cursor' (a copy of start) during the scatter.
@group(0) @binding(3) var<storage, read_write> heads: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> items: array<u32>;

// 2-D fold of the dispatch, as in prefixSum.ts: one thread per point crosses 65535
// workgroups at 16.8M points, past which a 1-D dispatch silently does nothing.
fn pointOf(lid: vec3u, wid: vec3u) -> u32 {
  return (wid.x + wid.y * U.gridX) * ${WG}u + lid.x;
}

// The division (not a multiply by 1/cell) and the clamp are bucketGrid.ts's, kept so a point
// on a cell boundary floors the same way on both sides.
fn cellOfPoint(i: u32) -> u32 {
  let x = pts[U.stride * i];
  let y = pts[U.stride * i + 1u];
  let cx = clamp(i32(floor((x - U.minX) / U.cell)), 0, i32(U.cols) - 1);
  let cy = clamp(i32(floor((y - U.minY) / U.cell)), 0, i32(U.rows) - 1);
  var cz = 0i;
  if (U.depth > 0u) {
    let z = pts[U.stride * i + 2u];
    cz = clamp(i32(floor((z - U.minZ) / U.cell)), 0, i32(U.depth) - 1);
  }
  return u32(cx) + U.cols * (u32(cy) + U.rows * u32(cz));
}

@compute @workgroup_size(${WG})
fn histogram(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let i = pointOf(lid, wid);
  if (i >= U.n) { return; }
  let b = cellOfPoint(i);
  cellOf[i] = b;
  atomicAdd(&heads[b], 1u);
}

@compute @workgroup_size(${WG})
fn scatter(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let i = pointOf(lid, wid);
  if (i >= U.n) { return; }
  let slot = atomicAdd(&heads[cellOf[i]], 1u);
  items[slot] = i;
}
`;

export interface GridIndexCtx {
  scan: ScanCtx;
  layout: GPUBindGroupLayout;
  histogram: GPUComputePipeline;
  scatter: GPUComputePipeline;
}

let ctxCache: Promise<GridIndexCtx> | undefined;

export function getGridIndexCtx(): Promise<GridIndexCtx> {
  ctxCache ??= (async () => {
    const scan = await getScanCtx();
    const { device } = scan;
    // One explicit layout for both entry points (each leaves one binding untouched, which
    // `layout: "auto"` would drop from that entry point's derived layout).
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const module = await compileShader(device, SHADER, "gridIndex");
    const mk = (entryPoint: string) => device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
    return { scan, layout, histogram: mk("histogram"), scatter: mk("scatter") };
  })();
  return ctxCache;
}

const storageRw = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

/**
 * Record an index build over `n` points already resident in `points` (`stride` floats each,
 * `STORAGE` usage) into `enc`. The caller owns submission, so a consumer fuses the build with
 * its own query pass — `crossPcf.ts` does.
 */
export function encodeGridIndex(
  ctx: GridIndexCtx,
  points: GPUBuffer,
  n: number,
  lattice: GridLattice,
  enc: GPUCommandEncoder,
  opts: Omit<GridIndexOptions, "cell" | "bounds"> = {},
): GridIndexResident {
  const { device } = ctx.scan;
  const stride = opts.stride ?? 2;
  const maxWg = opts.maxWorkgroupsPerDim ?? MAX_WORKGROUPS_PER_DIM;
  const key = opts.keyPrefix ?? "gridIndex";
  if (lattice.depth !== undefined && stride !== 3) throw new Error(`gridIndex: a 3D lattice needs stride 3, got ${stride}`);
  const M = numCells(lattice);
  // A sparse lattice (M ≫ n) is bounded by the scan over M + 1 cells, not by n. Occupied-cell
  // compaction is the real fix for that case; until then this is a loud failure, not a silent one.
  checkBindingSize(device, `gridIndex: ${M} cells`, (M + 1) * 4);
  checkBindingSize(device, `gridIndex: ${n} points`, Math.max(n, 1) * stride * 4);

  const uni = ensureBuf(device, `${key}:uni`, UNI_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const cellOf = ensureBuf(device, `${key}:cellOf`, Math.max(n, 1) * 4, storageRw());
  const counts = ensureBuf(device, `${key}:counts`, (M + 1) * 4, storageRw());
  const cursor = ensureBuf(device, `${key}:cursor`, (M + 1) * 4, storageRw());
  const pointIds = ensureBuf(device, `${key}:pointIds`, Math.max(n, 1) * 4, storageRw());

  const { x: gridX, y: gridY } = dispatchGrid(Math.ceil(n / WG), maxWg);
  const u = new ArrayBuffer(UNI_BYTES);
  const dv = new DataView(u);
  dv.setUint32(0, n, true);
  dv.setUint32(4, gridX, true);
  dv.setUint32(8, lattice.cols, true);
  dv.setUint32(12, lattice.rows, true);
  dv.setFloat32(16, lattice.minX, true);
  dv.setFloat32(20, lattice.minY, true);
  dv.setFloat32(24, lattice.cell, true);
  dv.setUint32(28, stride, true);
  dv.setUint32(32, lattice.depth ?? 0, true);
  dv.setFloat32(36, lattice.minZ ?? 0, true);
  device.queue.writeBuffer(uni, 0, u);

  const group = (heads: GPUBuffer) =>
    device.createBindGroup({
      layout: ctx.layout,
      entries: [
        { binding: 0, resource: sized(uni, UNI_BYTES) },
        { binding: 1, resource: sized(points, Math.max(n, 1) * stride * 4) },
        { binding: 2, resource: sized(cellOf, Math.max(n, 1) * 4) },
        { binding: 3, resource: sized(heads, (M + 1) * 4) },
        { binding: 4, resource: sized(pointIds, Math.max(n, 1) * 4) },
      ],
    });

  // Histogram. The trailing count is cleared and never incremented, which is what makes the
  // exclusive scan's last entry equal n.
  enc.clearBuffer(counts, 0, (M + 1) * 4);
  if (n > 0) {
    const pass = enc.beginComputePass({ label: "gridIndex:histogram" });
    pass.setPipeline(ctx.histogram);
    pass.setBindGroup(0, group(counts));
    pass.dispatchWorkgroups(gridX, gridY);
    pass.end();
  }

  // Scan (its own pass inside), then seed the write heads from it.
  const { dst: start } = encodeScan(ctx.scan, "u32", counts, M + 1, enc, maxWg, `${key}:scan`);
  enc.copyBufferToBuffer(start, 0, cursor, 0, (M + 1) * 4);

  // Scatter.
  if (n > 0) {
    const pass = enc.beginComputePass({ label: "gridIndex:scatter" });
    pass.setPipeline(ctx.scatter);
    pass.setBindGroup(0, group(cursor));
    pass.dispatchWorkgroups(gridX, gridY);
    pass.end();
  }

  return { cellOffsets: start, pointIds, M, n, lattice };
}

/**
 * Build the index on the GPU and read it back as the host `BucketGrid`, so anything written
 * against `buildBucketGrid` can take this instead. `points` is `[x0, y0, x1, y1, …]` (or xyz
 * with `stride: 3`; `dims: 3` indexes the z as well).
 *
 * This is the test and drop-in path. A kernel that walks the index on the device should use
 * `encodeGridIndex` and never bring `cellOffsets`/`pointIds` to the host.
 */
export async function buildGridIndexGpu(points: Float32Array, opts: GridIndexOptions): Promise<BucketGrid> {
  const ctx = await getGridIndexCtx();
  const { device } = ctx.scan;
  const stride = opts.stride ?? 2;
  const dims = opts.dims ?? 2;
  if (dims === 3 && stride !== 3) throw new Error(`gridIndex: dims 3 needs stride 3, got ${stride}`);
  const n = Math.floor(points.length / stride);
  const key = opts.keyPrefix ?? "gridIndex";

  const xs = new Float32Array(n);
  const ys = new Float32Array(n);
  const zs = dims === 3 ? new Float32Array(n) : undefined;
  for (let i = 0; i < n; i++) {
    xs[i] = points[stride * i]!;
    ys[i] = points[stride * i + 1]!;
    if (zs) zs[i] = points[stride * i + 2]!;
  }
  const lattice = latticeFor(xs, ys, opts.cell, opts.bounds, zs);

  const ptsBuf = ensureBuf(device, `${key}:pts`, Math.max(n, 1) * stride * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  // The view's own buffer, not the view: under TS 5.9's lib a `Float32Array` may sit on a
  // SharedArrayBuffer, which `BufferSource` excludes but `writeBuffer` itself allows.
  if (n > 0) device.queue.writeBuffer(ptsBuf, 0, points.buffer, points.byteOffset, n * stride * 4);

  const enc = device.createCommandEncoder();
  const idx = encodeGridIndex(ctx, ptsBuf, n, lattice, enc, opts);
  device.queue.submit([enc.finish()]);

  const M = idx.M;
  const cellOffsets = new Int32Array(await readBack(device, `${key}:stagingOffsets`, idx.cellOffsets, 0, (M + 1) * 4));
  const pointIds = n > 0 ? new Int32Array(await readBack(device, `${key}:stagingIds`, idx.pointIds, 0, n * 4)) : new Int32Array(0);
  return { ...lattice, cellOffsets, pointIds };
}
