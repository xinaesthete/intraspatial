// Device-side query helpers for the uniform-grid index (`gridIndex.ts`), the pair
// `docs/gpu-spatial-index-and-scan.md` §3.4 describes. `cellCoord` floors a position into the
// lattice (clamped, so an on-edge query still has a home cell); `cellRange` turns a cell
// coordinate into a `[lo, hi)` slice of `items`, or `(0, 0)` when the coordinate lies outside
// the lattice, so a 3×3 stencil at a border cell simply visits fewer cells.
//
// TGSL (`"use gpu"`) `tgpu.fn`s, not a WGSL template: no atomics, no workgroup memory
// (ADR-0003). TGSL has no closures, so the stencil loop is the consumer's — the six lines in
// `nnDistance.ts`'s indexed kernel are the 2D shape; WGSL-template kernels
// (`kthNeighborDistance.ts`, `knn.ts`) call the same functions as resolver externals.
//
// Lattice arithmetic is `latticeFor`'s (`src/spatial/bucketGrid.ts`) and the build kernel's:
// a division by `cell` (not a multiply by its reciprocal) and a clamp, so a query and the
// build agree on which cell a boundary point belongs to. Integer maths is spelled out — in
// TGSL `/` on u32 is float division.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { type GridLattice, latticeFor } from "../../spatial/bucketGrid";
import { ensureBuf } from "../scan/prefixSum";
import { encodeGridIndex, type GridIndexCtx, type GridIndexResident } from "./gridIndex";

/** The 2D lattice as a uniform: `gridIndex.ts`'s `Uni` minus the build-only fields. */
export const Lattice = d.struct({
  cols: d.u32,
  rows: d.u32,
  minX: d.f32,
  minY: d.f32,
  cell: d.f32,
});
export type LatticeStruct = typeof Lattice;

/** Bytes of a `Lattice` uniform (padded to the 16-byte uniform alignment). */
export const LATTICE_BYTES = 32;

/** Serialise a host `GridLattice` for a `Lattice` uniform; the layout is the struct's. */
export function latticeBytes(lat: GridLattice): ArrayBuffer {
  const u = new ArrayBuffer(LATTICE_BYTES);
  const dv = new DataView(u);
  dv.setUint32(0, lat.cols, true);
  dv.setUint32(4, lat.rows, true);
  dv.setFloat32(8, lat.minX, true);
  dv.setFloat32(12, lat.minY, true);
  dv.setFloat32(16, lat.cell, true);
  return u;
}

/** Cell coordinate of a world position, clamped to the lattice — the build's `cellOfPoint`. */
export const cellCoord = tgpu.fn(
  [d.vec2f, Lattice],
  d.vec2i,
)((p, lat) => {
  "use gpu";
  const cx = d.i32(std.floor((p.x - lat.minX) / lat.cell));
  const cy = d.i32(std.floor((p.y - lat.minY) / lat.cell));
  return d.vec2i(std.clamp(cx, d.i32(0), d.i32(lat.cols) - 1), std.clamp(cy, d.i32(0), d.i32(lat.rows) - 1));
});

/** Offset list type the index exposes (`start[M + 1]`). */
export const StartArray = d.arrayOf(d.u32, 0);

/** `[lo, hi)` into `items` for cell coordinate `c`, or `(0, 0)` off-lattice. Consumers loop
 *  `dy`, `dx` over `[-1, 1]` around `cellCoord(p)` and call this per cell; border cells get
 *  empty ranges for their missing neighbours. */
export const cellRange = tgpu.fn(
  [d.vec2i, Lattice, d.ptrStorage(StartArray, "read")],
  d.vec2u,
)((c, lat, start) => {
  "use gpu";
  let lo = d.u32(0);
  let hi = d.u32(0);
  if (c.x >= 0 && c.y >= 0 && c.x < d.i32(lat.cols) && c.y < d.i32(lat.rows)) {
    const b = d.u32(c.y) * lat.cols + d.u32(c.x);
    lo = start[b]!;
    hi = start[b + 1]!;
  }
  return d.vec2u(lo, hi);
});

/** Options the indexed neighbour queries share. Without `cell` they are brute force. */
export interface IndexedQueryOptions {
  /**
   * Uniform-grid cell side in the points' units. When given, the index is built on the device in
   * the query's own command buffer and the query walks the 3×3 stencil around each point instead
   * of every other point.
   *
   * Contract: the answer equals brute force for every point whose true k-th neighbour (nearest,
   * for k = 1) lies within `cell`. A neighbour further away than `cell` can fall outside the
   * stencil and be missed, so an indexed distance is only ever ≥ brute force, never <. A point
   * with fewer than k candidates in its stencil gets `+Infinity` for the missing distances (and
   * index `0xFFFFFFFF` in `knnGpu`).
   */
  readonly cell?: number;
  /** `[minX, minY, maxX, maxY]` for the lattice; default is the points' own bounds. */
  readonly bounds?: readonly [number, number, number, number];
}

/** What a query kernel binds after `encodeQueryIndex`: the lattice uniform and the index. */
export interface QueryIndex {
  readonly lat: GPUBuffer;
  readonly index: GridIndexResident;
}

/**
 * Record the index build for `n` points already resident in `points` (`[x, y]` pairs, `STORAGE`
 * usage) into `enc`, and write the matching `Lattice` uniform. The caller binds `lat`,
 * `index.start`, `index.items` with `sized()` and submits; the build and the query share one
 * command buffer. Pools are namespaced by `keyPrefix` (grow-only, never destroyed).
 */
export function encodeQueryIndex(
  ctx: GridIndexCtx,
  points: GPUBuffer,
  n: number,
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  cell: number,
  bounds: IndexedQueryOptions["bounds"],
  enc: GPUCommandEncoder,
  keyPrefix: string,
): QueryIndex {
  if (!(cell > 0)) throw new Error(`${keyPrefix}: cell must be > 0 (got ${cell})`);
  const { device } = ctx.scan;
  const lattice = latticeFor(xs, ys, cell, bounds);
  const lat = ensureBuf(device, `${keyPrefix}:lat`, LATTICE_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(lat, 0, latticeBytes(lattice));
  const index = encodeGridIndex(ctx, points, n, lattice, enc, { keyPrefix });
  return { lat, index };
}
