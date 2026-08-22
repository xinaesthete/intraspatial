# ADR-0022 — GPU uniform-grid spatial index

Status: **accepted, partial** (2026-08-22). Promoted from
[`gpu-spatial-index-and-scan.md`](../gpu-spatial-index-and-scan.md) §3 the day the first slice
landed; that note keeps the reasoning, this records what is built.
Implementation: [`src/gpu/spatial/gridIndex.ts`](../../src/gpu/spatial/gridIndex.ts) (build),
[`src/spatial/bucketGrid.ts`](../../src/spatial/bucketGrid.ts) (`latticeFor`, shared lattice + CPU golden),
[`src/gpu/scan/prefixSum.ts`](../../src/gpu/scan/prefixSum.ts) (`keyPrefix`), consumers
[`crossPcf.ts`](../../src/gpu/spatial/crossPcf.ts) and [`tcm.ts`](../../src/gpu/spatial/tcm.ts),
`pnpm bench:grid-index`.

## Decision

The 2D `BucketGrid` offset list (`start[M+1]` / `items[n]`, cell = query radius, 3×3 stencil)
is built **on the device, in the caller's command buffer**: histogram (atomics) → `encodeScan`
→ atomic scatter, one submit. The host never sees `start`/`items` unless it asks
(`buildGridIndexGpu`, the test and drop-in path).

| | |
| --- | --- |
| Lattice | `latticeFor(xs, ys, cell, bounds?)` in `bucketGrid.ts` — the CPU build's own arithmetic, so both builds agree cell for cell; the kernel repeats its floor-and-clamp with a division, not a reciprocal |
| API | `encodeGridIndex(ctx, points, n, lattice, enc, { stride, keyPrefix, maxWorkgroupsPerDim })` → `{ start, items, M, n, lattice }` resident; `buildGridIndexGpu(points, { cell, bounds, stride })` → `BucketGrid` |
| Pools | grow-only, never destroyed, namespaced by `keyPrefix`; `encodeScan` gained the same so two builds in one submit (`computeTcmGpu`) do not alias |
| Contract | order within a cell unspecified; golden compares per-cell **sets** plus the 3D note's invariants |
| Limits | `checkBindingSize` on `M+1` and `n` (loud, not silent); 2-D dispatch fold |

Measured (`bench:grid-index`, ~8 points/cell): 1M points 0.9 ms, 10M points 5.5 ms on-device
(host counting sort: 20 ms / 202 ms); readback adds 5–30 ms, which is the case for staying resident.

## Query side (2026-08-22)

[`gridIndexQuery.ts`](../../src/gpu/spatial/gridIndexQuery.ts): TGSL `cellCoord(p, lat) -> vec2i`
(clamped) and `cellRange(c, lat, &start) -> vec2u` (`[lo, hi)` into `items`, `(0, 0)`
off-lattice), usable from `"use gpu"` kernels (`d.ref(layout.$.start)`) and as WGSL-template
externals. `nearestNeighborDistancesGpu`, `kthNeighborDistanceGpu`, `knnGpu` (2-D only) take an
optional `cell`: the index is built via `encodeGridIndex` in the query's own command buffer and
the kernel walks the 3×3 stencil. Contract (`IndexedQueryOptions`): equal to brute force for every
point whose true k-th neighbour is within `cell`; beyond that a neighbour can be missed, so indexed
≥ brute, never <; fewer than k candidates in the stencil → `+Infinity` (and index `0xFFFFFFFF`,
`KNN_NO_NEIGHBOUR`, in `knnGpu`), padded at the tail, rows still ascending. Brute force stays the
default and the golden (`gridIndexQuery.gpu.test.ts`). TypeGPU 0.11 note: the runtime-sized array
schema for pointer params must be `d.arrayOf(d.u32, 0)` — the count-less form is a comptime
placeholder that neither indexes nor resolves.

## Not yet

- `dims: 3` — kernel takes `stride: 3` (indexes xy of xyz), no third lattice axis until
  `buildBucketGrid` has a `zs` golden; the query helpers are `vec2`-shaped for the same reason.
- Graph op (three resident ports, note §3.3) and a resident `separate` / `spring` force op
  (note §5 step 3); CkNN and fuzzy adjacency still call `kthNeighborDistanceGpu` brute force.
- Occupied-cell compaction for sparse lattices (needs `encodeCompact`, note §2.4).
