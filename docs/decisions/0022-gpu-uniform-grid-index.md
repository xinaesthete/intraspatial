# ADR-0022 — GPU uniform-grid spatial index

Status: **accepted, partial** (2026-08-22). Promoted from
[`gpu-spatial-index-and-scan.md`](../gpu-spatial-index-and-scan.md) §3 the day the first slice
landed; that note keeps the reasoning, this records what is built.
Implementation: [`src/gpu/spatial/gridIndex.ts`](../../src/gpu/spatial/gridIndex.ts) (build),
[`src/spatial/bucketGrid.ts`](../../src/spatial/bucketGrid.ts) (`latticeFor`, shared lattice + CPU golden),
[`src/gpu/scan/prefixSum.ts`](../../src/gpu/scan/prefixSum.ts) (`keyPrefix`), consumers
[`crossPcf.ts`](../../src/gpu/spatial/crossPcf.ts) and [`tcm.ts`](../../src/gpu/spatial/tcm.ts),
[`ops/gridIndex.ts`](../../src/gpu/graph/ops/gridIndex.ts) (graph op), `pnpm bench:grid-index`.

## Decision

The `BucketGrid` offset list (`start[M+1]` / `items[n]`, cell = query radius, 3×3[×3] stencil)
is built **on the device, in the caller's command buffer**: histogram (atomics) → `encodeScan`
→ atomic scatter, one submit. The host never sees `start`/`items` unless it asks
(`buildGridIndexGpu`, the test and drop-in path).

| | |
| --- | --- |
| Lattice | `latticeFor(xs, ys, cell, bounds?, zs?)` in `bucketGrid.ts` — the CPU build's own arithmetic, so both builds agree cell for cell; the kernel repeats its floor-and-clamp with a division, not a reciprocal. `zs` adds `depth`/`minZ` (2026-08-22); `cellId = cx + cols·(cy + rows·cz)`, x-fastest |
| API | `encodeGridIndex(ctx, points, n, lattice, enc, { stride, keyPrefix, maxWorkgroupsPerDim })` → `{ start, items, M, n, lattice }` resident (a 3D lattice requires `stride: 3`); `buildGridIndexGpu(points, { cell, bounds, stride, dims })` → `BucketGrid` |
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

## Graph op (2026-08-22) — superseded by [ADR-0023](0023-composite-values-and-borrowed-leases.md)

The three-port shape below shipped for one afternoon and was replaced the same day: `start` and
`items` are both `points`/u32, so a graph could wire them from *different* indexes and get a
plausible wrong answer. The op now emits ONE `gridIndex` bundle on a port named `buckets`;
`gridIndex.start` and its siblings take a part out. In the composer the whole thing is called a
*bucket grid* — `index` reads as "an array index", which is what the `items` part actually holds. Everything below still describes what the op computes — only the port
shape changed.

## Original three-port shape

[`ops/gridIndex.ts`](../../src/gpu/graph/ops/gridIndex.ts), the note's §3.3 shape: **three output
ports**, `start` (`points{M+1}`, u32), `items` (`points{n}`, u32) and `lattice` (opaque
`gridLattice` payload — the `GridLattice` plus `cells`, `n` and the derived placement). Not one
opaque handle carrying buffers: the executor tracks resident ownership through `v.buffer` only, so
hidden leases would never be released.

Three consequences worth stating:

- **One lease per port, and the kernel's buffers are pooled**, so `execute` copies `start`/`items`
  into their own leases inside the same submit. That is what makes two `gridIndex` nodes in one
  tick safe while they share the global `encodeGridIndex` pool.
- **`pull()` on `start`/`items` throws** — the executor's host bridge is f32-only (`residentF32Count`,
  ADR-0017 stage 1), and reinterpreting u32 as f32 would return mangled numbers. `pullResident` is
  the supported path; `mode: "cpu"` runs `cpuGolden` and yields host `Uint32Array`s.
- **The extent is params, not a measurement.** `inferShapes` runs at graph-build time with no
  values, so `minX`/`minY`/`maxX`/`maxY` (+ `cell`) fix the lattice, exactly as `buildBucketGrid`'s
  optional `bounds` does. Out-of-bounds points clamp into the edge cells.

`inferPlacement` gives the lattice `worldFromArray · translate(origin) · scale(cell)` in the points'
system (corner-indexed, as `decimatedPlacement`); `start`/`items` are index lists and stay
array-space.

## Not yet

- A resident `separate` / `spring` force op reading the index (note §5 step 3); CkNN and fuzzy
  adjacency still call `kthNeighborDistanceGpu` brute force.
- `cellCounts` (bundle → per-cell counts) is the only graph-level consumer so far.
- The query helpers (`gridIndexQuery.ts`) are `vec2`-shaped: no `dz` loop yet for a 3D lattice.
- Occupied-cell compaction for sparse lattices (needs `encodeCompact`, note §2.4).
