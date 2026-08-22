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

## Not yet

- Graph op (three resident ports, note §3.3) and indexed `nnDistance`/`knn` (note §5 steps 2–3).
- Occupied-cell compaction for sparse lattices (needs `encodeCompact`, note §2.4).
