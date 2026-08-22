# GPU spatial index, scan/compaction, and the one support mask

Status: **design note** (2026-08-22) — promote to an ADR when implementation starts.

Three items in [`gap-analysis.md`](gap-analysis.md) share one substrate: the scan /
stream-compaction kernel (built, `src/gpu/scan/`), the uniform-grid spatial index (specced
for 3D in [`gpu-spatial-index-3d.md`](gpu-spatial-index-3d.md), built nowhere), and
ADR-0005's `support` facet, which terrain has rediscovered as "nodata/validity". This note
documents the first, designs the second on it for **2D and 3D in one op**, and pins the third
to one facet. It extends the 3D note with the 2D case, the scan, the ADR-0017 handle, the
query API, and the sequencing.

## 1. Context and consumers

**Decision: the GPU index produces exactly the layout `bucketGrid.ts` builds on the host,
and the first slice swaps that host build out from under `crossPcf.ts` and `tcm.ts`.**

| Consumer | Today | Needs |
| --- | --- | --- |
| [`nnDistance.ts`](../src/gpu/spatial/nnDistance.ts), [`emptySpace.ts`](../src/gpu/spatial/emptySpace.ts), [`cknn.ts`](../src/gpu/spatial/cknn.ts), [`fuzzyAdjacency.ts`](../src/gpu/spatial/fuzzyAdjacency.ts) | `"use gpu"` TGSL, all-N loop, O(N²) | radius-bounded neighbour query |
| [`knn.ts`](../src/gpu/spatial/knn.ts), [`kthNeighborDistance.ts`](../src/gpu/spatial/kthNeighborDistance.ts) | WGSL templates (private k-array TGSL cannot express), O(N²·k) | same |
| [`splatDensity.ts`](../src/gpu/spatial/splatDensity.ts) | scatter through the blend unit; no index needed in 2D; no 3D equivalent (ADR-0004) | the 3D note's gather |
| [`crossPcf.ts`](../src/gpu/spatial/crossPcf.ts), [`tcm.ts`](../src/gpu/spatial/tcm.ts) | `buildBucketGrid` from [`src/spatial/bucketGrid.ts`](../src/spatial/bucketGrid.ts) on the host, two arrays uploaded, 3×3 walk in WGSL | the same arrays built on-device |
| [`forces.ts`](../src/gpu/sim/forces.ts) `separateForce`/`cohereForce`/`springForce`, [`danceForces.ts`](../src/gpu/graph/ops/danceForces.ts) | O(N) host loops per body; **no GPU pairwise-force kernel exists** (`execute` and `cpuGolden` are the same `computeForce`) | neighbour query with `cell = radius` (the force-directed layout behind TerraCognita's displacement-field terrain, which this engine serves as a consumer) |
| [`morphology.ts`](../src/gpu/spatial/morphology.ts), [`convolveSeparable.ts`](../src/gpu/spatial/convolveSeparable.ts) | clamp-to-edge gathers with no invalid tap: a nodata sentinel is a hole erosion spreads or a value the mean averages in | §4 |

The consumer contract `crossPcf.ts` and `tcm.ts` already run, and its host producer:

```wgsl
@group(0) @binding(3) var<storage, read> start: array<u32>;       // bucket offsets over B
@group(0) @binding(4) var<storage, read> items: array<u32>;
...
let b = u32(rr * cols + cc);
let lo = start[b];
let hi = start[b + 1u];
for (var k = lo; k < hi; k = k + 1u) { let j = items[k]; ... }
```

```ts
export interface BucketGrid {
  readonly cols: number; readonly rows: number;
  /** Bucket side length in world units (= the query radius the grid was built for). */
  readonly cell: number;
  readonly minX: number; readonly minY: number;
  /** `cols*rows + 1` prefix offsets into `items`. */
  readonly start: Int32Array;
  /** Point indices grouped by bucket; `items[start[b] .. start[b+1])` is bucket b. */
  readonly items: Int32Array;
}
export function buildBucketGrid(xs, ys, cell, bounds?): BucketGrid;
```

This is an offset list (`start[M+1]` / `items[n]`): the structure `bucketGrid.ts` builds,
oriented cell→points because every query iterates the points of a cell; it is not a sparse
feature matrix, so the CSR-vs-CSC question that matters for MDV/AnnData column access does
not arise here — the point→cell direction is just the dense `cellOf[n]` array.

Constraints: ADR-0003 (TGSL unless atomics / workgroup memory / barriers → WGSL template via
`resolveWithContext`); 8 storage buffers per stage and 65535 workgroups per dimension
(`dispatchGrid` in [`device.ts`](../src/gpu/device.ts) folds to 2-D); ~2 s watchdog; TGSL
`i / w` on u32 is **float** division (`textureBridge.ts`, `splatDensity.ts:108`,
`tcm.ts:141`); every op ships a `cpuGolden` ([`op.ts`](../src/gpu/graph/op.ts)).

## 2. What exists

### 2.1 The scan — `src/gpu/scan/prefixSum.ts`

```ts
export const SCAN_WG = 256;          // threads per workgroup
export const SCAN_PER_THREAD = 4;    // elements each thread scans serially first
export const SCAN_BLOCK = SCAN_WG * SCAN_PER_THREAD;   // 1024 elements per workgroup
export type ScanElement = "u32" | "f32";
export interface ScanOptions { readonly maxWorkgroupsPerDim?: number; }   // test seam
export interface ScanResult<T> { scan: T; total: number; }
export function exclusiveScanGpu(values: Uint32Array, opts?: ScanOptions): Promise<ScanResult<Uint32Array>>;
export function exclusiveScanGpu(values: Float32Array, opts?: ScanOptions): Promise<ScanResult<Float32Array>>;

export interface ScanCtx { device: GPUDevice; layout: GPUBindGroupLayout; pipes: Record<ScanElement, ScanPipes>; }
export function getScanCtx(): Promise<ScanCtx>;
export function ensureBuf(device: GPUDevice, key: string, bytes: number, usage: number): GPUBuffer;
export interface EncodedScan { dst: GPUBuffer; totalBuf: GPUBuffer; }
export function encodeScan(ctx: ScanCtx, elem: ScanElement, src: GPUBuffer, n: number,
                           enc: GPUCommandEncoder, maxWorkgroupsPerDim?: number): EncodedScan;
export { dispatchGrid, MAX_WORKGROUPS_PER_DIM };   // re-exported from device.ts
```

`exclusiveScanGpu` is host-in/host-out; `encodeScan` is encoder-level (caller owns
submission), so the index build records histogram, scan and scatter in one command buffer.

| Aspect | Choice |
| --- | --- |
| Algorithm | recursive reduce-then-scan, not decoupled lookback (WebGPU gives no inter-workgroup forward-progress guarantee) |
| Per level | `scanBlocks`: serial 4-element scan per thread, Hillis-Steele over the 256 totals in `var<workgroup> sdata`, block slice + total to `blockSums`; recurse until one block; `addOffsets` walks back down |
| Depth | 1024/block: 2 levels under 1M, 3 at 35M; `levels` scans + `levels−1` adds, one compute pass |
| Bindings | 16-byte uniform **per level** (one rewritten uniform leaves every pass reading the last `n` — the `umapLayoutGpu.ts` bug) + `src`, `dst`, `blockSums` |
| Geometry | 256×1×1, `dispatchGrid(ceil(n/1024))` |
| Track | WGSL template (`var<workgroup>`, `workgroupBarrier()`); one explicit layout for both entry points and element types |
| Numerics | u32 wraps at 2³²; f32 sums in tree order (compare with tolerance); not in-place (`src` ≠ `dst`) |
| Golden | `prefixSum.gpu.test.ts`: serial host scan at several n, forced 2-D fold via `maxWorkgroupsPerDim`, `dispatchGrid` pinned |
| Bench | `pnpm bench:scan` ([`scripts/bench-scan.ts`](../scripts/bench-scan.ts), a plain process — vitest's fork cannot time `mapAsync`): **46.6 ms, 33M-element scan incl. readback**; 10.4 ms, 33M-row compaction at ~6%. `gap-analysis.md` says "~34 ms"; the 33.8 ms figure is not in the repo. Cite 34–47 ms; re-run first |

Hazard recorded in `684c87e` (from codec `9144dbe`): a grow-only pooled buffer bound
**whole** crossed `maxStorageBufferBindingSize` (128 MiB default = 33,554,432 u32), the bind
group was invalid, every dispatch did nothing, and the readback returned the *previous
call's* answer — stale, not zero, so the non-zero assertion misses it. Hence `sized()` and
`checkBindingSize()` in `device.ts` on every binding, pinned by
`src/gpu/bindingSize.gpu.test.ts`. 35M MDV rows exceed the default; the adapter reports
4 GiB (§6.6).

**Explainer candidate.** The recursive reduce-then-scan is a good subject for a docs-site
interactive: blocks of numbers collapse to block totals, the totals get scanned (the same step,
one level up), and the result is broadcast back down — three pictures, no jargon needed. Same
family as the DWT-draw and filtration demos. Noted here so the "illustrative visualisation +
lay description" pass finds it; not part of this design.

### 2.2 Stream compaction — `src/gpu/scan/streamCompact.ts`

```ts
export const COMPACT_WG = 256;
export type MaskArray = Uint8Array | Uint32Array | Float32Array;
export interface CompactOptions { readonly pass?: "gt" | "eq"; readonly value?: number; readonly maxWorkgroupsPerDim?: number; }
export interface CompactResult { indices: Uint32Array; count: number; }
export async function streamCompactGpu(mask: MaskArray, opts?: CompactOptions): Promise<CompactResult>;
export type MaskEncoding = "u8" | "u32" | "f32";
export interface EncodeCompactOptions extends CompactOptions { readonly mask?: MaskEncoding; readonly keyPrefix?: string; }
export interface EncodedCompact { indices: GPUBuffer; countBuf: GPUBuffer; }
export function getCompactCtx(): Promise<CompactCtx>;
export function encodeCompact(ctx: CompactCtx, maskBuf: GPUBuffer, n: number, enc: GPUCommandEncoder, opts?: EncodeCompactOptions): EncodedCompact;
```

`encodeCompact` records the same three passes into a caller-owned encoder, **one submit**,
worst-case `n`-sized output, count on-device (see §2.4.1). `streamCompactGpu` does
`predicate` → `encodeScan` → `scatter` (`outIdx[offsets[i]] = i`) in **two submits**: the
output is sized to the count read back after the scan. u8/u32/f32 × eq/gt lets MDV's
byte-per-row `filterArray` upload verbatim and ADR-0005's soft f32 thresholds share the
`bitcast` binding. Output is stable and ascending (scan slot per row, no atomic bump), hence
content-addressable as `support`. WGSL; uniform + 4 storage; `dispatchGrid(ceil(n/256))`.
Golden: host filter loop at several n plus a 78,125-workgroup crossing of the 65535 cap.

### 2.3 Mask-as-field — `src/gpu/graph/ops/filterOps.ts`

Exists: ADR-0005's `support` in its *dense* encoding as `points{n}` f32 columns
([`mdv-dimension-vs-support-facet.md`](mdv-dimension-vs-support-facet.md) §6): `maskRange`
(lo/hi + `softness`), `maskEquals` (categorical, f32 tolerance), `maskCount` (rows > ½ and
total weight — MDV's `filterSize`), `maskedDensity` (mask-weighted KDE). Combinators are
**`min` / `max` / `1−a`**, not the ADR's `a·b`: `minFieldsOp` / `maxFieldsOp` /
`invertFieldOp` in [`fieldArithmetic.ts`](../src/gpu/graph/ops/fieldArithmetic.ts);
`filterOps.test.ts` pins idempotence, De Morgan, involution, `min(a, 1−b)`.

Missing: every op is **host-only Tier-1** (`execute` and `cpuGolden` are the same JS loop, no
kernel, no `resident: true`). Nothing outside `src/gpu/scan/` and `bindingSize.gpu.test.ts`
imports `streamCompactGpu` or `encodeScan`: no compaction op, no mask ⇄ index duality, no
`support` facet on `GpuField` / `FieldValue`, no propagation hook, no weighted reduction.
`maskedDensity` is not the resident `splatDensity` because the stride-2 points buffer has
nowhere to carry a weight.

### 2.4 Gaps in the primitive

1. ~~**No encoder-level compaction.**~~ Landed: `encodeCompact(ctx, maskBuf, n, enc, opts) → { indices, countBuf }`
   in `streamCompact.ts` — one submit into a worst-case `n`-sized pooled buffer, count left in
   the scan's `totalBuf`; `opts.mask` picks u8/u32/f32, `opts.keyPrefix` namespaces the pool.
2. ~~**Pool keys are global.**~~ Fixed: `encodeScan` takes a trailing `keyPrefix` (default
   `"scan"`), so a fused build-plus-occupied-cells pass can give each scan its own pool.
3. **No segmented scan, no counting-sort helper.** Not needed for the index; a segmented scan
   matters only for per-cell reductions over the sorted list.
4. **Resident bridge is f32-only.** `executor.ts` throws on non-f32 resident values
   (`residentF32Count`), so a u32 index buffer reaches the host only through its own
   builder's readback. Goldens below are written accordingly.

## 3. The uniform-grid index

### 3.1 Build

**Decision: one op, `dims: 2 | 3`, histogram + `encodeScan` + atomic scatter, one submit,
producing the `BucketGrid` offset list.** Lattice: `cell` (scalar, world units) over `origin`
and `gridDims` (Dx,Dy[,Dz]); `cellId = cx + Dx*(cy + Dy*cz)`, x-fastest, as the 3D note §2 and
`crossPcf.ts` (`rr * cols + cc`). `Shape` has no `dim` yet (ADR-0004's `points{n, dim}` is
unbuilt), so the op takes `stride: 2 | 3` until it lands.

| Pass | Kernel | Track | Bindings | Geometry |
| --- | --- | --- | --- | --- |
| 0 | `clearBuffer(counts)` | encoder command | — | — |
| 1 | `cellHistogram`: `cellOf[i] = cellId(p_i)`, `atomicAdd(counts[cellOf[i]], 1)` | WGSL (atomics) | uniform + points + cellOf + counts = 4 | 256, `dispatchGrid(ceil(n/256))` |
| 2 | `encodeScan(ctx, "u32", counts, M+1, enc)` → `start` | existing | 1 + 3 | as §2.1 |
| 3 | `copyBufferToBuffer(start → cursor)` | encoder command | — | — |
| 4 | `scatter`: `slot = atomicAdd(cursor[cellOf[i]], 1)`, `items[slot] = i` | WGSL (atomics) | uniform + cellOf + cursor + items = 4 | 256, `dispatchGrid(ceil(n/256))` |

`counts` is M+1 with a trailing zero so the exclusive scan yields `start[M] == n`; the scan's
`totalBuf` is `n`, a free sanity word. Order within a cell is unspecified by contract (a
stable counting-sort variant can come later without changing the layout). One uniform per
build in flight (§2.1's per-level lesson). Sparse lattices (M ≫ N): the scan is over M, so
512³ is a 134M-element scan, over the binding limit — cap M there and throw with the number;
occupied-cell compaction (3D note) is the real fix and waits on gap 2.4.1.

### 3.2 Cell size, extent, placement

`cell` = query radius (`bucketGrid.ts`, the toolbox, the 3D note); the 3^d stencil is exact
for compact kernels at `cell ≥ R`. Extent comes from the points' `placement` (ADR-0018
decision 3) or their own bounds (`buildBucketGrid`'s optional `bounds`); `cell` is in the
placement's world units (ADR-0018 decision 5). The lattice gets its own `ResolvedPlacement`,
`worldFromArray = translate(origin) · scale(cell)` in the points' `system`, so per-cell
consumers are already placed and `systemsAgree` rejects a foreign query cloud at build time
via `inferPlacement`.

### 3.3 Handle shape

**Decision: three output ports, no new `Shape` kind.**

```ts
outputs: [
  { name: "start",   kind: "points", dtype: "u32" },   // points{M+1}: offset list
  { name: "items",   kind: "points", dtype: "u32" },   // points{n}:   indices grouped by cell
  { name: "lattice", kind: "opaque", dtype: "f32" },   // { dims, cell, origin, gridDims, stride }
]
```

`start` and `items` are resident u32 leases from `ctx.backend.lease`, one per port (`op.ts`:
the same `ResidentBuffer` on two ports is released twice). `lattice` is a host payload
(`{ kind: "opaque", name: "gridLattice" }`) carrying the placement. `cellOf[n]` is build
scratch, not a port: O(1) recomputable. Because the host bridge is f32-only (gap 2.4.4),
`start`/`items` feed **resident** consumers only; test readback comes from the builder.

Rejected: one opaque value with buffers in `payload` — the executor tracks ownership only
via `v.buffer` / `v.texture` (`executor.ts` `own(...)`), so hidden leases never release. A
new `Shape` kind `"index"` — `shapesEqual`, `numCells`, port typing and memo keys all switch
on `kind`; `points{n}` with dtype `u32` (`FieldValue.data` already allows `Uint32Array`) is
lighter. The 3D note's `opaque` handle predates ADR-0017; this supersedes it.

### 3.4 Query API

```ts
// landed (ADR-0022), src/gpu/spatial/gridIndex.ts — 2D and 3D (`dims: 3` needs `stride: 3`)
export interface GridIndexOptions { cell: number; stride?: 2 | 3; dims?: 2 | 3; bounds?: Bounds2 | Bounds3; keyPrefix?: string; maxWorkgroupsPerDim?: number; }
export interface GridIndexResident { start: GPUBuffer; items: GPUBuffer; M: number; n: number; lattice: GridLattice; }
export function encodeGridIndex(ctx: GridIndexCtx, points: GPUBuffer, n: number, lattice: GridLattice, enc: GPUCommandEncoder, opts?): GridIndexResident;
export async function buildGridIndexGpu(points: Float32Array, opts: GridIndexOptions): Promise<BucketGrid>;
export function latticeFor(xs, ys, cell, bounds?, zs?): GridLattice;   // src/spatial/bucketGrid.ts, shared with the CPU build; `zs` ⇒ depth/minZ
```

`encodeGridIndex` fuses build + query in one submit; `buildGridIndexGpu` reads back into the
exact `BucketGrid` struct so `crossPcf.ts` / `tcm.ts` switch with a one-line change.

TGSL has no closures, so the helper is two `tgpu.fn` pieces and the consumer writes the 3^d
loop itself (six lines; `crossPcf.ts` shows it):

```ts
/** Cell coordinate of a world position, clamped to the lattice. Integer maths spelled out —
 *  no `/` on u32. */
cellCoord(p: vec3f, lat: Lattice) -> vec3i
/** [lo, hi) range into `items` for cell coordinate c, or (0,0) when c is off-lattice.
 *  Callers loop dz in [-1,1] only when lat.dims == 3, so 2D is 9 cells and 3D is 27. */
cellRange(c: vec3i, lat: Lattice, start: array<u32>) -> vec2u
```

Graph-level, a `nnDistance` / `knn` / `separate` op gains optional inputs `start`, `items`,
`lattice` and uses the stencil when they are connected; brute force stays its golden.

### 3.5 Golden

Compare **per-cell index sets**: for every `b`, `sort(items[start[b]..start[b+1]))` equals
the same slice of `cpu = buildBucketGrid(xs, ys, cell, bounds)` with the same origin. Plus
the 3D note §5 checks: `start` is the exclusive scan of the counts, `start[M] == n`, every
item lands in its implied cell, ranges partition `[0,n)`. Fixtures: one point per cell,
points on boundaries and corners (pins `floor` + clamp), an empty region, n = 0, a forced
2-D fold via `maxWorkgroupsPerDim`. For 3D, add a `zs` argument to `buildBucketGrid` rather
than a second CPU reference. End-to-end: indexed `nnDistance` equals brute force to 1e-6
when `cell ≥` every true NN distance, plus one case where it does not, so the failure mode
is visible.

## 4. The validity / support facet

### 4.1 One facet, named `support`

**Decision: ADR-0005's `support` and terrain's "nodata/validity" are one facet; keep the
ADR's name.** One weight per sample in [0,1]; a nodata sentinel is the boxcar 0, a soft
brush the ramp. A nodata source becomes a mask via `maskRange` (`lo = hi = sentinel`, then
`invertField`) or a `maskNotEqual` sibling of `maskEquals` — a leaf op. No `isValid` facet,
no `nodata` field on `FieldValue`, no NaN convention (NaN is invisible to `min`/`max` and
poisons a tree reduction).

### 4.2 Where it lives

On both `GpuField` (build time) and `FieldValue` (run time), exactly as `placement` does:

```ts
// handle.ts
readonly support?: SupportRef;            // GpuField
support?: FieldValue;                     // FieldValue — a points{numCells(shape)} f32 column, host or resident
// op.ts
inferSupport?(inputs: (SupportRef | undefined)[], params: Params): (SupportRef | undefined)[];
```

`SupportRef` at build time is `{ field: GpuField }`. The executor stamps
`node.outSupports?.[i]` onto outputs the op left unset, in the same `forEach` as
`outPlacements` (`executor.ts:398–413`), and resolves it to a concrete `FieldValue` before
`execute`, bridged like the inputs. Absent ⇒ all-valid; every existing op is unchanged.

### 4.3 Propagation rules per op category

Default (no `inferSupport`): **intersection** — `min` over same-shape input supports,
`undefined` if none. Right for pointwise ops, silently wrong for the rest:

| Category | Rule | Kernel |
| --- | --- | --- |
| Pointwise (`fieldArithmetic.ts`, `threshold.ts`, `complexOps.ts`) | intersection; the mask rides along, unread | none |
| Stencil / morphology ([`convolveSeparable.ts`](../src/gpu/spatial/convolveSeparable.ts), [`morphology.ts`](../src/gpu/spatial/morphology.ts), `getisOrd.ts`) | min/max **skip** invalid taps (output valid iff any tap was). Convolution **renormalises**: the same separable pass over the mask gives `W`, output `Σ w·m·v / W`, mask `W / Σ w` (or boxcar `W > 0`) — psychogeo's `block_mean` pairing, separable because the mask pass uses the same 1-D kernel | TGSL, 2×2 passes or one fused: `params, src, mask, wts, dst, dstMask` = 6; `WG = 64`, one thread per texel |
| Reduction / decimation (terrain's N×N→1; `maskCount`) | value `Σ m·v / Σ m`, mask `Σ m / N²`; all-invalid window → 0 with mask 0, never NaN. `inferPlacement` scales `worldFromArray` by N (ADR-0018); `inferSupport` decimates the mask the same way | TGSL, 1 pass: `params, src, mask, dst, dstMask` = 5; `dispatchGrid(ceil(wOut·hOut/64))` |
| Neighbour query (every §3 consumer) | invalid points never enter the grid: `encodeCompact(mask) → indices` (gap 2.4.1), index built over the compacted cloud, `items` holds original row ids. Soft masks ride as a companion f32 column on `items`. Work is O(selected) — ADR-0005's "two encodings chosen by scale" | §3 |

### 4.4 Storage

**Decision: the facet is a reference to a mask field; the mask is its own `FieldValue`
(one buffer, one lease, its own memo node) shared by reference — not a second edge the user
wires.** The mask is already a field (filterOps), the fuzzy-set ops already combine masks as
fields, ADR-0017 forbids packing value and mask into one buffer, and a mask port on every
input would double the port count of binary ops.

Migration from `filterOps`: (1) mask leaves unchanged; (2) add `attachSupport(value, mask) → value`
and `supportOf(value) → mask`; (3) `maskedDensity` becomes `splatDensity` reading
`points.support` as weight via a second vertex buffer, not a stride-3 zip; (4) `maskCount`
gains the resident form (`exclusiveScanGpu`'s `total` is `Σ m` for a hard mask); (5) stencil
ops add the mask binding behind a check on `inputs[0].support`, so an unmasked graph
compiles today's kernel and the morphology bit-exact test stays green.

## 5. Sequencing

1. ~~**2D index, host-in/host-out, golden against `bucketGrid.ts`**; swap `crossPcf.ts` and
   `tcm.ts` onto it. Proves the build; adds `scripts/bench-grid-index.ts` (like `bench-scan`).~~
   **Done 2026-08-22 → [ADR-0022](decisions/0022-gpu-uniform-grid-index.md).**
2. ~~**Indexed `nnDistance` / `kthNeighborDistance` / `knn`** via `cellCoord`/`cellRange`,
   brute force as golden. Unblocks O(N·k) point statistics past `knn.ts`'s O(N²) wall; CkNN
   and fuzzy adjacency follow since they compose `kthNeighborDistanceGpu`.~~
   **Done 2026-08-22** — `src/gpu/spatial/gridIndexQuery.ts` (2D helpers, `vec2f`/`vec2i`
   for now; the 3D signatures above wait on step 6) + an optional `cell` on all three
   kernels; contract in `IndexedQueryOptions`. CkNN / fuzzy adjacency still call brute force.
3. **Resident index op** (§3.3) and a resident `separate` / `spring` force op reading it.
   Unblocks force-directed layout at scale (TerraCognita's displacement-field terrain) and the first
   per-tick resident chain (ADR-0017 stage 3 feedback state).
4. **`encodeCompact` + the `support` facet** (§4.2, §4.4 steps 1–4), pointwise and reduction
   rules. Unblocks the mask ⇄ index duality, weighted null models, MDV filtered density with
   zero marshalling.
5. **Nodata-aware stencils and the N×N→1 decimation** (§4.3). Unblocks the terrain chain:
   morphology over a LIDAR raster with holes, bare-earth opening that does not grow them,
   block means psychogeo can call instead of its own `block_mean`.
6. ~~**`dims: 3`** once ADR-0004's `points{n, dim}` lands (or via `stride: 3` before it).~~
   **Done 2026-08-22**: `latticeFor`/`buildBucketGrid` take `zs` (+ 6-number bounds) and the
   lattice gains `depth`/`minZ`; the kernel reads z when the lattice has it (`stride: 3`
   required, `dims: 3` on `buildGridIndexGpu`). Still open: the 3D note's `splatVolume`
   gather over the same three ports, and the occupied-cell list as one `encodeCompact` over
   `counts`; `dims` folds into `Shape.dim` when ADR-0004 lands.

Promoted at step 1 (ADR-0022); the ADR records what is built, this note keeps the reasoning.

## 6. Open questions

1. **`stride: 3` before `Shape` has `dim`, or wait?** Default: ship `stride` as an op param,
   validated against `data.length`, fold into `dim` when ADR-0004 lands.
2. **Sparse lattices (M ≫ N).** Default: throw at the binding limit with the number; build
   occupied-cell compaction only when a 3D consumer hits it (needs gap 2.4.1, changes the
   consumer loop).
3. **Deterministic order within a cell?** Default: no — contract and test as a set. Revisit
   if a parity test depends on summation order (`tcm.ts` notes its residual is exactly that).
4. **Stamp `support` onto placement-less (array-space) grids?** Default: yes; the facet is
   independent of placement and terrain rasters arrive array-space.
5. **Soft masks in the neighbour path — compact at `m > 0` and carry the weight, or
   threshold at ½?** Default: compact at `> 0` and carry the weight; dropping half-weighted
   points jumps like a hard filter.
6. **Raise `maxStorageBufferBindingSize` in `getDevice()`?** Default: not here; device-wide,
   belongs to the MDV integration; `checkBindingSize` makes the ceiling loud.

## References

- [`gpu-spatial-index-3d.md`](gpu-spatial-index-3d.md) — the 3D spec this extends. Stale
  there: "read back via TypeGPU `.read()`, not `mapAsync`" (the Dawn instance-lifetime bug was
  fixed 2026-07-29; `prefixSum.ts` uses pooled `mapAsync`, pinned by
  `test/dawn-limits-sweep.gpu.test.ts`), and the `opaque` handle (§3.3).
- [ADR-0003](decisions/0003-use-gpu-tgsl-kernels.md) TGSL vs WGSL template;
  [ADR-0004](decisions/0004-field-type-model-and-volumetric-splat.md) `dim: 2 | 3`, gather-splat;
  [ADR-0005](decisions/0005-columnar-filters-and-sparse-support.md) `support`;
  [ADR-0017](decisions/0017-tier2-resident-buffer-edges.md) leases, one per port;
  [ADR-0018](decisions/0018-field-domains-placement-and-resolution.md) placement, units of spatial params.
- [`mdv-dimension-vs-support-facet.md`](mdv-dimension-vs-support-facet.md) — operator
  correction and the "build the scan first" call, now discharged.
- [`gpu-spatial-analysis-toolbox.md`](gpu-spatial-analysis-toolbox.md) — `GpuGridIndex` as
  "the foundational new kernel"; [`fuzzy-tda-and-windowing.md`](fuzzy-tda-and-windowing.md)
  — a mask is a window, hard is its boxcar.
