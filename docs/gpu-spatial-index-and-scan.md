# GPU spatial index, scan/compaction, and the one support mask

Status: **design note** (2026-08-22) — promote to an ADR when implementation starts.

Three things in [`gap-analysis.md`](gap-analysis.md) share one substrate: the scan /
stream-compaction kernel (now built, at `src/gpu/scan/`), the uniform-grid spatial index
(specced for 3D in [`gpu-spatial-index-3d.md`](gpu-spatial-index-3d.md), built nowhere), and
ADR-0005's `support` facet, which the terrain work has just rediscovered under the name
"nodata/validity". This note documents the first as it exists, designs the second on top of
it for **2D and 3D in one op**, and pins the third down as a single facet so that two never
get built. It extends the 3D note rather than replacing it: what it adds is the 2D case,
the scan it now rests on, the resident handle shape under ADR-0017, the consumer-facing
query API, and the sequencing.

## 1. Context and consumers

**What is brute-force today.** Every per-point spatial kernel loops over all N points, one
thread per query:

- [`nnDistance.ts`](../src/gpu/spatial/nnDistance.ts) (TGSL, `for (let j = d.u32(0); j < count; j++)`),
  [`emptySpace.ts`](../src/gpu/spatial/emptySpace.ts), [`cknn.ts`](../src/gpu/spatial/cknn.ts),
  [`fuzzyAdjacency.ts`](../src/gpu/spatial/fuzzyAdjacency.ts) — all `"use gpu"`, all O(N²);
- [`knn.ts`](../src/gpu/spatial/knn.ts) and [`kthNeighborDistance.ts`](../src/gpu/spatial/kthNeighborDistance.ts)
  — WGSL templates (a private k-array that TGSL could not express), O(N²·k);
- [`splatDensity.ts`](../src/gpu/spatial/splatDensity.ts) is the exception: it is a *scatter*
  (instanced quads through the blend unit) and does not need an index in 2D. ADR-0004 records
  that this path has no 3D equivalent, which is why the 3D note exists.

**What already consumes a grid index — on the GPU, from a CPU build.** This is the finding
that sets the handle shape. [`crossPcf.ts`](../src/gpu/spatial/crossPcf.ts) and
[`tcm.ts`](../src/gpu/spatial/tcm.ts) call `buildBucketGrid` from
[`src/spatial/bucketGrid.ts`](../src/spatial/bucketGrid.ts) on the host, upload its two arrays,
and walk the 3×3 block in WGSL:

```wgsl
@group(0) @binding(3) var<storage, read> start: array<u32>;       // CSR bucket offsets over B
@group(0) @binding(4) var<storage, read> items: array<u32>;
...
let b = u32(rr * cols + cc);
let lo = start[b];
let hi = start[b + 1u];
for (var k = lo; k < hi; k = k + 1u) { let j = items[k]; ... }
```

with the CPU side:

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

So the consumer contract — `start[cells+1]`, `items[n]`, a 3^d loop — is already written and
tested twice. The GPU index should produce *exactly this layout* and the first slice should
be swapping the host build out from under those two callers.

**Force-directed layout.** [`src/gpu/sim/forces.ts`](../src/gpu/sim/forces.ts) `separateForce` /
`cohereForce` / `springForce` are O(N) host loops per body, and the graph ops in
[`danceForces.ts`](../src/gpu/graph/ops/danceForces.ts) run them on the host (`execute` and
`cpuGolden` are the same `computeForce`). There is no GPU pairwise-force kernel at all. A
springy layout at any size needs radius-bounded repulsion, and that is the index's neighbour
query with `cell = radius`. Note the brief for this note describes a "displacement-field
flagship" fed by that layout; `gap-analysis.md` does not mention it, so it is cited here
from the brief, not from a committed doc.

**Terrain.** The terrain block of `gap-analysis.md` lists the decimating N×N→1 reduction and
a nodata/validity facet, and says "unify with the ADR-0005 `support` facet, one mask facet
not two". [`morphology.ts`](../src/gpu/spatial/morphology.ts) and
[`convolveSeparable.ts`](../src/gpu/spatial/convolveSeparable.ts) are clamp-to-edge gathers
with no notion of an invalid tap: a nodata sentinel is either a hole the erosion spreads or a
value the mean averages in. §4 fixes that.

**Constraints every kernel below is written against.** ADR-0003: `"use gpu"` TGSL unless the
kernel needs atomics, workgroup memory or barriers, in which case a WGSL template through
`resolveWithContext`. Eight storage buffers per stage — a ninth makes the layout invalid and
every dispatch a silent no-op. 65535 workgroups per dimension — `dispatchGrid` in
[`device.ts`](../src/gpu/device.ts) folds a linear count into 2-D and the kernel rebuilds
`wid.x + wid.y * gridX`. Dispatches past ~2 s are killed silently. In TGSL, `i / w` on u32
operands is **float** division (`textureBridge.ts`, `splatDensity.ts:108`, `tcm.ts:141` all
carry the workaround); any index math in a TGSL kernel must spell integer division out.
Every op ships a `cpuGolden` ([`op.ts`](../src/gpu/graph/op.ts)).

## 2. What exists

### 2.1 The scan — `src/gpu/scan/prefixSum.ts`

Interface:

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

`exclusiveScanGpu` is the host-array convenience (upload, encode, submit, read back);
`encodeScan` is the resident, encoder-level form — the caller owns submission, so the index
build below records its histogram, the scan, and its scatter into one command buffer.

Algorithm: a **recursive reduce-then-scan**, not decoupled lookback (WebGPU gives no
forward-progress guarantee between workgroups, so a lookback spin can deadlock on a device
that serialises them). Per level, `scanBlocks` does a serial 4-element exclusive scan per
thread, then a Hillis-Steele scan over the 256 per-thread totals in `var<workgroup> sdata`,
writes the block's slice and its total to `blockSums`; the same kernel runs on `blockSums`
one level up until a level has a single block; then `addOffsets` walks back down adding each
parent's scanned offset into its children. 1024 elements per block means 2 levels under 1M
and 3 at 35M. Passes: `levels` scans + `levels−1` adds, all in one compute pass. Bindings: a
16-byte uniform **per level** (a single rewritten uniform would leave every pass reading the
last level's `n` — the `umapLayoutGpu.ts` bug) plus 3 storage buffers (`src`, `dst`,
`blockSums`). Workgroup geometry 256×1×1, `dispatchGrid(ceil(n/1024))`. u32 wraps at 2³²;
f32 accumulates in tree order, so compare with a tolerance. Not in-place (`src` ≠ `dst`,
because aliasing a buffer across a read-only and a writable binding is undefined).

WGSL template, not TGSL, because it needs `var<workgroup>` and `workgroupBarrier()` — exactly
the case ADR-0003 routes to templates. One explicit bind-group layout serves both entry
points and both element types (`layout: "auto"` would derive a different layout per entry
point, and `addOffsets` never mentions `blockSums`).

The original commit (`684c87e`, migrated from codec `9144dbe`, "Build the scan/compaction
primitive, and find a fourth silent-zero mode") records the hazard that shaped the file:
grow-only pooled buffers bound **whole** put 132 MB of data in a 160 MB pooled buffer, the
binding crossed `maxStorageBufferBindingSize` (128 MiB default = 33,554,432 u32 rows), the
bind group was invalid, every dispatch did nothing, and the readback returned the *previous
call's* answer — plausible stale data, not zeroes, so the non-zero assertion that catches the
other three silent modes does not catch this one. Hence `sized()` and `checkBindingSize()` in
`device.ts` on every binding, pinned by `src/gpu/bindingSize.gpu.test.ts`. The 35M-row MDV
case does not fit the default device limit; the adapter reports 4 GiB, so raising it is a
`getDevice()` decision left to the integration.

Golden and measurements. `prefixSum.gpu.test.ts` compares against a serial host scan at
several n, forces the 2-D fold through `maxWorkgroupsPerDim`, and pins `dispatchGrid`
directly. `pnpm bench:scan` ([`scripts/bench-scan.ts`](../scripts/bench-scan.ts), a plain
process because vitest's fork cannot time `mapAsync`): the file header records **46.6 ms for
a 33M-element scan including the full readback** and 10.4 ms for a 33M-row compaction at ~6%
selectivity, both readback-dominated. `gap-analysis.md` quotes "~34 ms"; the 33.8 ms figure
the migration PR is said to have recorded is not written anywhere in the repo. Treat 34–47 ms
as the range and re-run the bench before citing a number — and either way the dispatch is
two orders of magnitude inside the watchdog.

**Explainer candidate.** The recursive reduce-then-scan is a good subject for a docs-site
interactive: blocks of numbers collapse to block totals, the totals get scanned (the same step, one
level up), and the result is broadcast back down — three pictures, no jargon needed. Same family as
the DWT-draw and filtration demos. Noted here so the "illustrative visualisation + lay description"
pass finds it; not part of this design.

### 2.2 Stream compaction — `src/gpu/scan/streamCompact.ts`

```ts
export const COMPACT_WG = 256;
export type MaskArray = Uint8Array | Uint32Array | Float32Array;
export interface CompactOptions { readonly pass?: "gt" | "eq"; readonly value?: number; readonly maxWorkgroupsPerDim?: number; }
export interface CompactResult { indices: Uint32Array; count: number; }
export async function streamCompactGpu(mask: MaskArray, opts?: CompactOptions): Promise<CompactResult>;
```

Three passes — `predicate` (mask → 0/1 flags), `encodeScan` (flags → write offsets + total),
`scatter` (`outIdx[offsets[i]] = i` where `flags[i] == 1`) — in **two submits**, because the
output is sized to the count read back after the scan. Mask encodings u8/u32/f32 × eq/gt are
there so MDV's byte-per-row `filterArray` uploads verbatim (4 rows per word, 35 MB not 140)
and ADR-0005's soft f32 weight thresholds through the same `bitcast` binding. Output is
stable and ascending (a scan slot per row, no atomic bump), which is what makes it a
content-addressable `support` encoding. WGSL; 5 bindings (uniform + 4 storage);
`dispatchGrid(ceil(n/256))`. Golden: host filter loop at several n, plus a genuine
78,125-workgroup crossing of the 65535 cap.

### 2.3 Mask-as-field — `src/gpu/graph/ops/filterOps.ts`

What is there: ADR-0005's `support` in its *dense* encoding as ordinary `points{n}` f32
columns, per the argument in [`mdv-dimension-vs-support-facet.md`](mdv-dimension-vs-support-facet.md)
§6. Ops: `maskRange` (lo/hi + `softness`, boxcar at 0, smoothstep ramp above), `maskEquals`
(categorical, with an f32 tolerance), `maskCount` (rows with weight > ½, and total weight —
MDV's `filterSize`), `maskedDensity` (the KDE weighted by the mask). The combinators are the
corrected operator set — **`min` / `max` / `1−a`**, not the ADR's `a·b` — and live in
[`fieldArithmetic.ts`](../src/gpu/graph/ops/fieldArithmetic.ts) as `minFieldsOp`,
`maxFieldsOp`, `invertFieldOp`; `filterOps.test.ts` pins idempotence, De Morgan, involution,
and set difference `min(a, 1−b)`.

What is missing, precisely. Every one of these ops is **host-only Tier-1**: `execute` and
`cpuGolden` are the same JS loop, no kernel, no `resident: true`. Nothing in `src/` outside
`src/gpu/scan/` and `bindingSize.gpu.test.ts` imports `streamCompactGpu` or `encodeScan` —
no `materializeSupport`, no compaction op, no mask ⇄ index duality in the graph. There is no
`support` facet on `GpuField` / `FieldValue`, no propagation hook, and no weighted reduction
that reads a mask. `maskedDensity` is deliberately not the resident `splatDensity` op
because the stride-2 points buffer has nowhere to carry a weight.

### 2.4 Gaps in the primitive, for the consumers below

1. **No encoder-level compaction.** `streamCompactGpu` is host-in/host-out and two submits.
   The support facet needs `encodeCompact(ctx, maskBuf, n, enc) → { indices, countBuf }`
   into a worst-case `n`-sized buffer with the count left on-device (one submit, no
   readback); the 3D note's occupied-cell list wants the same thing.
2. **Pool keys are global.** `encodeScan` leases `scan:${elem}:dst${li}` etc. — fixed names,
   so two scans recorded into one command buffer alias each other's output. The index build
   records one scan per build and is fine; a fused build-plus-occupied-cells pass is not.
   Add a `keyPrefix` parameter (the signature `gap-analysis.md` prints already has one; the
   code does not — the gap-analysis signature is wrong about argument order as well).
3. **No segmented scan, no counting-sort helper.** Neither is needed for the index (the
   scan over cell counts is a flat scan; the sort is histogram + atomic bump). A segmented
   scan becomes interesting only for per-cell reductions over the sorted list.
4. **Resident bridge is f32-only.** `executor.ts` throws on a non-f32 resident value
   (`residentF32Count`), so a u32 index buffer can cross to the host only through its own
   builder's readback, not through the executor. Goldens below are written accordingly.

## 3. The uniform-grid index

### 3.1 Build

One op, `dims: 2 | 3`, producing the `BucketGrid` layout. The lattice is `cell` (scalar, world
units) over `origin` and `gridDims` (Dx,Dy[,Dz]); `cellId = cx + Dx*(cy + Dy*cz)`, x-fastest,
exactly as the 3D note §2 and as `crossPcf.ts` (`rr * cols + cc`) already index. Points
pack `[x0,y0,x1,y1,…]` today; `Shape` has no `dim` yet (ADR-0004's `points{n, dim}` is
unbuilt), so the op takes a `stride: 2 | 3` that is 2 until that lands, then reads `dim`.

| Pass | Kernel | Track | Bindings | Geometry |
| --- | --- | --- | --- | --- |
| 0 | `clearBuffer(counts)` | encoder command | — | — |
| 1 | `cellHistogram`: `cellOf[i] = cellId(p_i)`, `atomicAdd(counts[cellOf[i]], 1)` | WGSL (atomics) | uniform + points + cellOf + counts = 4 | 256, `dispatchGrid(ceil(n/256))` |
| 2 | `encodeScan(ctx, "u32", counts, M+1, enc)` → `start` | existing | 1 + 3 | as §2.1 |
| 3 | `copyBufferToBuffer(start → cursor)` | encoder command | — | — |
| 4 | `scatter`: `slot = atomicAdd(cursor[cellOf[i]], 1)`, `items[slot] = i` | WGSL (atomics) | uniform + cellOf + cursor + items = 4 | 256, `dispatchGrid(ceil(n/256))` |

One submit. `counts` is sized M+1 with a trailing zero so the exclusive scan yields the
`cells+1` offsets `bucketGrid.ts` promises (`start[M] == n`). The `totalBuf` from the scan is
`n` and doubles as a free sanity word. Order within a cell is whatever the atomics gave —
unspecified, by contract. A uniform `{ n, gridX, dims, Dx, Dy, Dz, stride, cell, origin… }`
is written once per build; two builds in flight need two uniforms (the per-level lesson
from the scan). The 3D note's cost concern for sparse lattices (M ≫ N) stands: the scan is
over M, so a 512³ lattice is a 134M-element scan — over the binding limit. Cap M at the
binding limit in the builder and throw with the number; the occupied-cell compaction the 3D
note proposes is the real fix and waits on gap 2.4.1.

Why atomics rather than a sort: the 3D note already chose this; with the scan in hand it
is two trivial one-thread-per-point kernels, and non-determinism within a cell is harmless
for every consumer (they reduce over the cell). A deterministic variant (stable counting sort
via a second scan over `cellOf` ranks) is possible later without changing the layout.

### 3.2 Cell size, extent, and placement

`cell` is the query radius — the toolbox's rule, `bucketGrid.ts`'s doc comment, and the 3D
note's "cell size = kernel support radius" are the same rule; the 3^d stencil is exact for
compact kernels at `cell ≥ R`. Extent comes from the points' `placement` when they have one
(ADR-0018 decision 3: bbox is a placement constructor at sources, not a free parameter) and
from the points' own bounds otherwise, which is what `buildBucketGrid`'s optional `bounds`
does today. `cell` is declared in the placement's world units (ADR-0018 decision 5: ops
declare the units of their spatial params). The lattice itself gets a `ResolvedPlacement`:
`worldFromArray = translate(origin) · scale(cell)` in the points' `system` — so a grid-shaped
consumer (a density gathered per cell) is already placed, and `systemsAgree` rejects a
query cloud from another system at build time through `inferPlacement`.

### 3.3 Handle shape — a struct of fields, not a new value kind

Recommendation: **three output ports, no new `Shape` kind.**

```ts
outputs: [
  { name: "start",   kind: "points", dtype: "u32" },   // points{M+1}: CSR offsets
  { name: "items",   kind: "points", dtype: "u32" },   // points{n}:   indices grouped by cell
  { name: "lattice", kind: "opaque", dtype: "f32" },   // { dims, cell, origin, gridDims, stride }
]
```

`start` and `items` are resident u32 buffers leased from `ctx.backend.lease` — one lease per
port, as `op.ts` requires ("returning the same `ResidentBuffer` on two ports makes the
executor release it twice"). `lattice` is a tiny host payload (`{ kind: "opaque", name:
"gridLattice" }`) and carries the lattice's placement. `cellOf[n]` is **not** a port: it is a
build scratch, recomputable in O(1) from a point and the lattice, and exposing it would cost
a third lease and a third consumer input for nothing.

Rejected alternatives. A single opaque value with buffers in its `payload`: the executor
tracks ownership only through `v.buffer` / `v.texture` (`executor.ts`, the `own(...)` calls),
so leases hidden in a payload would never be released — it needs an executor change for no
gain. A new `Shape` kind `"index"`: `shapesEqual`, `numCells`, the composer's port typing and
the memo keys all switch on `kind`; a fourth arm for one op is heavier than reusing
`points{n}` with dtype `u32`, which `FieldValue.data` already allows (`Uint32Array`). The
3D note's `opaque` suggestion predates ADR-0017 and assumed host payloads; this supersedes
it for the resident case.

The u32 dtype has one consequence worth stating: the executor's host bridge is f32-only, so
`start`/`items` can only feed **resident** consumers. That is the intended use; the host
copy for tests comes from the standalone builder's own `readBack`.

### 3.4 Query API

Standalone builder, sibling of `buildBucketGrid` and `splatDensityGpu`:

```ts
export interface GridIndexOptions { cell: number; dims: 2 | 3; stride?: 2 | 3; bounds?: number[]; }
export interface GridIndexResident { start: GPUBuffer; items: GPUBuffer; M: number; n: number; lattice: GridLattice; }
export function encodeGridIndex(points: GPUBuffer, n: number, opts: GridIndexOptions, enc: GPUCommandEncoder): GridIndexResident;
export async function buildGridIndexGpu(points: Float32Array, opts: GridIndexOptions): Promise<BucketGrid | BucketGrid3D>;
```

`encodeGridIndex` is the encoder-level form (the consumer fuses build + query in one
submit, the pattern `encodeScan` set); `buildGridIndexGpu` reads back into the exact
`BucketGrid` struct so `crossPcf.ts` / `tcm.ts` can drop their host build with a one-line
change.

TGSL-callable helper. TGSL has no closures, so the helper is not "for each neighbour, call
f" but the two pieces a consumer loop needs, both `tgpu.fn` with `"use gpu"` bodies:

```ts
/** Cell coordinate of a world position, clamped to the lattice. Integer maths spelled out —
 *  no `/` on u32. */
cellCoord(p: vec3f, lat: Lattice) -> vec3i
/** [lo, hi) range into `items` for cell coordinate c, or (0,0) when c is off-lattice.
 *  Callers loop dz in [-1,1] only when lat.dims == 3, so 2D is 9 cells and 3D is 27. */
cellRange(c: vec3i, lat: Lattice, start: array<u32>) -> vec2u
```

A consumer writes the 3^d loop itself (it is six lines, and `crossPcf.ts` shows it); the
helper owns the index math, the clamp, and the dims switch. Graph-level, the query is
whatever op consumes the three ports — a `nnDistance`, `knn`, or `separate` op gains two
optional inputs (`start`, `items`) plus `lattice` and switches from the all-N loop to the
stencil when they are connected. That keeps the brute-force path as the golden for the
indexed one.

### 3.5 Golden

Compare **per-cell index sets**, not arrays: for every cell `b`, `sort(items[start[b]..
start[b+1]))` equals `sort(cpu.items[cpu.start[b]..cpu.start[b+1]))` with `cpu =
buildBucketGrid(xs, ys, cell, bounds)` and the same origin. Then the structural checks from
the 3D note §5: `start` is the exclusive scan of the counts, `start[M] == n`, every item
lands in the cell its coordinates imply, the ranges partition `[0,n)`. Fixtures: a regular
lattice with one point per cell (counts all ones), points exactly on cell boundaries and at
the corners (pins `floor` + clamp), an empty lattice region, n = 0, and a forced 2-D dispatch
fold through the `maxWorkgroupsPerDim` seam. For 3D, extend `buildBucketGrid` with a `zs`
argument (a twenty-line change) rather than writing a second CPU reference. End-to-end: the
indexed `nnDistance` must equal the brute-force one to 1e-6 for `cell ≥` the true NN
distance of every point — and the test should also include a case where it does not hold,
to show the failure mode is visible.

## 4. The validity / support facet

### 4.1 One facet, named `support`

ADR-0005's `support` and the terrain "nodata/validity" mask are the same object: one weight
per sample in [0,1], where a nodata sentinel is the boxcar 0 and a soft brush is the ramp.
Keep the ADR's name. A nodata source turns its sentinel into a mask with `maskRange`
(`lo = hi = sentinel`, then `invertField`) or a `maskNotEqual` sibling of `maskEquals` — a
leaf op, not a new mechanism. Do not add an `isValid` facet, a `nodata` field on
`FieldValue`, or a NaN convention; NaN in particular is invisible to `min`/`max` and
poisons a tree reduction.

### 4.2 Where it lives

On both `GpuField` (build time) and `FieldValue` (run time), exactly as `placement` does:

```ts
// handle.ts
readonly support?: SupportRef;            // GpuField
support?: FieldValue;                     // FieldValue — a points{numCells(shape)} f32 column, host or resident
// op.ts
inferSupport?(inputs: (SupportRef | undefined)[], params: Params): (SupportRef | undefined)[];
```

`SupportRef` at build time is `{ field: GpuField }` — which mask node gates this value. The
executor stamps `node.outSupports?.[i]` onto the output when the op left it unset, in the
same `forEach` as `outPlacements` (`executor.ts:398–413`), and resolves the reference to a
concrete `FieldValue` (host `data` or resident `buffer`) before `execute`, bridging it with
the same upload/download the inputs get. Absent ⇒ all-valid, today's behaviour, so every
existing op is unchanged.

### 4.3 Propagation rules per op category

The default (no `inferSupport`) is **intersection**: the output's support is `min` over the
inputs' supports of the same shape, and `undefined` if none have one. That is the right
rule for pointwise ops and is wrong, silently, for everything else — which is why the
categories below each declare theirs.

- **Pointwise** (`fieldArithmetic.ts`, `threshold.ts`, `complexOps.ts`): intersection,
  free. The mask is not read by the kernel; the facet just rides along.
- **Stencil / morphology** ([`convolveSeparable.ts`](../src/gpu/spatial/convolveSeparable.ts),
  [`morphology.ts`](../src/gpu/spatial/morphology.ts), `getisOrd.ts`): the kernel must read
  the mask. Min/max **skip** invalid taps (an erosion over a hole must not spread the hole;
  the output is valid iff any tap was). Convolution **renormalises**: run the same separable
  pass over the mask to get the per-texel weight sum `W`, output `Σ w·m·v / W`, and the
  output mask is `W / Σ w` (or the boxcar `W > 0`). This is psychogeo's `block_mean`
  pairing, and it is separable exactly because the mask pass uses the same 1-D kernel.
  Kernels: TGSL, 2 passes × 2 (value, mask) or one fused pass with bindings `params, src,
  mask, wts, dst, dstMask` = 6 ≤ 8; geometry as today (`WG = 64`, one thread per texel).
- **Reduction / decimation** (the N×N→1 op the terrain block wants; `maskCount`): paired
  count. Output value `Σ m·v / Σ m` over the window, output mask `Σ m / N²`; an all-invalid
  window yields 0 with mask 0, never NaN. The op changes grid size, so its `inferPlacement`
  scales `worldFromArray` by N (ADR-0018) and its `inferSupport` decimates the mask the same
  way. TGSL, 1 pass, bindings `params, src, mask, dst, dstMask` = 5; one thread per output
  texel, `dispatchGrid(ceil(wOut·hOut/64))`.
- **Neighbour query** (every consumer of §3): invalid points never enter the grid. This is
  where compaction turns a masked column into a sparse one: `encodeCompact(mask) → indices`
  (gap 2.4.1), then the index is built over the compacted cloud and `items` holds original
  row ids, so results index the original table. Weighted variants (a soft mask as KDE
  weight) are the 0 < m < 1 case of the same path — the mask value rides on `items` as a
  companion f32 column. For a hard mask this is strictly cheaper than weighting: work is
  O(selected), the whole point of ADR-0005's "two encodings chosen by scale".

### 4.4 Storage — companion field on the value, not a second edge

Recommendation: **the facet carries a reference to a mask field; the mask is stored as its
own `FieldValue` and shared by reference.** It is a separate value (one buffer, one lease,
memoised under its own node) but it is not a separate *edge* the user wires — the executor
follows `support` the way it follows `placement`. This is the right split because the mask
is already a field (filterOps), the fuzzy-set ops already combine masks as fields, and
ADR-0017's one-lease-per-port rule forbids packing value and mask into one buffer; and it
avoids the second-edge cost the 3D note's gather and every stencil kernel would pay in ports
(a mask on every input port doubles the port count on binary ops).

Migration from today's `filterOps`: (1) no change to the mask leaves — they keep producing
plain f32 columns; (2) add `attachSupport(value, mask) → value` (sets the facet; pointwise,
`inferSupport` returns the mask ref) and `supportOf(value) → mask` (reads it) so a graph
can move between the explicit and carried forms; (3) `maskedDensity` becomes
`splatDensity` reading `points.support` as its weight — the resident splat gets a second
vertex buffer for the weight rather than a stride-3 zip, which is why that op was kept
Tier-1 until now; (4) `maskCount` gains the resident reduction form (`exclusiveScanGpu`'s
`total` is exactly `Σ m` for a hard mask — a scan with the readback of one word); (5) the
stencil ops add the mask binding behind a feature check on `inputs[0].support`, so an
unmasked graph compiles the same kernel as today and the morphology bit-exact test stays
green.

## 5. Sequencing

Smallest vertical slice first; each step names what it unblocks.

1. **2D index, host-in/host-out, golden against `bucketGrid.ts`.** `cellHistogram` +
   `encodeScan` + `scatter` + the per-cell-set test. Swap `crossPcf.ts` and `tcm.ts` onto it
   (they already consume the layout). Unblocks: nothing user-visible yet, but it proves the
   build and gives a benchmark (`scripts/bench-grid-index.ts`, standalone like `bench-scan`).
2. **Indexed `nnDistance` / `kthNeighborDistance` / `knn`** via `cellCoord`/`cellRange`,
   brute-force as golden. Unblocks: O(N·k) point statistics at the sizes where `knn.ts`'s
   comment says the O(N²) wall is real; CkNN and fuzzy adjacency follow for free since they
   compose `kthNeighborDistanceGpu`.
3. **Resident index op** (three ports, §3.3) and a resident `separate` / `spring` force op
   that reads it. Unblocks the force-directed layout at scale — the springy layout that the
   brief says feeds the displacement-field flagship — and the first resident op chain that
   runs per tick without a host round trip (ADR-0017 stage 3's feedback state owning the
   pooled buffers).
4. **`encodeCompact` + the `support` facet** (§4.2, §4.4 steps 1–4) with the pointwise and
   reduction rules. Unblocks: the mask ⇄ index duality ADR-0005 was written for, the
   weighted null models the toolbox doc plans, MDV's filtered density with zero marshalling.
5. **Nodata-aware stencils and the N×N→1 decimation** (§4.3). Unblocks the terrain chain:
   morphology over a LIDAR raster with holes, bare-earth opening that does not grow the
   holes, block means psychogeo can call instead of its own `block_mean`.
6. **`dims: 3`** once ADR-0004's `points{n, dim}` lands on `Shape` (or via `stride: 3`
   before it) — then the 3D note's `splatVolume` gather becomes a consumer of the same
   three ports and the occupied-cell list is one `encodeCompact` over `counts`.

Promotion to an ADR happens at step 1, and the ADR should absorb the 3D note and this one.

## 6. Open questions

1. **`stride: 3` before `Shape` has `dim`, or wait?** Default: ship `stride` as an op param
   now, validated against `data.length`, and fold it into `dim` when ADR-0004 lands; the
   alternative serialises the index behind a `Shape` refactor nobody has started.
2. **Sparse lattices (M ≫ N).** Default: throw at the binding limit with the number, as the
   scan does; build occupied-cell compaction only when a 3D consumer hits it, since it needs
   gap 2.4.1 and changes the consumer loop (a cell-to-slot lookup).
3. **Deterministic order within a cell?** Default: no — document the contract as a set and
   test it as a set. Revisit only if a consumer's summation order turns out to matter for
   a parity test (tcm.ts already notes its residual *is* summation order).
4. **Should `support` be stamped onto `placement`-less (array-space) grids?** Default: yes;
   the facet is independent of placement, and the terrain rasters arrive array-space.
5. **Soft masks in the neighbour path — compact at `m > 0` and carry the weight, or
   threshold at ½?** Default: compact at `> 0` and carry the weight, because a ramped brush
   that drops half-weighted points jumps exactly the way a hard filter does.
6. **Raise `maxStorageBufferBindingSize` in `getDevice()`?** Default: not in this work. It is
   device-wide and belongs to the MDV integration; `checkBindingSize` makes the ceiling loud.

## References

- [`gpu-spatial-index-3d.md`](gpu-spatial-index-3d.md) — the 3D spec this extends. Two
  details there are stale: it says to read back via TypeGPU `.read()` not `mapAsync`, but
  the Dawn instance-lifetime bug was fixed 2026-07-29 and `prefixSum.ts` reads back through
  a pooled `mapAsync` pinned by `test/dawn-limits-sweep.gpu.test.ts`; and its `opaque` handle
  predates ADR-0017 (§3.3 above).
- [ADR-0003](decisions/0003-use-gpu-tgsl-kernels.md) TGSL vs WGSL template;
  [ADR-0004](decisions/0004-field-type-model-and-volumetric-splat.md) `dim: 2 | 3`, gather-splat;
  [ADR-0005](decisions/0005-columnar-filters-and-sparse-support.md) `support`;
  [ADR-0017](decisions/0017-tier2-resident-buffer-edges.md) leases, one per port;
  [ADR-0018](decisions/0018-field-domains-placement-and-resolution.md) placement, units of spatial params.
- [`mdv-dimension-vs-support-facet.md`](mdv-dimension-vs-support-facet.md) — the operator
  correction and the "build the scan first" call, now discharged.
- [`gpu-spatial-analysis-toolbox.md`](gpu-spatial-analysis-toolbox.md) — `GpuGridIndex` as
  "the foundational new kernel"; [`fuzzy-tda-and-windowing.md`](fuzzy-tda-and-windowing.md)
  — why a mask is a window, and hard is its boxcar.
