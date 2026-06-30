# Spec — a 3D uniform-grid spatial index (GPU primitive)

Status: **proposed / planned** (2026-06-29)

This is a **design + interface + algorithm + test plan** for the 3D uniform-grid
spatial index. It is a spec, **not an implementation** — no production `.ts` kernel
is written here. It grounds the "build it first" call from
[ADR-0004](decisions/0004-field-type-model-and-volumetric-splat.md) and slots into
the discrete-cell front described in
[`gpu-spatial-analysis-toolbox.md`](gpu-spatial-analysis-toolbox.md), generalising
the index that doc already names as *"the foundational new kernel … the analogue of
the line kernel for this front"* from 2D to 3D.

## 1. Motivation & role

This index is **load-bearing under two distinct things**, and both already exist as
commitments elsewhere in the codebase:

**(a) Volumetric gather-splat (points → voxel density).** ADR-0004 settled that the
2D additive-render splat ([`splatDensity.ts`](../src/gpu/spatial/splatDensity.ts)) —
which draws each point as an instanced quad and lets the fixed-function **blend unit**
accumulate overlaps into an `r32float` target, *no atomics* — **does not generalise
to 3D**: there is no render-blend into a 3D texture, and the slice-stack workaround is
exact only for *separable* kernels (a radial Epanechnikov, the spatial front's
preferred compact-support kernel, is not separable). For the general case, and
decisively for **gene-tensor volumes** (element `scalar` + an open `gene` axis of
length G), the only sane path is **gather**: one thread per voxel pulls points from
its neighbouring cells and accumulates the whole G-vector **in registers / workgroup
memory, written once, coalesced**. The reason gather beats scatter here is sharp —
scatter needs an **atomic per channel per voxel**, so a G-gene volume is G atomics per
footprint-voxel, a non-starter (and core WGSL has no f32 atomics at all; see §3). The
index is what makes "for this voxel, which points are near?" answerable in O(1)
cells instead of O(N) points. **Render-blend has no 3D equivalent; the index *is* the
3D replacement for the additive-render splat.**

**(b) The compaction kernel shared with columnar filters / sparse support.** Building
the index *is* a **prefix-sum (scan)** plus a **counting-sort scatter**. The spatial
toolbox doc already flags this as high-leverage beyond neighbour search: *"prefix sum
(scan) and counting sort / scatter … histograms, compaction, stream partition all
fall out."* The same scan+sort that buckets points into cells is the compaction
primitive a columnar filter (keep rows where predicate) or a sparse-volume
materialiser (list occupied voxels only) needs. We build it once and three callers
share it.

So this primitive sits beneath the gather-splat, beneath the discrete-cell front's
neighbour queries, and beneath generic compaction. It is the 3D counterpart of the
2D `GpuGridIndex` the toolbox already planned, and the thing the volumetric ADR points
at when it says *"the 3D uniform-grid spatial index is the load-bearing primitive …
build it first."*

## 2. Algorithm

Uniform-grid bucketing via **histogram → scan → counting-sort scatter**, then a
**gather** pass for the consumer (splat / neighbour query). The grid is an axis-aligned
lattice of cubic cells over the point cloud's bounding box.

### Geometry & indexing math

Given an `origin` (the min corner of the box), a scalar `cellSize`, and integer
`gridDims = (Dx, Dy, Dz)`, a point at world position `p = (px, py, pz)` maps to an
integer **cell coordinate**:

```
cx = floor((px - origin.x) / cellSize)
cy = floor((py - origin.y) / cellSize)
cz = floor((pz - origin.z) / cellSize)
```

clamped to `[0, Dx)`, `[0, Dy)`, `[0, Dz)`. The **linear cell id** is x-fastest
(z-major), so the z-neighbour stride is the whole `Dx·Dy` plane:

```
cellId = cx + Dx * (cy + Dy * cz)            // in [0, Dx·Dy·Dz)
```

**Cell size = kernel support radius.** Choosing `cellSize ≈ R` (the kernel's
truncation radius, e.g. `radiusSigma·σ`, or the exact finite support of an
Epanechnikov / tricube kernel) guarantees a point's footprint touches at most a fixed
**3×3×3 = 27-cell neighbourhood**. This is the whole point of a uniform grid: a query
never scans more than 27 cells regardless of N. The distance-decay argument from the
toolbox doc ("truncating at a few σ is a principled approximation with a closed-form
error bound, not a corner cut") is what makes the bounded 27-cell stencil exact-enough
for Gaussians and *exactly exact* for compact-support kernels.

### The four build passes

Let `M = Dx·Dy·Dz` be the cell count and `N` the point count.

1. **Cell-index + histogram.** One thread per point computes its `cellId` (above) and
   does an `atomicAdd(counts[cellId], 1u)`. Output: `counts[M]` (u32). This is the
   only scatter, and it is an **integer** atomic (legal in core WGSL; see §3). Store
   each point's `cellId` to a scratch `pointCell[N]` so pass 4 needn't recompute it.

2. **Exclusive prefix-sum (scan)** of `counts[M]` → `cellStart[M]` (u32), the
   per-cell start offset into the sorted array. `cellStart[c] = Σ_{c'<c} counts[c']`,
   and `cellStart[c] + counts[c]` is the (exclusive) end. A work-efficient
   Blelloch scan (up-sweep / down-sweep) over workgroup-shared tiles, with a
   second-level scan over per-workgroup totals for `M` beyond one workgroup. Keep a
   copy of the exclusive offsets as a mutable `cellCursor[M]` for the next pass.

3. **Counting-sort scatter.** One thread per point: `slot = atomicAdd(cellCursor[pc], 1u)`
   where `pc = pointCell[i]`, then `sortedPointIds[slot] = i`. After this pass,
   `sortedPointIds[cellStart[c] .. cellStart[c]+counts[c])` are exactly the point ids
   in cell `c`. This is the classic **CSR-like layout**: `cellStart[]` is the row
   pointer, `sortedPointIds[]` is the column index. (Sorting *ids*, not coordinates,
   keeps the scatter narrow and lets the consumer fetch any per-point attribute by id.)

4. **Gather** (the consumer pass — splat or neighbour query). One thread per **voxel**
   (or per query point). Compute the voxel's cell coord, loop the 27 neighbour cells
   `(cx+dx, cy+dy, cz+dz)` for `dx,dy,dz ∈ {-1,0,+1}`, skip out-of-range cells, and for
   each in-range cell `c` iterate `sortedPointIds[cellStart[c] .. cellStart[c]+counts[c])`,
   accumulating the kernel contribution `w_p · K(‖voxelCentre − p‖)` into a register
   (scalar) or a small register/workgroup tile (the G-vector for a gene volume).
   **One coalesced write per voxel, no atomics.** This is the pass ADR-0004 describes
   as "accumulates the whole G-vector in registers … written once."

```
points ──cellId+atomicAdd──▶ counts[M] ──scan──▶ cellStart[M]
   │                                                  │
   └────────────counting-sort scatter────────────────┘
                       │
                       ▼
              sortedPointIds[N]   (CSR: cellStart + sortedPointIds)
                       │
   voxels ──27-cell gather over ranges──▶ density / G-vector  (one write/voxel)
```

### Data structures (the index handle)

```
SpatialIndex3D = {
  cellStart:      Uint32Array(M)   // CSR row pointer: start offset per cell
  counts:         Uint32Array(M)   // points per cell (cellStart[c]+counts[c] = end)
  sortedPointIds: Uint32Array(N)   // CSR column index: point ids grouped by cell
  gridDims:       [Dx, Dy, Dz]
  cellSize:       number
  origin:         [ox, oy, oz]
}
```

A consumer needs only `cellStart`, `counts`, `sortedPointIds`, plus the geometry
(`gridDims`, `cellSize`, `origin`) to recompute any voxel's cell coord. The point
coordinates themselves stay in the caller's packed `points` buffer (see §4); the index
adds no copy of them.

### Empty-cell sparsity optimisation (large sparse volumes)

For a large but sparse volume — biology's common case, a thin tissue slab in a big
bounding box — most of the `M` cells are empty, and `M` can dwarf `N`. Two costs scale
with `M` rather than `N` if left naive: the scan (over all `M` counts) and the gather
(one thread per voxel, including empty regions). The fix is to **compact the list of
occupied cells** so work tracks point density, not volume size:

- After the histogram, stream-compact the cells with `counts[c] > 0` into an
  `occupiedCells[]` list (itself a scan + scatter — the **same compaction primitive**
  from §1(b), reused). The scan can then run over occupied cells only; the gather can
  dispatch one thread per **occupied** voxel-neighbourhood rather than per voxel of the
  full lattice.
- This is also the seam to **ADR-0004's dense-vs-lazy fork**: the gather/index path
  "yields the *occupied* voxels naturally," so a gene-density volume is preferably a
  **lazy field only ever sampled / fused into its consumer**, never a stored dense
  grid (128³ × 200 genes × f32 ≈ 1.7 GB — must not be materialised at scale). The
  occupied-cell list is exactly the support set a lazy volume iterates.

This keeps the index **output-sensitive**: cost ≈ O(N + occupied), not O(N + M).

## 3. Kernel authoring (TGSL vs WGSL templates)

Per [ADR-0003](decisions/0003-use-gpu-tgsl-kernels.md), kernel authoring is a
two-track choice decided by **whether the kernel needs workgroup shared memory,
barriers, or atomics** — TGSL (`"use gpu"`) in TypeGPU 0.11.x does not cover those
cleanly, so they stay as **WGSL templates** (`resolveWithContext({ template, externals }`),
the `kthNeighborDistance.ts` pattern).

Mapping the four passes onto that rule:

| Pass | Needs | Track |
| --- | --- | --- |
| 1. cell-index + histogram | integer `atomicAdd` into `counts[]` | **WGSL template** |
| 2. prefix-sum (scan) | `var<workgroup>` tiles + `workgroupBarrier()` | **WGSL template** |
| 3. counting-sort scatter | `atomicAdd` into `cellCursor[]` | **WGSL template** |
| 4. gather (splat / query) | register/workgroup accumulation, **no atomics** | **TGSL `"use gpu"`** (eligible) |

So the **scan / counting-sort / histogram are WGSL templates** (atomics + shared
memory + barriers), and **the gather kernel may be authored in TGSL** if it needs no
atomics — which, by design, it does not (one write per voxel). The gather is the same
shape as `nnDistance.ts`'s per-point loop, just looping over the 27-cell ranges instead
of all N, so it fits TGSL the way `nnDistance` does.

Hard constraints to honour (all from ADR-0003 / the toolbox constraints section):

- **No f32 atomics in core WGSL.** Counts use **integer atomics** (`atomic<u32>`).
  There is no float accumulation in the index build at all — float accumulation
  happens only in the gather, and there it is in private registers, not shared/atomic.
  (This is the same constraint that pushed 2D density onto the render-blend path; in 3D
  there is no render path, so we sidestep it by *gathering* instead of *scattering*
  floats.)
- **Cell-index range.** `cellId` fits in `u32` for any realistic lattice; the toolbox
  notes ≤2048² 2D bins fit in i32, and 3D `M` up to ~hundreds of millions still fits
  u32. No 64-bit-int dependency. Very large domains coarsen the lattice or tile.
- **Dawn-on-Node teardown discipline.** Build the **layout-bound pipeline once**
  (`resolveWithContext` → pipeline against a `bindGroupLayout`, *not* the guarded
  `"use gpu"` pipeline that closes over buffers and rebuilds on growth — that
  segfaulted Dawn's exit teardown). **Pool buffers and grow them, never `.destroy()`**
  mid-process. **Read results back via TypeGPU `.read()`, not a raw `mapAsync`** on a
  pooled `MAP_READ` buffer (which crashed the vitest worker on teardown). **Keep ~1
  heavy scenario per gpu test file** — "enough" GPU work segfaults the exit teardown
  regardless, so large-N belongs in the browser harness (§5).

## 4. Interface

### Standalone builder function

A raw builder, the sibling of `nearestNeighborDistancesGpu` / `splatDensityGpu`, living
at `src/gpu/spatial/spatialIndex3d.ts`:

```ts
export interface SpatialIndex3DOptions {
  /** Min corner of the lattice in world units. Default: points' min corner. */
  origin?: [number, number, number];
  /** Cubic cell edge length in world units. Default: ≈ kernel support radius R,
   *  so a footprint touches a fixed 3×3×3 neighbourhood. */
  cellSize: number;
  /** Lattice extent in cells. Default: derived from points' bbox / cellSize. */
  gridDims?: [number, number, number];
  /** Compact the occupied-cell list (output-sensitive cost). Default true. */
  compactOccupied?: boolean;
}

export interface SpatialIndex3D {
  cellStart: Uint32Array;      // length M = Dx·Dy·Dz
  counts: Uint32Array;         // length M
  sortedPointIds: Uint32Array; // length N
  occupiedCells?: Uint32Array; // present iff compactOccupied
  gridDims: [number, number, number];
  cellSize: number;
  origin: [number, number, number];
}

/** Build a 3D uniform-grid index over a packed xyz point cloud.
 *  `points` is `[x0,y0,z0, x1,y1,z1, ...]` (3·N f32). */
export async function buildSpatialIndex3D(
  points: Float32Array,
  opts: SpatialIndex3DOptions,
): Promise<SpatialIndex3D>;
```

This matches the `points`/`grid` model in [`handle.ts`](../src/gpu/graph/handle.ts):
2D `points` pack as `[x0,y0,x1,y1,...]` (`unpackPoints`); the **planned 3D `points`**
(ADR-0004 gives `points` a `dim: 2 | 3`) pack as `[x0,y0,z0,...]`, and `grid` grows to
a 2-or-3 tuple. `buildSpatialIndex3D` consumes the `dim:3` packing directly. The index
itself is **not a numeric field** — it is a bundle of CSR buffers + geometry — so as a
graph value it is an **`opaque`-shaped** payload (the same shape-kind `handle.ts`
reserves for "a non-numeric payload … `name` tags the concrete type"), e.g.
`{ kind: "opaque", name: "spatialIndex3d" }`.

### Graph op(s) in the registry

Following the [`splatDensity.ts`](../src/gpu/graph/ops/splatDensity.ts) /
[`kthNeighborDistance.ts`](../src/gpu/graph/ops/kthNeighborDistance.ts) adapter pattern
(a Tier-1 `OpType` that marshals `FieldValue`s in/out and delegates to the legacy
function), two ops cover the two roles:

**(i) `buildIndex3D`** — `points` input → an `opaque` index output (explicit index,
reusable by several consumers in one graph):

```ts
export const buildIndex3DOp: OpType = {
  name: "buildIndex3D",
  label: "3D spatial index",
  describe: "Bucket a 3D point cloud into a uniform-grid CSR index (scan + counting-sort).",
  inputs: [{ name: "points", kind: "points" }],         // dim:3 packed xyz
  outputs: [{ name: "index", kind: "opaque", dtype: "u32" }],
  params: [
    { name: "cellSize", type: "number", default: 1, min: 1e-3, max: 1e6, step: 0.1,
      describe: "cell edge ≈ kernel support radius (world units)" },
    { name: "compactOccupied", type: "bool", default: true,
      describe: "output-sensitive: skip empty cells" },
  ],
  inferShapes() { return [{ kind: "opaque", name: "spatialIndex3d" }]; },
  async execute(_ctx, inputs, params) {
    const idx = await buildSpatialIndex3D(inputs[0]!.data as Float32Array, {
      cellSize: params.cellSize as number,
      compactOccupied: params.compactOccupied as boolean,
    });
    return [{ shape: { kind: "opaque", name: "spatialIndex3d" }, dtype: "u32", payload: idx }];
  },
};
```

**(ii) `splatVolume`** — the gather consumer, `points` (+ optionally a precomputed
`index`) → a 3D `grid` density / gene-volume output. This is the 3D analogue of
`splatDensityOp`, but it **gathers off the index instead of additive-rendering**. Port
& param sketch:

- inputs: `{ name: "points", kind: "points" }`, optional `{ name: "index", kind: "opaque" }`
  (if absent, `splatVolume` builds one internally with `cellSize = radiusSigma·σ`).
- outputs: `{ name: "volume", kind: "grid", dtype: "f32" }` — a 3D grid (`gridDims`),
  element `scalar`, optionally with a `gene` tensor axis (ADR-0004) when the points
  carry per-point G-vectors.
- params: `sigma` (bandwidth, world units), `radiusSigma` (support in σ; sets
  `cellSize`), `gridDims` (or derive from bbox), `kernel` (`enum`: `gaussian` /
  `epanechnikov` / `tricube` — compact-support kernels make the 27-cell stencil
  *exactly* exact).

Splitting build from gather lets one index feed several consumers (a gene-volume
splat *and* a proximity-network query) in a single graph without rebucketing — the
shared-compaction payoff from §1(b) made concrete in the op registry.

## 5. Test plan / golden

Correctness is **N-independent** (the index is exact bucketing; the gather is an exact
finite sum over the 27-cell stencil), so small-N CPU-golden tests in Node fully
validate the kernel, and large-N is purely a performance/stability concern for the
browser harness — mirroring `nnDistance` / `splatDensity`, where Node's Dawn teardown
caps point counts (§3) but small-N still pins correctness.

**CPU golden.** A pure-JS reference in the `*.gpu.test.ts` file (the `cpuGolden`
pattern): bucket the points by the same `floor((p-origin)/cellSize)` math into a
`Map<cellId, number[]>`, then for each voxel **brute-force gather** — loop all points,
keep those within R, sum `w·K(d)` — and compare against the GPU index+gather
(≤1e-3, the repo's bit-close tolerance). Validate the index structure directly too:
`cellStart` is a correct exclusive scan of `counts`; every `sortedPointIds` entry lands
in the cell its coordinates imply; the per-cell ranges partition `[0,N)`.

**Concrete fixtures.**

1. **Known regular grid of points** — e.g. points on a `5×5×5` lattice at spacing `s`,
   `cellSize = s`. Each cell holds exactly one point; `counts` is all-ones, `cellStart`
   is `0,1,2,…`, and each voxel's 27-cell gather sees a predictable, hand-checkable set
   of neighbours. Pins the indexing math and the scan with no float ambiguity.
2. **Single Gaussian blob → known voxel density** — one point (or a tight cluster) at a
   known centre; the gathered density at each voxel must equal `w·exp(−d²/2σ²)`
   (Gaussian) or the closed-form Epanechnikov value, sampled at voxel centres. Closed
   form, so it pins the gather kernel and the kernel-support truncation.
3. **Sparsity fixture** — a handful of points in a large `gridDims`, `compactOccupied`
   on: assert `occupiedCells.length` equals the number of non-empty cells and that the
   compacted gather matches the dense gather (output-sensitivity is correct, not just
   faster).
4. **Edge/clamp fixture** — points exactly on cell boundaries and at the lattice corners,
   to pin the `floor` + clamp and the out-of-range neighbour skipping in pass 4.

**Harness split.** Node: small-N (≤ a few hundred points, ≤ ~32³ lattice) for
correctness, one heavy scenario per file (Dawn teardown ceiling). Browser harness:
large-N / large-lattice for throughput and stability, where correctness is already
established and only performance is in question.

## 6. Open questions / future

- **2D reuse.** The same scan + counting-sort *is* the 2D `GpuGridIndex` the
  [spatial toolbox](gpu-spatial-analysis-toolbox.md) already planned — the
  discrete-cell front's **uniform-grid proximity network** (Fig 3F cell-cell
  interaction network, without Delaunay) and the **Monte-Carlo null models** (the
  flagship: keep points resident, recompute the statistic over permutation replicates
  in one extra dispatch dimension) both want exactly this index, dropped to 2D
  (`dim:2`, `gridDims = (Dx,Dy)`). Worth factoring the build as `dim`-parametric so 2D
  and 3D share one kernel rather than forking. The 2D index unblocks cross-PCF, ANNI
  nulls, and contact networks at index-accelerated O(N·k) instead of brute-force O(N²).
- **Anisotropic cells.** `cellSize` is currently one scalar (cubic cells). Tissue is
  often anisotropic (thin z, wide xy); a per-axis `cellSize = (sx,sy,sz)` with a
  matching anisotropic kernel support would let the stencil stay 3×3×3 while matching
  the data's aspect. Changes only the cell-coord math, not the CSR machinery.
- **Periodic boundaries.** For toroidal / periodic domains (some simulation and
  null-model contexts), the 27-cell neighbour loop would wrap `(cx±1) mod Dx` instead
  of clamping. A build-time flag; affects only pass 4's neighbour enumeration.
- **Dense vs lazy volume materialisation** (the ADR-0004 fork). The occupied-cell list
  makes a gene-density volume naturally **lazy** (sampled / fused into its consumer,
  never stored dense — mandatory at the 1.7 GB scale). Open: where the lazy-pull
  executor draws the line between materialising a small interactive pull and
  fusing/streaming a large one, and whether the gather op exposes a "sample at these
  query points" mode (lazy) alongside a "fill this dense sub-block" mode (eager) behind
  the same op. This is the scale-equivariance seam ADR-0004 elevates to a design goal.
- **Index reuse across frames.** For interactive / animated point clouds, can the index
  be incrementally updated rather than rebuilt? Likely not worth it (a full rebuild is
  cheap and the toolbox already keeps points resident), but worth a note before
  committing to rebuild-every-frame.
