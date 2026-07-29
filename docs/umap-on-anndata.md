# UMAP over AnnData matrices — GPU k-NN, animated subsets, and offline `obsm`

Two things are wanted, and they are less similar than they look:

1. **In-browser, animated transitions** between UMAPs of different gene / cell subsets.
2. **Offline processing** that writes an embedding into `obsm` — the project's first
   real work done outside the browser.

They share all their maths (`src/spatial/umap.ts` is the single entry point) and differ
in exactly two places: which k-NN runs, and who owns the epoch loop.

---

## 1. What was already here

More of UMAP existed in this repo than expected, in 2-D spatial form:

| Piece | Existing | Gap |
| --- | --- | --- |
| Fuzzy simplicial set | [`fuzzyAdjacencyAdaptive.ts`](../src/gpu/spatial/fuzzyAdjacencyAdaptive.ts) — per-point bandwidth + probabilistic t-conorm, i.e. UMAP's symmetrisation exactly | σ_i taken as `scale·ρ_i` rather than **calibrated**; 2-D only; dense N×N |
| Local bandwidth ρ_i | [`kthNeighborDistance.ts`](../src/gpu/spatial/kthNeighborDistance.ts) | 2-D only, no neighbour **indices** |
| Self-tuning distance | [`cknn.ts`](../src/gpu/spatial/cknn.ts) — the same density-adaptive idea from Berry & Sauer | 2-D, dense |
| Eigensolver for PCA | [`eigenSym.ts`](../src/spatial/eigenSym.ts) — f64 Jacobi, accurate on small matrices | none; it is exactly right for a G×G gene covariance |
| Headless GPU | [`backend.node.ts`](../src/gpu/graph/backend.node.ts) | see §5 — an Instance-lifetime bug, since fixed |
| Reading `X` | [`varMatrix.ts`](../playground/src/datasource/varMatrix.ts) + [`sparseColumns.ts`](../src/datasource/sparseColumns.ts) | lived in `playground/`, browser-only |
| Writing anything to zarr | **nothing** | the whole write path |

So the work was: generalise to D dimensions, go sparse, calibrate σ, add a layout
optimiser, and build the first zarr writer.

## 2. Why sparse, everywhere

Every pre-existing neighbourhood primitive emits a dense N×N matrix. That is the right
shape for a persistence sweep over a few hundred points and impossible here: 100k cells
is 10¹⁰ floats, 40 GB. [`knn.ts`](../src/gpu/spatial/knn.ts) therefore returns `[N, k]`
index/distance arrays, and [`umapGraph.ts`](../src/spatial/umapGraph.ts) carries a COO
edge list from there on. Nothing downstream ever materialises N×N.

The graph build stays on the host deliberately. It is O(N·k) with k in the tens — 1.5M
edges of arithmetic at N=100k — which is milliseconds, and being on the host lets it be
**bit-faithful to the reference implementation**. That matters because the entire point
of writing into `obsm` is that somebody else compares it to their own scanpy output; a
merely "UMAP-like" graph makes the comparison meaningless. The inherited oddities are
reproduced on purpose and named where they appear:

- `n_neighbors` counts the point itself, so the k-NN is asked for `n_neighbors - 1`.
- The σ bisection targets `log2(n_neighbors)` while summing over `n_neighbors - 1` terms.
- `SMOOTH_K_TOLERANCE = 1e-5`, `MIN_K_DIST_SCALE = 1e-3`.

One consequence worth knowing: σ is only pinned to the width of the tolerance band, not
uniquely. For a point whose neighbours are equidistant the target is only approached as
σ → 0, and bisection halts on the dyadic grid — a 10× rescale of the data can move σ by
8×. The **memberships** are nonetheless scale-invariant to ~5 decimals, and that is what
the tests assert, because that is what downstream reads.

## 3. The k-NN, and the wall at ~20k cells

`knnGpu` is exact brute force, O(N²·D), one thread per query point, k smallest kept in a
private array. Exact-first is the ADR-0016 pattern (reproduce, then improve) and it is
genuinely adequate to ~10–20k cells: at N=10k, D=50 the inner loop is 5·10⁹ FMAs.

Past that the O(N²) wall is real and a bigger GPU does not fix it. The fix is an
approximate index — RP-trees or a generalised bucket grid — dropped in behind the same
`KnnResult` interface, which is why that interface is a named type and the k-NN is
**injected** rather than imported. A full Xenium section is not reachable today.

PCA first, always, past ~50 genes ([`pca.ts`](../src/spatial/pca.ts)). The covariance is
G×G where G is the number of *selected* genes, so the eigenproblem is tiny and
`eigenSym` handles it exactly. Component signs are pinned (largest-magnitude entry made
positive) so two runs produce byte-identical output — otherwise a sign flip makes two
identical `obsm` entries look unrelated.

## 4. The animation: relax, don't tween

**Do not compute two UMAPs and interpolate between them.** A UMAP embedding is defined
only up to rotation and reflection, and it is genuinely unstable under subsetting, so a
tween between two independent runs is mostly meaningless motion — the points move a long
way and none of the movement means anything.

Instead keep **one** embedding and keep optimising it while the graph underneath is
swapped:

```ts
// once
const { graph } = await umapGraphFor(data, n, dim, { nNeighbors: 15, knn: gpuKnn });
let state = initLayout(graph, { nEpochs: 500, seed: 1 });

// every frame
optimizeLayoutStep(state, graph, { nEpochs: 500 });
draw(state.embedding);

// when the user changes the gene or cell subset
const next = await umapGraphFor(subsetColumns(data, n, dim, genes), n, genes.length, opts);
state = initLayout(next.graph, { nEpochs: 500, initialAlpha: 0.3 }, state.embedding); // ← carry the coordinates
```

The embedding then *relaxes* from the old layout into the new one. The animation is the
optimisation, which is cheap (no extra work at all — the SGD was running anyway) and
honest: what you see moving is structure that actually changed. `umapLayout.test.ts`
pins the claim — continuing from a settled layout drifts less than a third as far as a
cold start on the same new graph.

This is why `optimizeLayoutStep` takes and returns state instead of a `fit()` that owns
its loop, and why `initLayout` accepts an existing embedding.

For a **cell** subset the embedding rows must be re-indexed alongside the data;
`subsetRows` returns the mapping for that reason — getting it wrong is the easiest way
to write a scrambled `obsm`.

## 5. Offline: `pnpm umap:obsm`

```bash
pnpm umap:obsm path/to/store.zarr --key X_umap --n-neighbors 15
```

Auto-detects the AnnData group (SpatialData nests it under `tables/<name>`; a bare store
has it at the root), reads `X` or a layer, computes, and writes `obsm/<key>`. Refuses to
overwrite an existing key without `--force`, and refuses a row count that disagrees with
`obs`.

`--obsp` additionally writes the **graph**, which is the more useful half of the
handover: `obsp/connectivities` (the fuzzy simplicial set), `obsp/distances` (the k-NN),
and the `uns/neighbors` block that points scanpy at them. `sc.tl.leiden` clusters on
`connectivities`, so a collaborator can re-cluster on our manifold instead of building
their own and wondering why the labels disagree — and without `uns/neighbors` scanpy
silently recomputes its own graph, so that block is not optional. Validated by loading
both matrices with scipy: `connectivities` comes back symmetric with sorted indices,
`distances` correctly asymmetric with exactly `n * (n_neighbors - 1)` entries.

**Two findings that shaped this.**

*The writer must match the store's zarr format.* `zarrita`'s `create` only emits **zarr
v3** (`zarr.json`), while most AnnData / SpatialData stores in the wild are **v2**
(`.zarray` / `.zattrs`). A v3 array inside a v2 store is a mixed hierarchy that
`anndata.read_zarr` will not load — the write appears to succeed and the result is
unreadable by the tools it exists to feed. So the format is detected and v2 is
hand-written (uncompressed, trailing chunk zero-padded as v2 requires). Verified by
round-tripping through **zarr-python 2.18**, not just by reading back through the same
library that wrote it.

*"Dawn-on-Node is unreliable" was our own bug, and it is fixed.* This is the important
one, because it was load-bearing for the whole non-browser story.

`create([])` returns Dawn's **Instance**, which owns the native event loop and the
mutexes every later device call takes. `device.ts` let it — and the adapter — fall out of
scope the moment `getDevice()` resolved. V8 collected them whenever it chose, and the
N-API finaliser destroyed the Instance out from under a live device; the next dispatch
hit a destroyed mutex and produced `mutex lock failed: Invalid argument` or a segfault.

Being **GC-timing dependent** is exactly why it presented as ambient flakiness. It also
explains a red herring worth recording: a *non-allocating* busy loop never reproduced it
while `knnBruteForceCpu` or writing zarr chunks did — allocation provokes collection, CPU
work alone does not.

The fix is module-level references in [`device.ts`](../src/gpu/device.ts), with a matching
rule: a retained Instance keeps a libuv handle alive, so a CLI must call `releaseDevice()`
**exactly once, as the very last thing the process does**. Releasing mid-run faults on the
finalisers of still-live pooled buffers and pipelines — measured 3/3 crash when released
early, 3/3 exit 0 when released at the end.

What it bought:

| | before | after |
| --- | --- | --- |
| GPU suite | `fileParallelism: false`, 9.5 s | parallel, **3.2 s** |
| `dangerouslyIgnoreUnhandledErrors` | required | **removed** |
| GPU tests | ~5 files lost per run to fork death | **95 pass, deterministic over 3 runs** |
| offline k-NN | quarantined in a child process, CPU default | **in-process, GPU default** |
| 6000 cells x 60 genes | 5.7 s (CPU) | **3.2 s (GPU)**, identical graph |

[`deviceLifetime.gpu.test.ts`](../src/gpu/deviceLifetime.gpu.test.ts) guards the
regression. It provokes GC between dispatches deliberately — a "call the kernel twice"
test would not catch it, because without heap pressure the Instance is never collected.
Removing the references makes it kill the fork, 3 runs of 3.

The Node-22 volta pin was justified by Dawn atexit crashes on Node 24/26, which was
plausibly this same bug from another angle. Worth re-testing; not yet re-tested.

## 6. The layout on the GPU

[`umapLayoutGpu.ts`](../src/gpu/spatial/umapLayoutGpu.ts) runs the SGD edge-parallel over
a **resident** embedding: one thread per edge, coordinates never leave the device between
epochs, one readback per *frame* rather than per epoch. Measured at n=4000 / 75k edges:
**16.55 ms/epoch host → 0.20 ms/epoch GPU, 82x**. In the page at 3000 cells the GPU path
is rAF-capped (960 epochs/s at 8 epochs/frame, i.e. still idle) while the host plateaus
at ~144 regardless of how many epochs are asked for.

It is a stateful handle (`GpuUmapLayout`), not a pure function, because the animation
model is one embedding that keeps being optimised — a `layout(graph) -> coords` signature
would upload and download every frame, which is the cost this exists to remove.

**It races, deliberately.** Threads for edges sharing an endpoint read-modify-write the
same position unsynchronised — the Hogwild! regime every GPU UMAP uses. So output is not
reproducible run to run even at a fixed seed, and the tests assert `trustworthiness`
rather than coordinates. The offline `obsm` path deliberately keeps the **host** SGD: a
written `obsm` should be reproducible. The page offers both, which is also the cheapest
way to confirm the racy kernel agrees with the exact one.

One bug worth recording, because it was silent: batching every epoch's dispatch into a
single command buffer looks like an obvious win and is wrong. The per-epoch uniform is
written with `queue.writeBuffer`, so all of those writes land before the single submit
and every pass runs with the *last* epoch's parameters. The layout still moves and still
looks plausible — trustworthiness just collapses to ~0.49, barely above random.

## 7. Testing notes

The GPU suite is deterministic and runs files in parallel. The only failing files are
those importing `rust/htj2k-core/pkg`, which need `pnpm build:wasm` first.

The layout SGD is checked on **trustworthiness**, not coordinates. A parallel GPU
implementation races against a shared position buffer (the Hogwild! regime every GPU UMAP
uses) and cannot reproduce host coordinates; what it must reproduce is a layout preserving
the same neighbourhoods.

`knnGpu` and `knnBruteForceCpu` are drop-in substitutes and are diffed directly, which
also gives the offline path a cheap cross-check: `--cpu` and the default must produce the
same graph (they do — 139790 edges on the 6000-cell fixture either way).

## 8. Not done

- **Approximate k-NN** (§3) — the blocker for a full section.
