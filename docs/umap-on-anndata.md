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

## 3. The k-NN, and where it actually runs out

`knnGpu` is exact brute force, O(N²·D), one thread per query point. Exact-first is the
ADR-0016 pattern, and with the tiling below it reaches further than expected: **n=50000,
D=50 in 14.6 s**.

**Tiling is a correctness fix, not a tuning knob.** One dispatch covering every query row
is O(n²·D) of work in a single command, and past roughly two seconds the OS GPU watchdog
kills it — Dawn reports *no error*, the output buffer keeps its zeroes, and the caller
gets a complete-looking result whose every index is 0. Measured: n=26000 and n=30000 came
back entirely zero (recall 0.000) while n=28000 happened to survive, so it presented as an
intermittent wrong answer rather than a failure. Dispatches are now bounded to 4·10⁹
row×column×dimension products — about 3× below the largest dispatch observed to complete —
and `knnInvariants.gpu.test.ts` pins the tiled path.

[`knnDescent.ts`](../src/spatial/knnDescent.ts) is the approximate path: NN-descent
(Dong, Charikar & Li 2011), on the observation that a neighbour of my neighbour is a good
candidate for being my neighbour. Cost is `maxIters · n · candidates²` rather than n², and
recall is measured rather than assumed (`knnRecall`).

**The crossover depends on which exact search you are racing** (`pnpm umap:bench`, Apple
M2 Max, idle — see the note below on why that matters):

| n | host exact | host descent | `knnGpu` (tiled) |
| --- | --- | --- | --- |
| 3000 | 551 ms | 938 ms | — |
| 6000 | 2197 ms | 1814 ms | — |
| 16,000 | — | 2.7 s | **255 ms** |
| 50,000 | — | 9.5 s | **2.5 s** |
| 100,000 | — | 17.7 s | **14.0 s** |
| 100,000 (D=51) | — | 39.8 s | 103.8 s |
| 200,000 | — | 50.1 s | — |

Against a *host* brute force descent wins from about 5–6k points. Against the tiled
`knnGpu` the exact search holds on much longer at low dimension — still ahead at 100k
cells — but D matters as much as n: at D=51 the quadratic is already four times slower at
100k, and D≈50 is what the real pipeline produces.

**Benchmark on an idle machine.** An earlier version of this table was taken while the GPU
had other work on it, and every figure in it was wrong by up to 3× in both directions —
including some that were fast because the kernel had been silently killed and returned
zeros. On an idle machine the same measurements repeat to within 2%. Timings taken under
contention are not conservative, they are meaningless.

**NN-descent now runs its local join on the GPU** ([`knnDescentGpu.ts`](../src/gpu/spatial/knnDescentGpu.ts)),
and that is the change that lifts the ceiling — it replaces the O(N²·D) term with a fixed
number of O(N·c²·D) passes, where `c` is the candidate width (~21) rather than N:

| n | dim | exact (GPU) | descent (host) | descent (GPU) | recall |
| --- | --- | --- | --- | --- | --- |
| 16,000 | 18 | 255 ms | 2.7 s | **445 ms** | 0.983 |
| 50,000 | 18 | 2.5 s | 9.5 s | **1.4 s** | 0.966 |
| 100,000 | 18 | 14.0 s | 17.7 s | **2.7 s** | 0.951 |
| 100,000 | 51 | 103.8 s | 39.8 s | **3.9 s** | 0.798 |
| 200,000 | 51 | — | 94.5 s | **7.7 s** | — |

6–12× over the host descent, and **27× over the exact search at n=100k, dim=51** — which is
the realistic shape, since the pipeline reduces to 30–50 principal components. A full
200k-cell section is 7.7 s.

**It parallelises cleanly for a reason worth stating.** The textbook local join updates both
endpoints of every pair, which on a device means many threads writing one point's list. The
host implementation here never did that: thread `i` walks its own candidates and their
candidates and offers them **only to `i`'s own list**. So there are no atomics, no locks and
no races — and the kernel can be held to a far stronger contract than any other GPU code
here. `knnDescentGpu.gpu.test.ts` asserts the device result is **element-for-element
identical** to the host's, which is the opposite of `umapLayoutGpu`, where the racy kernel
can only be tested statistically.

At large n the two do drift apart (81% per-slot agreement at 100k), because the device
compares squared distances and the host rooted ones; one differently-ordered near-tie
changes a list, which changes the candidate lists built from it. Recall is unmoved — 0.951
on both — so recall is the property to hold it to at scale, and exact agreement the one to
hold it to in tests.

**Is the approximation good enough? Measured, not assumed.** At n=50,000, dim=51, recall
0.867, the final embedding scores **trustworthiness 0.9486 against the exact search's
0.9484** — indistinguishable. That is the claim this document previously asserted about
UMAP tolerating recall loss; it is now a measurement.

Recall does fall with n at a fixed `maxIters`, and more passes only partly buy it back:
at n=100k, dim=51 it goes 0.798 → 0.835 → 0.853 → 0.860 for 12 → 20 → 30 → 45 iterations,
plateauing. Closing the rest would need a wider candidate list or better seeding, and on the
evidence above it is not worth doing for UMAP.

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
plausibly this same bug from another angle. **Re-tested 2026-08-01: it was.** Full
`pnpm test` (78 CPU files / 732 tests, 48 GPU files / 131 tests) passes on **24.18.0 and
26.5.0**, and `pnpm bench:readback` — the heaviest GPU work in the repo, ending in
`releaseDevice()`, i.e. exactly the atexit path — exits 0 on **9 of 9** runs across 22/24/26.
Nothing now justifies the pin.

One caveat found while testing, and it is *not* about Node: `umapLayoutGpu.gpu.test.ts`
fails about **3 runs in 12 on every version including the pinned 22**. The edge-parallel
SGD races by design, so `gpuTrust` is nondeterministic, and the `hostTrust - 0.05`
threshold sits inside the spread (observed failures 0.872–0.911 against a 0.912 bar).
A 5-run sample showed 0/5 on Node 22 and 1/5 on 26, which reads as a Node 26 regression
and is not one — worth remembering before attributing a marginal statistical test to a
toolchain change.

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

Two bugs worth recording, because both were silent and both reported healthy numbers
while producing a broken layout.

**Batching epochs into one command buffer.** An obvious win, and wrong. The per-epoch
uniform is written with `queue.writeBuffer`, so all of those writes land before the single
submit and every pass runs with the *last* epoch's parameters. The layout still moves and
still looks plausible — trustworthiness just collapses to ~0.49, barely above random.
Hence one submit per epoch, deliberately.

**The 65535-workgroup limit.** `maxComputeWorkgroupsPerDimension` is 65535, so a 1-D
dispatch of one thread per edge covers at most 65535 × 64 ≈ 4.19M edges. A branching
manifold at 200k cells produces 4.46M, and past the cap Dawn *invalidates the command
buffer*: the kernel never runs. The tells were an epoch that got **faster** as the graph
grew — 0.17 ms at 4.46M edges against 1.58 ms at half that — and trustworthiness at 0.500.
Dawn does print a validation message, but a benchmark that only reads timings will happily
tabulate the result. The dispatch is now a 2-D grid of workgroups folded back into one edge
index in the kernel, which costs nothing and raises the ceiling to 65535² × 64 workgroups.
After the fix, 200k cells / 4.5M edges runs at 2.80 ms/epoch with trustworthiness 0.933.

This is the same family as the watchdog bug in §3: **on the GPU, "suspiciously fast" is a
correctness signal.** Both were found by benchmarking at a size nothing had been run at
before, which is the argument for `pnpm umap:bench` existing at all.

## 7. What the page shows, and why the shapes changed

The page used to generate isotropic Gaussian blobs. Blobs are the one case where any
projection works, so the embedding settled into coloured dots and there was nothing to
watch. [`syntheticManifolds.ts`](../src/spatial/syntheticManifolds.ts) replaces them with
nine generators chosen so the embedding has something to get right or wrong: a branching
trajectory, a swiss roll, linked rings, a hollow sphere, clusters-of-clusters, cycling
types, a 1% rare population, blobs (kept as the baseline), and uniform noise as a null
control.

Each emits a low-dimensional **latent** plus ground truth, and `expressManifold` maps each
latent axis onto a block of genes. A gene programme is therefore exactly one latent axis,
so the page's toggles delete a dimension of the true manifold and the embedding relaxes
into what is left — the same control works identically on real data, where the blocks are
principal components instead.

**The finding that shaped all of them: UMAP shatters clean low-dimensional manifolds.** A
trajectory drawn as a near-exact curve comes out as a few hundred disconnected beads, each
internally perfect (trustworthiness 1.000) and globally scattered. This was checked against
umap-learn on the identical matrix, and it produces the same picture — median edge length
0.199 against our 0.192, extent 33 against 33, largest connected piece 129 points in both,
with spectral init and with random. It is not our optimiser, and it is not the
initialisation: seeding the layout with the *true* tree still shatters.

So the generators give their continua the width real data has — cells scatter around a
trajectory, they do not sit on it. `branching` at noise 0.45 is a clean three-armed tree;
at 0.10 it is beads. Two related corrections came out of the same exercise, both recorded
in the module:

- The swiss roll's sheet is **square**. Deriving its height from `n` to hold sampling
  density fixed produced a 20:1 ribbon, and a ribbon is globally a 1-D object, so it
  shattered for exactly the reason above.
- The roll is sampled uniformly in **arc length**, not in angle. Uniform in angle makes the
  outer turns ~3× sparser, so the graph short-circuits across the gap at the outside of the
  roll while the inside is fine.

**The null control does not do what folklore says.** Uniform noise was expected to come out
as an archipelago of fake cell types; at 6, 30 and 80 latent dimensions it comes out as a
featureless disc every time. The clumping UMAP is criticised for is real, but it is what
happens to clean low-dimensional *structure*, not to noise. The generator's description and
the page caption say so, and `syntheticManifolds.test.ts` pins it — largest connected piece
0.999 for the null against 0.200 for `blobs`.

## 8. Real data in the browser

The same page loads an AnnData table out of a SpatialData store
([`umapSource.ts`](../playground/src/datasource/umapSource.ts)): genes → `log1p` → highly
variable → PCA → graph, with the principal components becoming the feature-block toggles.
Both sources produce the same `UmapDataset`, so nothing downstream knows which it holds —
which is the point, since the question worth asking is whether the behaviour you learn to
read on the synthetic shapes still reads the same way on real data.

It is a *selection*, not a full load (ADR-0005): a dense `[nCells, nGenes]` matrix at 100k
cells and 2000 genes is 1.6 GB as f64, so the gene count is capped. When a panel exceeds
the cap the genes are drawn as a deterministic **random sample** rather than the first N —
var order in a store is alphabetical or arbitrary, so "the first 400" is a biased slice
dressed up as a default. What the cap does *not* bound is the zarr read: CSR scatters each
gene across every cell's row, so 20 genes cost the same full scan as 2000.

**Cells are capped too, and that came out of trying it.** Pointed at a 162,254-cell Xenium
table, the first version read and reduced the whole thing in 11.5 s and then sat in an
exact GPU k-NN for over two minutes with the page looking hung. With NN-descent on the
device that search is seconds, so `maxCells` (default 100,000, a uniform deterministic
subsample) is now set by the READ and the memory it costs rather than by the search. The
subsample happens *after* the variance pass, so highly-variable genes are still chosen from
every cell — the gene selection is cheap to do properly and is what a subsample would bias
most.

**The k-NN selector's `auto` is calibrated in the browser, and that is not the same
calibration as offline.** NN-descent reads its neighbour lists back once per pass, because
the next pass's candidate lists are built on the host. In Node a buffer map is nearly free;
in Chrome it is a round trip that costs more than the kernel it follows. The first version
of the page read three buffers per pass and a 32k-cell build took **11.3 s** against ~1 s in
Node; dropping the two that are never used between passes — the distances, which
`buildCandidates` does not read, and the change counts, which only decide early exit and are
now sampled every fourth pass — brought that to **4.9 s**. What remains is one map per pass,
and closing it properly means building the candidate lists on the device too.

So `auto` switches on **predicted exact work** (`n² × features`) rather than a cell count,
with the threshold set from measurements taken *in the page*: at 32k cells × 42 features the
exact search wins (3.3 s against 4.9 s); at 60k × 30 it loses badly (9.8 s against 2.9 s).

End to end on that store: 60,000 of 162,254 cells, 150 genes, 30 PCs, k-NN and graph in
2.9 s, **16 s from click to a running layout**.

### 8.1 PCA on the device

With the k-NN dealt with, the reduction became the slowest thing in the load, and the worst
kind of slow: `pca` is synchronous, so its cost is not "the page takes a while" but "the
page is frozen". Loading 100,000 cells × 377 genes from the Xenium table blocked Chrome's
main thread for **14.4 seconds in one unbroken block**.

Two of PCA's four steps are O(N·G·…) and both are now tiled matmuls in
[`pcaGpu.ts`](../src/gpu/spatial/pcaGpu.ts) — the covariance `CᵀC` and the projection onto
the retained components. Measured on an M2 Max, 30 components, standardised:

| n × G | cov host | cov gpu | proj host | proj gpu | **pca host** | **pca gpu** | cov residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20,000 × 150 | 521 ms | 11 ms | 196 ms | 25 ms | 806 ms | 117 ms | 1.6e-6 |
| 60,000 × 150 | 1,559 ms | 17 ms | 559 ms | 63 ms | 2,230 ms | 181 ms | 2.7e-6 |
| 60,000 × 300 | 6,008 ms | 45 ms | 1,108 ms | 69 ms | 7,533 ms | 701 ms | 1.9e-6 |
| 100,000 × 300 | 9,793 ms | 45 ms | 1,861 ms | 126 ms | 12,199 ms | 756 ms | 1.6e-6 |
| 200,000 × 300 | 20,165 ms | 80 ms | 3,726 ms | 246 ms | 24,167 ms | 1,008 ms | 1.3e-6 |

In the browser, on the real table, 14.4 s → **1.4 s**.

Three things are worth reading off that rather than just the speedup.

**The naive kernel shape would have wasted most of it.** One thread per covariance entry
`(p, q)` striding the rows reads *columns* of a row-major matrix, so consecutive threads
touch addresses `G` floats apart and every memory transaction is discarded. The 16×16 tile
stages a block of rows in workgroup memory, where each thread's load is contiguous along
the gene axis, and gets 16 multiply-adds per element loaded.

**f32 is fine here, and that is a measurement rather than a hope.** The host oracle is f64
and the device is not, so the contract is a residual, not equality. The interesting column
is the last one: the residual **does not grow with n** (1.3e-6 at 200k against 1.6e-6 at
20k), because the products go into a per-tile accumulator that folds into the running total
once per 16 rows — the longest dependent add chain is `n/16`, not `n`. Kahan compensation
was therefore not needed. Downstream, the eigenvalues agree with the host's to 3 decimal
places and the scores to a relative 1e-3.

**The remaining cost is now the eigensolve, and more GPU work will not touch it.**
`eigenSym` is O(G³) and does not care how many cells there are: invisible at G=150, about
500 ms at G=300, which is most of what the device path still spends. The next move for PCA,
if one is wanted, is a truncated Lanczos or randomised solve for the leading d components
instead of the full G×G Jacobi.

Both the page and `pnpm umap:obsm` use the device path; the page falls back to the host if
there is no usable GPU and *says so in the status line*, because a 14-second freeze and a
1.4-second pause are different enough that the user should know which one they got.

## 9. Testing notes

The GPU suite is deterministic and runs files in parallel. The only failing files are
those importing `rust/htj2k-core/pkg`, which need `pnpm build:wasm` first.

The layout SGD is checked on **trustworthiness**, not coordinates. A parallel GPU
implementation races against a shared position buffer (the Hogwild! regime every GPU UMAP
uses) and cannot reproduce host coordinates; what it must reproduce is a layout preserving
the same neighbourhoods.

`knnGpu` and `knnBruteForceCpu` are drop-in substitutes and are diffed directly, which
also gives the offline path a cheap cross-check: `--cpu` and the default must produce the
same graph (they do — 139790 edges on the 6000-cell fixture either way).

**Which GPU kernels can be held to exact agreement, and why it differs, is worth being
explicit about** — three kernels here sit in three different places:

- `knnDescentGpu` **is** diffed element for element against the host, because it is
  race-free by construction: thread `i` offers candidates only to `i`'s own list.
- `pcaGpu` cannot be, because the host accumulates in f64 and the device in f32. Its
  contract is a *measured residual* (§8.1). What is asserted exactly instead are the
  structural properties a tiling bug would break while leaving the numbers plausible: the
  covariance is bit-exactly symmetric, and the projection is bit-identical however the rows
  are split across dispatches.
- `umapLayoutGpu` cannot be either, for a different reason — it is deliberately racy
  (Hogwild!), so it is checked statistically on trustworthiness. Note that its
  `gpuTrust > hostTrust - 0.05` assertion is genuinely marginal and fails roughly one run in
  six *in the full suite* (never in isolation), which is a real flake predating this work.

Row tiling exists in every one of them for the watchdog reason, and in all three it is
forced in a test rather than trusted, because at test sizes a single dispatch covers
everything and the tiling would otherwise never run.

## 10. How far it scales

`pnpm umap:bench` sweeps `n` over one generator and times each stage separately, because
k-NN is O(n²·D) and the layout is O(edges) per epoch — they hit their walls at completely
different sizes, and a single wall-clock number hides which one you are waiting for.
Measured on an Apple M2 Max, `branching`, 15 neighbours, 200 epochs, 18 genes:

| cells | edges | exact (gpu) | descent (gpu) | graph | epoch host | epoch gpu | speedup | trust | rss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16,000 | 349k | 255 ms | 445 ms | 121 ms | 41.83 ms | 0.27 ms | 157× | 0.940 | 244 MB |
| 50,000 | 1.11M | 2.5 s | 1.4 s | 467 ms | 65.49 ms | 0.67 ms | 98× | 0.938 | 394 MB |
| 100,000 | 2.23M | 14.0 s | 2.7 s | 1.41 s | 124.87 ms | 1.31 ms | 95× | 0.938 | 536 MB |
| 200,000 | 4.50M | — | 5.3 s | 3.31 s | 259.57 ms | 2.56 ms | 101× | 0.938 | 732 MB |

Reading it:

- **The layout is not the problem.** 200 epochs at 100k cells is 0.3 s on the GPU. Even at
  4.5M edges an epoch is 2.8 ms, so the interactive page stays interactive at sizes where
  the host path (269 ms/epoch) is four frames per second.
- **The k-NN is no longer the wall.** With the local join on the device, 200k cells is
  5.3 s of k-NN — less than twice the cost of the graph build that follows it. The
  quadratic exact search is now the fallback for small n and for checking recall, not the
  path anything large takes.
- **Quality holds.** Trustworthiness stays at 0.933–0.946 across two orders of magnitude,
  so the speed is not being bought with a worse embedding.
- **Memory is not the binding constraint** at these sizes — 604 MB at 200k cells, and the
  dense feature matrix is the largest single term.

In the browser the same path runs 32k cells with an exact GPU k-NN in 1.2 s and 2.43
ms/epoch, which is where the page's cell slider tops out.

## 11. Not done

- **Candidate-tiled exact k-NN.** Restructuring `knnGpu` to keep every query row resident
  and tile over candidates instead measured 1.8× at 100k/D=18 and 4.6× at D=51, because
  the current row tiling shrinks the dispatch as `1/n` (35 workgroups at 100k). It also
  returns silently wrong results at n ≥ 45k about one run in three, and the cause is not
  understood — not contention, device loss, a validation error, buffer lifetime, or a
  params race, all of which were ruled out. Parked deliberately: NN-descent removed the
  motivation, and a silent wrong answer in the k-NN is the one thing this file keeps having
  to warn about.
- **Building the candidate lists on the device.** The one remaining per-pass buffer map
  (§8) is what keeps in-browser descent several times slower than the same code in Node.
  Moving `buildCandidates` to the GPU would remove it entirely — at the cost of the
  reverse-neighbour reservoir sampling, which is sequential, and with it the
  element-for-element agreement with the host that the test currently asserts. Worth doing,
  but the correctness property is worth more than the constant until someone is actually
  waiting on it.
- **A truncated eigensolve.** After §8.1, `eigenSym`'s full G×G Jacobi is the largest
  remaining term in the device PCA (~500 ms at G=300, independent of cell count). A Lanczos
  or randomised solve for just the leading d components would remove most of it. Not urgent:
  the whole reduction is now under 1.5 s in the page.
- **The `columnStats` pass is still on the host.** O(N·G) and 100–200 ms, so it is the
  smallest of PCA's four steps by an order of magnitude — but it is *synchronous*, and it is
  the last synchronous O(N·G) loop in the load. Cheap to move if the freeze budget ever gets
  tight.
- **A cheap post-condition on the k-NN.** Counting rows whose neighbours are all index 0
  during the readback costs nothing and would have turned both silent-truncation bugs into
  a thrown error. Worth doing regardless of the above.
- **A spectral initialiser.** Reference UMAP defaults to one and we use a random init. It
  was tested as a candidate fix for the shattering in §7 and is *not* one — a perfect
  init shatters too — so this is a fidelity gap rather than a quality one, and it is
  cheap to leave open.
- **The bench sweeps one shape at a time.** The generators differ enough in edge count per
  cell that a per-shape table would be more useful than the single `branching` column here.
