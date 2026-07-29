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

**The crossover depends on which exact search you are racing, and this is the honest
summary** (`pnpm umap:bench`, Apple M2 Max, D=18):

| n | host exact | host descent | recall | `knnGpu` (tiled) |
| --- | --- | --- | --- | --- |
| 3000 | 551 ms | 938 ms | 0.990 | — |
| 6000 | 2197 ms | 1814 ms | 0.970 | — |
| 16000 | — | 2.8 s | 0.983 | **270 ms** |
| 25000 | 38.0 s | 8.9 s | 0.881 | — |
| 50000 | — | 10.6 s | 0.966 | **3.5 s** |
| 100000 | — | 24.8 s | 0.951 | **17.3 s** |
| 200000 | — | **57–78 s** | 0.934 | 94.3 s |

So descent overtakes a *host* brute force around 5–6k points — and the tiled GPU exact
search stays ahead of descent all the way to **100k**, where it is both faster (17.3 s
against 24.8 s) and exact. The quadratic finally wins out somewhere around 150k; at 200k
descent is ahead.

That is a much later crossover than the 50k previously recorded here, and it changes the
advice: **on a machine with a GPU, prefer the exact search up to about 100k cells.**
`pickKnn` takes the crossover as a parameter rather than pretending one number fits both;
the CLI passes 5k when forced to the host and 50k when it has a GPU — conservative, since
the exact path is still winning there.

Note also that descent's recall falls as n grows at a fixed `maxIters` (0.999 → 0.934).
It is a knob, not a constant, and a recall figure quoted without its n is meaningless.

The consequence worth being clear about: **descent's value today is machines without a
usable GPU, plus being the algorithm that will actually scale once its inner local join
moves to the device.** That is the next piece of work — the flat `NeighbourHeap` and
candidate arrays are laid out to upload verbatim when it happens.

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
exact GPU k-NN for over two minutes with the page looking hung — 162k² × 30 is 7.9·10¹¹
products, and a script's patience budget is not a page's. So the store path takes
`maxCells` (default 40,000, a uniform deterministic subsample) and the k-NN selector
defaults to **auto**, taking the exact GPU path below 40k cells and NN-descent above. The
subsample happens *after* the variance pass, so highly-variable genes are still chosen from
every cell — the gene selection is cheap to do properly and is what a subsample would bias
most.

End to end on that store, at the defaults: 40,000 of 162,254 cells, all 377 genes, 30 PCs,
graph in 3.2 s, **14.7 s from click to a running layout** at 2.4 ms/epoch over 967k edges.

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

## 10. How far it scales

`pnpm umap:bench` sweeps `n` over one generator and times each stage separately, because
k-NN is O(n²·D) and the layout is O(edges) per epoch — they hit their walls at completely
different sizes, and a single wall-clock number hides which one you are waiting for.
Measured on an Apple M2 Max, `branching`, 15 neighbours, 200 epochs, 18 genes:

| cells | edges | knn (gpu, exact) | knn (descent) | graph | epoch host | epoch gpu | speedup | trust | rss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1,000 | 20k | 21 ms | 101 ms | 13 ms | 4.51 ms | 0.06 ms | 74× | 0.942 | 129 MB |
| 4,000 | 85k | 38 ms | 506 ms | 28 ms | 16.29 ms | 0.12 ms | 138× | 0.946 | 192 MB |
| 16,000 | 349k | 270 ms | 2.8 s | 136 ms | 43.50 ms | 0.28 ms | 157× | 0.941 | 238 MB |
| 50,000 | 1.11M | 3.5 s | 10.6 s | 519 ms | 68.87 ms | 0.83 ms | 83× | 0.939 | 342 MB |
| 100,000 | 2.24M | 17.3 s | 24.8 s | 1.55 s | 129.37 ms | 1.54 ms | 84× | 0.938 | 578 MB |
| 200,000 | 4.50M | 94.3 s | 57.0 s | 3.98 s | 268.70 ms | 2.80 ms | 96× | 0.933 | 604 MB |

Reading it:

- **The layout is not the problem.** 200 epochs at 100k cells is 0.3 s on the GPU. Even at
  4.5M edges an epoch is 2.8 ms, so the interactive page stays interactive at sizes where
  the host path (269 ms/epoch) is four frames per second.
- **The k-NN is the wall**, and it is the quadratic one. Everything above ~100k wants the
  approximate path, and the approximate path wants its local join on the device (§3).
- **Quality holds.** Trustworthiness stays at 0.933–0.946 across two orders of magnitude,
  so the speed is not being bought with a worse embedding.
- **Memory is not the binding constraint** at these sizes — 604 MB at 200k cells, and the
  dense feature matrix is the largest single term.

In the browser the same path runs 32k cells with an exact GPU k-NN in 1.2 s and 2.43
ms/epoch, which is where the page's cell slider tops out.

## 11. Not done

- **NN-descent on the GPU** (§3). The host version is in and correct; the local join is
  what needs to move to the device for a full section. The measured crossover (§3) says
  this only starts to matter past ~100k cells on a machine with a GPU.
- **A spectral initialiser.** Reference UMAP defaults to one and we use a random init. It
  was tested as a candidate fix for the shattering in §7 and is *not* one — a perfect
  init shatters too — so this is a fidelity gap rather than a quality one, and it is
  cheap to leave open.
- **The bench sweeps one shape at a time.** The generators differ enough in edge count per
  cell that a per-shape table would be more useful than the single `branching` column here.
