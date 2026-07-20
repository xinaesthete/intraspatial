# ADR-0016 — TopACT: reproduce the uniform-box multiscale classifier before an adaptive-KDE variant

Status: **draft / proposed** (2026-07-13)

## Context

We want to demonstrate this engine's usefulness to the **TopACT** group (Benjamin et al., Oxford;
Nature 2024, "Multiscale topology classifies cells in subcellular spatial transcriptomics"; ref
impl `/Users/ptodd/code/py/topact`). The shipped `topact` code is the **multiscale local SVM
classifier** (the multiparameter persistent homology is a separate, downstream stage not in that
repo). Its core pooling primitive is a **uniform square (L∞) box sum over gene counts at fixed
integer radii**, on a pre-binned integer lattice, fed to a calibrated `LinearSVC`.

This collides with a standing bias in our spatial/TDA front, which is built on **smooth,
adaptive-bandwidth** density estimation, not uniform pooling:

- `splatDensity` is a **Gaussian KDE** ([splatDensity.ts](../../src/gpu/spatial/splatDensity.ts));
- local bandwidth is **adaptive**, `ρ_i` = k-th-NN distance (CkNN,
  [kthNeighborDistance.ts](../../src/gpu/spatial/kthNeighborDistance.ts));
- fuzzy adjacency uses `exp(−d²/2σ²)` with per-point σ.

The bias is a *philosophy*, not a missing primitive: a box kernel already exists
(`boxKernel` in [convolveSeparable.ts](../../src/gpu/spatial/convolveSeparable.ts), used by
[getisOrd.ts:42](../../src/gpu/spatial/getisOrd.ts)). The risk is that our **default** density ops
are smooth/isotropic/adaptive, so a later contributor wiring TopACT through them would **silently
change the cell calls** and mistake it for an improvement.

## Decision

**Reproduce-then-improve, as an ordered gate — the second step may not be claimed without the
first.**

1. **Reproduce faithfully, first.** Implement the **exact** TopACT pooling: uniform square (L∞)
   neighbourhood sum of gene counts at integer radii `s ∈ [min,max]`, on the integer lattice,
   reusing their trained calibrated `LinearSVC`. Prove **bit-for-bit agreement** with their
   `predict_proba` `(H,W,scales,classes)` confidence tensor via the repo's CPU-golden discipline.
   This is the credibility precondition for any improvement claim to the collaborators.
2. **The reproduction op is named and distinct** (e.g. `boxPool`/`squarePool`), **never** the
   default `kdeSplat`. A TopACT reproduction must not resolve to a smooth-kernel op by default —
   the guardrail against silent cell-call drift.
3. **Box gets the box-specific fast path.** A **summed-area table** (parallel prefix sum) — or an
   incremental separable running-sum — makes each scale's box-sum `O(1)`, replacing TopACT's
   redundant `O(H·W·Σ_s(2s+1)²)` re-gather. Note this speedup is **box-specific**: it exists
   *because* the kernel is a uniform box (see Consequences).
4. **Only then, an explicitly re-trained adaptive-KDE variant**, offered as a *scientific
   alternative*, not a drop-in: distance-weighted, adaptive bandwidth `ρ_i`, continuous
   coordinates (no pre-binning), continuous scale. The classifier is **re-trained on KDE-pooled
   features** (a box pool and a KDE pool produce different local-composition estimates, so their
   feature distributions differ). Its value is an **empirical question to test** — does it change
   or improve classification in density-heterogeneous tissue? — not a claim to assert.

## Why

- **The trained SVM is coupled to the box.** Their pooled feature is a sum of counts in radius `s`,
  row-normalised + `log1p`. A distance-weighted KDE reshapes the pooled *composition* (even after
  normalisation removes magnitude), so substituting it feeds the trained classifier
  out-of-distribution features — a bug, not an improvement. Faithful reproduction *requires* the box.
- **The adaptive-KDE critique is statistically motivated, but must be validated.** Subcellular
  transcript density is highly heterogeneous (dense nuclei vs sparse cytoplasm/immune cells), so a
  global fixed radius is simultaneously too large where dense (mixes adjacent cells) and too small
  where sparse — the classic case for adaptive bandwidth. And because nearby transcripts are more
  likely to belong to the *same* cell, distance-weighting is a better estimator of a cell's own
  composition (it suppresses neighbour contamination a hard box admits at full weight). These are
  real arguments, which is exactly why they deserve a controlled test rather than a silent swap.
- **Kernel quality matters more downstream.** The density estimator also becomes the density axis
  of Half 2's density–radius **bifiltration**; persistent homology is acutely sensitive to density
  noise and grid anisotropy, so our smooth/adaptive density is plausibly a larger win for the
  topological half than for the classifier — aligning with the part the collaborators value most.

## Consequences / open questions

- **Kernel ⇄ speed tradeoff is real.** Box: SAT-fast `O(1)`/scale, but crude (uniform weight, hard
  cutoff, L∞ grid-anisotropy, integer-scale quantisation, lossy pre-binning). Gaussian: separable
  and smooth, still fast (recursive-IIR `O(1)`), isotropic. Radial **Epanechnikov** (the spatial
  front's *preferred* compact kernel): **non-separable → gather** (ADR-0004), so choosing it
  forfeits the multiscale speedup. The kernels our stack prefers are the expensive ones; the fast
  primitive is the box we're averse to. Gaussian is the pragmatic middle if we want smooth + fast.
- **Box has virtues we'd be dropping:** exact interpretable counts (transcripts within radius `s`),
  a clean radius→micron length scale, and no bandwidth to tune. The improvement case must beat
  these, not just be "smoother".
- **L∞-square exactness must be preserved in reproduction** — our defaults lean isotropic
  disc/Gaussian; a disc window would already diverge. The `boxPool` op must be the Chebyshev square.
- **Whole-transcriptome memory:** dense per-gene SATs are `O(H·W·G)` — fine for targeted panels
  (G~hundreds), but whole-transcriptome G needs the lazy/gather-fused path (ADR-0004 dense-vs-lazy).
- **Open:** whether the re-trained KDE variant is evaluated on TopACT's own labelled benchmarks or
  needs new ground truth; and whether "improvement" is measured as classification accuracy, spatial
  smoothness/robustness, or downstream persistence stability.

## References

- ADR-0004 (scale-equivariance; gene axis; Epanechnikov non-separable ⇒ gather), ADR-0015
  (metric linear algebra over the gene axis — TopACT's `W·x` SVM step is exactly this).
- `topact` ref impl (`/Users/ptodd/code/py/topact`): `spatial.py` (square_nbhd, classify_parallel,
  confidence tensor), `classifier.py` (calibrated `LinearSVC` on log-normalised pooled counts).
- Ours: `splatDensity.ts` (Gaussian KDE), `kthNeighborDistance.ts`/CkNN (adaptive `ρ_i`),
  `convolveSeparable.ts` (`boxKernel`/`gaussianKernel`), `getisOrd.ts` (existing box windowed sum).
- Nature 2024, DOI 10.1038/s41586-024-07563-1 (paper); the [[topact-collaboration-target]] memory.
