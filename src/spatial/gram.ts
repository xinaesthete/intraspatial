// The N-way association matrix in Gram form, and the eigen-projection it makes well-posed.
// CPU reference (f64); `src/gpu/spatial/gramMatrix.ts` is the render twin this is the oracle for.
//
// ## The reformulation
//
// `crossPCFMatrix` counts pairs: for every ordered type pair (a,b) it walks a bucket grid and
// tallies `|x_i − x_j| < r`. Cost is `O(n · ρ · πr²)` — **quadratic in the radius**, because a
// bigger disk holds more neighbours to walk.
//
// Rasterise instead. Let `R` be the K×P matrix whose row `a` is channel `a`'s point mass deposited
// on a P-pixel grid, and let `J` be a unit-mass radial kernel. Then with `M = J ⊛ R` (one splat
// per channel) the whole K×K matrix is a single product:
//
//     C = M Mᵀ,      C_ab = ∫ (J ⊛ R_a)(x) · (J ⊛ R_b)(x) dx
//
// which, writing the point sets as sums of deltas, is
//
//     C_ab = Σ_{i∈a} Σ_{j∈b} w_i w_j · (J ⊛ J)(x_i − x_j)
//
// — the same pairwise sum, with the hard `1[d < r]` replaced by the smooth `(J ⊛ J)(d)`. Cost is
// `O(K·P)` to splat plus `O(K²·P)` to multiply, both **independent of r**. So there is a crossover
// radius past which the raster form wins outright, and the paper's regime (r = 100 µm, bins out to
// 300 µm) is on the far side of it.
//
// ## Why this is not just "the same statistic, faster"
//
// An eigendecomposition is only readable as a decomposition of variance if the matrix is positive
// semi-definite, and **symmetry is not definiteness**. Two independent things can cost us that
// property, and it is worth keeping them apart because they have different fixes.
//
// **1. The mark kernel.** Written as an operator, the association matrix is `C = R K Rᵀ` with `K`
// the kernel's convolution operator, and that is PSD iff `K` is a positive-definite kernel — iff
// its Fourier transform is non-negative (Bochner). `kernelSpectrum.ts` measures that **no kernel
// in this codebase qualifies**: the top-hat's 2-D transform dips to −13.2% of its DC value, and
// even the truncated Gaussian reaches −0.13%. So computing the statistic with the paper's disk
// gives a matrix whose negative eigenvalues are partly a property of the *kernel*.
//   The Gram form escapes this for free. Its effective kernel is `J ⊛ J`, whose transform is
// `|Ĵ|² ≥ 0` — positive-definite whatever `J` was. Equivalently and more simply: `M Mᵀ` is a Gram
// matrix of real vectors, so it is PSD by construction, exactly, for any kernel, and even in f32,
// because the property is structural rather than numerical.
//
// **2. The normalisation — and this is the one that actually bites.** `g` divides by per-channel
// mass and drops self-pairs, which is exactly what removes the diagonal dominance that would
// otherwise carry the matrix. On self-clustering populations `g` usually stays PSD by accident
// (each type's own clustering inflates `g_aa`), but there is no guarantee, and interdigitated
// populations destroy it outright: two types alternating on a lattice measure
// `g = [[0, 2.09], [2.09, 0]]` at the pitch radius, whose eigenvalues are ±2.09 — maximally
// indefinite. This happens in the published statistic and in the Gram form alike, because it is a
// property of the normalisation rather than of how the pair sum was computed. Both are pinned in
// `gram.test.ts`.
//
// **The conclusion, which corrects `muspan-cell-stats-plan.md` §7.** That plan proposed
// "eigenvectors of the symmetrised g-matrix" as the co-location modes. Symmetrising is not the
// missing ingredient — `crossPCFMatrix`'s `g` is *already* exactly symmetric — and it does not
// confer definiteness. The modes must be taken from `corr` (or from raw `C`), which is PSD by
// construction; `g` remains the right thing to *report* and to draw a network from, but its
// spectrum is not a variance decomposition.
//
// ## Cell types are the one-hot case of a general mark
//
// Nothing above uses the fact that a cell belongs to exactly one type. `R`'s rows are arbitrary
// non-negative per-cell weights, and the cell-type matrix is just the case where those weights are
// one-hot. Hand the same code a gene's expression column from an AnnData `X` and it computes the
// spatially-smoothed gene–gene co-expression Gram matrix, with the identical normalisation. This
// is the mark cross-correlation function of point-process theory (Stoyan & Stoyan), which the
// cell-type cross-PCF is a special case of — not a new statistic.
//
// **One caveat is load-bearing in the weighted case.** The double sum includes `i = j`. For
// one-hot types that self term only touches the diagonal, but when every cell carries a weight in
// every channel it lands in *every* entry, contributing `(J⊛J)(0) · Σ_i w_a(i) w_b(i)` — pure
// within-cell co-expression, with no spatial content whatsoever, masquerading as co-location. It
// is reported as `selfTerm` so it can be subtracted or at least seen. Subtracting it is what
// `crossPCFMatrix` does (it skips `j == i`) and it is what makes `g` comparable to the published
// statistic — but it also destroys the PSD property, which is why `corr`, the matrix the modes are
// computed from, does not subtract it.

import { eigenSym, psdDefect } from "./eigenSym";
import { EPANECHNIKOV, type KernelSpec, kernelAt, roughness } from "./kernels";

/**
 * One channel of the mark matrix: a point cloud with optional per-point weights.
 *
 * For cell types, pass one cloud per type with no weights (the one-hot case). For expression, pass
 * the *same* xs/ys once per gene, each with that gene's column as `weights`.
 */
export interface ChannelCloud {
  readonly label: string;
  readonly xs: ArrayLike<number>;
  readonly ys: ArrayLike<number>;
  /** Per-point weight, index-aligned to xs/ys. Absent means 1 for every point. Must be
   *  non-negative: the normalisation divides by total mass, which a signed weight can zero. */
  readonly weights?: ArrayLike<number>;
}

export interface GramParams {
  /** World extent the raster spans, `[minX, minY, maxX, maxY]`. */
  readonly bbox: readonly [number, number, number, number];
  readonly width: number;
  readonly height: number;
  /** Support radius of the splat kernel `J`, in world units. Note the *effective* pair kernel is
   *  `J ⊛ J`, whose support is `2·radius` — see `effectiveRadius`. */
  readonly radius: number;
  /** Splat kernel. Default Epanechnikov: `kernels.ts` measures it as the best-behaved member, and
   *  nothing here needs the top-hat's discontinuity. */
  readonly kernel?: KernelSpec;
}

export interface GramResult {
  readonly labels: string[];
  /** `W_a` — total mark mass per channel. Equals the cell count in the one-hot case. */
  readonly mass: Float64Array;
  /** K×K raw Gram `∫ M_a M_b`, row-major. PSD by construction. */
  readonly c: Float64Array;
  /** K×K association matrix, normalised so complete spatial randomness gives **1**. The self-pair
   *  term is subtracted, matching `crossPCFMatrix`'s `j != i` — so this is the quantity comparable
   *  to the published cross-PCF, and it is *not* guaranteed PSD. */
  readonly g: Float64Array;
  /** K×K spatial **correlation** of the smoothed channel densities: centred, standardised, so the
   *  diagonal is exactly 1 and the trace is exactly K. PSD by construction. **This is the matrix
   *  the modes are computed from.** */
  readonly corr: Float64Array;
  /** K×K self-pair contribution `(J⊛J)(0) · Σ_i w_a(i) w_b(i)`, in the units of `c`. Zero
   *  off-diagonal for one-hot channels; the within-cell co-expression confound otherwise. */
  readonly selfTerm: Float64Array;
  /** K×P smoothed channel densities `M`, planar (`rasters[a*P + p]`). Kept because the mode
   *  projection is a recombination of exactly these — no second pass over the points. */
  readonly rasters: Float64Array;
  /** Per-channel mean and standard deviation of `M` over pixels; the centring `corr` used. */
  readonly mean: Float64Array;
  readonly sd: Float64Array;
  readonly width: number;
  readonly height: number;
  readonly bbox: readonly [number, number, number, number];
  readonly pixelArea: number;
}

/** Support radius of the *effective* pair kernel `J ⊛ J` — twice the splat radius. Two cells this
 *  far apart contribute nothing to `C`, which is the honest answer to "what radius is this?" and
 *  is not the `radius` parameter. */
export function effectiveRadius(p: GramParams): number {
  return 2 * p.radius;
}

function weightAt(c: ChannelCloud, i: number): number {
  return c.weights ? (c.weights[i] ?? 0) : 1;
}

/**
 * Splat one channel through `J`, evaluated at pixel centres: `M(p) = Σ_i w_i J(x_p − x_i)`.
 *
 * The result is a **density** (mass per unit world area), so `Σ_p M(p)·A_pix ≈ W`. That is what
 * makes the normalisation below kernel-agnostic, exactly as in `kernels.ts`.
 */
export function splatChannel(cloud: ChannelCloud, p: GramParams): Float64Array {
  const [minX, minY, maxX, maxY] = p.bbox;
  const { width, height, radius } = p;
  const kernel = p.kernel ?? EPANECHNIKOV;
  const out = new Float64Array(width * height);
  const spanX = maxX - minX;
  const spanY = maxY - minY;
  const dx = spanX / width;
  const dy = spanY / height;
  const n = cloud.xs.length;
  const r2 = radius * radius;

  for (let i = 0; i < n; i++) {
    const w = weightAt(cloud, i);
    if (w === 0) continue;
    const x = cloud.xs[i] ?? 0;
    const y = cloud.ys[i] ?? 0;
    // Pixel window covering the kernel support. Row 0 is the TOP of the bbox (worldY = maxY),
    // matching ScalarField / splatDensity, so the row range is derived from maxY downwards.
    const c0 = Math.max(0, Math.floor((x - radius - minX) / dx));
    const c1 = Math.min(width - 1, Math.ceil((x + radius - minX) / dx));
    const r0 = Math.max(0, Math.floor((maxY - (y + radius)) / dy));
    const r1 = Math.min(height - 1, Math.ceil((maxY - (y - radius)) / dy));
    for (let row = r0; row <= r1; row++) {
      const py = maxY - ((row + 0.5) / height) * spanY;
      const ddy = py - y;
      const rowBase = row * width;
      for (let col = c0; col <= c1; col++) {
        const px = minX + ((col + 0.5) / width) * spanX;
        const ddx = px - x;
        const d2 = ddx * ddx + ddy * ddy;
        if (d2 < r2) out[rowBase + col]! += w * kernelAt(kernel, d2, radius);
      }
    }
  }
  return out;
}

/**
 * The N-way association matrix in Gram form, plus everything the eigen-projection needs.
 *
 * `channels` may share point positions (the gene case) or partition them (the cell-type case);
 * nothing here cares which.
 */
export function gramMatrix(channels: readonly ChannelCloud[], p: GramParams): GramResult {
  const K = channels.length;
  const P = p.width * p.height;
  const [minX, minY, maxX, maxY] = p.bbox;
  const roiArea = Math.max((maxX - minX) * (maxY - minY), 1e-12);
  const pixelArea = roiArea / P;
  const kernel = p.kernel ?? EPANECHNIKOV;

  const rasters = new Float64Array(K * P);
  const mass = new Float64Array(K);
  for (let a = 0; a < K; a++) {
    rasters.set(splatChannel(channels[a]!, p), a * P);
    let w = 0;
    for (let i = 0; i < channels[a]!.xs.length; i++) w += weightAt(channels[a]!, i);
    mass[a] = w;
  }

  // `(J⊛J)(0) = ∫J²` — the kernel's roughness, already a closed form in kernels.ts.
  const selfAtZero = roughness(kernel, p.radius);
  const selfTerm = new Float64Array(K * K);
  for (let a = 0; a < K; a++) {
    for (let b = a; b < K; b++) {
      // Σ_i w_a(i)·w_b(i) — only meaningful when the channels index the SAME points. Channels that
      // partition the cells (one-hot types) share no point, so every off-diagonal entry is 0.
      const ca = channels[a]!;
      const cb = channels[b]!;
      let s = 0;
      if (a === b || (ca.xs === cb.xs && ca.ys === cb.ys)) {
        const n = Math.min(ca.xs.length, cb.xs.length);
        for (let i = 0; i < n; i++) s += weightAt(ca, i) * weightAt(cb, i);
      }
      selfTerm[a * K + b] = selfAtZero * s;
      selfTerm[b * K + a] = selfAtZero * s;
    }
  }

  // Means and standard deviations over pixels, for the correlation form.
  const mean = new Float64Array(K);
  const sd = new Float64Array(K);
  for (let a = 0; a < K; a++) {
    let s = 0;
    for (let q = 0; q < P; q++) s += rasters[a * P + q]!;
    const mu = s / P;
    mean[a] = mu;
    let v = 0;
    for (let q = 0; q < P; q++) v += (rasters[a * P + q]! - mu) ** 2;
    sd[a] = Math.sqrt(v / P);
  }

  const c = new Float64Array(K * K);
  const g = new Float64Array(K * K);
  const corr = new Float64Array(K * K);
  for (let a = 0; a < K; a++) {
    for (let b = a; b < K; b++) {
      let raw = 0;
      let cov = 0;
      for (let q = 0; q < P; q++) {
        const va = rasters[a * P + q]!;
        const vb = rasters[b * P + q]!;
        raw += va * vb;
        cov += (va - mean[a]!) * (vb - mean[b]!);
      }
      const cab = raw * pixelArea;
      c[a * K + b] = cab;
      c[b * K + a] = cab;

      // CSR expectation: E[C_ab] = W_a·W_b/|ROI| for a unit-mass kernel — kernel-agnostic, with no
      // πr² anywhere, which is why this matches crossPCFMatrix's g despite a different kernel.
      const expected = (mass[a]! * mass[b]!) / roiArea;
      const gab = expected > 0 ? (cab - selfTerm[a * K + b]!) / expected : 0;
      g[a * K + b] = gab;
      g[b * K + a] = gab;

      const denom = sd[a]! * sd[b]!;
      const rab = denom > 0 ? cov / P / denom : 0;
      corr[a * K + b] = rab;
      corr[b * K + a] = rab;
    }
  }
  // Pin the diagonal exactly: accumulated in a different order it lands a few ulp off 1, and a
  // trace that is not exactly K would put a spurious offset in every variance fraction.
  for (let a = 0; a < K; a++) if (sd[a]! > 0) corr[a * K + a] = 1;

  return {
    labels: channels.map((ch) => ch.label),
    mass,
    c,
    g,
    corr,
    selfTerm,
    rasters,
    mean,
    sd,
    width: p.width,
    height: p.height,
    bbox: p.bbox,
    pixelArea,
  };
}

export interface CoLocationModes {
  /** Eigenvalues of `corr`, descending. They sum to K, so each is "how many channels' worth of
   *  spatial variance this mode carries". */
  readonly values: Float64Array;
  /** Mode-major loadings: mode `k`'s weighting over channels is `vectors[k*K .. k*K+K)`. */
  readonly vectors: Float64Array;
  /** `values[k] / K` — the fraction of total spatial variance in mode `k`. Meaningful precisely
   *  because `corr` is PSD; see `psdDefect`. */
  readonly explained: Float64Array;
  /** Indefiniteness of the matrix that was decomposed, as a fraction of its spectral radius. For
   *  `corr` this is machine-epsilon; it is reported so a caller who decomposes `g` instead can
   *  see what that costs. */
  readonly psdDefect: number;
  readonly labels: string[];
}

/**
 * Co-location modes: the eigenvectors of the spatial correlation matrix.
 *
 * Mode `k` is a signed weighting over channels whose recombined density field
 * `Σ_a v_ka · (M_a − μ_a)/σ_a` has maximal spatial variance orthogonal to the earlier modes — so
 * "channels that vary together in space" load together, with opposite signs for channels that
 * exclude one another. `projectMode` renders that field.
 *
 * Pass `matrix: "g"` to decompose the association matrix instead. That is the one the literature's
 * network diagrams are drawn from, and it is *not* PSD (see the module header), so `explained` is
 * not a variance share; `psdDefect` reports how badly.
 */
export function coLocationModes(
  // Only the three fields it actually reads, so the GPU path's narrower result works here too
  // without either widening that type or duplicating the decomposition.
  gram: Pick<GramResult, "labels" | "corr" | "g">,
  opts: { matrix?: "corr" | "g" } = {},
): CoLocationModes {
  const K = gram.labels.length;
  const source = opts.matrix === "g" ? gram.g : gram.corr;
  const { values, vectors } = eigenSym(source, K);
  let total = 0;
  for (let k = 0; k < K; k++) total += values[k]!;
  const explained = new Float64Array(K);
  for (let k = 0; k < K; k++) explained[k] = total !== 0 ? values[k]! / total : 0;
  return { values, vectors, explained, psdDefect: psdDefect(values, K), labels: gram.labels };
}

/**
 * Render mode `k` as a pixel field: `y_k(p) = Σ_a v_ka · (M_a(p) − μ_a)/σ_a`.
 *
 * This is the "spatial map coloured by dominant co-location mode". Note it costs one pass over the
 * rasters already in hand — no re-splatting, no second neighbour search — because the modes are a
 * change of basis on exactly the fields that produced the matrix.
 *
 * The result is signed and centred at 0, so it wants a diverging ramp (`src/color/ramps.ts`), and
 * its sign is only meaningful relative to the mode's loadings.
 */
export function projectMode(gram: GramResult, modes: CoLocationModes, k: number): Float64Array {
  const K = gram.labels.length;
  const P = gram.width * gram.height;
  const out = new Float64Array(P);
  for (let a = 0; a < K; a++) {
    const v = modes.vectors[k * K + a]!;
    if (v === 0 || gram.sd[a]! === 0) continue;
    const mu = gram.mean[a]!;
    const inv = v / gram.sd[a]!;
    for (let q = 0; q < P; q++) out[q]! += inv * (gram.rasters[a * P + q]! - mu);
  }
  return out;
}

/**
 * Build one-hot channels from a flat labelled cloud — the cell-type case, in the shape
 * `crossPCFMatrix` already takes.
 */
export function channelsFromLabels(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  typeId: ArrayLike<number>,
  labelOf?: (id: number) => string,
): ChannelCloud[] {
  const ids = [...new Set(Array.from({ length: typeId.length }, (_, i) => typeId[i] ?? 0))].sort((a, b) => a - b);
  return ids.map((id) => {
    const cx: number[] = [];
    const cy: number[] = [];
    for (let i = 0; i < typeId.length; i++) {
      if (typeId[i] === id) {
        cx.push(xs[i] ?? 0);
        cy.push(ys[i] ?? 0);
      }
    }
    return { label: labelOf ? labelOf(id) : String(id), xs: cx, ys: cy };
  });
}

/**
 * Build channels from an expression matrix — the AnnData `X` case.
 *
 * `x` is column-major over the selected genes (`x[g * n + i]`), which is the layout a CSC slice or
 * a per-gene column read produces naturally. All channels share the one `xs`/`ys`, which is also
 * what makes `selfTerm` non-trivial: see the module header.
 */
export function channelsFromExpression(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  x: ArrayLike<number>,
  geneLabels: readonly string[],
): ChannelCloud[] {
  const n = xs.length;
  // A typed-array input is sliced as a view — no copy, which matters when the caller has already
  // paid to densify a few genes' columns out of a sparse X.
  const column =
    ArrayBuffer.isView(x) && "subarray" in x
      ? (g: number) => (x as Float64Array).subarray(g * n, (g + 1) * n)
      : (g: number) => Array.from({ length: n }, (_, i) => x[g * n + i] ?? 0);
  return geneLabels.map((label, g) => ({ label, xs, ys, weights: column(g) }));
}
