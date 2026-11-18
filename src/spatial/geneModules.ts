// Gene modules from a whole-transcriptome points sample: a PCA gene embedding, clustered by e-means
// into disjoint gene sets that the transcript Gram can take as channels.
//
// **Why modules, not PCA axes, as channels.** A whole-transcriptome panel (Atera: ~18k genes) is
// far too many channels to splat, so the genes have to be reduced first. The obvious reduction,
// PCA, gives each gene a *signed* weight on *every* axis. `transcriptChannels` refuses both:
// weights must be non-negative (the Gram's `g` divides by mass) and channels disjoint (the
// self-term and permutation logic decide sharing by identity). PCA axes would also make the
// co-location modes redundant, since the modes of the Gram over PCA-projected fields are mostly
// the PCA again. So the PCA is used to *embed* the genes, and e-means turns the embedding into a
// partition. Each module is a gene set with weight 1 per molecule. That is exactly the channel
// the page already understands, so `g`, the modes, the lock and the wand all work unchanged.
//
// **The embedding.** The sample is binned into square bins. Each bin's counts are normalised to a
// fixed total and log1p-transformed, then each gene is standardised over the bins (the scanpy
// recipe: `normalize_total`, `log1p`, `scale(max_value=10)`). A randomized PCA (Halko, Martinsson &
// Tropp 2011) of that bins × genes matrix runs without densifying it: centring and scaling are
// applied implicitly in each product, and the clip at +10 SD is folded into the stored values. A
// gene's embedding is its row of correlations with the leading components. Its squared norm is
// the share of the gene's spatial variance those components explain (its *communality*), so
// genes with no spatial structure sit near the origin. Only components above the Marchenko–Pastur
// noise edge are kept, so that noise genes cannot borrow structure from noise components.
//
// **Unstructured genes are left out.** Genes whose communality is below `minCommunality` are not
// put in any module. They are mostly noise at this sample depth, and clustering them would
// produce modules made of noise. They are reported, not hidden.

import { emeans } from "./emeans";
import { eigenSym } from "./eigenSym";
import { mulberry32 } from "#src/evo/rng";

export interface GeneEmbeddingOptions {
  /** Side of the square bins, in the points' units (µm on Xenium/Atera). */
  binSize: number;
  /** Most principal components to compute (default 30). Only those above the noise edge are kept. */
  components?: number;
  /** Genes with fewer molecules than this in the sample are dropped (default 30). */
  minGeneCount?: number;
  /** Bins with fewer molecules than this are dropped as background (default 20). */
  minBinCount?: number;
  /** Feature codes to leave out, e.g. negative-control probes. */
  exclude?: ReadonlySet<number>;
  /** `normalize_total` target (default 1e4). */
  targetSum?: number;
  /** Clip standardised values above this many SDs (default 10, as scanpy's `max_value`). */
  maxValue?: number;
  /** Power iterations of the randomized PCA (default 4). */
  powerIterations?: number;
  seed?: number;
}

export interface GeneEmbedding {
  /** The feature codes embedded, in row order. */
  codes: Int32Array;
  /** genes × components, row-major: each gene's correlation with each kept component. */
  loadings: Float64Array;
  /** Components kept: those whose variance is above `noiseEdge`. */
  components: number;
  /** Variance of every computed component over the bins, as a share of the total (all genes
   *  standardised). Longer than `components` when some fell below the noise edge. */
  explained: Float64Array;
  /** The largest share a component of pure noise would reach (Marchenko–Pastur edge). */
  noiseEdge: number;
  /** Molecules per embedded gene in the sample. */
  geneCounts: Float64Array;
  /** Bins used, and molecules in them. */
  bins: number;
  molecules: number;
  /** Genes dropped for being excluded, or too rare in the sample. */
  dropped: { excluded: number; rare: number };
}

/** A bin side that gives about `perBin` molecules per bin (default 750) for `molecules` spread over
 *  `area`. On Atera's 1/256 level (4.6M molecules) that is ~117 µm. Smaller bins at that depth leave
 *  too few molecules per bin for most genes to show structure. */
export function binSizeFor(molecules: number, area: number, perBin = 750): number {
  return Math.sqrt((perBin * area) / Math.max(1, molecules));
}

/** A sparse bins × genes matrix in CSR form, values already log-normalised. */
interface Csr {
  rows: number;
  cols: number;
  ptr: Int32Array;
  col: Int32Array;
  val: Float64Array;
}

/** Bin the points and build the log-normalised CSR matrix over the kept genes. */
function binMatrix(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  codes: ArrayLike<number>,
  nFeatures: number,
  o: Required<Pick<GeneEmbeddingOptions, "binSize" | "minGeneCount" | "minBinCount" | "targetSum">> & { exclude?: ReadonlySet<number> },
) {
  const n = codes.length;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < n; i++) {
    const x = xs[i]!;
    const y = ys[i]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
  }
  const nx = Math.max(1, Math.floor((maxX - minX) / o.binSize) + 1);
  const binOf = (i: number) => Math.floor((ys[i]! - minY) / o.binSize) * nx + Math.floor((xs[i]! - minX) / o.binSize);

  // Which genes are kept: not excluded, and common enough in the sample.
  const total = new Float64Array(nFeatures);
  for (let i = 0; i < n; i++) total[codes[i]!]! += 1;
  const colOf = new Int32Array(nFeatures).fill(-1);
  const kept: number[] = [];
  let excluded = 0;
  let rare = 0;
  for (let c = 0; c < nFeatures; c++) {
    if (total[c] === 0) continue;
    if (o.exclude?.has(c)) excluded++;
    else if (total[c]! < o.minGeneCount) rare++;
    else colOf[c] = kept.push(c) - 1;
  }

  // Group the kept molecules by bin (counting sort), then count genes within each bin.
  const binCount = new Map<number, number>();
  for (let i = 0; i < n; i++) if (colOf[codes[i]!]! >= 0) binCount.set(binOf(i), (binCount.get(binOf(i)) ?? 0) + 1);
  const rowOfBin = new Map<number, number>();
  for (const [b, c] of binCount) if (c >= o.minBinCount) rowOfBin.set(b, rowOfBin.size);
  const rows = rowOfBin.size;
  const start = new Int32Array(rows + 1);
  for (const [b, r] of rowOfBin) start[r + 1] = binCount.get(b)!;
  for (let r = 0; r < rows; r++) start[r + 1]! += start[r]!;
  const fill = start.slice(0, rows);
  const members = new Int32Array(start[rows]!);
  for (let i = 0; i < n; i++) {
    const g = colOf[codes[i]!]!;
    if (g < 0) continue;
    const r = rowOfBin.get(binOf(i));
    if (r !== undefined) members[fill[r]!++] = g;
  }

  const ptr = new Int32Array(rows + 1);
  const col: number[] = [];
  const val: number[] = [];
  const scratch = new Float64Array(kept.length);
  const touched: number[] = [];
  let molecules = 0;
  for (let r = 0; r < rows; r++) {
    const lo = start[r]!;
    const hi = start[r + 1]!;
    for (let m = lo; m < hi; m++) {
      const g = members[m]!;
      if (scratch[g] === 0) touched.push(g);
      scratch[g]! += 1;
    }
    const size = hi - lo;
    molecules += size;
    touched.sort((a, b) => a - b);
    for (const g of touched) {
      col.push(g);
      val.push(Math.log1p((scratch[g]! * o.targetSum) / size));
      scratch[g] = 0;
    }
    touched.length = 0;
    ptr[r + 1] = col.length;
  }
  const geneCounts = new Float64Array(kept.length);
  for (let g = 0; g < kept.length; g++) geneCounts[g] = total[kept[g]!]!;
  const m: Csr = { rows, cols: kept.length, ptr, col: Int32Array.from(col), val: Float64Array.from(val) };
  return { m, codes: Int32Array.from(kept), geneCounts, molecules, dropped: { excluded, rare } };
}

/** Orthonormalise the columns of a rows × l matrix (column-major), twice for stability. */
function orthonormalise(a: Float64Array, rows: number, l: number): void {
  for (let pass = 0; pass < 2; pass++) {
    for (let j = 0; j < l; j++) {
      const cj = j * rows;
      for (let i = 0; i < j; i++) {
        const ci = i * rows;
        let dot = 0;
        for (let r = 0; r < rows; r++) dot += a[ci + r]! * a[cj + r]!;
        for (let r = 0; r < rows; r++) a[cj + r]! -= dot * a[ci + r]!;
      }
      let norm = 0;
      for (let r = 0; r < rows; r++) norm += a[cj + r]! ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let r = 0; r < rows; r++) a[cj + r]! /= norm;
    }
  }
}

/**
 * The PCA gene embedding of a points sample: bin, normalise, standardise, randomized PCA. Returns
 * each kept gene's correlations with the leading components.
 *
 * Cost: a few passes of `nnz × (components + 10)` over the sparse matrix. A 4.6M-molecule sample
 * at 100 µm bins (Atera kidney, ~18k genes) takes a few seconds on the host.
 */
export function geneEmbedding(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  codes: ArrayLike<number>,
  nFeatures: number,
  opts: GeneEmbeddingOptions,
): GeneEmbedding {
  if (!(opts.binSize > 0)) throw new Error(`geneEmbedding: binSize must be positive (got ${opts.binSize})`);
  if (xs.length !== codes.length || ys.length !== codes.length) throw new Error("geneEmbedding: xs, ys and codes differ in length");
  const built = binMatrix(xs, ys, codes, nFeatures, {
    binSize: opts.binSize,
    minGeneCount: opts.minGeneCount ?? 30,
    minBinCount: opts.minBinCount ?? 20,
    targetSum: opts.targetSum ?? 1e4,
    ...(opts.exclude ? { exclude: opts.exclude } : {}),
  });
  const { m } = built;
  const B = m.rows;
  const G = m.cols;
  const d = Math.min(opts.components ?? 30, G, Math.max(1, B - 1));
  if (B < 2 || G < 1) throw new Error(`geneEmbedding: too little data (${B} bins, ${G} genes); lower minBinCount or minGeneCount`);
  const l = Math.min(d + 10, G, B);

  // Column statistics over all B bins (zeros included), then the clip folded into the values.
  const mean = new Float64Array(G);
  const sq = new Float64Array(G);
  for (let k = 0; k < m.val.length; k++) {
    mean[m.col[k]!]! += m.val[k]!;
    sq[m.col[k]!]! += m.val[k]! ** 2;
  }
  const sd = new Float64Array(G);
  for (let g = 0; g < G; g++) {
    mean[g]! /= B;
    sd[g] = Math.sqrt(Math.max(sq[g]! / B - mean[g]! ** 2, 0)) || 1;
  }
  const maxValue = opts.maxValue ?? 10;
  const val = m.val.map((v, k) => Math.min(v, mean[m.col[k]!]! + maxValue * sd[m.col[k]!]!));
  // Z = (X − 1μᵀ) D⁻¹, applied implicitly: Z·W = X·(D⁻¹W) − 1·(μᵀD⁻¹W), Zᵀ·Y = D⁻¹(XᵀY − μ·1ᵀY).
  const zTimes = (w: Float64Array, cols: number): Float64Array => {
    // w: G × cols column-major → out: B × cols column-major.
    const out = new Float64Array(B * cols);
    for (let c = 0; c < cols; c++) {
      const wc = c * G;
      let shift = 0;
      for (let g = 0; g < G; g++) shift += (mean[g]! / sd[g]!) * w[wc + g]!;
      const oc = c * B;
      for (let r = 0; r < B; r++) {
        let s = 0;
        for (let k = m.ptr[r]!; k < m.ptr[r + 1]!; k++) s += (val[k]! / sd[m.col[k]!]!) * w[wc + m.col[k]!]!;
        out[oc + r] = s - shift;
      }
    }
    return out;
  };
  const zTransposeTimes = (y: Float64Array, cols: number): Float64Array => {
    // y: B × cols column-major → out: G × cols column-major.
    const out = new Float64Array(G * cols);
    for (let c = 0; c < cols; c++) {
      const yc = c * B;
      const oc = c * G;
      let colSum = 0;
      for (let r = 0; r < B; r++) {
        const yr = y[yc + r]!;
        colSum += yr;
        for (let k = m.ptr[r]!; k < m.ptr[r + 1]!; k++) out[oc + m.col[k]!]! += val[k]! * yr;
      }
      for (let g = 0; g < G; g++) out[oc + g] = (out[oc + g]! - mean[g]! * colSum) / sd[g]!;
    }
    return out;
  };

  const rng = mulberry32(opts.seed ?? 0);
  const omega = new Float64Array(G * l);
  for (let i = 0; i < omega.length; i++) omega[i] = Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
  let q = zTimes(omega, l);
  orthonormalise(q, B, l);
  for (let it = 0; it < (opts.powerIterations ?? 4); it++) {
    const w = zTransposeTimes(q, l);
    orthonormalise(w, G, l);
    q = zTimes(w, l);
    orthonormalise(q, B, l);
  }
  // Small problem: Bt = Zᵀ Q (G × l). Its Gram BtᵀBt (l × l) = S² in Q's basis; V = Bt·U·S⁻¹.
  const bt = zTransposeTimes(q, l);
  const small = new Float64Array(l * l);
  for (let i = 0; i < l; i++) {
    for (let j = i; j < l; j++) {
      let s = 0;
      for (let g = 0; g < G; g++) s += bt[i * G + g]! * bt[j * G + g]!;
      small[i * l + j] = s;
      small[j * l + i] = s;
    }
  }
  const eig = eigenSym(small, l);
  const explained = new Float64Array(d);
  for (let k = 0; k < d; k++) explained[k] = Math.max(eig.values[k]!, 0) / (B * G);
  // Components of pure noise: a B × G matrix of independent unit-variance entries has covariance
  // eigenvalues up to (1 + √(G/B))² (Marchenko–Pastur), a share of (1 + √(G/B))² / G. Components
  // below that are indistinguishable from noise, and keeping them lets noise genes load on them and
  // look structured. At least one component is always kept.
  const noiseEdge = (1 + Math.sqrt(G / B)) ** 2 / G;
  let kept = 0;
  while (kept < d && explained[kept]! > noiseEdge) kept++;
  kept = Math.max(kept, 1);
  // A gene's correlation with component k is V[g,k]·S_k / sqrt(B), since its column has unit
  // variance over the B bins: (Bt·u_k)[g] / sqrt(B).
  const loadings = new Float64Array(G * kept);
  for (let k = 0; k < kept; k++) {
    const u = eig.vectors.subarray(k * l, k * l + l);
    for (let g = 0; g < G; g++) {
      let s = 0;
      for (let i = 0; i < l; i++) s += bt[i * G + g]! * u[i]!;
      loadings[g * kept + k] = s / Math.sqrt(B);
    }
  }
  return {
    codes: built.codes,
    loadings,
    components: kept,
    explained,
    noiseEdge,
    geneCounts: built.geneCounts,
    bins: B,
    molecules: built.molecules,
    dropped: built.dropped,
  };
}

export interface GeneModule {
  /** Feature codes in this module, most strongly structured first. */
  codes: number[];
  /** Molecules of the module's genes in the sample. */
  molecules: number;
}

export interface GeneModuleOptions {
  /** Genes whose communality (share of spatial variance explained by the embedding) is below this
   *  are left out of every module (default 0.1). */
  minCommunality?: number;
  epochs?: number;
  /** e-means starts; the lowest-MSE one is kept (default 5). */
  restarts?: number;
  seed?: number;
}

export interface GeneModules {
  /** Disjoint gene sets, largest molecule count first. */
  modules: GeneModule[];
  /** Codes left out for low communality. */
  unstructured: number[];
}

/**
 * Partition the structured genes of an embedding into `k` disjoint modules with e-means on their
 * loading rows (Euclidean, so a gene's strength of structure counts as well as its direction).
 */
export function geneModules(emb: GeneEmbedding, k: number, opts: GeneModuleOptions = {}): GeneModules {
  const d = emb.components;
  const G = emb.codes.length;
  const minCommunality = opts.minCommunality ?? 0.1;
  const communality = new Float64Array(G);
  const keep: number[] = [];
  const unstructured: number[] = [];
  for (let g = 0; g < G; g++) {
    let s = 0;
    for (let c = 0; c < d; c++) s += emb.loadings[g * d + c]! ** 2;
    communality[g] = s;
    if (s >= minCommunality) keep.push(g);
    else unstructured.push(emb.codes[g]!);
  }
  if (keep.length < k)
    throw new Error(`geneModules: only ${keep.length} genes are structured enough for ${k} modules; lower minCommunality or k`);
  const data = new Float64Array(keep.length * d);
  keep.forEach((g, i) => data.set(emb.loadings.subarray(g * d, g * d + d), i * d));
  // Gene clusters can be well separated, and e-means cannot move a centroid across empty space, so
  // seed with k-means++ and keep the best of a few starts. Cheap: only the structured genes (a few
  // hundred on Atera) are clustered.
  let fit = emeans(data, keep.length, d, k, { epochs: opts.epochs ?? 10, seed: opts.seed ?? 0, seeding: "kmeans++" });
  for (let r = 1; r < (opts.restarts ?? 5); r++) {
    const next = emeans(data, keep.length, d, k, { epochs: opts.epochs ?? 10, seed: (opts.seed ?? 0) + r, seeding: "kmeans++" });
    if (next.mse < fit.mse) fit = next;
  }
  const groups: number[][] = Array.from({ length: k }, () => []);
  keep.forEach((g, i) => groups[fit.labels[i]!]!.push(g));
  const modules = groups
    .filter((gs) => gs.length > 0)
    .map((gs) => {
      gs.sort((a, b) => communality[b]! - communality[a]!);
      return { codes: gs.map((g) => emb.codes[g]!), molecules: gs.reduce((s, g) => s + emb.geneCounts[g]!, 0) };
    })
    .sort((a, b) => b.molecules - a.molecules);
  return { modules, unstructured };
}
