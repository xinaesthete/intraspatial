// UMAP end to end — the one entry point both consumers use.
//
// The browser page and the offline `obsm` writer run the SAME code: the only thing
// that differs between them is which k-NN implementation is injected and whether the
// caller drives the epochs itself. That is deliberate — the whole point of writing
// into `obsm` offline is that it agrees with what the interactive view shows, and the
// cheapest way to guarantee that is to have one implementation.
//
// Pipeline: `[N, G] expression → (PCA) → [N, d] → k-NN → fuzzy graph → SGD → [N, 2]`.
//
//   • PCA          `pca.ts`         — optional but on by default past ~`pcaThreshold` genes
//   • k-NN         injected         — `knnGpu` in the browser, `knnBruteForceCpu` in tests
//   • fuzzy graph  `umapGraph.ts`   — smooth-kNN calibration + t-conorm, sparse
//   • layout       `umapLayout.ts`  — negative-sampling SGD
//
// For the ANIMATED case do not call `umap()` in a loop. Use `umapGraphFor()` to build
// the graph for a subset and hand it, plus the previous embedding, to
// `initLayout`/`optimizeLayoutStep` from `umapLayout.ts`. Two independent UMAP runs
// differ by an arbitrary rotation and reflection even on nearly identical data, so
// tweening between them shows motion that means nothing; continuing one optimiser
// across a graph swap shows the layout actually relaxing. See
// `docs/umap-on-anndata.md` §4.

import { type PcaOptions, pca } from "./pca";
import { type FuzzyGraph, type FuzzyGraphOptions, fuzzySimplicialSet, type KnnResult, knnBruteForceCpu } from "./umapGraph";
import { type AbParams, fitAB, type LayoutOptions, optimizeLayout } from "./umapLayout";

/** Injected k-NN. `knnBruteForceCpu` is the default; pass `knnGpu` in the browser. */
export type KnnFn = (data: ArrayLike<number>, n: number, dim: number, k: number) => Promise<KnnResult> | KnnResult;

export interface UmapOptions extends Omit<FuzzyGraphOptions, "nNeighbors">, LayoutOptions, PcaOptions {
  /** `n_neighbors` in reference semantics (counts the point itself). Default 15. */
  readonly nNeighbors?: number;
  /** Reduce to this many PCs before the k-NN. Set `pca: false` to skip entirely. */
  readonly pca?: boolean;
  /** Skip PCA when the feature count is at or below this. Default 50 — below it the
   *  reduction would not be reducing anything. */
  readonly pcaThreshold?: number;
  readonly knn?: KnnFn;
  /** Reuse an existing layout as the starting point (the continuation path). */
  readonly initialEmbedding?: Float32Array;
  readonly minDist?: number;
  readonly spread?: number;
}

export interface UmapResult {
  /** Row-major `[n, dim]`, `dim` = 2 unless overridden. */
  readonly embedding: Float32Array;
  readonly graph: FuzzyGraph;
  readonly knn: KnnResult;
  /** Present when PCA ran; the reduced matrix the k-NN actually saw. */
  readonly reduced?: Float32Array;
  readonly reducedDim?: number;
  readonly ab: AbParams;
}

const DEFAULTS = { nNeighbors: 15, minDist: 0.1, spread: 1, nEpochs: 200, pcaThreshold: 50, nComponents: 50 };

/**
 * Build the fuzzy graph for a feature matrix — everything up to, but not including,
 * the layout.
 *
 * Split out from `umap()` because the animated path needs exactly this: a new graph
 * for the new gene/cell subset, to hand to an optimiser that is already running.
 */
export async function umapGraphFor(
  data: ArrayLike<number>,
  n: number,
  dim: number,
  opts: UmapOptions = {},
): Promise<{ graph: FuzzyGraph; knn: KnnResult; reduced?: Float32Array; reducedDim?: number }> {
  const nNeighbors = opts.nNeighbors ?? DEFAULTS.nNeighbors;
  if (nNeighbors < 2) throw new Error("umap: nNeighbors must be >= 2");
  if (n < nNeighbors) throw new Error(`umap: need n >= nNeighbors (n=${n}, nNeighbors=${nNeighbors})`);

  // Reference semantics: `n_neighbors` counts the point itself, so the search asks for
  // one fewer. `umapGraph.ts` documents why the off-by-one is reproduced rather than
  // tidied away.
  const k = nNeighbors - 1;

  let searchData: ArrayLike<number> = data;
  let searchDim = dim;
  let reduced: Float32Array | undefined;
  const wantPca = opts.pca ?? dim > (opts.pcaThreshold ?? DEFAULTS.pcaThreshold);
  if (wantPca && dim > 1) {
    const res = pca(data, n, dim, { nComponents: opts.nComponents ?? DEFAULTS.nComponents, standardise: opts.standardise });
    reduced = res.scores;
    searchData = res.scores;
    searchDim = res.nComponents;
  }

  const knnFn = opts.knn ?? ((d, nn, dd, kk) => knnBruteForceCpu(d, nn, dd, kk));
  const knn = await knnFn(searchData, n, searchDim, k);
  const graph = fuzzySimplicialSet(knn, {
    nNeighbors,
    localConnectivity: opts.localConnectivity,
    bandwidth: opts.bandwidth,
    setOpMixRatio: opts.setOpMixRatio,
  });
  return { graph, knn, reduced, reducedDim: reduced ? searchDim : undefined };
}

/**
 * Run the whole pipeline and return an embedding.
 *
 * The batch path — offline `obsm` writing, tests, and the first frame of an
 * interactive session. It owns its own epoch loop, so it is not the one to call per
 * frame.
 */
export async function umap(data: ArrayLike<number>, n: number, dim: number, opts: UmapOptions = {}): Promise<UmapResult> {
  const { graph, knn, reduced, reducedDim } = await umapGraphFor(data, n, dim, opts);
  const ab = opts.ab ?? fitAB(opts.minDist ?? DEFAULTS.minDist, opts.spread ?? DEFAULTS.spread);
  const embedding = optimizeLayout(
    graph,
    {
      dim: opts.dim ?? 2,
      nEpochs: opts.nEpochs ?? DEFAULTS.nEpochs,
      negativeSampleRate: opts.negativeSampleRate,
      repulsionStrength: opts.repulsionStrength,
      initialAlpha: opts.initialAlpha,
      seed: opts.seed,
      ab,
    },
    opts.initialEmbedding,
  );
  return { embedding, graph, knn, reduced, reducedDim, ab };
}

/**
 * Take a row-subset of a row-major `[n, dim]` matrix.
 *
 * The cell-subset half of "animated transitions between subsets of genes/cells".
 * Kept here rather than left to callers because the index array it returns is what
 * maps embedding rows back to original `obs` rows — and getting that mapping wrong is
 * the easiest way to write a scrambled `obsm`.
 */
export function subsetRows(data: ArrayLike<number>, dim: number, rows: ArrayLike<number>): Float32Array {
  const out = new Float32Array(rows.length * dim);
  for (let r = 0; r < rows.length; r++) {
    const src = rows[r]! * dim;
    const dst = r * dim;
    for (let c = 0; c < dim; c++) out[dst + c] = data[src + c]!;
  }
  return out;
}

/**
 * Take a column-subset — the gene-selection half.
 *
 * `columns` indexes features; order is preserved, so the caller's gene-name list lines
 * up with the output columns one for one.
 */
export function subsetColumns(data: ArrayLike<number>, n: number, dim: number, columns: ArrayLike<number>): Float32Array {
  const out = new Float32Array(n * columns.length);
  for (let i = 0; i < n; i++) {
    const src = i * dim;
    const dst = i * columns.length;
    for (let c = 0; c < columns.length; c++) out[dst + c] = data[src + columns[c]!]!;
  }
  return out;
}
