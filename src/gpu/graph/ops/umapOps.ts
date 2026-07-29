// UMAP as op-graph nodes — three primitives and one high-level composite.
//
//   knn         matrix [n,dim]  -> opaque "knn"          (GPU; the expensive step)
//   fuzzyGraph  opaque "knn"    -> opaque "fuzzyGraph"   (host; O(N·k), exact)
//   umapLayout  opaque "fuzzy"  -> matrix [n,embedDim]   (host SGD)
//   umap        matrix [n,dim]  -> matrix [n,embedDim]   (all of the above, one node)
//
// **Why both granularities.** The composite is what you want on a canvas when UMAP is
// simply a step in a bigger pipeline. The primitives are what you want when the
// interesting object is the *intermediate*: the k-NN graph feeds the existing fuzzy-TDA
// front (`membershipToDistance` → `vietorisRips`) just as happily as it feeds a layout,
// and the fuzzy graph is a legitimate output in its own right — it is what scanpy stores
// in `obsp` and clusters on. Splitting the stages lets those wires exist. It also means
// a gene-subset change re-runs only `knn` onward, with PCA memoised, which is what makes
// the interactive path affordable.
//
// The two opaque payloads are structural, not lazy typing: a `KnnResult` is two arrays of
// different dtypes and a `FuzzyGraph` is three, and neither is a `matrix`. Flattening
// them into numeric ports would misrepresent both and force a repack at every edge.
//
// Distance/embedding conventions and the reference-semantics `nNeighbors` off-by-one are
// documented in `src/spatial/umapGraph.ts`; this module only wires them up.

import { pca } from "../../../spatial/pca";
import { type FuzzyGraph, fuzzySimplicialSet, type KnnResult, knnBruteForceCpu } from "../../../spatial/umapGraph";
import { fitAB, optimizeLayout } from "../../../spatial/umapLayout";
import { KNN_MAX_K, knnGpu } from "../../spatial/knn";
import type { Shape } from "../handle";
import type { OpType, Params } from "../op";

/** `[n, dim]` from a matrix port, with the error a user can act on. */
function matrixDims(s: Shape, who: string): { n: number; dim: number } {
  if (s.kind !== "matrix") throw new Error(`${who}: input must be a matrix [cells, features]`);
  return { n: s.rows, dim: s.cols };
}

function payloadOf<T>(value: { payload?: unknown }, who: string, what: string): T {
  if (!value.payload) throw new Error(`${who}: input carries no ${what} payload`);
  return value.payload as T;
}

/** Shared param definitions, so the composite and the primitives cannot drift apart. */
const NEIGHBORS_PARAM = {
  name: "nNeighbors",
  type: "number" as const,
  default: 15,
  min: 2,
  max: KNN_MAX_K + 1,
  step: 1,
  describe: "n_neighbors, counting the point itself (reference semantics)",
};
const MIN_DIST_PARAM = {
  name: "minDist",
  type: "number" as const,
  default: 0.1,
  min: 0,
  max: 1,
  step: 0.01,
  describe: "how tightly points may clump",
};
const EPOCHS_PARAM = { name: "nEpochs", type: "number" as const, default: 200, min: 10, max: 2000, step: 10 };
const EMBED_DIM_PARAM = { name: "embedDim", type: "number" as const, default: 2, min: 1, max: 3, step: 1, describe: "embedding dimension" };
const SEED_PARAM = { name: "seed", type: "number" as const, default: 42, min: 0, max: 100000, step: 1 };
const COMPONENTS_PARAM = {
  name: "nComponents",
  type: "number" as const,
  default: 50,
  min: 2,
  max: 200,
  step: 1,
  describe: "PCA dimensions before the k-NN (0 = no PCA)",
};

/** PCA-reduce when asked and when it would actually reduce anything. */
function maybeReduce(
  data: Float32Array | Int32Array | Uint32Array,
  n: number,
  dim: number,
  nComponents: number,
): { values: ArrayLike<number>; dim: number } {
  if (nComponents <= 0 || dim <= nComponents) return { values: data, dim };
  const res = pca(data, n, dim, { nComponents });
  return { values: res.scores, dim: res.nComponents };
}

// --- Primitive 1: k-NN ------------------------------------------------------------

export const knnOp: OpType = {
  name: "knn",
  label: "k-nearest neighbours",
  category: "Manifold",
  describe: "Exact k-NN over an [n, features] matrix (GPU). The expensive step of a UMAP.",
  help: {
    detail:
      "For each row, the k closest other rows by Euclidean distance. Brute force O(N^2 * D) on the GPU, exact. The approximate index that lifts the ~20k-cell ceiling drops in behind this same port, so nothing downstream changes.",
    math: "\\mathrm{knn}(i)=\\underset{j\\neq i}{\\arg\\min}^{\\,k}\\;\\lVert x_i-x_j\\rVert_2",
  },
  inputs: [{ name: "features", kind: "matrix" }],
  outputs: [{ name: "knn", kind: "opaque", dtype: "f32" }],
  params: [NEIGHBORS_PARAM, COMPONENTS_PARAM],
  inferShapes(inputs) {
    matrixDims(inputs[0]!, "knn");
    return [{ kind: "opaque", name: "knn" }];
  },
  async execute(_ctx, inputs, params) {
    const { n, dim } = matrixDims(inputs[0]!.shape, "knn");
    const k = (params.nNeighbors as number) - 1;
    if (k < 1) throw new Error("knn: nNeighbors must be at least 2");
    if (k >= n) throw new Error(`knn: nNeighbors-1 must be < rows (got ${k}, rows ${n})`);
    const reduced = maybeReduce(inputs[0]!.data!, n, dim, params.nComponents as number);
    const result = await knnGpu(reduced.values, { n, dim: reduced.dim, k });
    return [{ shape: { kind: "opaque", name: "knn" }, dtype: "f32", payload: result }];
  },
  cpuGolden(inputs, params) {
    const { n, dim } = matrixDims(inputs[0]!.shape, "knn");
    const reduced = maybeReduce(inputs[0]!.data!, n, dim, params.nComponents as number);
    const result = knnBruteForceCpu(reduced.values, n, reduced.dim, (params.nNeighbors as number) - 1);
    return [{ shape: { kind: "opaque", name: "knn" }, dtype: "f32", payload: result }];
  },
};

// --- Primitive 2: fuzzy simplicial set --------------------------------------------

export const fuzzyGraphOp: OpType = {
  name: "fuzzyGraph",
  label: "Fuzzy simplicial set",
  category: "Manifold",
  describe: "Calibrated smooth-kNN memberships, symmetrised by the probabilistic t-conorm. UMAP's graph.",
  help: {
    detail:
      "Solves a per-point bandwidth sigma_i so every point has the same smooth neighbour count log2(n_neighbors), then unions the two one-sided views with the probabilistic t-conorm. Density adaptation lives in rho_i (the nearest-neighbour offset); sigma_i carries the residual spread. This graph is what scanpy stores in obsp.",
    math: "\\mu_{i|j}=e^{-\\max(0,\\,d_{ij}-\\rho_i)/\\sigma_i},\\quad \\mu_{ij}=\\mu_{i|j}+\\mu_{j|i}-\\mu_{i|j}\\mu_{j|i}",
  },
  inputs: [{ name: "knn", kind: "opaque" }],
  outputs: [{ name: "graph", kind: "opaque", dtype: "f32" }],
  params: [
    {
      name: "localConnectivity",
      type: "number",
      default: 1,
      min: 0.5,
      max: 8,
      step: 0.5,
      describe: "neighbours assumed locally connected",
    },
    { name: "setOpMixRatio", type: "number", default: 1, min: 0, max: 1, step: 0.05, describe: "1 = fuzzy union, 0 = intersection" },
  ],
  inferShapes() {
    return [{ kind: "opaque", name: "fuzzyGraph" }];
  },
  async execute(_ctx, inputs, params) {
    return [buildGraph(inputs[0]!, params)];
  },
  cpuGolden(inputs, params) {
    // Host-only op: the golden IS the implementation, so they cannot disagree.
    return [buildGraph(inputs[0]!, params)];
  },
};

function buildGraph(input: { payload?: unknown }, params: Params) {
  const knn = payloadOf<KnnResult>(input, "fuzzyGraph", "k-NN");
  const graph = fuzzySimplicialSet(knn, {
    // `k` is the non-self count, so the reference `n_neighbors` is one more.
    nNeighbors: knn.k + 1,
    localConnectivity: params.localConnectivity as number,
    setOpMixRatio: params.setOpMixRatio as number,
  });
  return { shape: { kind: "opaque" as const, name: "fuzzyGraph" }, dtype: "f32" as const, payload: graph };
}

// --- Primitive 3: layout ----------------------------------------------------------

export const umapLayoutOp: OpType = {
  name: "umapLayout",
  label: "UMAP layout",
  category: "Manifold",
  describe: "Negative-sampling SGD over a fuzzy graph -> an [n, embedDim] embedding.",
  help: {
    detail:
      "Minimises the fuzzy cross-entropy between the high-dimensional graph and the embedding's own membership curve: attraction along every edge, repulsion against uniformly sampled non-edges, with a linearly decaying learning rate. a and b are fitted from minDist.",
    math: "\\Psi(d)=\\frac{1}{1+a\\,d^{2b}}",
  },
  inputs: [{ name: "graph", kind: "opaque" }],
  outputs: [{ name: "embedding", kind: "matrix", dtype: "f32" }],
  params: [MIN_DIST_PARAM, EPOCHS_PARAM, EMBED_DIM_PARAM, SEED_PARAM],
  inferShapes(_inputs, params) {
    // The row count lives in the opaque payload, not the shape, so it is not knowable
    // until execution. Report 0 rows at build time; the executor stamps the real shape.
    return [{ kind: "matrix", rows: 0, cols: params.embedDim as number }];
  },
  async execute(_ctx, inputs, params) {
    return [runLayout(inputs[0]!, params)];
  },
  cpuGolden(inputs, params) {
    return [runLayout(inputs[0]!, params)];
  },
};

function runLayout(input: { payload?: unknown }, params: Params) {
  const graph = payloadOf<FuzzyGraph>(input, "umapLayout", "fuzzy graph");
  const embedDim = params.embedDim as number;
  const embedding = optimizeLayout(graph, {
    dim: embedDim,
    nEpochs: params.nEpochs as number,
    seed: params.seed as number,
    ab: fitAB(params.minDist as number, 1),
  });
  return { shape: { kind: "matrix" as const, rows: graph.n, cols: embedDim }, dtype: "f32" as const, data: embedding };
}

// --- High-level composite ---------------------------------------------------------

export const umapOp: OpType = {
  name: "umap",
  label: "UMAP",
  category: "Manifold",
  describe: "Expression matrix -> 2-D embedding. PCA, k-NN, fuzzy graph and layout in one node.",
  help: {
    detail:
      "The whole pipeline as one node, for when UMAP is a step rather than the subject. Wire knn -> fuzzyGraph -> umapLayout by hand instead when you want the intermediates: the k-NN graph also feeds the fuzzy-TDA front, and the fuzzy graph is a legitimate output in its own right.",
  },
  inputs: [{ name: "features", kind: "matrix" }],
  outputs: [{ name: "embedding", kind: "matrix", dtype: "f32" }],
  params: [NEIGHBORS_PARAM, COMPONENTS_PARAM, MIN_DIST_PARAM, EPOCHS_PARAM, EMBED_DIM_PARAM, SEED_PARAM],
  inferShapes(inputs, params) {
    const { n } = matrixDims(inputs[0]!, "umap");
    return [{ kind: "matrix", rows: n, cols: params.embedDim as number }];
  },
  async execute(ctx, inputs, params) {
    const knn = await knnOp.execute(ctx, inputs, params);
    const graph = await fuzzyGraphOp.execute(ctx, knn, { localConnectivity: 1, setOpMixRatio: 1 });
    return umapLayoutOp.execute(ctx, graph, params);
  },
  cpuGolden(inputs, params) {
    const knn = knnOp.cpuGolden!(inputs, params);
    const graph = fuzzyGraphOp.cpuGolden!(knn, { localConnectivity: 1, setOpMixRatio: 1 });
    return umapLayoutOp.cpuGolden!(graph, params);
  },
};

export const UMAP_OPS = [knnOp, fuzzyGraphOp, umapLayoutOp, umapOp];
