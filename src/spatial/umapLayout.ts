// UMAP's layout optimisation — the fuzzy graph → embedding half, host-side.
//
// Given the weighted graph from `umapGraph.ts`, find low-dimensional coordinates whose
// own membership function matches it. The objective is the fuzzy cross-entropy between
// the high-D graph and the low-D one; the low-D membership is the smooth curve
//
//     Ψ(d) = 1 / (1 + a · d^(2b))
//
// with `a`, `b` fitted so Ψ approximates the piecewise target
// `1 if d ≤ min_dist else exp(-(d - min_dist)/spread)`. Minimising the cross-entropy by
// SGD gives an attractive force along every graph edge and a repulsive force against
// sampled non-edges.
//
// **This module is the golden and the small-N path; it is not where the animation
// lives.** The interactive story — see `docs/umap-on-anndata.md` §4 — keeps ONE
// embedding buffer resident on the GPU and keeps stepping it while the edge list
// underneath is swapped for a different gene or cell subset, so the picture relaxes
// continuously from one layout into the next. That needs the SGD to be a resumable
// step over persistent state rather than a `fit()` call that owns its own loop, which
// is why `optimizeLayoutStep` takes and returns the embedding rather than hiding it.
//
// **On determinism.** The GPU counterpart runs edges in parallel against a shared
// position buffer, so it races (the Hogwild! regime every GPU UMAP uses). It therefore
// cannot be checked against this module elementwise, and the tests do not pretend
// otherwise: parity is asserted on the *structure* the embedding preserves
// (`trustworthiness`), not on coordinates. See `umapLayout.gpu.test.ts`.

/** Curve parameters for the low-dimensional membership Ψ(d) = 1/(1 + a·d^(2b)). */
export interface AbParams {
  readonly a: number;
  readonly b: number;
}

/**
 * Fit `a`, `b` so that `1/(1 + a·d^(2b))` approximates UMAP's piecewise target.
 *
 * `minDist` is how tightly points may clump (the flat region of the target); `spread`
 * sets the scale over which membership decays. The reference implementation calls
 * scipy's `curve_fit`; this is Gauss-Newton with a Levenberg damping term on 300
 * samples, which converges in a handful of iterations on a 2-parameter problem and
 * avoids a dependency for what is ultimately two numbers.
 */
export function fitAB(minDist = 0.1, spread = 1): AbParams {
  const M = 300;
  const xs = new Float64Array(M);
  const ys = new Float64Array(M);
  for (let i = 0; i < M; i++) {
    const x = (3 * spread * i) / (M - 1);
    xs[i] = x;
    ys[i] = x <= minDist ? 1 : Math.exp(-(x - minDist) / spread);
  }

  let a = 1;
  let b = 1;
  let lambda = 1e-3;
  for (let iter = 0; iter < 100; iter++) {
    // Normal equations for the 2-parameter Gauss-Newton step.
    let h00 = 0;
    let h01 = 0;
    let h11 = 0;
    let g0 = 0;
    let g1 = 0;
    let sse = 0;
    for (let i = 0; i < M; i++) {
      const x = xs[i]!;
      if (x === 0) continue; // x^(2b) and its log-derivative are both singular at 0
      const p = x ** (2 * b);
      const denom = 1 + a * p;
      const f = 1 / denom;
      const r = f - ys[i]!;
      sse += r * r;
      // ∂f/∂a = -p/denom² ;  ∂f/∂b = -2a·p·ln(x)/denom²
      const da = -p / (denom * denom);
      const db = (-2 * a * p * Math.log(x)) / (denom * denom);
      h00 += da * da;
      h01 += da * db;
      h11 += db * db;
      g0 += da * r;
      g1 += db * r;
    }
    const d00 = h00 * (1 + lambda);
    const d11 = h11 * (1 + lambda);
    const det = d00 * d11 - h01 * h01;
    if (Math.abs(det) < 1e-300) break;
    const stepA = (-(d11 * g0) + h01 * g1) / det;
    const stepB = (h01 * g0 - d00 * g1) / det;
    const na = a + stepA;
    const nb = b + stepB;
    if (!(na > 0) || !(nb > 0)) {
      // Left the feasible region — damp harder and retry from where we are.
      lambda *= 10;
      if (lambda > 1e10) break;
      continue;
    }
    let nsse = 0;
    for (let i = 0; i < M; i++) {
      const x = xs[i]!;
      if (x === 0) continue;
      const r = 1 / (1 + na * x ** (2 * nb)) - ys[i]!;
      nsse += r * r;
    }
    if (nsse < sse) {
      a = na;
      b = nb;
      lambda = Math.max(lambda * 0.3, 1e-12);
      if (Math.abs(stepA) + Math.abs(stepB) < 1e-12) break;
    } else {
      lambda *= 10;
      if (lambda > 1e10) break;
    }
  }
  return { a, b };
}

/** Deterministic 32-bit PRNG — the repo's standard seeded generator, so a run with a
 *  given seed is reproducible on the host and the GPU kernel can mirror the same
 *  recurrence per thread. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Per-edge sampling period, in epochs.
 *
 * The SGD does not visit every edge every epoch — it visits each edge at a rate
 * proportional to its weight, which is what makes the cost independent of how skewed
 * the weight distribution is. An edge with the maximum weight is applied every epoch;
 * one at half that, every other epoch. Edges too weak to be sampled even once over the
 * whole run get `Infinity` and are never touched.
 */
export function makeEpochsPerSample(weight: Float32Array, nEpochs: number): Float64Array {
  let maxW = 0;
  for (let e = 0; e < weight.length; e++) if (weight[e]! > maxW) maxW = weight[e]!;
  const out = new Float64Array(weight.length);
  for (let e = 0; e < weight.length; e++) {
    const nSamples = (nEpochs * weight[e]!) / Math.max(maxW, 1e-12);
    out[e] = nSamples > 0 ? nEpochs / nSamples : Number.POSITIVE_INFINITY;
  }
  return out;
}

/** Gradients are clamped to this magnitude per component. Without it a pair that lands
 *  almost on top of each other produces an unbounded repulsion and the embedding
 *  explodes on the first epoch. */
const GRAD_CLIP = 4;

function clip(v: number): number {
  return v > GRAD_CLIP ? GRAD_CLIP : v < -GRAD_CLIP ? -GRAD_CLIP : v;
}

export interface LayoutState {
  /** Row-major `[n, dim]`, mutated in place by each step. */
  readonly embedding: Float32Array;
  readonly n: number;
  readonly dim: number;
  /** Next epoch at which each edge is due to be sampled. */
  readonly epochOfNextSample: Float64Array;
  readonly epochsPerSample: Float64Array;
  /** Same, for the repulsive (negative-sample) side — negatives are drawn at a rate
   *  tied to the edge's own rate, so dense regions repel proportionally more. */
  readonly epochOfNextNegativeSample: Float64Array;
  readonly epochsPerNegativeSample: Float64Array;
  /** Epochs completed so far. */
  epoch: number;
}

export interface LayoutOptions {
  readonly dim?: number;
  readonly nEpochs?: number;
  readonly negativeSampleRate?: number;
  /** Repulsion strength γ. Reference default 1. */
  readonly repulsionStrength?: number;
  readonly initialAlpha?: number;
  readonly seed?: number;
  readonly ab?: AbParams;
}

/**
 * Initialise persistent SGD state for a graph.
 *
 * `embedding` may be supplied to **continue from an existing layout** — that is the
 * animated-transition path: build the new subset's graph, hand it the coordinates the
 * previous subset ended at, and the optimiser relaxes rather than restarting. Omit it
 * for a fresh random init.
 */
export function initLayout(
  graph: { readonly n: number; readonly weight: Float32Array },
  opts: LayoutOptions = {},
  embedding?: Float32Array,
): LayoutState {
  const dim = opts.dim ?? 2;
  const nEpochs = opts.nEpochs ?? 200;
  const negativeSampleRate = opts.negativeSampleRate ?? 5;
  const rnd = mulberry32(opts.seed ?? 42);

  let emb = embedding;
  if (!emb) {
    emb = new Float32Array(graph.n * dim);
    // Uniform in [-10, 10], the reference's random-init scale. Spectral init would be
    // better conditioned, but it needs an eigensolve on an N×N Laplacian and the
    // continuation path above makes a good init much less load-bearing here.
    for (let t = 0; t < emb.length; t++) emb[t] = rnd() * 20 - 10;
  } else if (emb.length !== graph.n * dim) {
    throw new Error(`initLayout: embedding has ${emb.length} entries, expected ${graph.n * dim}`);
  }

  const epochsPerSample = makeEpochsPerSample(graph.weight, nEpochs);
  const epochsPerNegativeSample = Float64Array.from(epochsPerSample, (v) => v / negativeSampleRate);
  return {
    embedding: emb,
    n: graph.n,
    dim,
    epochsPerSample,
    epochOfNextSample: Float64Array.from(epochsPerSample),
    epochsPerNegativeSample,
    epochOfNextNegativeSample: Float64Array.from(epochsPerNegativeSample),
    epoch: 0,
  };
}

/**
 * Rewind an existing layout's anneal so it starts moving again.
 *
 * Resetting `state.epoch` alone is NOT enough, and getting this wrong is silent: the
 * per-edge schedule (`epochOfNextSample`) has already advanced past wherever you rewind
 * to, so no edge is ever due again and the optimiser does nothing while reporting a
 * healthy non-zero learning rate. The layout simply freezes. The schedule has to be
 * re-based onto the new epoch as well, which is what this does.
 *
 * Use it whenever an already-settled layout should resume — perturbing the embedding and
 * watching it re-converge, or continuing after the decay horizon has been reached.
 * Carrying an embedding across a *graph swap* does not need it, because that path builds
 * a fresh state via `initLayout`.
 *
 * `epoch` is where to rewind to: 0 restarts the full schedule, and something part-way
 * (the default) gives a gentler recovery than re-running the whole anneal.
 */
export function reheatLayout(state: LayoutState, epoch = 0): void {
  state.epoch = Math.max(0, epoch);
  for (let e = 0; e < state.epochsPerSample.length; e++) {
    state.epochOfNextSample[e] = state.epoch + state.epochsPerSample[e]!;
    state.epochOfNextNegativeSample[e] = state.epoch + state.epochsPerNegativeSample[e]!;
  }
}

/**
 * Run one epoch of the layout SGD, in place.
 *
 * Returns the learning rate that was used, which decays linearly to zero over
 * `nEpochs` — a caller driving an animation can watch it to know when the layout has
 * settled, or override `nEpochs` upward to keep it live.
 */
export function optimizeLayoutStep(
  state: LayoutState,
  graph: { readonly head: Uint32Array; readonly tail: Uint32Array; readonly nEdges: number },
  opts: LayoutOptions = {},
): number {
  const { embedding, dim, n } = state;
  const nEpochs = opts.nEpochs ?? 200;
  const gamma = opts.repulsionStrength ?? 1;
  const { a, b } = opts.ab ?? fitAB();
  const alpha = (opts.initialAlpha ?? 1) * Math.max(0, 1 - state.epoch / nEpochs);
  const rnd = mulberry32((opts.seed ?? 42) + state.epoch * 0x9e3779b9);
  const epoch = state.epoch;

  for (let e = 0; e < graph.nEdges; e++) {
    if (state.epochOfNextSample[e]! > epoch) continue;
    const j = graph.head[e]!;
    const kk = graph.tail[e]!;
    const jb = j * dim;
    const kb = kk * dim;

    let d2 = 0;
    for (let c = 0; c < dim; c++) {
      const delta = embedding[jb + c]! - embedding[kb + c]!;
      d2 += delta * delta;
    }

    // Attractive: ∂/∂d of the cross-entropy's "edge exists" term.
    let coeff = 0;
    if (d2 > 0) coeff = (-2 * a * b * d2 ** (b - 1)) / (a * d2 ** b + 1);
    for (let c = 0; c < dim; c++) {
      const grad = clip(coeff * (embedding[jb + c]! - embedding[kb + c]!));
      embedding[jb + c] = embedding[jb + c]! + grad * alpha;
      embedding[kb + c] = embedding[kb + c]! - grad * alpha;
    }

    state.epochOfNextSample[e] = state.epochOfNextSample[e]! + state.epochsPerSample[e]!;

    // Repulsive: a number of uniformly-sampled non-neighbours proportional to how
    // often this edge itself has come due.
    const nNeg = Math.floor((epoch - state.epochOfNextNegativeSample[e]!) / state.epochsPerNegativeSample[e]!);
    for (let p = 0; p < nNeg; p++) {
      const other = Math.floor(rnd() * n);
      if (other === j) continue;
      const ob = other * dim;
      let nd2 = 0;
      for (let c = 0; c < dim; c++) {
        const delta = embedding[jb + c]! - embedding[ob + c]!;
        nd2 += delta * delta;
      }
      let ncoeff = 0;
      if (nd2 > 0) ncoeff = (2 * gamma * b) / ((0.001 + nd2) * (a * nd2 ** b + 1));
      else if (j === other) continue;
      for (let c = 0; c < dim; c++) {
        // A coincident pair has no direction to separate along; nudge it arbitrarily
        // rather than dividing by zero.
        const grad = ncoeff > 0 ? clip(ncoeff * (embedding[jb + c]! - embedding[ob + c]!)) : 4;
        embedding[jb + c] = embedding[jb + c]! + grad * alpha;
      }
    }
    if (nNeg > 0) {
      state.epochOfNextNegativeSample[e] = state.epochOfNextNegativeSample[e]! + nNeg * state.epochsPerNegativeSample[e]!;
    }
  }

  state.epoch = epoch + 1;
  return alpha;
}

/** Convenience: run `nEpochs` steps and return the embedding. The small-N / golden
 *  path; the interactive path drives `optimizeLayoutStep` from its own frame loop. */
export function optimizeLayout(
  graph: {
    readonly n: number;
    readonly head: Uint32Array;
    readonly tail: Uint32Array;
    readonly weight: Float32Array;
    readonly nEdges: number;
  },
  opts: LayoutOptions = {},
  embedding?: Float32Array,
): Float32Array {
  const nEpochs = opts.nEpochs ?? 200;
  const ab = opts.ab ?? fitAB();
  const state = initLayout(graph, { ...opts, nEpochs }, embedding);
  for (let e = 0; e < nEpochs; e++) optimizeLayoutStep(state, graph, { ...opts, nEpochs, ab });
  return state.embedding;
}

/**
 * Trustworthiness — the fraction of each point's low-D neighbours that were also its
 * high-D neighbours, penalised by how far down the high-D ranking the intruders came
 * from. 1 is perfect, ~0.5 is what random coordinates score.
 *
 * This is the assertion the GPU layout is tested against. A racing parallel SGD cannot
 * reproduce host coordinates and it is not supposed to — what it must reproduce is a
 * layout that preserves the same neighbourhoods, and this measures exactly that.
 */
export function trustworthiness(
  high: ArrayLike<number>,
  low: ArrayLike<number>,
  n: number,
  highDim: number,
  lowDim: number,
  k = 10,
): number {
  const rank = highRanks(high, n, highDim);
  let total = 0;
  for (let i = 0; i < n; i++) {
    const lowNeighbours = nearestK(low, n, lowDim, i, k);
    for (const j of lowNeighbours) {
      const r = rank[i * n + j]!;
      if (r > k) total += r - k;
    }
  }
  const norm = (2 / (n * k * (2 * n - 3 * k - 1))) * total;
  return 1 - norm;
}

/** Rank of every point in every other point's high-D ordering (1 = nearest). */
function highRanks(data: ArrayLike<number>, n: number, dim: number): Uint32Array {
  const rank = new Uint32Array(n * n);
  const order: number[] = Array.from({ length: n }, (_, i) => i);
  const dist = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) dist[j] = sqDist(data, dim, i, j);
    order.sort((p, q) => dist[p]! - dist[q]!);
    // order[0] is i itself; ranks start at 1 for the nearest other point.
    for (let r = 0; r < n; r++) rank[i * n + order[r]!] = r;
    order.sort((p, q) => p - q);
  }
  return rank;
}

function nearestK(data: ArrayLike<number>, n: number, dim: number, i: number, k: number): number[] {
  const idx: number[] = [];
  for (let j = 0; j < n; j++) if (j !== i) idx.push(j);
  idx.sort((p, q) => sqDist(data, dim, i, p) - sqDist(data, dim, i, q));
  return idx.slice(0, k);
}

function sqDist(data: ArrayLike<number>, dim: number, i: number, j: number): number {
  let acc = 0;
  for (let c = 0; c < dim; c++) {
    const delta = data[i * dim + c]! - data[j * dim + c]!;
    acc += delta * delta;
  }
  return acc;
}
