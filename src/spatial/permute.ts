// The null the envelopes are drawn against: **random labelling**.
//
// ## Why not complete spatial randomness
//
// The textbook null for a point-process statistic is CSR — throw the points down uniformly and
// recompute. For tissue that is a straw man. Cells are nowhere near uniform: there is dense
// epithelium and sparse stroma and empty lumen, so a CSR null rejects for essentially every channel
// pair, and what it has detected is that the section has anatomy. Nobody needed a test for that.
//
// Random labelling keeps every cell exactly where it is and shuffles the **marks** between them. The
// question becomes the one actually of interest: *given where the cells are, is the assignment of
// types (or expression) to them spatially arranged?* Positions carry the anatomy, so holding them
// fixed conditions the test on it.
//
// ## One permutation for all channels, not one each
//
// This is the decision that changes what the answer means, so it is not configurable.
//
// Permuting each channel independently destroys within-cell co-expression as well as spatial
// arrangement. The null then reads "these genes are unrelated *and* unarranged", and the test
// rejects on cells that co-express — which has no spatial content at all and is exactly the
// confound `gram.ts`'s `selfTerm` exists to document. Applying a single permutation to every
// channel moves each cell's whole profile together, so co-expression is preserved exactly and only
// the geography is destroyed. That isolates the spatial question, which is the only one the
// statistic is entitled to answer.
//
// For one-hot cell types the two are identical (a cell has one mark either way), so the distinction
// is invisible until the marks come from `X` — and then it is the whole ball game.
//
// ## The analytic mean, and why it is here
//
// `nullMeanGram` computes `E[C]` under this null in closed form. It is not an optimisation: it is
// the check that the shuffle is doing what it claims. A Monte Carlo mean that does not converge to
// it means the permutation is wrong somewhere, and that is a failure mode with no other symptom —
// biased simulations produce a perfectly plausible-looking envelope in the wrong place.

import type { ChannelCloud, GramParams } from "./gram";
import { gramMatrix } from "./gram";
import { EPANECHNIKOV, roughness } from "./kernels";

/**
 * A permutation of `0..n-1`, Fisher–Yates. `rnd` returns uniform `[0, 1)`.
 *
 * Pass `out` to fill an existing array instead of allocating one; a simulation loop reshuffles the
 * same buffer thousands of times and has no use for the old contents.
 */
export function randomPermutation(n: number, rnd: () => number, out?: Uint32Array): Uint32Array {
  const perm = out && out.length === n ? out : new Uint32Array(n);
  for (let i = 0; i < n; i++) perm[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    const t = perm[i]!;
    perm[i] = perm[j]!;
    perm[j] = t;
  }
  return perm;
}

/** True when every channel indexes the same point set — the expression case, where a permutation is
 *  a reshuffle of weights. False when the channels partition the points (one-hot types), where it
 *  is a reshuffle of membership. Identity of the arrays is the test `gram.ts` already uses to decide
 *  whether `selfTerm` has off-diagonal content, so the two agree about what "shared points" means. */
function sharesPoints(channels: readonly ChannelCloud[]): boolean {
  const first = channels[0];
  if (!first) return false;
  return channels.every((c) => c.xs === first.xs && c.ys === first.ys);
}

const weightAt = (c: ChannelCloud, i: number): number => (c.weights ? (c.weights[i] ?? 0) : 1);

/** The number of cells a permutation for these channels must cover. */
export function cellCount(channels: readonly ChannelCloud[]): number {
  if (channels.length === 0) return 0;
  return sharesPoints(channels) ? channels[0]!.xs.length : channels.reduce((s, c) => s + c.xs.length, 0);
}

export interface ChannelPermuter {
  /** Cells the permutation must cover. */
  readonly cells: number;
  /**
   * Move every cell's mark to the position named by `perm`, leaving the positions alone.
   *
   * **The returned channels alias reused scratch buffers and are valid only until the next
   * `apply`.** That is the same contract `gramMatrixGpu`'s resident rasters carry, and for the same
   * reason: a simulation loop consumes each realisation immediately, and allocating a fresh set
   * per realisation dominated the run — at 162k cells and 8 channels it was ~10 MB of garbage each
   * time, and roughly 200 ms of the 340 ms per realisation was allocation and collection rather
   * than GPU work. Keep a realisation past the next call and it will silently be a later one.
   */
  apply(perm: Uint32Array): ChannelCloud[];
}

/**
 * A reusable permuter over one channel set.
 *
 * Handles both channel shapes, because the caller should not have to know which it has:
 *
 *   * **shared points** — every channel's weight vector is permuted by the *same* `perm`, so each
 *     cell's whole profile lands together, and the positions are handed back by reference;
 *   * **partitioned points** — the concatenated cells are regrouped by permuted membership. Each
 *     channel's cell *count* is invariant under relabelling, which is what lets the destination
 *     arrays be allocated once and filled by index instead of pushed into.
 */
export function channelPermuter(channels: readonly ChannelCloud[]): ChannelPermuter {
  const cells = cellCount(channels);
  if (channels.length === 0) return { cells: 0, apply: () => [] };

  const check = (perm: Uint32Array) => {
    if (perm.length !== cells) throw new Error(`permuteChannels: perm length ${perm.length} != ${cells} cells`);
  };

  if (sharesPoints(channels)) {
    const first = channels[0]!;
    const K = channels.length;
    const buf = Array.from({ length: K }, () => new Float64Array(cells));
    // Source weights read once into a dense array: `channelsFromExpression` hands back subarray
    // views, and going through `weightAt` per element inside the hot loop costs a branch each time.
    const src = channels.map((c) => {
      const w = new Float64Array(cells);
      for (let i = 0; i < cells; i++) w[i] = weightAt(c, i);
      return w;
    });
    return {
      cells,
      apply(perm) {
        check(perm);
        const out: ChannelCloud[] = [];
        for (let a = 0; a < K; a++) {
          const dst = buf[a]!;
          const s = src[a]!;
          for (let i = 0; i < cells; i++) dst[i] = s[perm[i]!]!;
          out.push({ label: channels[a]!.label, xs: first.xs, ys: first.ys, weights: dst });
        }
        return out;
      },
    };
  }

  // Partitioned: flatten once, then regroup per call.
  const counts = channels.map((c) => c.xs.length);
  const flatX = new Float64Array(cells);
  const flatY = new Float64Array(cells);
  const flatW = new Float64Array(cells);
  const owner = new Int32Array(cells);
  let at = 0;
  channels.forEach((c, k) => {
    for (let i = 0; i < c.xs.length; i++) {
      flatX[at] = c.xs[i] ?? 0;
      flatY[at] = c.ys[i] ?? 0;
      flatW[at] = weightAt(c, i);
      owner[at] = k;
      at++;
    }
  });
  const outX = counts.map((n) => new Float64Array(n));
  const outY = counts.map((n) => new Float64Array(n));
  const outW = counts.map((n) => new Float64Array(n));
  const fill = new Int32Array(channels.length);
  return {
    cells,
    apply(perm) {
      check(perm);
      fill.fill(0);
      for (let i = 0; i < cells; i++) {
        const src = perm[i]!;
        const k = owner[src]!;
        const j = fill[k]!;
        outX[k]![j] = flatX[i]!;
        outY[k]![j] = flatY[i]!;
        outW[k]![j] = flatW[src]!;
        fill[k] = j + 1;
      }
      return channels.map((c, k) => ({ label: c.label, xs: outX[k]!, ys: outY[k]!, weights: outW[k]! }));
    },
  };
}

/**
 * One-shot permutation, allocating its own buffers.
 *
 * The convenient form, for tests and single uses. A loop over many realisations should build one
 * `channelPermuter` and reuse it — see its aliasing note for what that costs you in exchange.
 */
export function permuteChannels(channels: readonly ChannelCloud[], perm: Uint32Array): ChannelCloud[] {
  return channelPermuter(channels).apply(perm);
}

/**
 * `E[C]` under random labelling, in closed form — K×K, row-major.
 *
 * Splitting the double sum at `i = j` and taking the expectation over a uniform permutation, the
 * marks and the geometry separate completely:
 *
 *     E[C_ab] = φ(0)·S_ab + Φ·(W_a·W_b − S_ab) / (n(n−1))
 *
 * with `φ = J ⊛ J`, `S_ab = Σ_k w_a(k)·w_b(k)`, `W_a = Σ_k w_a(k)`, and `Φ = Σ_{i≠j} φ(x_i − x_j)`
 * the *only* geometric quantity involved. `Φ` in turn is `∫(J⊛ρ)² − n·φ(0)` for the total point
 * measure `ρ`, so it costs one splat of all cells and one reduction — **independent of how many
 * permutations you were going to run**. The mean of the null is free; only its spread needs Monte
 * Carlo.
 *
 * Assumes every cell lies inside the analysis window. With an apron populated, `C` integrates over
 * the window while `Φ` here is the unrestricted double sum, and the two stop matching; this is a
 * check on the shuffle, not a general-purpose null.
 */
export function nullMeanGram(channels: readonly ChannelCloud[], p: GramParams): Float64Array {
  const K = channels.length;
  const out = new Float64Array(K * K);
  if (K === 0) return out;

  // The flat cell list, and a per-channel weight lookup over it.
  const shared = sharesPoints(channels);
  const first = channels[0]!;
  const n = shared ? first.xs.length : channels.reduce((s, c) => s + c.xs.length, 0);
  if (n < 2) return out;

  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  const w = Array.from({ length: K }, () => new Float64Array(n));
  if (shared) {
    for (let i = 0; i < n; i++) {
      xs[i] = first.xs[i] ?? 0;
      ys[i] = first.ys[i] ?? 0;
      for (let a = 0; a < K; a++) w[a]![i] = weightAt(channels[a]!, i);
    }
  } else {
    let at = 0;
    channels.forEach((c, k) => {
      for (let i = 0; i < c.xs.length; i++) {
        xs[at] = c.xs[i] ?? 0;
        ys[at] = c.ys[i] ?? 0;
        w[k]![at] = weightAt(c, i);
        at++;
      }
    });
  }

  const phi0 = roughness(p.kernel ?? EPANECHNIKOV, p.radius);
  // One unweighted splat of every cell: c[0] is Σ_i Σ_j φ(x_i − x_j), including the n self terms.
  const all = gramMatrix([{ label: "all", xs, ys }], p);
  const bigPhi = (all.c[0] ?? 0) - n * phi0;

  for (let a = 0; a < K; a++) {
    for (let b = a; b < K; b++) {
      let sab = 0;
      let wa = 0;
      let wb = 0;
      for (let i = 0; i < n; i++) {
        sab += w[a]![i]! * w[b]![i]!;
        wa += w[a]![i]!;
        wb += w[b]![i]!;
      }
      const v = phi0 * sab + (bigPhi * (wa * wb - sab)) / (n * (n - 1));
      out[a * K + b] = v;
      out[b * K + a] = v;
    }
  }
  return out;
}
