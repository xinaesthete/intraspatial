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

/** A permutation of `0..n-1`, Fisher–Yates. `rnd` returns uniform `[0, 1)`. */
export function randomPermutation(n: number, rnd: () => number): Uint32Array {
  const perm = new Uint32Array(n);
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

/**
 * Move every cell's mark to the position named by `perm`, leaving the positions alone.
 *
 * Handles both channel shapes, because the caller should not have to know which it has:
 *
 *   * **shared points** — every channel's weight vector is permuted by the *same* `perm`, so each
 *     cell's whole profile lands together;
 *   * **partitioned points** — the concatenated cells are regrouped by permuted membership, which
 *     keeps every channel's cell count exactly and only moves which cells are in it.
 *
 * `perm` must be a permutation of `0..n-1` for the relevant `n`; a shorter one would silently drop
 * cells, so it is checked.
 */
export function permuteChannels(channels: readonly ChannelCloud[], perm: Uint32Array): ChannelCloud[] {
  if (channels.length === 0) return [];
  if (sharesPoints(channels)) {
    const first = channels[0]!;
    const n = first.xs.length;
    if (perm.length !== n) throw new Error(`permuteChannels: perm length ${perm.length} != ${n} cells`);
    return channels.map((c) => {
      const w = new Float64Array(n);
      for (let i = 0; i < n; i++) w[i] = weightAt(c, perm[i]!);
      return { label: c.label, xs: c.xs, ys: c.ys, weights: w };
    });
  }

  // Partitioned: one flat cell list, membership permuted, regrouped.
  const counts = channels.map((c) => c.xs.length);
  const n = counts.reduce((a, b) => a + b, 0);
  if (perm.length !== n) throw new Error(`permuteChannels: perm length ${perm.length} != ${n} cells`);
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  const owner = new Int32Array(n);
  const weight = new Float64Array(n);
  let at = 0;
  channels.forEach((c, k) => {
    for (let i = 0; i < c.xs.length; i++) {
      xs[at] = c.xs[i] ?? 0;
      ys[at] = c.ys[i] ?? 0;
      weight[at] = weightAt(c, i);
      owner[at] = k;
      at++;
    }
  });
  const out = channels.map((c) => ({ label: c.label, xs: [] as number[], ys: [] as number[], weights: [] as number[] }));
  for (let i = 0; i < n; i++) {
    const k = owner[perm[i]!]!;
    const dst = out[k]!;
    dst.xs.push(xs[i]!);
    dst.ys.push(ys[i]!);
    dst.weights.push(weight[perm[i]!]!);
  }
  return out;
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
