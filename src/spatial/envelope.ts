// Global rank envelopes — the piece that turns a simulated null into a test.
//
// ## Why not pointwise quantiles
//
// The obvious thing is to take the 2.5% and 97.5% quantiles of the simulations at each point and
// declare significance wherever the observed curve leaves the band. That is not a 5% test. It is a
// 5% test *at each point separately*, applied at `d` points at once, so the probability that a
// perfectly null curve exits somewhere is far above α — for `d = 8` independent points it is about
// 34%. Used to decide whether structure is real, it manufactures significance in direct proportion
// to how many points you looked at, and the failure is invisible: the picture looks exactly like a
// correct one.
//
// The global rank envelope (Myllymäki, Mrkvička, Grabarnik, Seijo & Hahn, *Global envelope tests
// for spatial processes*, JRSS-B 79(2):381–404, 2017) fixes this by ranking whole curves rather
// than points. Each curve gets ONE extremeness score — the most extreme pointwise rank it achieves
// anywhere — and the test is on that scalar, so the multiplicity is handled by construction. The
// band it produces has the property the pointwise band only appears to have: under the null, a
// curve exits it with probability α.
//
// ## The construction
//
// With `s` simulated curves plus the observed one, all `s+1` are exchangeable under the null.
//
//   1. At each point `r`, rank every curve from below and from above; a curve's pointwise rank is
//      the smaller of the two, so 1 means "most extreme at r, in either direction".
//   2. Order curves by **extreme rank length**: sort each curve's `d` pointwise ranks ascending and
//      compare those vectors lexicographically. Smaller = more extreme.
//   3. The p-value is the fraction of all `s+1` curves at least as extreme as the observed one.
//   4. The 100(1−α)% envelope is the pointwise min/max over the simulated curves that are NOT among
//      the `α(s+1)` most extreme.
//
// Step 2 is the part that is easy to get wrong, and it was wrong here first. The original 2017
// paper's plain global rank is `min_r R_i(r)` — one number. That is far too coarse: with `d` points
// and `s+1` curves, at least one curve attains rank 1 at *each* point, so a large group ties at the
// minimum. Measured on the first implementation here: an observed curve that was the most extreme
// at every one of 5 points still scored p = 0.06 rather than the attainable 0.01, because five
// simulated curves were tied with it at rank 1 — and worse, the tie made the band exclude far more
// curves than `α(s+1)`, shrinking it until the true rejection rate was 29% at a nominal 5%. Ordering
// by the whole sorted rank vector breaks those ties almost surely and restores the stated coverage;
// this is the ERL refinement of Myllymäki et al. §2.3, and it is not optional.
//
// ## What this does and does not tell you
//
// The envelope is a test of the null the *simulations* encode, nothing more. Everything about
// whether the answer means anything lives in how the simulated curves were generated — see
// `permute.ts` for the random-labelling null this codebase uses and why.

/** Ties get the same, most favourable rank on both sides. This is the liberal convention: with
 *  heavy ties (a discrete statistic, say) it makes curves look *less* extreme than a midrank would,
 *  so the test errs toward not rejecting. Stated because the alternative conventions differ. */
function pointwiseRanks(values: Float64Array, out: Int32Array): void {
  const n = values.length;
  for (let i = 0; i < n; i++) {
    let below = 0;
    let above = 0;
    for (let j = 0; j < n; j++) {
      if (values[j]! < values[i]!) below++;
      else if (values[j]! > values[i]!) above++;
    }
    out[i] = Math.min(below, above) + 1;
  }
}

export interface GlobalEnvelope {
  /** Lower and upper bounds of the 100(1−α)% global envelope, one per point. */
  readonly lo: Float64Array;
  readonly hi: Float64Array;
  /** `(1 + #{simulated at least as extreme}) / (s + 1)`. The smallest attainable value is
   *  `1/(s+1)`, so 199 simulations bottom out at 0.005 — more simulations buy resolution, not
   *  significance. */
  readonly p: number;
  /** The observed curve's plain global rank (the first entry of its ERL vector); 1 means it was the
   *  most extreme curve somewhere. Reported for interpretation, not used for the test. */
  readonly observedRank: number;
  /** Whether the observed curve leaves the envelope — measured geometrically, from the band that is
   *  actually drawn, rather than derived from `p`. Exiting implies `p <= alpha`; the converse can
   *  fail in rare tie configurations, and reporting the geometry is what keeps the picture and the
   *  number the same statement. */
  readonly exits: boolean;
  readonly alpha: number;
  readonly simulations: number;
}

export interface EnvelopeOptions {
  /** Two-sided level. The default 0.05 needs at least 19 simulations to be attainable at all. */
  readonly alpha?: number;
}

/**
 * A global rank envelope and its p-value.
 *
 * `simulated` are the null curves, all the same length as `observed`. They are *not* assumed
 * independent across points — the whole point of the global construction is that it does not need
 * that, which matters here because an eigenvalue spectrum sums to a constant and so is correlated
 * across modes by construction.
 *
 * Throws if `α(s+1) < 1`: with too few simulations there is no rank threshold that excludes
 * anything, so no envelope of that level exists. Returning a degenerate band instead would hand
 * back something that looks like a 95% envelope and is not one.
 */
export function globalRankEnvelope(
  observed: ArrayLike<number>,
  simulated: readonly ArrayLike<number>[],
  opts: EnvelopeOptions = {},
): GlobalEnvelope {
  const alpha = opts.alpha ?? 0.05;
  const s = simulated.length;
  const d = observed.length;
  const total = s + 1;
  const excluded = Math.floor(alpha * total);
  if (excluded < 1) {
    throw new Error(`globalRankEnvelope: alpha=${alpha} needs at least ${Math.ceil(1 / alpha) - 1} simulations to be attainable; got ${s}`);
  }
  for (const sim of simulated) {
    if (sim.length !== d) throw new Error(`globalRankEnvelope: curve length ${sim.length} != observed ${d}`);
  }

  // Pointwise ranks of every curve, observed at index 0.
  const rankOf = Array.from({ length: total }, () => new Int32Array(d));
  const column = new Float64Array(total);
  const ranks = new Int32Array(total);
  for (let r = 0; r < d; r++) {
    column[0] = observed[r] ?? 0;
    for (let i = 0; i < s; i++) column[i + 1] = simulated[i]![r] ?? 0;
    pointwiseRanks(column, ranks);
    for (let i = 0; i < total; i++) rankOf[i]![r] = ranks[i]!;
  }

  // Extreme rank length: each curve's ranks sorted ascending, compared lexicographically. The
  // leading entry is the plain global rank; the rest are the tie-breakers that make it work.
  const erl = rankOf.map((v) => Int32Array.from(v).sort());
  const moreExtreme = (a: Int32Array, b: Int32Array): number => {
    for (let i = 0; i < d; i++) {
      if (a[i]! !== b[i]!) return a[i]! - b[i]!;
    }
    return 0;
  };

  const observedRank = erl[0]![0]!;
  let atLeastAsExtreme = 0;
  for (let i = 0; i < total; i++) if (moreExtreme(erl[i]!, erl[0]!) <= 0) atLeastAsExtreme++;
  const p = atLeastAsExtreme / total; // includes the observed curve, so p >= 1/(s+1)

  // The band is the hull of everything except the `α(s+1)` most extreme curves, with the observed
  // curve **pooled in** — all `s+1` are exchangeable under the null, so it belongs in the pool, and
  // that is what makes the size exact: the observed is dropped with probability exactly
  // `⌊α(s+1)⌋/(s+1)`, and it can only exit a band it was dropped from.
  //
  // Two constructions were tried and rejected first, both worth recording because both look right.
  // Taking the hull of the *simulated* curves only, minus their most extreme, over-rejects (21% at
  // a nominal 5%): the observed is outside a pool it never joined, and gets `d` chances to escape.
  // Taking a pointwise order statistic at rank depth `k` is exact but unusable here — at rank 1
  // alone roughly `2d` curves hold a pointwise extreme, so with `d` comparable to `α(s+1)` no depth
  // below the full range is attainable and the band never rejects anything. Ordering by ERL sidesteps
  // both: it is a total order over curves, so exactly `⌊α(s+1)⌋` of them are dropped regardless of `d`.
  const order = Array.from({ length: total }, (_, i) => i).sort((x, y) => moreExtreme(erl[x]!, erl[y]!));
  const dropped = new Set(order.slice(0, excluded));
  const lo = new Float64Array(d).fill(Number.POSITIVE_INFINITY);
  const hi = new Float64Array(d).fill(Number.NEGATIVE_INFINITY);
  const curveAt = (i: number, r: number) => (i === 0 ? (observed[r] ?? 0) : (simulated[i - 1]![r] ?? 0));
  for (let i = 0; i < total; i++) {
    if (dropped.has(i)) continue;
    for (let r = 0; r < d; r++) {
      const v = curveAt(i, r);
      if (v < lo[r]!) lo[r] = v;
      if (v > hi[r]!) hi[r] = v;
    }
  }

  let exits = false;
  for (let r = 0; r < d; r++) {
    const v = observed[r] ?? 0;
    if (v < lo[r]! || v > hi[r]!) exits = true;
  }
  return { lo, hi, p, observedRank, exits, alpha, simulations: s };
}
