// A permutation envelope for the co-location spectrum — the piece that turns "mode 1 carries 45% of
// the spatial variance" from a number into a claim.
//
// ## What is being tested
//
// The scree chart is uninterpretable on its own. 45% sounds like a lot, but K channels sharing a
// tissue will always concentrate variance in the leading mode to *some* degree, and without knowing
// how much of that is inevitable there is nothing to conclude. The null says: keep every cell where
// it is, shuffle the marks between them (`src/spatial/permute.ts`), and see what the spectrum looks
// like when the geography is destroyed but the anatomy and the marks are not.
//
// The statistic is the whole explained-variance vector, not one mode, and the test is a global rank
// envelope (`src/spatial/envelope.ts`) — because asking "is mode 1 unusual?" and "is mode 2
// unusual?" and so on, each at 5%, is not a 5% test of anything.
//
// One property of this particular vector is worth stating: the explained variances sum to 1 by
// construction, so the K values are compositional and strongly correlated across modes. That breaks
// nothing — the global envelope makes no independence assumption, which is precisely why it is the
// right tool here — but it does mean the band is narrower than K independent bands would be, and
// that a high mode 1 forces the others down.
//
// ## Cost
//
// One simulation is one full `gramMatrixGpu`, because a permutation changes which cell carries
// which mark and therefore has to be re-splatted. There is no shortcut to the *spread*; the mean
// alone is analytic (see `nullMeanGram`) and is not what an envelope needs. At the demo's settings
// that is ~140 ms each, so 99 simulations is around 15 seconds — a button, not a live control.

import { type GlobalEnvelope, globalRankEnvelope } from "../../spatial/envelope";
import { type ChannelCloud, coLocationModes, type GramParams } from "../../spatial/gram";
import { permuteChannels, randomPermutation } from "../../spatial/permute";
import { gramMatrixGpu } from "./gramMatrix";

/** Deterministic RNG so a reported p-value can be reproduced. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface SpectrumEnvelopeOptions {
  /** Null realisations. The p-value cannot go below `1/(s+1)`, and α=0.05 needs at least 19 for a
   *  band to exist at all. 99 is the default: one decimal place of p, ~15 s at demo settings. */
  readonly simulations?: number;
  readonly alpha?: number;
  readonly seed?: number;
  /** Called after each realisation. Awaited, so a browser caller can yield and repaint. */
  readonly onProgress?: (done: number, total: number) => void | Promise<void>;
}

export interface SpectrumEnvelopeResult {
  /** Observed explained variance per mode, `λ_k / K`. */
  readonly observed: Float64Array;
  readonly envelope: GlobalEnvelope;
  readonly labels: string[];
  /** Wall-clock milliseconds, for a caller that wants to warn before re-running. */
  readonly elapsedMs: number;
}

/**
 * Simulate the co-location spectrum under random labelling and envelope it.
 *
 * The observed realisation is computed **last**, deliberately. `gramMatrixGpu` pools its rasters and
 * overwrites them on every call, so whatever ran most recently is what a subsequent
 * `paintGramModes` or `paintGramTerrain` will draw — finishing on the observed data leaves the
 * caller's views showing the thing they were showing before, instead of the last random shuffle.
 */
export async function spectrumEnvelope(
  channels: readonly ChannelCloud[],
  params: GramParams,
  opts: SpectrumEnvelopeOptions = {},
): Promise<SpectrumEnvelopeResult> {
  const simulations = Math.max(1, Math.floor(opts.simulations ?? 99));
  const rnd = mulberry32(opts.seed ?? 0x5eed);
  const n = channels[0]?.xs.length ?? 0;
  const cells = channels.every((c) => c.xs === channels[0]?.xs) ? n : channels.reduce((s, c) => s + c.xs.length, 0);
  const started = performance.now();

  const explainedOf = async (chs: readonly ChannelCloud[]): Promise<Float64Array> => {
    const res = await gramMatrixGpu(chs, params);
    return coLocationModes(res).explained;
  };

  const simulated: Float64Array[] = [];
  for (let i = 0; i < simulations; i++) {
    simulated.push(await explainedOf(permuteChannels(channels, randomPermutation(cells, rnd))));
    await opts.onProgress?.(i + 1, simulations + 1);
  }
  const observed = await explainedOf(channels);
  await opts.onProgress?.(simulations + 1, simulations + 1);

  return {
    observed,
    envelope: globalRankEnvelope(observed, simulated, { alpha: opts.alpha }),
    labels: channels.map((c) => c.label),
    elapsedMs: performance.now() - started,
  };
}

/** How many modes' explained variance falls outside the band, and in which direction — a one-line
 *  summary for a readout, since "the curve exits somewhere" is not by itself informative. */
export function envelopeExcursions(res: SpectrumEnvelopeResult): { mode: number; observed: number; bound: number; above: boolean }[] {
  const out: { mode: number; observed: number; bound: number; above: boolean }[] = [];
  const { lo, hi } = res.envelope;
  for (let k = 0; k < res.observed.length; k++) {
    const v = res.observed[k]!;
    if (v > hi[k]!) out.push({ mode: k + 1, observed: v, bound: hi[k]!, above: true });
    else if (v < lo[k]!) out.push({ mode: k + 1, observed: v, bound: lo[k]!, above: false });
  }
  return out;
}
