// Anchor screen: per-sample Spearman ρ between one anchor column and every feature of a regions ×
// features table, with a within-sample shuffle null on the anchor (R5 of region statistics; see
// `docs/anchor-screen.md`).
//
// The semantics are SIMBA's continuous-anchor scorer (`compute_feature_block_scores` in TNBC-MMB),
// rule for rule:
//
// - **Pairs.** Feature j is scored on the regions where the anchor AND feature j are finite. Fewer
//   than `minFinite` (default 3) such regions → skipped as `insufficient`, ρ NaN.
// - **Ranks.** Average ranks (ties share the mean rank), taken over those regions only, for the
//   anchor and the feature alike. ρ is the Pearson correlation of the centred ranks.
// - **Constant.** If either side's centred ranks are all zero on those regions → `constant`, ρ NaN.
// - **Shuffle.** Each shuffle permutes the FINITE anchor values among the finite-anchor regions;
//   missing anchors stay where they are. A feature with its own missing values then sees the
//   shuffled anchor at its regions only, and those values are RE-RANKED there. So its null anchor is
//   a random draw of k of the anchor's values, not a permutation of the k it was observed with.
// - **Null ρ.** NaN when the shuffled anchor is constant on the feature's regions; such a shuffle
//   never counts as extreme, but N stays the configured count.
// - **p.** Two-sided, `(1 + #{|ρ_null| ≥ |ρ_obs| − tol}) / (N + 1)`, tol = 1e-12 (the SIMBA check's
//   tie margin: ranks of small n give exactly tied |ρ| that rounding would otherwise split).
//
// ## Cost
//
// Features are ranked once. Features are grouped by their finite mask, so each group shares one
// set of anchor ranks per shuffle:
//
// - The group whose mask is the whole finite-anchor set (usually nearly every feature) needs no
//   re-ranking: average ranks commute with a permutation, so the shuffled anchor's centred ranks
//   are the observed ones permuted and the norm is unchanged.
// - Any other group re-ranks k values per shuffle by counting sort over the anchor's dense rank
//   ids (ranking a subset by value equals ranking it by its global ranks), O(k + distinct values).
//
// What is left is one length-k dot product per (shuffle, feature): N·m·k multiply-adds per sample.

import { mulberry32 } from "#src/evo/rng";
import { randomPermutation } from "./permute";

/** SIMBA's `DEFAULT_N_PERMUTATIONS`. */
export const ANCHOR_SCREEN_PERMUTATIONS = 999;
/** SIMBA's `CONTINUOUS_MIN_FINITE_REGIONS`. */
export const ANCHOR_SCREEN_MIN_FINITE = 3;
/** The tie margin on `|ρ_null| ≥ |ρ_obs|`. */
export const ANCHOR_SCREEN_TIE_TOLERANCE = 1e-12;

/** Largest shuffle count: exceedance tallies are 32-bit. */
export const ANCHOR_SCREEN_MAX_PERMUTATIONS = 2 ** 31 - 2;

/** Throws unless `permutations` is an integer in [0, ANCHOR_SCREEN_MAX_PERMUTATIONS]; 0 is ρ only. */
export function checkAnchorPermutations(permutations: number): number {
  if (!Number.isInteger(permutations) || permutations < 0 || permutations > ANCHOR_SCREEN_MAX_PERMUTATIONS) {
    throw new Error(`anchorScreen: permutations must be an integer in [0, ${ANCHOR_SCREEN_MAX_PERMUTATIONS}], got ${permutations}`);
  }
  return permutations;
}

/** Why a feature has no ρ. */
export const ANCHOR_SKIP = { none: 0, insufficient: 1, constant: 2 } as const;

export interface AnchorScreenInput {
  /** One anchor value per region; a non-finite value is missing. */
  readonly anchor: ArrayLike<number>;
  /** Feature columns, one value per region (e.g. R3's `ZonalTable.columns`). */
  readonly features: ReadonlyMap<string, ArrayLike<number>>;
  /** The sample (specimen) of each region. Omitted: all regions are one sample. */
  readonly samples?: ArrayLike<string | number>;
}

export interface AnchorScreenOptions {
  /** Anchor shuffles per sample (default 999). 0 gives ρ only, p NaN. */
  readonly permutations?: number;
  /** Fewest regions with both values finite (default 3, at least 2). */
  readonly minFinite?: number;
  /** Run seed. Each sample's stream is derived from it and the sample's own key, so a sample's
   *  null does not depend on which other samples are present or their order. */
  readonly seed?: number;
  readonly tieTolerance?: number;
}

/** The long table: one row per (sample, feature), samples in first-seen order, features in input
 *  order. */
export interface AnchorScreenResult {
  readonly sample: (string | number)[];
  readonly feature: string[];
  readonly rho: Float64Array;
  /** NaN where ρ is NaN or there are no shuffles. */
  readonly p: Float64Array;
  /** Regions where the anchor and the feature are both finite. */
  readonly nFinite: Int32Array;
  /** An `ANCHOR_SKIP` code. */
  readonly skip: Uint8Array;
  readonly permutations: number;
}

/** Average ranks (1-based, ties share their mean rank) of `values[idx[t]]`. */
export function averageRanks(values: ArrayLike<number>, idx: ArrayLike<number>): Float64Array {
  const k = idx.length;
  const order = Array.from({ length: k }, (_, t) => t);
  const v = Float64Array.from({ length: k }, (_, t) => values[idx[t]!]!);
  order.sort((a, b) => v[a]! - v[b]!);
  const ranks = new Float64Array(k);
  for (let lo = 0; lo < k;) {
    let hi = lo + 1;
    while (hi < k && v[order[hi]!] === v[order[lo]!]) hi++;
    const r = (lo + 1 + hi) / 2;
    for (let t = lo; t < hi; t++) ranks[order[t]!] = r;
    lo = hi;
  }
  return ranks;
}

/** Centre in place; returns the Euclidean norm of the result. */
function centre(x: Float64Array): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i]!;
  const mean = s / x.length;
  let ss = 0;
  for (let i = 0; i < x.length; i++) {
    const c = x[i]! - mean;
    x[i] = c;
    ss += c * c;
  }
  return Math.sqrt(ss);
}

/** Features that share one finite mask, ready for the null: their centred ranks over the mask,
 *  feature-major (`ranks[j * k + t]`). */
export interface AnchorMaskGroup {
  /** Positions of the mask's regions in the finite-anchor list, ascending. */
  readonly slots: Int32Array;
  /** True when the mask is every finite-anchor region. */
  readonly full: boolean;
  /** Indices into the block's features. */
  readonly features: Int32Array;
  readonly ranks: Float64Array;
  readonly norms: Float64Array;
}

/** One sample's anchor and features, scored, and grouped for the null. */
export interface AnchorBlock {
  /** Regions with a finite anchor. */
  readonly nAnchor: number;
  /** Centred average ranks of the finite anchor values, in finite-region order. */
  readonly anchorRanks: Float64Array;
  readonly anchorNorm: number;
  /** Dense rank id (0 = smallest distinct value) of each finite anchor value. */
  readonly anchorDense: Int32Array;
  readonly distinct: number;
  readonly rho: Float64Array;
  readonly nFinite: Int32Array;
  readonly skip: Uint8Array;
  /** Only features with a ρ appear in a group. */
  readonly groups: readonly AnchorMaskGroup[];
}

/** 0 for the smallest distinct value, 1 for the next, … */
function denseRankIds(ranks: Float64Array): { ids: Int32Array; distinct: number } {
  const order = Array.from({ length: ranks.length }, (_, t) => t).sort((a, b) => ranks[a]! - ranks[b]!);
  const ids = new Int32Array(ranks.length);
  let id = -1;
  let prev = Number.NaN;
  for (const t of order) {
    if (ranks[t] !== prev) {
      id++;
      prev = ranks[t]!;
    }
    ids[t] = id;
  }
  return { ids, distinct: id + 1 };
}

/**
 * Score one sample: observed ρ, support and skip code per feature, plus the mask groups the null
 * needs. `features[j]` is feature j's column over the same regions as `anchor`.
 */
export function prepareAnchorBlock(
  anchor: ArrayLike<number>,
  features: readonly ArrayLike<number>[],
  minFinite = ANCHOR_SCREEN_MIN_FINITE,
): AnchorBlock {
  if (!Number.isFinite(minFinite) || minFinite < 2)
    throw new Error(`anchorScreen: minFinite must be a finite number of at least 2, got ${minFinite}`);
  const n = anchor.length;
  const m = features.length;
  const finite: number[] = [];
  const slotOf = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(anchor[i]!)) {
      slotOf[i] = finite.length;
      finite.push(i);
    }
  }
  const nAnchor = finite.length;
  const anchorRanks = averageRanks(anchor, finite);
  const { ids: anchorDense, distinct } = denseRankIds(anchorRanks);
  const anchorNorm = centre(anchorRanks);

  const rho = new Float64Array(m).fill(Number.NaN);
  const nFinite = new Int32Array(m);
  const skip = new Uint8Array(m);
  // Group features by mask: the key is the list of finite regions (as slot positions).
  const byMask = new Map<string, { slots: Int32Array; members: number[] }>();
  for (let j = 0; j < m; j++) {
    const col = features[j]!;
    if (col.length !== n) throw new Error(`anchorScreen: feature ${j} has ${col.length} values for ${n} regions`);
    const slots: number[] = [];
    for (const i of finite) if (Number.isFinite(col[i]!)) slots.push(slotOf[i]!);
    nFinite[j] = slots.length;
    if (slots.length < minFinite) {
      skip[j] = ANCHOR_SKIP.insufficient;
      continue;
    }
    const key = slots.length === nAnchor ? "*" : slots.join(",");
    let g = byMask.get(key);
    if (!g) byMask.set(key, (g = { slots: Int32Array.from(slots), members: [] }));
    g.members.push(j);
  }

  const groups: AnchorMaskGroup[] = [];
  for (const { slots, members } of byMask.values()) {
    const k = slots.length;
    const full = k === nAnchor;
    const regions = Int32Array.from(slots, (t) => finite[t]!);
    // The anchor's ranks over this mask: the observed statistic re-ranks them, like SIMBA does.
    let aRanks = anchorRanks;
    let aNorm = anchorNorm;
    if (!full) {
      aRanks = averageRanks(anchor, regions);
      aNorm = centre(aRanks);
    }
    const kept: number[] = [];
    const ranks: number[][] = [];
    const norms: number[] = [];
    for (const j of members) {
      const r = averageRanks(features[j]!, regions);
      const norm = centre(r);
      if (aNorm * norm === 0) {
        skip[j] = ANCHOR_SKIP.constant;
        continue;
      }
      let dot = 0;
      for (let t = 0; t < k; t++) dot += aRanks[t]! * r[t]!;
      rho[j] = dot / (aNorm * norm);
      kept.push(j);
      ranks.push(Array.from(r));
      norms.push(norm);
    }
    if (kept.length === 0) continue;
    const flat = new Float64Array(kept.length * k);
    ranks.forEach((r, g) => flat.set(r, g * k));
    groups.push({ slots, full, features: Int32Array.from(kept), ranks: flat, norms: Float64Array.from(norms) });
  }
  return { nAnchor, anchorRanks, anchorNorm, anchorDense, distinct, rho, nFinite, skip, groups };
}

/** Extreme-shuffle counts for one block, from a stream of permutations of `0..nAnchor-1`. Each call
 *  of `nextPermutation` must return a fresh permutation (it may reuse the buffer it returned last).
 *  `onNull`, when given, sees every null ρ (feature, shuffle, value; NaN included). */
export function anchorNullExceedances(
  block: AnchorBlock,
  permutations: number,
  nextPermutation: () => Uint32Array,
  tieTolerance = ANCHOR_SCREEN_TIE_TOLERANCE,
  onNull?: (feature: number, shuffle: number, rho: number) => void,
): Int32Array {
  checkAnchorPermutations(permutations);
  const exceed = new Int32Array(block.rho.length);
  // Nothing scored: draw no permutations at all.
  if (block.groups.length === 0) return exceed;
  const { anchorRanks, anchorNorm, anchorDense, distinct } = block;
  const thresholds = block.groups.map((g) => Float64Array.from(g.features, (j) => Math.abs(block.rho[j]!) - tieTolerance));
  const maxK = block.groups.reduce((a, g) => Math.max(a, g.slots.length), 0);
  const u = new Float64Array(maxK);
  const count = new Int32Array(distinct);
  const start = new Float64Array(distinct);
  for (let s = 0; s < permutations; s++) {
    const perm = nextPermutation();
    block.groups.forEach((g, gi) => {
      const k = g.slots.length;
      let norm = anchorNorm;
      if (g.full) {
        for (let t = 0; t < k; t++) u[t] = anchorRanks[perm[t]!]!;
      } else {
        // Counting-sort average ranks of the shuffled values that land on this mask.
        count.fill(0);
        for (let t = 0; t < k; t++) count[anchorDense[perm[g.slots[t]!]!]!]!++;
        let below = 0;
        for (let d = 0; d < distinct; d++) {
          start[d] = below + (count[d]! + 1) / 2;
          below += count[d]!;
        }
        const mean = (k + 1) / 2;
        let ss = 0;
        for (let t = 0; t < k; t++) {
          const c = start[anchorDense[perm[g.slots[t]!]!]!]! - mean;
          u[t] = c;
          ss += c * c;
        }
        norm = Math.sqrt(ss);
      }
      const thr = thresholds[gi]!;
      const { ranks, norms, features } = g;
      for (let f = 0; f < features.length; f++) {
        let value = Number.NaN;
        if (norm > 0) {
          let dot = 0;
          const off = f * k;
          for (let t = 0; t < k; t++) dot += u[t]! * ranks[off + t]!;
          value = dot / (norm * norms[f]!);
          if (Math.abs(value) >= thr[f]!) exceed[features[f]!]!++;
        }
        onNull?.(features[f]!, s, value);
      }
    });
  }
  return exceed;
}

/** p from an exceedance count; NaN where ρ is NaN or there are no shuffles. */
export function anchorEmpiricalP(rho: ArrayLike<number>, exceed: ArrayLike<number>, permutations: number): Float64Array {
  return Float64Array.from({ length: rho.length }, (_, j) =>
    permutations > 0 && Number.isFinite(rho[j]!) ? (1 + exceed[j]!) / (permutations + 1) : Number.NaN,
  );
}

/** A stable 32-bit seed from the run seed and a sample key (FNV-1a over `seed|key`). */
export function anchorSampleSeed(seed: number, key: string | number): number {
  let h = 0x811c9dc5;
  for (const ch of `${seed}|${key}`) h = Math.imul(h ^ ch.charCodeAt(0), 0x01000193);
  return h >>> 0;
}

/** Regions of each sample, samples in first-seen order. */
function groupBySample(rows: number, samples?: ArrayLike<string | number>): Map<string | number, number[]> {
  const out = new Map<string | number, number[]>();
  if (!samples) {
    out.set(
      0,
      Array.from({ length: rows }, (_, i) => i),
    );
    return out;
  }
  if (samples.length !== rows) throw new Error(`anchorScreen: ${samples.length} sample labels for ${rows} regions`);
  for (let i = 0; i < rows; i++) {
    const key = samples[i]!;
    let list = out.get(key);
    if (!list) out.set(key, (list = []));
    list.push(i);
  }
  return out;
}

/** Check that every column has one value per region; the regions of each sample. */
export function anchorSamples(input: AnchorScreenInput): Map<string | number, number[]> {
  const n = input.anchor.length;
  for (const [name, col] of input.features) {
    if (col.length !== n) throw new Error(`anchorScreen: feature ${name} has ${col.length} values for ${n} regions`);
  }
  return groupBySample(n, input.samples);
}

/** The anchor and feature columns restricted to one sample's regions. */
export function sliceSample(input: AnchorScreenInput, regions: readonly number[]): { anchor: Float64Array; features: Float64Array[] } {
  const take = (col: ArrayLike<number>) => Float64Array.from(regions, (i) => col[i]!);
  return { anchor: take(input.anchor), features: Array.from(input.features.values(), take) };
}

/** Run the screen on the CPU (Fisher–Yates shuffles, f64 throughout). */
export function anchorScreen(input: AnchorScreenInput, opts: AnchorScreenOptions = {}): AnchorScreenResult {
  const permutations = checkAnchorPermutations(opts.permutations ?? ANCHOR_SCREEN_PERMUTATIONS);
  const names = [...input.features.keys()];
  const bySample = anchorSamples(input);
  const out = anchorResultTable(names.length, bySample, permutations);
  let row = 0;
  for (const [key, regions] of bySample) {
    const { anchor, features } = sliceSample(input, regions);
    const block = prepareAnchorBlock(anchor, features, opts.minFinite);
    const rnd = mulberry32(anchorSampleSeed(opts.seed ?? 0, key));
    const buf = new Uint32Array(block.nAnchor);
    const exceed = anchorNullExceedances(block, permutations, () => randomPermutation(block.nAnchor, rnd, buf), opts.tieTolerance);
    writeAnchorRows(out, row, key, names, block, anchorEmpiricalP(block.rho, exceed, permutations));
    row += names.length;
  }
  return out;
}

export function anchorResultTable(features: number, samples: ReadonlyMap<unknown, unknown>, permutations: number): AnchorScreenResult {
  const rows = features * samples.size;
  return {
    sample: new Array<string | number>(rows),
    feature: new Array<string>(rows),
    rho: new Float64Array(rows),
    p: new Float64Array(rows),
    nFinite: new Int32Array(rows),
    skip: new Uint8Array(rows),
    permutations,
  };
}

export function writeAnchorRows(
  out: AnchorScreenResult,
  row: number,
  key: string | number,
  names: readonly string[],
  block: AnchorBlock,
  p: Float64Array,
): void {
  for (let j = 0; j < names.length; j++) {
    out.sample[row + j] = key;
    out.feature[row + j] = names[j]!;
  }
  out.rho.set(block.rho, row);
  out.p.set(p, row);
  out.nFinite.set(block.nFinite, row);
  out.skip.set(block.skip, row);
}
