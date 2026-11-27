import { describe, expect, it } from "vite-plus/test";
import { mulberry32 } from "#src/evo/rng";
import { ANCHOR_SKIP, anchorNullExceedances, anchorScreen, anchorSampleSeed, averageRanks, prepareAnchorBlock } from "./anchorScreen";
import { randomPermutation } from "./permute";

/** SIMBA's steps literally: mask, rank both sides over the mask, Pearson of centred ranks. NaN
 *  where support is short or a side is constant. */
function spearmanMasked(a: ArrayLike<number>, f: ArrayLike<number>, minFinite = 3): number {
  const idx: number[] = [];
  for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i]!) && Number.isFinite(f[i]!)) idx.push(i);
  if (idx.length < minFinite) return Number.NaN;
  const ra = averageRanks(a, idx);
  const rf = averageRanks(f, idx);
  const ma = ra.reduce((s, v) => s + v, 0) / idx.length;
  const mf = rf.reduce((s, v) => s + v, 0) / idx.length;
  let dot = 0;
  let na = 0;
  let nf = 0;
  for (let t = 0; t < idx.length; t++) {
    dot += (ra[t]! - ma) * (rf[t]! - mf);
    na += (ra[t]! - ma) ** 2;
    nf += (rf[t]! - mf) ** 2;
  }
  return na * nf === 0 ? Number.NaN : dot / Math.sqrt(na * nf);
}

/** SIMBA's `generate_permuted_anchor` for one permutation: finite values shuffled among finite
 *  slots, missing slots kept. `perm[t]` is the finite value that slot t receives. */
function permutedAnchor(a: ArrayLike<number>, perm: Uint32Array): Float64Array {
  const finite: number[] = [];
  for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i]!)) finite.push(i);
  const out = Float64Array.from(a);
  finite.forEach((i, t) => (out[i] = a[finite[perm[t]!]!]!));
  return out;
}

function lcgData(seed: number, n: number, m: number, opts: { ties?: number; nanRate?: number } = {}) {
  const rnd = mulberry32(seed);
  const q = (v: number) => (opts.ties ? Math.round(v * opts.ties) / opts.ties : v);
  const anchor = Float64Array.from({ length: n }, () => q(rnd()));
  const cols: Float64Array[] = [];
  for (let j = 0; j < m; j++) {
    const w = rnd() * 2 - 1;
    cols.push(Float64Array.from(anchor, (a) => (rnd() < (opts.nanRate ?? 0) ? Number.NaN : q(w * a + rnd()))));
  }
  return { anchor, cols };
}

describe("averageRanks", () => {
  it("shares the mean rank across ties", () => {
    expect(Array.from(averageRanks([3, 1, 3, 2, 3], [0, 1, 2, 3, 4]))).toEqual([4, 1, 4, 2, 4]);
    expect(Array.from(averageRanks([10, 20, 30, 10], [3, 0, 2]))).toEqual([1.5, 1.5, 3]);
  });
});

describe("prepareAnchorBlock", () => {
  it("gives the textbook Spearman ρ", () => {
    const b = prepareAnchorBlock(
      [1, 2, 3, 4, 5],
      [
        [2, 1, 4, 3, 5],
        [5, 4, 3, 2, 1],
      ],
    );
    expect(b.rho[0]).toBeCloseTo(0.8, 14);
    expect(b.rho[1]).toBeCloseTo(-1, 14);
  });

  it("masks pairs, re-ranks over the mask, and flags insufficient / constant", () => {
    const N = Number.NaN;
    const anchor = [1, 5, N, 2, 4, 3, 7];
    const feats = [
      [2, 9, 1, 3, N, 4, 8], // 5 pairs
      [N, N, 1, 2, N, N, 3], // 2 pairs -> insufficient
      [1, 1, 1, 1, 1, 1, 1], // constant feature
      [N, 5, 5, 2, 4, 3, 7], // anchor slots match; ties in feature
    ];
    const b = prepareAnchorBlock(anchor, feats);
    expect(Array.from(b.nFinite)).toEqual([5, 2, 6, 5]);
    expect(Array.from(b.skip)).toEqual([ANCHOR_SKIP.none, ANCHOR_SKIP.insufficient, ANCHOR_SKIP.constant, ANCHOR_SKIP.none]);
    for (const j of [0, 3]) expect(b.rho[j]).toBeCloseTo(spearmanMasked(anchor, feats[j]!), 14);
    expect(b.rho[1]).toBeNaN();
    expect(b.rho[2]).toBeNaN();
    // Only scored features are grouped for the null; the partial masks stay distinct.
    expect(b.groups.flatMap((g) => Array.from(g.features)).sort((a, b) => a - b)).toEqual([0, 3]);
  });

  it("is constant when the anchor is constant on a feature's mask", () => {
    const N = Number.NaN;
    const b = prepareAnchorBlock(
      [1, 1, 1, 2, 3],
      [
        [1, 2, 3, N, N],
        [1, 2, 3, 4, 5],
      ],
    );
    expect(b.skip[0]).toBe(ANCHOR_SKIP.constant);
    expect(b.skip[1]).toBe(ANCHOR_SKIP.none);
  });

  it("refuses minFinite below 2 or not finite", () => {
    expect(() => prepareAnchorBlock([1, 2], [[1, 2]], 1)).toThrow(/minFinite/);
    expect(() => prepareAnchorBlock([1, 2], [[1, 2]], Number.NaN)).toThrow(/minFinite/);
    expect(prepareAnchorBlock([1, 2], [[1, 2]], 2).rho[0]).toBeCloseTo(1, 14);
  });
});

describe("anchorNullExceedances", () => {
  // The fast path (permuted ranks for the full mask, counting-sort re-ranks for partial masks)
  // against SIMBA's steps run literally on the same permutations.
  for (const [label, opts] of [
    ["continuous, no missing", {}],
    ["heavy ties and missing values", { ties: 4, nanRate: 0.25 }],
  ] as const) {
    it(`matches the literal null exactly: ${label}`, () => {
      const { anchor, cols } = lcgData(7, 23, 12, opts);
      anchor[3] = Number.NaN;
      anchor[11] = Number.NaN;
      const block = prepareAnchorBlock(anchor, cols);
      const S = 200;
      const perms: Uint32Array[] = [];
      const rnd = mulberry32(99);
      for (let s = 0; s < S; s++) perms.push(randomPermutation(block.nAnchor, rnd));
      const seen = new Map<string, number>();
      let s = 0;
      const exceed = anchorNullExceedances(
        block,
        S,
        () => perms[s++]!,
        1e-12,
        (j, sh, v) => seen.set(`${j}:${sh}`, v),
      );
      const want = new Int32Array(cols.length);
      for (let sh = 0; sh < S; sh++) {
        const pa = permutedAnchor(anchor, perms[sh]!);
        cols.forEach((f, j) => {
          // The observed score, recomputed literally rather than taken from the block.
          const obs = spearmanMasked(anchor, f);
          expect(Number.isFinite(obs)).toBe(Number.isFinite(block.rho[j]!));
          if (!Number.isFinite(obs)) return;
          const v = spearmanMasked(pa, f);
          const got = seen.get(`${j}:${sh}`)!;
          if (Number.isNaN(v)) expect(got).toBeNaN();
          else expect(Math.abs(got - v)).toBeLessThan(1e-13);
          if (Math.abs(v) >= Math.abs(obs) - 1e-12) want[j]!++;
        });
      }
      expect(Array.from(exceed)).toEqual(Array.from(want));
    });
  }
});

describe("anchorScreen", () => {
  it("returns a long table per sample, p NaN where ρ is", () => {
    const { anchor, cols } = lcgData(3, 40, 3);
    cols[2] = new Float64Array(40).fill(2);
    const features = new Map(cols.map((c, j) => [`f${j}`, c]));
    const samples = Array.from({ length: 40 }, (_, i) => (i < 25 ? "A" : "B"));
    const r = anchorScreen({ anchor, features, samples }, { permutations: 99 });
    expect(r.sample).toEqual(["A", "A", "A", "B", "B", "B"]);
    expect(r.feature).toEqual(["f0", "f1", "f2", "f0", "f1", "f2"]);
    expect(Array.from(r.nFinite)).toEqual([25, 25, 25, 15, 15, 15]);
    expect(r.p[2]).toBeNaN();
    expect(r.skip[2]).toBe(ANCHOR_SKIP.constant);
    for (const j of [0, 1, 3, 4]) {
      expect(r.p[j]).toBeGreaterThanOrEqual(1 / 100);
      expect(r.p[j]).toBeLessThanOrEqual(1);
      expect(Math.abs(r.p[j]! * 100 - Math.round(r.p[j]! * 100))).toBeLessThan(1e-9);
    }
    const a = Array.from(anchor.slice(0, 25));
    expect(r.rho[0]).toBeCloseTo(spearmanMasked(a, cols[0]!.slice(0, 25)), 14);
  });

  it("gives a sample the same null whatever else is in the table", () => {
    const { anchor, cols } = lcgData(5, 60, 4);
    const features = new Map(cols.map((c, j) => [`f${j}`, c]));
    const samples = Array.from({ length: 60 }, (_, i) => (i % 2 ? 10 : 20));
    const both = anchorScreen({ anchor, features, samples }, { permutations: 199, seed: 4 });
    const keep = Array.from({ length: 60 }, (_, i) => i).filter((i) => i % 2 === 0);
    const sub = new Map(cols.map((c, j) => [`f${j}`, keep.map((i) => c[i]!)]));
    const alone = anchorScreen(
      { anchor: keep.map((i) => anchor[i]!), features: sub, samples: keep.map(() => 20) },
      { permutations: 199, seed: 4 },
    );
    // sample 20 is first-seen in `both`
    expect(Array.from(alone.p)).toEqual(Array.from(both.p.slice(0, 4)));
    expect(anchorSampleSeed(4, 20)).not.toBe(anchorSampleSeed(4, 10));
  });

  it("floors p for a perfect association and is roughly uniform under the null", () => {
    const n = 30;
    const rnd = mulberry32(1);
    const anchor = Float64Array.from({ length: n }, () => rnd());
    const features = new Map<string, Float64Array>([["mono", anchor.map((v) => v * 3 + 1)]]);
    for (let j = 0; j < 300; j++)
      features.set(
        `noise${j}`,
        Float64Array.from({ length: n }, () => rnd()),
      );
    const r = anchorScreen({ anchor, features }, { permutations: 199 });
    expect(r.rho[0]).toBeCloseTo(1, 14);
    expect(r.p[0]).toBe(1 / 200);
    const ps = Array.from(r.p.slice(1));
    const below = (t: number) => ps.filter((p) => p <= t).length / ps.length;
    // 300 independent features: binomial sd at 0.1 is ~0.017, at 0.5 ~0.029.
    expect(Math.abs(below(0.1) - 0.1)).toBeLessThan(0.06);
    expect(Math.abs(below(0.5) - 0.5)).toBeLessThan(0.1);
  });

  it("rejects a non-integer or negative shuffle count, and draws nothing when nothing is scored", () => {
    const features = new Map([["x", [1, 2, 3, 4]]]);
    for (const permutations of [1.5, -1, Number.NaN])
      expect(() => anchorScreen({ anchor: [1, 2, 3, 4], features }, { permutations })).toThrow(/permutations/);
    expect(anchorScreen({ anchor: [1, 2, 3, 4], features }, { permutations: 0 }).p[0]).toBeNaN();
    const empty = prepareAnchorBlock([1, 2, 3], [[Number.NaN, 1, 2]]);
    let drawn = 0;
    anchorNullExceedances(empty, 99, () => {
      drawn++;
      return new Uint32Array(3);
    });
    expect(drawn).toBe(0);
  });

  it("rejects ragged columns", () => {
    expect(() => anchorScreen({ anchor: [1, 2, 3], features: new Map([["x", [1, 2]]]) })).toThrow(/2 values for 3/);
  });
});
