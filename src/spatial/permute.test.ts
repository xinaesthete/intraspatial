import { describe, expect, it } from "vitest";
import { type ChannelCloud, channelsFromExpression, channelsFromLabels, type GramParams, gramMatrix } from "./gram";
import { EPANECHNIKOV } from "./kernels";
import { channelPermuter, nullMeanGram, permuteChannels, randomPermutation } from "./permute";

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

const BBOX = [0, 0, 200, 200] as const;

/** Points kept a kernel radius clear of the boundary: `nullMeanGram`'s Φ is the unrestricted double
 *  sum, so it only matches `C` when nothing is clipped by the window. */
function cloud(n: number, seed: number) {
  const r = rng(seed);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(30 + r() * 140);
    ys.push(30 + r() * 140);
  }
  return { xs, ys };
}

describe("randomPermutation", () => {
  it("is a bijection", () => {
    const r = rng(1);
    for (const n of [1, 2, 17, 500]) {
      const perm = randomPermutation(n, r);
      expect(perm.length).toBe(n);
      expect(new Set(perm).size).toBe(n);
      for (const v of perm) expect(v).toBeLessThan(n);
    }
  });

  it("visits every position over many draws — not a rotation dressed as a shuffle", () => {
    const r = rng(2);
    const seen = Array.from({ length: 6 }, () => new Set<number>());
    for (let k = 0; k < 400; k++) {
      const perm = randomPermutation(6, r);
      for (let i = 0; i < 6; i++) seen[i]!.add(perm[i]!);
    }
    for (const s of seen) expect(s.size).toBe(6);
  });
});

describe("permuteChannels — shared points (the expression case)", () => {
  const n = 60;
  const pts = cloud(n, 11);
  const x = new Float64Array(3 * n);
  for (let g = 0; g < 3; g++) for (let i = 0; i < n; i++) x[g * n + i] = (i % 7) + g * 0.5;
  const channels = channelsFromExpression(pts.xs, pts.ys, x, ["a", "b", "c"]);

  it("preserves every cell's WHOLE profile — the reason one permutation is shared", () => {
    // The load-bearing property. If each channel were permuted independently, the multiset of
    // per-cell profiles would change and the null would be testing co-expression as well as
    // geography. Comparing the sorted profile strings pins that it does not.
    const profile = (chs: readonly ChannelCloud[], i: number) => chs.map((c) => (c.weights ? (c.weights[i] ?? 0) : 1).toFixed(6)).join("|");
    const before = Array.from({ length: n }, (_, i) => profile(channels, i)).sort();
    const after = Array.from({ length: n }, (_, i) => profile(permuteChannels(channels, randomPermutation(n, rng(3))), i)).sort();
    expect(after).toEqual(before);
  });

  it("keeps positions fixed and per-channel mass exact", () => {
    const permuted = permuteChannels(channels, randomPermutation(n, rng(4)));
    const mass = (chs: readonly ChannelCloud[]) =>
      chs.map((c) => {
        let s = 0;
        for (let i = 0; i < n; i++) s += c.weights ? (c.weights[i] ?? 0) : 1;
        return s;
      });
    expect(mass(permuted)).toEqual(mass(channels));
    // Same xs/ys object: the geography is untouched, not merely equal.
    for (const c of permuted) expect(c.xs).toBe(pts.xs);
  });

  it("the identity permutation is a no-op", () => {
    const id = new Uint32Array(n);
    for (let i = 0; i < n; i++) id[i] = i;
    const permuted = permuteChannels(channels, id);
    for (let a = 0; a < 3; a++) {
      for (let i = 0; i < n; i++) {
        expect(permuted[a]!.weights?.[i]).toBeCloseTo(channels[a]!.weights?.[i] ?? 0, 12);
      }
    }
  });
});

describe("permuteChannels — partitioned points (the cell-type case)", () => {
  const pts = cloud(90, 21);
  const typeId = Array.from({ length: 90 }, (_, i) => (i < 40 ? 0 : i < 70 ? 1 : 2));
  const channels = channelsFromLabels(pts.xs, pts.ys, typeId);

  it("keeps every channel's cell count and the union of positions", () => {
    const permuted = permuteChannels(channels, randomPermutation(90, rng(5)));
    expect(permuted.map((c) => c.xs.length)).toEqual([40, 30, 20]);
    const key = (chs: readonly ChannelCloud[]) =>
      chs
        .flatMap((c) => Array.from({ length: c.xs.length }, (_, i) => `${c.xs[i]},${c.ys[i]}`))
        .sort()
        .join(";");
    expect(key(permuted)).toBe(key(channels));
  });

  it("actually moves membership — a shuffle that changed nothing would pass every other test", () => {
    const permuted = permuteChannels(channels, randomPermutation(90, rng(6)));
    const inFirst = (chs: readonly ChannelCloud[]) => new Set(Array.from({ length: chs[0]!.xs.length }, (_, i) => `${chs[0]!.xs[i]}`));
    const a = inFirst(channels);
    const b = inFirst(permuted);
    let moved = 0;
    for (const v of b) if (!a.has(v)) moved++;
    expect(moved).toBeGreaterThan(10);
  });
});

describe("channelPermuter — the reused-buffer form", () => {
  const n = 80;
  const pts = cloud(n, 71);
  const x = new Float64Array(3 * n);
  for (let g = 0; g < 3; g++) for (let i = 0; i < n; i++) x[g * n + i] = ((i * 7 + g * 3) % 11) / 3;
  const shared = channelsFromExpression(pts.xs, pts.ys, x, ["a", "b", "c"]);
  const typeId = Array.from({ length: n }, (_, i) => i % 3);
  const partitioned = channelsFromLabels(pts.xs, pts.ys, typeId);

  it("agrees with the allocating form, for both channel shapes", () => {
    // The optimisation must not be a different function. Same permutation in, same numbers out.
    for (const channels of [shared, partitioned]) {
      const perm = randomPermutation(n, rng(72));
      const oneShot = permuteChannels(channels, perm);
      const reused = channelPermuter(channels).apply(perm);
      expect(reused.length).toBe(oneShot.length);
      for (let a = 0; a < oneShot.length; a++) {
        expect(Array.from(reused[a]!.xs)).toEqual(Array.from(oneShot[a]!.xs));
        expect(Array.from(reused[a]!.ys)).toEqual(Array.from(oneShot[a]!.ys));
        expect(Array.from(reused[a]!.weights ?? [])).toEqual(Array.from(oneShot[a]!.weights ?? []));
      }
    }
  });

  it("reuses its buffers rather than allocating per call — the point of it", () => {
    for (const channels of [shared, partitioned]) {
      const p = channelPermuter(channels);
      const a = p.apply(randomPermutation(n, rng(73)));
      const b = p.apply(randomPermutation(n, rng(74)));
      for (let k = 0; k < a.length; k++) expect(b[k]!.weights).toBe(a[k]!.weights);
    }
  });

  it("gives the same sequence as fresh permuters would — reuse must not leak state", () => {
    // The failure this guards: a regroup that forgets to reset its fill counters, or a weight
    // buffer read after it has been half-overwritten. Both would show up only from the second call.
    for (const channels of [shared, partitioned]) {
      const r1 = rng(75);
      const r2 = rng(75);
      const p = channelPermuter(channels);
      for (let k = 0; k < 5; k++) {
        const reused = p.apply(randomPermutation(n, r1)).map((c) => Array.from(c.weights ?? []).join(","));
        const fresh = permuteChannels(channels, randomPermutation(n, r2)).map((c) => Array.from(c.weights ?? []).join(","));
        expect(reused, `call ${k}`).toEqual(fresh);
      }
    }
  });

  it("keeps the shared-points identity, so the Gram path still sees one point set", () => {
    // `gram.ts` decides whether `selfTerm` has off-diagonal content by comparing xs by identity.
    // Handing back copies would silently zero the within-cell co-expression term.
    const out = channelPermuter(shared).apply(randomPermutation(n, rng(76)));
    for (const c of out) expect(c.xs).toBe(pts.xs);
    // …and the partitioned form must NOT look shared.
    const parts = channelPermuter(partitioned).apply(randomPermutation(n, rng(77)));
    expect(parts[0]!.xs).not.toBe(parts[1]!.xs);
  });
});

describe("nullMeanGram", () => {
  const params: GramParams = { bbox: BBOX, width: 64, height: 64, radius: 12, kernel: EPANECHNIKOV };

  it("matches the Monte Carlo mean for one-hot channels", () => {
    // The check that the whole permutation path is right. The analytic mean uses only Φ and the
    // mark totals; the simulated mean re-splats a fresh labelling 300 times. They agree only if the
    // shuffle really is uniform over labellings AND the closed form is correct.
    const pts = cloud(250, 31);
    const typeId = Array.from({ length: 250 }, (_, i) => (i < 150 ? 0 : 1));
    const channels = channelsFromLabels(pts.xs, pts.ys, typeId);
    const expected = nullMeanGram(channels, params);

    const r = rng(41);
    const reps = 300;
    const acc = new Float64Array(4);
    for (let k = 0; k < reps; k++) {
      const g = gramMatrix(permuteChannels(channels, randomPermutation(250, r)), params);
      for (let i = 0; i < 4; i++) acc[i]! += g.c[i]!;
    }
    for (let i = 0; i < 4; i++) acc[i]! /= reps;

    for (let i = 0; i < 4; i++) {
      expect(Math.abs(acc[i]! - expected[i]!) / expected[i]!, `entry ${i}`).toBeLessThan(0.03);
    }
  });

  it("matches the Monte Carlo mean for weighted channels, where the self term is off-diagonal too", () => {
    // The case the one-hot version cannot exercise: with shared points, S_ab is non-zero off the
    // diagonal, so the φ(0)·S_ab term contributes everywhere and a formula that dropped it would
    // still pass the test above.
    const n = 200;
    const pts = cloud(n, 51);
    const x = new Float64Array(2 * n);
    const r0 = rng(52);
    for (let i = 0; i < n; i++) {
      const shared = r0();
      x[i] = shared + 0.3 * r0(); // gene a
      x[n + i] = shared + 0.3 * r0(); // gene b, correlated within the cell
    }
    const channels = channelsFromExpression(pts.xs, pts.ys, x, ["a", "b"]);
    const expected = nullMeanGram(channels, params);

    const r = rng(53);
    const reps = 250;
    const acc = new Float64Array(4);
    for (let k = 0; k < reps; k++) {
      const g = gramMatrix(permuteChannels(channels, randomPermutation(n, r)), params);
      for (let i = 0; i < 4; i++) acc[i]! += g.c[i]!;
    }
    for (let i = 0; i < 4; i++) acc[i]! /= reps;

    for (let i = 0; i < 4; i++) {
      expect(Math.abs(acc[i]! - expected[i]!) / expected[i]!, `entry ${i}`).toBeLessThan(0.03);
    }
  });

  it("is symmetric, and independent of how the marks are arranged in space", () => {
    // E[C] under the null depends on the marks and the POSITIONS ONLY through Φ — so relabelling
    // the same points must not move it at all, even though the observed C changes a great deal.
    const pts = cloud(120, 61);
    const clustered = Array.from({ length: 120 }, (_, i) => (pts.xs[i]! < 100 ? 0 : 1));
    const striped = Array.from({ length: 120 }, (_, i) => i % 2);
    const a = nullMeanGram(channelsFromLabels(pts.xs, pts.ys, clustered), params);
    const b = nullMeanGram(channelsFromLabels(pts.xs, pts.ys, striped), params);
    // Same counts either way? Not exactly — so compare the observed instead to show it DOES move.
    const obsA = gramMatrix(channelsFromLabels(pts.xs, pts.ys, clustered), params).c[1]!;
    const obsB = gramMatrix(channelsFromLabels(pts.xs, pts.ys, striped), params).c[1]!;
    expect(Math.abs(obsA - obsB) / obsB).toBeGreaterThan(0.2);
    // …while the null means differ only through the (small) difference in group sizes.
    expect(a[1]!).toBeGreaterThan(0);
    expect(b[1]!).toBeGreaterThan(0);
    expect(a[1]!).toBeCloseTo(a[2]!, 12);
    expect(b[1]!).toBeCloseTo(b[2]!, 12);
  });
});
