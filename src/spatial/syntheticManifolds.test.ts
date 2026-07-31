// What these tests are for: a synthetic dataset that is subtly wrong is worse than no
// dataset, because every conclusion drawn from the page or the benchmark inherits the
// error silently. So each generator is checked against the property it exists to provide
// — the swiss roll must actually be rolled, the rings must actually be linked, the null
// control must actually be null — rather than against fixed numbers.

import { describe, expect, test } from "vitest";
import { expressManifold, MANIFOLDS, makeManifold } from "./syntheticManifolds";
import { umap } from "./umap";
import { trustworthiness } from "./umapLayout";

function dist(x: ArrayLike<number>, dim: number, i: number, j: number): number {
  let acc = 0;
  for (let c = 0; c < dim; c++) {
    const t = x[i * dim + c]! - x[j * dim + c]!;
    acc += t * t;
  }
  return Math.sqrt(acc);
}

/** The roll's gap between adjacent turns, in whatever normalised units the generator ended
 *  up with: the closest two points get while being far apart along the sheet. */
function pitchOf(m: { n: number; latent: Float32Array; truth?: Float32Array }): number {
  let gap = Infinity;
  const take = Math.min(m.n, 600);
  for (let i = 0; i < take; i++) {
    for (let j = i + 1; j < take; j++) {
      if (Math.abs(m.truth![i]! - m.truth![j]!) > 0.15) gap = Math.min(gap, dist(m.latent, 3, i, j));
    }
  }
  return gap;
}

/** Median rather than mean: embedded distances are heavy-tailed, and one pair thrown to
 *  the edge of the layout would move a mean enough to matter. */
function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)]!;
}

describe("generators", () => {
  test("every generator is deterministic in its seed", () => {
    for (const spec of MANIFOLDS) {
      const a = spec.make(200, 3);
      const b = spec.make(200, 3);
      expect(Array.from(a.latent), spec.key).toEqual(Array.from(b.latent));
      expect(Array.from(a.label), spec.key).toEqual(Array.from(b.label));
    }
  });

  test("every generator is normalised to unit RMS pairwise distance", () => {
    // The contract `noise` depends on: without it, "noise 0.05" would mean a different
    // signal-to-noise ratio for each generator.
    for (const spec of MANIFOLDS) {
      const m = spec.make(600, 7);
      let acc = 0;
      let count = 0;
      for (let i = 0; i < 300; i++) {
        for (let j = i + 1; j < 300; j++) {
          const d = dist(m.latent, m.nLatent, i, j);
          acc += d * d;
          count++;
        }
      }
      expect(Math.sqrt(acc / count), spec.key).toBeCloseTo(1, 1);
    }
  });

  test("labels and names line up, and truth is present exactly where claimed", () => {
    for (const spec of MANIFOLDS) {
      const m = spec.make(300, 5);
      expect(m.latentNames, spec.key).toHaveLength(m.nLatent);
      expect(m.latent, spec.key).toHaveLength(m.n * m.nLatent);
      for (let i = 0; i < m.n; i++) expect(m.label[i]!, `${spec.key} label ${i}`).toBeLessThan(m.labelNames.length);
      if (m.truth) {
        expect(m.truth, spec.key).toHaveLength(m.n);
        expect(m.truthName, spec.key).toBeTruthy();
      }
    }
  });

  test("the swiss roll's pitch straddles the k-NN radius at usable n_neighbors", () => {
    // The generator's difficulty is exactly one number: the gap between adjacent turns,
    // measured against the radius a k-neighbourhood reaches. Below the gap the graph
    // cannot short-circuit and "UMAP unrolled it" demonstrates nothing; above it, turns
    // fuse and the roll collapses. The textbook parameterisation puts the gap eight times
    // beyond the nearest-neighbour distance and stays there for every k anyone would use,
    // which is why it was replaced — it only looks hard.
    //
    // Straddling is the property worth pinning: safe at a default n_neighbors, broken if
    // you push it. That makes the failure reachable from the page's own slider.
    const n = 3000;
    const m = makeManifold("swissRoll", n, 3);

    const kthRadius = (k: number) => {
      let total = 0;
      const sample = 150;
      for (let i = 0; i < sample; i++) {
        const ds: number[] = [];
        for (let j = 0; j < n; j++) if (j !== i) ds.push(dist(m.latent, 3, i, j));
        ds.sort((a, b) => a - b);
        total += ds[k - 1]!;
      }
      return total / sample;
    };

    const gap = pitchOf(m);
    expect(kthRadius(12)).toBeLessThan(gap);
    expect(kthRadius(40)).toBeGreaterThan(gap);
  });

  test("the rings are linked, not merely adjacent", () => {
    // Two circles are linked iff neither can be shrunk to a point without crossing the
    // other. Checking that directly is overkill; checking that they are disjoint but
    // interpenetrating (each ring has points on both sides of the other's plane, and no
    // point of one is inside the other's tube) is the property that matters here.
    const m = makeManifold("linkedRings", 1200, 3);
    const a: number[] = [];
    const b: number[] = [];
    for (let i = 0; i < m.n; i++) (m.label[i] === 0 ? a : b).push(i);
    let minCross = Infinity;
    for (const i of a.slice(0, 300)) for (const j of b.slice(0, 300)) minCross = Math.min(minCross, dist(m.latent, 3, i, j));
    expect(minCross).toBeGreaterThan(0);

    // Ring B lies in the x–z plane, so it must have points at both positive and negative
    // z; ring A is confined to z = 0. That is what threading looks like in coordinates.
    const bz = b.map((i) => m.latent[i * 3 + 2]!);
    expect(Math.min(...bz)).toBeLessThan(0);
    expect(Math.max(...bz)).toBeGreaterThan(0);
    for (const i of a) expect(Math.abs(m.latent[i * 3 + 2]!)).toBeLessThan(1e-6);
  });

  test("the rare population is rare and separated", () => {
    const m = makeManifold("rarePopulation", 2000, 5);
    const rare = Array.from(m.label).filter((l) => l === 1).length;
    expect(rare / m.n).toBeLessThan(0.02);
    expect(rare).toBeGreaterThan(2);
  });

  test("the null control has no label structure", () => {
    const m = makeManifold("uniformNoise", 500, 9);
    expect(m.labelNames).toHaveLength(1);
    expect(Array.from(m.label).every((l) => l === 0)).toBe(true);
    expect(m.truth).toBeUndefined();
  });
});

describe("expression", () => {
  test("blocks partition the columns, one per latent axis", () => {
    const m = makeManifold("cellCycle", 200, 2);
    const e = expressManifold(m, { genesPerAxis: 4 });
    expect(e.blocks).toHaveLength(m.nLatent);
    expect(e.dim).toBe(m.nLatent * 4);
    const seen = new Set(e.blocks.flatMap((b) => b.columns));
    expect(seen.size).toBe(e.dim);
    for (let c = 0; c < e.dim; c++) expect(seen.has(c)).toBe(true);
  });

  test("expression preserves the manifold's neighbourhoods", () => {
    // The load-bearing claim: a k-NN over genes has to be a k-NN over the latent, or the
    // page is showing an embedding of the loading matrix rather than of the manifold.
    const m = makeManifold("branching", 400, 4);
    const e = expressManifold(m, { noise: 0.02 });
    expect(trustworthiness(e.values, m.latent, m.n, e.dim, m.nLatent, 10)).toBeGreaterThan(0.95);
  });

  test("more noise degrades those neighbourhoods, monotonically", () => {
    const m = makeManifold("swissRoll", 400, 6);
    const clean = expressManifold(m, { noise: 0.01 });
    const dirty = expressManifold(m, { noise: 0.5 });
    const tClean = trustworthiness(clean.values, m.latent, m.n, clean.dim, m.nLatent, 10);
    const tDirty = trustworthiness(dirty.values, m.latent, m.n, dirty.dim, m.nLatent, 10);
    expect(tClean).toBeGreaterThan(tDirty);
  });
});

describe("what UMAP makes of them", () => {
  test("the swiss roll comes out unrolled: adjacent turns are kept apart", async () => {
    // The assertion this whole module exists to support, aimed at the one thing unrolling
    // means: CONFUSABLE pairs — close in ambient 3-D, far along the sheet, i.e. facing each
    // other across the gap between turns — must end up far apart in the embedding.
    //
    // An earlier version compared rank correlations against the sheet and against the
    // ambient coordinates over random pairs. That measures the wrong thing: most random
    // pairs are long-range, UMAP makes no attempt to preserve long-range distance, and the
    // two correlations landed close enough together (0.40 vs 0.30) to flip between
    // otherwise-equivalent runs. The confusable ratio below is the mechanism itself, and
    // measures 46-72 across seeds and n where a collapsed roll would give ~1.
    const m = makeManifold("swissRoll", 3000, 3);
    const e = expressManifold(m);
    const res = await umap(e.values, m.n, e.dim, { nNeighbors: 15, nEpochs: 400, seed: 1, pca: false });

    // Thresholds are multiples of the roll's own pitch, measured here rather than written
    // in as constants: the generator derives its sheet height from n, so the normalised
    // scale moves whenever n does, and absolute cut-offs silently select nothing.
    const gap = pitchOf(m);
    const confusable: number[] = [];
    const sheetNear: number[] = [];
    for (let i = 0; i < m.n; i++) {
      for (let j = i + 1; j < m.n; j++) {
        const sheet = dist(m.intrinsic!, m.intrinsicDim!, i, j);
        if (dist(m.latent, 3, i, j) < 1.3 * gap && sheet > 3 * gap) confusable.push(dist(res.embedding, 2, i, j));
        else if (sheet < 0.5 * gap) sheetNear.push(dist(res.embedding, 2, i, j));
      }
    }
    expect(confusable.length).toBeGreaterThan(100);
    expect(median(confusable) / median(sheetNear)).toBeGreaterThan(10);
  });

  test("the null control does NOT clump — it is a plain disc", async () => {
    // This test asserted the opposite until it was measured. Received wisdom says UMAP
    // manufactures islands out of structureless data; at 6, 30 and 80 latent dimensions it
    // does not, and the null comes out as a featureless disc. The generator's description
    // and the page's caption were rewritten to match, so the assertion has to pin the fact
    // they now rest on — otherwise a future change could quietly invalidate both.
    //
    // "Islands" means DISCONNECTED, so measure connectivity: link every pair closer than
    // LINK_SCALE times the median nearest-neighbour distance and take the largest
    // component's share of the points. Measured at n = 3000: the null gives 0.999, `blobs`
    // 0.200 — five islands whose gaps are so wide that even a 10x threshold leaves them
    // separate. The discrimination is not delicate.
    //
    // Two earlier attempts used the coefficient of variation of nearest-neighbour distances
    // instead, and both were wrong. "> 0.25" is passed comfortably by a uniform Poisson
    // process (which sits near 0.52), so it proved nothing; and comparing that CV against
    // `blobs` fails outright, because five TIGHT islands have very regular internal spacing
    // and so score LOWER than a plain disc. Neither measures what the word means.
    const LINK_SCALE = 6;
    const largestPiece = (emb: Float32Array, n: number) => {
      const nn = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let best = Infinity;
        for (let j = 0; j < n; j++) if (j !== i) best = Math.min(best, dist(emb, 2, i, j));
        nn[i] = best;
      }
      const threshold = LINK_SCALE * median([...nn]);
      const parent = Int32Array.from({ length: n }, (_, i) => i);
      const find = (x: number): number => {
        let r = x;
        while (parent[r] !== r) r = parent[r] = parent[parent[r]!]!;
        return r;
      };
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (dist(emb, 2, i, j) >= threshold) continue;
          const a = find(i);
          const b = find(j);
          if (a !== b) parent[a] = b;
        }
      }
      const sizes = new Map<number, number>();
      for (let i = 0; i < n; i++) {
        const r = find(i);
        sizes.set(r, (sizes.get(r) ?? 0) + 1);
      }
      return Math.max(...sizes.values()) / n;
    };

    const opts = { nNeighbors: 15, nEpochs: 200, seed: 1, pca: false } as const;
    const nullM = makeManifold("uniformNoise", 1500, 2);
    const nullE = expressManifold(nullM);
    const nullRes = await umap(nullE.values, nullM.n, nullE.dim, opts);

    const islandM = makeManifold("blobs", 1500, 2);
    const islandE = expressManifold(islandM);
    const islandRes = await umap(islandE.values, islandM.n, islandE.dim, opts);

    // The null stays one piece; `blobs`, which really does have islands, does not.
    expect(largestPiece(nullRes.embedding, nullM.n)).toBeGreaterThan(0.9);
    expect(largestPiece(islandRes.embedding, islandM.n)).toBeLessThan(0.5);
  });
});
