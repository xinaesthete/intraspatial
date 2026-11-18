import { describe, expect, it } from "vite-plus/test";
import { mulberry32 } from "#src/evo/rng";
import { eigenSym } from "./eigenSym";
import { geneEmbedding, geneModules } from "./geneModules";

/**
 * Points on a 1000 × 1000 square in three vertical stripes. Genes 0–29 are programs: gene g
 * belongs to stripe g % 3 and is 8× enriched there. Genes 30–59 are uniform noise. Gene 60 is a
 * "control" to exclude, and gene 61 is too rare to keep.
 */
function stripes(n: number, seed = 1) {
  const rnd = mulberry32(seed);
  const xs = new Float32Array(n);
  const ys = new Float32Array(n);
  const codes = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const x = rnd() * 1000;
    xs[i] = x;
    ys[i] = rnd() * 1000;
    const stripe = Math.min(2, Math.floor(x / 333.34));
    const u = rnd();
    if (u < 0.001) codes[i] = 60;
    else if (i < 5) codes[i] = 61;
    else if (u < 0.5) {
      // A program gene: from this stripe's 10 genes w.p. 8/10, else from the other 20.
      const own = rnd() < 0.8;
      const pick = Math.floor(rnd() * (own ? 10 : 20));
      const others = [0, 1, 2].filter((s) => s !== stripe);
      const s = own ? stripe : others[pick % 2]!;
      codes[i] = 3 * (own ? pick : Math.floor(pick / 2)) + s;
    } else codes[i] = 30 + Math.floor(rnd() * 30);
  }
  return { xs, ys, codes, nFeatures: 62 };
}

describe("geneEmbedding", () => {
  const { xs, ys, codes, nFeatures } = stripes(200_000);
  const emb = geneEmbedding(xs, ys, codes, nFeatures, { binSize: 50, components: 6, exclude: new Set([60]) });

  it("drops excluded and rare genes, and reports them", () => {
    expect(emb.dropped).toEqual({ excluded: 1, rare: 1 });
    expect(Array.from(emb.codes)).toEqual(Array.from({ length: 60 }, (_, g) => g));
    expect(emb.bins).toBe(400);
  });

  it("keeps the two components three stripes make, and loadings are correlations", () => {
    expect(emb.components).toBe(2);
    expect(emb.explained).toHaveLength(6);
    expect(emb.explained[0]! + emb.explained[1]!).toBeGreaterThan(0.25);
    expect(emb.explained[2]!).toBeLessThan(emb.noiseEdge);
    for (let g = 0; g < 60; g++) {
      let s = 0;
      for (let c = 0; c < emb.components; c++) s += emb.loadings[g * emb.components + c]! ** 2;
      expect(s).toBeLessThanOrEqual(1 + 1e-9);
      // Program genes are well explained, noise genes are not.
      if (g < 30) expect(s).toBeGreaterThan(0.4);
      else expect(s).toBeLessThan(0.05);
    }
  });

  it("matches a dense PCA of the same standardised matrix", () => {
    // Rebuild the matrix densely from the same recipe, and compare explained variance.
    const B = 400;
    const G = 60;
    const counts = new Float64Array(B * G);
    // The same grid as the embedding: anchored at the data's minimum, not at 0.
    const x0 = xs.reduce((a, b) => Math.min(a, b), Number.POSITIVE_INFINITY);
    const y0 = ys.reduce((a, b) => Math.min(a, b), Number.POSITIVE_INFINITY);
    for (let i = 0; i < codes.length; i++) {
      const c = codes[i]!;
      if (c >= 60) continue;
      const b = Math.floor((ys[i]! - y0) / 50) * 20 + Math.floor((xs[i]! - x0) / 50);
      counts[b * G + c]! += 1;
    }
    const z = new Float64Array(B * G);
    for (let b = 0; b < B; b++) {
      let tot = 0;
      for (let g = 0; g < G; g++) tot += counts[b * G + g]!;
      for (let g = 0; g < G; g++) z[b * G + g] = Math.log1p((counts[b * G + g]! * 1e4) / tot);
    }
    for (let g = 0; g < G; g++) {
      let m = 0;
      let s = 0;
      for (let b = 0; b < B; b++) m += z[b * G + g]! / B;
      for (let b = 0; b < B; b++) s += (z[b * G + g]! - m) ** 2 / B;
      for (let b = 0; b < B; b++) z[b * G + g] = Math.min((z[b * G + g]! - m) / Math.sqrt(s), 10);
    }
    const cov = new Float64Array(G * G);
    for (let i = 0; i < G; i++)
      for (let j = 0; j < G; j++) for (let b = 0; b < B; b++) cov[i * G + j]! += (z[b * G + i]! * z[b * G + j]!) / (B * G);
    const dense = eigenSym(cov, G);
    for (let k = 0; k < 2; k++) expect(emb.explained[k]!).toBeCloseTo(dense.values[k]!, 8);
  });

  it("rejects malformed input", () => {
    expect(() => geneEmbedding(xs, ys, codes, nFeatures, { binSize: 0 })).toThrow(/binSize/);
    expect(() => geneEmbedding(xs.subarray(1), ys, codes, nFeatures, { binSize: 50 })).toThrow(/length/);
  });
});

describe("geneModules", () => {
  it("recovers the stripe programs as disjoint modules and leaves noise genes out", () => {
    const { xs, ys, codes, nFeatures } = stripes(200_000, 2);
    const emb = geneEmbedding(xs, ys, codes, nFeatures, { binSize: 50, components: 6, exclude: new Set([60]) });
    const { modules, unstructured } = geneModules(emb, 3);
    expect(modules).toHaveLength(3);
    const stripeOf = modules.map((m) => new Set(m.codes.map((c) => c % 3)));
    for (const s of stripeOf) expect(s.size).toBe(1);
    expect(new Set(stripeOf.map((s) => [...s][0]))).toEqual(new Set([0, 1, 2]));
    expect(modules.flatMap((m) => m.codes).sort((a, b) => a - b)).toEqual(Array.from({ length: 30 }, (_, g) => g));
    expect(unstructured.sort((a, b) => a - b)).toEqual(Array.from({ length: 30 }, (_, g) => 30 + g));
  });

  it("says so when too few genes are structured", () => {
    const { xs, ys, codes, nFeatures } = stripes(50_000, 3);
    const emb = geneEmbedding(xs, ys, codes, nFeatures, { binSize: 50, components: 4 });
    expect(() => geneModules(emb, 40)).toThrow(/structured enough/);
  });
});
