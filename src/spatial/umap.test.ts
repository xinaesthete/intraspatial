import { describe, expect, it } from "vitest";
import { pca } from "./pca";
import { subsetColumns, subsetRows, umap, umapGraphFor } from "./umap";
import { mulberry32, trustworthiness } from "./umapLayout";

/** Three separated blobs, each with `dim` features of which only the first few carry
 *  the cluster signal — the rest are noise, which is what PCA has to see through. */
function blobs(nPer: number, dim: number, nBlobs = 3, seed = 7, signalDims = 3) {
  const rnd = mulberry32(seed);
  const n = nPer * nBlobs;
  const data = new Float64Array(n * dim);
  const label = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const b = Math.floor(i / nPer);
    label[i] = b;
    for (let c = 0; c < dim; c++) {
      const u = Math.max(rnd(), 1e-12);
      const v = rnd();
      const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      data[i * dim + c] = g + (c < signalDims ? b * 20 * ((c % 2) - 0.5) * 2 : 0);
    }
  }
  return { data, label, n };
}

describe("pca", () => {
  it("recovers a known 2-D subspace embedded in 10-D", () => {
    // Points on a plane spanned by two axes, plus tiny noise elsewhere.
    const n = 200;
    const dim = 10;
    const rnd = mulberry32(3);
    const data = new Float64Array(n * dim);
    for (let i = 0; i < n; i++) {
      const s = rnd() * 10 - 5;
      const t = rnd() * 4 - 2;
      data[i * dim] = s;
      data[i * dim + 1] = t;
      for (let c = 2; c < dim; c++) data[i * dim + c] = (rnd() - 0.5) * 1e-3;
    }
    const res = pca(data, n, dim, { nComponents: 3 });
    // Almost all variance in the first two components, and PC1 the wider axis.
    expect(res.explainedVarianceRatio[0]! + res.explainedVarianceRatio[1]!).toBeGreaterThan(0.999);
    expect(res.explainedVariance[0]!).toBeGreaterThan(res.explainedVariance[1]!);
    expect(res.explainedVariance[2]!).toBeLessThan(1e-4);
  });

  it("orders components by descending variance", () => {
    const { data, n } = blobs(40, 12, 3, 11);
    const res = pca(data, n, 12, { nComponents: 6 });
    let ok = true;
    for (let c = 1; c < 6; c++) if (res.explainedVariance[c]! > res.explainedVariance[c - 1]! + 1e-9) ok = false;
    expect(ok).toBe(true);
  });

  it("is sign-stable across runs", () => {
    const { data, n } = blobs(30, 8, 3, 13);
    const a = pca(data, n, 8, { nComponents: 4 });
    const b = pca(data, n, 8, { nComponents: 4 });
    expect(Array.from(a.components)).toEqual(Array.from(b.components));
  });

  it("centres — the scores have zero mean", () => {
    const { data, n } = blobs(30, 8, 3, 17);
    const res = pca(data, n, 8, { nComponents: 3 });
    for (let c = 0; c < 3; c++) {
      let m = 0;
      for (let i = 0; i < n; i++) m += res.scores[i * 3 + c]!;
      expect(Math.abs(m / n)).toBeLessThan(1e-4);
    }
  });

  it("standardise:true equalises wildly different column scales", () => {
    const n = 100;
    const dim = 3;
    const rnd = mulberry32(23);
    const data = new Float64Array(n * dim);
    for (let i = 0; i < n; i++) {
      data[i * dim] = (rnd() - 0.5) * 1000; // huge scale, no structure
      data[i * dim + 1] = (rnd() - 0.5) * 2;
      data[i * dim + 2] = (rnd() - 0.5) * 2;
    }
    const raw = pca(data, n, dim, { nComponents: 3 });
    const std = pca(data, n, dim, { nComponents: 3, standardise: true });
    // Unstandardised, the big column swallows PC1 entirely; standardised it does not.
    expect(raw.explainedVarianceRatio[0]!).toBeGreaterThan(0.99);
    expect(std.explainedVarianceRatio[0]!).toBeLessThan(0.6);
  });

  it("tolerates a constant column without producing NaN", () => {
    const n = 50;
    const dim = 4;
    const rnd = mulberry32(29);
    const data = new Float64Array(n * dim);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < dim; c++) data[i * dim + c] = c === 2 ? 5 : rnd();
    }
    const res = pca(data, n, dim, { nComponents: 3, standardise: true });
    let finite = true;
    for (let t = 0; t < res.scores.length; t++) if (!Number.isFinite(res.scores[t]!)) finite = false;
    expect(finite).toBe(true);
  });
});

describe("umap", () => {
  it("separates three blobs through a PCA reduction of noisy 80-D data", async () => {
    const nPer = 40;
    const dim = 80;
    const { data, label, n } = blobs(nPer, dim, 3, 33);
    const res = await umap(data, n, dim, { nNeighbors: 12, nEpochs: 200, seed: 6, nComponents: 20 });

    expect(res.reduced).toBeDefined();
    expect(res.reducedDim).toBe(20);

    // Every point's nearest embedded neighbour should share its label.
    let sameLabel = 0;
    for (let i = 0; i < n; i++) {
      let bestJ = -1;
      let best = Number.POSITIVE_INFINITY;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dx = res.embedding[i * 2]! - res.embedding[j * 2]!;
        const dy = res.embedding[i * 2 + 1]! - res.embedding[j * 2 + 1]!;
        const d2 = dx * dx + dy * dy;
        if (d2 < best) {
          best = d2;
          bestJ = j;
        }
      }
      if (label[i] === label[bestJ]!) sameLabel++;
    }
    expect(sameLabel / n).toBeGreaterThan(0.95);
  });

  it("skips PCA below the threshold and runs it above", async () => {
    const { data, n } = blobs(25, 10, 2, 37);
    const small = await umapGraphFor(data, n, 10, { nNeighbors: 8 });
    expect(small.reduced).toBeUndefined();

    const { data: wide, n: n2 } = blobs(25, 90, 2, 37);
    const big = await umapGraphFor(wide, n2, 90, { nNeighbors: 8 });
    expect(big.reducedDim).toBe(50);
  });

  it("pca:false forces the raw feature space", async () => {
    const { data, n } = blobs(20, 90, 2, 41);
    const res = await umapGraphFor(data, n, 90, { nNeighbors: 8, pca: false });
    expect(res.reduced).toBeUndefined();
  });

  it("preserves high-D neighbourhood structure (trustworthiness)", async () => {
    const nPer = 35;
    const dim = 30;
    const { data, n } = blobs(nPer, dim, 3, 43);
    const res = await umap(data, n, dim, { nNeighbors: 12, nEpochs: 200, seed: 8, pca: false });
    expect(trustworthiness(data, res.embedding, n, dim, 2, 10)).toBeGreaterThan(0.9);
  });

  it("uses the injected k-NN", async () => {
    const { data, n } = blobs(20, 6, 2, 47);
    let calls = 0;
    await umapGraphFor(data, n, 6, {
      nNeighbors: 8,
      knn: (d, nn, dd, kk) => {
        calls++;
        // Delegate to the default so the result is still valid.
        return import("./umapGraph").then((m) => m.knnBruteForceCpu(d, nn, dd, kk));
      },
    });
    expect(calls).toBe(1);
  });

  it("asks the k-NN for nNeighbors - 1 (reference semantics)", async () => {
    const { data, n } = blobs(20, 6, 2, 53);
    let sawK = -1;
    await umapGraphFor(data, n, 6, {
      nNeighbors: 15,
      knn: (d, nn, dd, kk) => {
        sawK = kk;
        return import("./umapGraph").then((m) => m.knnBruteForceCpu(d, nn, dd, kk));
      },
    });
    expect(sawK).toBe(14);
  });

  it("rejects n < nNeighbors", async () => {
    const { data } = blobs(2, 4, 1, 59);
    await expect(umap(data, 2, 4, { nNeighbors: 15 })).rejects.toThrow(/n >= nNeighbors/);
  });
});

describe("subsetting", () => {
  it("subsetRows selects and reorders rows", () => {
    const data = [0, 1, 10, 11, 20, 21, 30, 31];
    const got = subsetRows(data, 2, [3, 0, 2]);
    expect(Array.from(got)).toEqual([30, 31, 0, 1, 20, 21]);
  });

  it("subsetColumns selects and reorders columns, preserving row order", () => {
    const data = [0, 1, 2, 10, 11, 12];
    const got = subsetColumns(data, 2, 3, [2, 0]);
    expect(Array.from(got)).toEqual([2, 0, 12, 10]);
  });

  it("a gene subset produces a different graph over the same cells", async () => {
    const { data, n } = blobs(30, 12, 3, 61);
    const all = await umapGraphFor(data, n, 12, { nNeighbors: 10, pca: false });
    // Keep only the noise columns — the cluster signal lives in the first 3.
    const noiseOnly = subsetColumns(data, n, 12, [5, 6, 7, 8, 9, 10, 11]);
    const sub = await umapGraphFor(noiseOnly, n, 7, { nNeighbors: 10, pca: false });
    expect(sub.graph.nEdges).toBeGreaterThan(0);
    // Dropping the signal must change which cells are neighbours at all.
    const key = (g: { head: Uint32Array; tail: Uint32Array; nEdges: number }) => {
      const s = new Set<string>();
      for (let e = 0; e < g.nEdges; e++) s.add(`${g.head[e]}-${g.tail[e]}`);
      return s;
    };
    const a = key(all.graph);
    const b = key(sub.graph);
    let shared = 0;
    for (const e of b) if (a.has(e)) shared++;
    expect(shared / b.size).toBeLessThan(0.5);
  });
});
