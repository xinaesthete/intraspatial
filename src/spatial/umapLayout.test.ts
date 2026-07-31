import { describe, expect, it } from "vitest";
import { fuzzySimplicialSet, knnBruteForceCpu } from "./umapGraph";
import { fitAB, initLayout, makeEpochsPerSample, mulberry32, optimizeLayout, optimizeLayoutStep, reheatLayout, trustworthiness } from "./umapLayout";

function twoBlobs(nPer: number, dim: number, seed = 7): { data: Float64Array; label: Uint8Array } {
  const rnd = mulberry32(seed);
  const n = nPer * 2;
  const data = new Float64Array(n * dim);
  const label = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const blob = i < nPer ? 0 : 1;
    label[i] = blob;
    for (let c = 0; c < dim; c++) {
      const u = Math.max(rnd(), 1e-12);
      const v = rnd();
      const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      data[i * dim + c] = g + (c === 0 && blob === 1 ? 30 : 0);
    }
  }
  return { data, label };
}

describe("fitAB", () => {
  it("recovers the reference a,b for the default min_dist/spread", () => {
    // The published values for min_dist=0.1, spread=1.0 — a ~1.577, b ~0.895.
    const { a, b } = fitAB(0.1, 1);
    expect(a).toBeCloseTo(1.577, 2);
    expect(b).toBeCloseTo(0.895, 2);
  });

  it("fits the piecewise target closely across the sampled range", () => {
    const minDist = 0.1;
    const spread = 1;
    const { a, b } = fitAB(minDist, spread);
    let worst = 0;
    for (let i = 1; i <= 100; i++) {
      const x = (3 * spread * i) / 100;
      const target = x <= minDist ? 1 : Math.exp(-(x - minDist) / spread);
      worst = Math.max(worst, Math.abs(1 / (1 + a * x ** (2 * b)) - target));
    }
    expect(worst).toBeLessThan(0.05);
  });

  it("a larger minDist gives a flatter curve near zero (points may clump less tightly)", () => {
    const tight = fitAB(0.001, 1);
    const loose = fitAB(0.5, 1);
    const psi = (p: { a: number; b: number }, x: number) => 1 / (1 + p.a * x ** (2 * p.b));
    // At a small separation the loose fit still reads as "fully connected".
    expect(psi(loose, 0.3)).toBeGreaterThan(psi(tight, 0.3));
  });

  it("returns positive parameters for a range of inputs", () => {
    for (const [md, sp] of [
      [0, 1],
      [0.1, 1],
      [0.5, 1],
      [0.1, 0.5],
      [0.25, 2],
    ]) {
      const { a, b } = fitAB(md, sp);
      expect(a).toBeGreaterThan(0);
      expect(b).toBeGreaterThan(0);
      expect(Number.isFinite(a)).toBe(true);
      expect(Number.isFinite(b)).toBe(true);
    }
  });
});

describe("makeEpochsPerSample", () => {
  it("samples the strongest edge every epoch and weaker ones proportionally less", () => {
    const eps = makeEpochsPerSample(Float32Array.from([1, 0.5, 0.25]), 100);
    expect(eps[0]!).toBeCloseTo(1, 6);
    expect(eps[1]!).toBeCloseTo(2, 6);
    expect(eps[2]!).toBeCloseTo(4, 6);
  });

  it("never-sampled (zero-weight) edges get Infinity", () => {
    const eps = makeEpochsPerSample(Float32Array.from([1, 0]), 100);
    expect(eps[1]!).toBe(Number.POSITIVE_INFINITY);
  });
});

describe("optimizeLayout", () => {
  it("separates two well-separated blobs in the embedding", () => {
    const nPer = 60;
    const { data, label } = twoBlobs(nPer, 8, 21);
    const n = nPer * 2;
    const knn = knnBruteForceCpu(data, n, 8, 14);
    const graph = fuzzySimplicialSet(knn, { nNeighbors: 15 });
    const emb = optimizeLayout(graph, { nEpochs: 200, seed: 3 });

    // Mean within-blob distance must be far below the between-blob centroid gap.
    const centroid = (want: number) => {
      let cx = 0;
      let cy = 0;
      let count = 0;
      for (let i = 0; i < n; i++) {
        if (label[i] !== want) continue;
        cx += emb[i * 2]!;
        cy += emb[i * 2 + 1]!;
        count++;
      }
      return [cx / count, cy / count] as const;
    };
    const [ax, ay] = centroid(0);
    const [bx, by] = centroid(1);
    const gap = Math.hypot(ax - bx, ay - by);

    let within = 0;
    for (let i = 0; i < n; i++) {
      const [cx, cy] = label[i] === 0 ? [ax, ay] : [bx, by];
      within += Math.hypot(emb[i * 2]! - cx, emb[i * 2 + 1]! - cy);
    }
    within /= n;
    expect(gap).toBeGreaterThan(within * 3);
  });

  it("improves trustworthiness over the random initialisation", () => {
    const nPer = 40;
    const { data } = twoBlobs(nPer, 6, 31);
    const n = nPer * 2;
    const knn = knnBruteForceCpu(data, n, 6, 10);
    const graph = fuzzySimplicialSet(knn, { nNeighbors: 11 });

    const before = initLayout(graph, { nEpochs: 150, seed: 5 }).embedding;
    const t0 = trustworthiness(data, before, n, 6, 2, 8);
    const after = optimizeLayout(graph, { nEpochs: 150, seed: 5 });
    const t1 = trustworthiness(data, after, n, 6, 2, 8);

    expect(t0).toBeLessThan(0.8);
    expect(t1).toBeGreaterThan(0.9);
    expect(t1).toBeGreaterThan(t0);
  });

  it("is deterministic for a fixed seed", () => {
    const { data } = twoBlobs(25, 4, 41);
    const knn = knnBruteForceCpu(data, 50, 4, 8);
    const graph = fuzzySimplicialSet(knn, { nNeighbors: 9 });
    const a = optimizeLayout(graph, { nEpochs: 60, seed: 17 });
    const b = optimizeLayout(graph, { nEpochs: 60, seed: 17 });
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it("produces finite coordinates (the gradient clip holds)", () => {
    const { data } = twoBlobs(30, 5, 51);
    const knn = knnBruteForceCpu(data, 60, 5, 10);
    const graph = fuzzySimplicialSet(knn, { nNeighbors: 11 });
    const emb = optimizeLayout(graph, { nEpochs: 200, seed: 9 });
    for (let t = 0; t < emb.length; t++) expect(Number.isFinite(emb[t]!)).toBe(true);
  });

  it("supports 3-D embeddings", () => {
    const { data } = twoBlobs(25, 5, 61);
    const knn = knnBruteForceCpu(data, 50, 5, 8);
    const graph = fuzzySimplicialSet(knn, { nNeighbors: 9 });
    const emb = optimizeLayout(graph, { dim: 3, nEpochs: 100, seed: 2 });
    expect(emb.length).toBe(50 * 3);
    expect(trustworthiness(data, emb, 50, 5, 3, 6)).toBeGreaterThan(0.85);
  });
});

describe("continuation — the animated-transition path", () => {
  it("initLayout accepts an existing embedding and starts from it", () => {
    const { data } = twoBlobs(20, 4, 71);
    const knn = knnBruteForceCpu(data, 40, 4, 8);
    const graph = fuzzySimplicialSet(knn, { nNeighbors: 9 });
    const seedEmb = Float32Array.from({ length: 80 }, (_, i) => i * 0.01);
    const state = initLayout(graph, { nEpochs: 50 }, seedEmb);
    expect(state.embedding).toBe(seedEmb);
  });

  it("rejects an embedding of the wrong size", () => {
    const graph = { n: 10, weight: Float32Array.from([1, 1]) };
    expect(() => initLayout(graph, { dim: 2 }, new Float32Array(8))).toThrow(/expected 20/);
  });

  it("relaxing a settled layout under a NEW graph moves less than a cold start", () => {
    // The claim the interactive story rests on: continuing from the previous subset's
    // coordinates lands near them, whereas a fresh random init lands somewhere
    // arbitrary. Measured as total displacement from the old layout.
    const nPer = 30;
    const n = nPer * 2;
    const { data } = twoBlobs(nPer, 6, 81);
    // A perturbed "different gene subset" over the same cells.
    const rnd = mulberry32(99);
    const data2 = Float64Array.from(data, (v) => v + (rnd() - 0.5) * 0.4);

    const g1 = fuzzySimplicialSet(knnBruteForceCpu(data, n, 6, 10), { nNeighbors: 11 });
    const g2 = fuzzySimplicialSet(knnBruteForceCpu(data2, n, 6, 10), { nNeighbors: 11 });

    const settled = optimizeLayout(g1, { nEpochs: 200, seed: 4 });
    const continued = optimizeLayout(g2, { nEpochs: 30, seed: 4, initialAlpha: 0.2 }, Float32Array.from(settled));
    const coldStart = optimizeLayout(g2, { nEpochs: 30, seed: 77 });

    const drift = (emb: Float32Array) => {
      let acc = 0;
      for (let t = 0; t < emb.length; t++) acc += Math.abs(emb[t]! - settled[t]!);
      return acc / emb.length;
    };
    expect(drift(continued)).toBeLessThan(drift(coldStart) / 3);
  });

  it("a step advances the epoch counter and decays the learning rate", () => {
    const { data } = twoBlobs(20, 4, 91);
    const knn = knnBruteForceCpu(data, 40, 4, 8);
    const graph = fuzzySimplicialSet(knn, { nNeighbors: 9 });
    const state = initLayout(graph, { nEpochs: 100, seed: 1 });
    const a0 = optimizeLayoutStep(state, graph, { nEpochs: 100, seed: 1 });
    expect(state.epoch).toBe(1);
    for (let i = 0; i < 50; i++) optimizeLayoutStep(state, graph, { nEpochs: 100, seed: 1 });
    const a50 = optimizeLayoutStep(state, graph, { nEpochs: 100, seed: 1 });
    expect(a50).toBeLessThan(a0);
    expect(a50).toBeGreaterThan(0);
  });
});

describe("trustworthiness", () => {
  it("is 1 when the embedding is the data itself", () => {
    const { data } = twoBlobs(20, 2, 101);
    const emb = Float32Array.from(data);
    expect(trustworthiness(data, emb, 40, 2, 2, 5)).toBeCloseTo(1, 6);
  });

  it("is much lower for random coordinates", () => {
    const { data } = twoBlobs(20, 4, 111);
    const rnd = mulberry32(5);
    const emb = Float32Array.from({ length: 80 }, () => rnd() * 20 - 10);
    expect(trustworthiness(data, emb, 40, 4, 2, 5)).toBeLessThan(0.85);
  });
});

describe("reheatLayout", () => {
  it("makes a settled layout move again", () => {
    const { data } = twoBlobs(25, 5, 121);
    const n = 50;
    const graph = fuzzySimplicialSet(knnBruteForceCpu(data, n, 5, 8), { nNeighbors: 9 });
    const state = initLayout(graph, { nEpochs: 100, seed: 2 });
    for (let e = 0; e < 100; e++) optimizeLayoutStep(state, graph, { nEpochs: 100, seed: 2 });

    // Perturb, then try to resume by rewinding the epoch counter ALONE. The learning
    // rate looks healthy but every edge's next-sample epoch is already past the horizon,
    // so nothing is due and the layout is frozen — the bug this function exists for.
    const perturb = () => {
      for (let t = 0; t < state.embedding.length; t++) state.embedding[t] = state.embedding[t]! + (t % 7) * 0.1;
    };
    perturb();
    state.epoch = 20;
    const naiveBefore = Float32Array.from(state.embedding);
    const alpha = optimizeLayoutStep(state, graph, { nEpochs: 100, seed: 2 });
    expect(alpha).toBeGreaterThan(0);
    let naiveMoved = 0;
    for (let t = 0; t < state.embedding.length; t++) naiveMoved += Math.abs(state.embedding[t]! - naiveBefore[t]!);
    expect(naiveMoved).toBe(0);

    // Re-basing the schedule as well actually resumes it.
    reheatLayout(state, 20);
    const before = Float32Array.from(state.embedding);
    for (let e = 0; e < 5; e++) optimizeLayoutStep(state, graph, { nEpochs: 100, seed: 2 });
    let moved = 0;
    for (let t = 0; t < state.embedding.length; t++) moved += Math.abs(state.embedding[t]! - before[t]!);
    expect(moved).toBeGreaterThan(0);
  });

  it("re-converges after a perturbation", () => {
    const { data } = twoBlobs(25, 5, 131);
    const n = 50;
    const graph = fuzzySimplicialSet(knnBruteForceCpu(data, n, 5, 8), { nNeighbors: 9 });
    const state = initLayout(graph, { nEpochs: 150, seed: 3 });
    for (let e = 0; e < 150; e++) optimizeLayoutStep(state, graph, { nEpochs: 150, seed: 3 });
    const settledTrust = trustworthiness(data, state.embedding, n, 5, 2, 8);

    const rnd = mulberry32(9);
    for (let t = 0; t < state.embedding.length; t++) state.embedding[t] = state.embedding[t]! + (rnd() - 0.5) * 8;
    const kickedTrust = trustworthiness(data, state.embedding, n, 5, 2, 8);
    expect(kickedTrust).toBeLessThan(settledTrust);

    reheatLayout(state, 0);
    for (let e = 0; e < 150; e++) optimizeLayoutStep(state, graph, { nEpochs: 150, seed: 3 });
    expect(trustworthiness(data, state.embedding, n, 5, 2, 8)).toBeGreaterThan(kickedTrust);
  });
});
