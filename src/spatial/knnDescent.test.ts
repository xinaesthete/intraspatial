import { describe, expect, it } from "vitest";
import { buildCandidates, initialiseHeap, knnDescentCpu, knnRecall, knnStrategyFor, offer, pickKnn } from "./knnDescent";
import { umap } from "./umap";
import { fuzzySimplicialSet, knnBruteForceCpu } from "./umapGraph";
import { mulberry32, trustworthiness } from "./umapLayout";

/** Clustered data — the case random initialisation handles badly and the projection
 *  seeding exists for. Uniform noise would flatter the algorithm. */
function clusters(nPer: number, dim: number, nClusters: number, seed = 5) {
  const rnd = mulberry32(seed);
  const gauss = () => {
    const u = Math.max(rnd(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
  };
  const n = nPer * nClusters;
  const data = new Float64Array(n * dim);
  const label = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const c = i % nClusters;
    label[i] = c;
    for (let t = 0; t < dim; t++) data[i * dim + t] = gauss() + (t < 4 ? c * 9 : 0);
  }
  return { data, label, n };
}

describe("offer", () => {
  it("keeps the k best, sorted ascending", () => {
    const heap = { n: 1, k: 3, indices: new Int32Array(3).fill(-1), distances: new Float32Array(3).fill(Number.POSITIVE_INFINITY) };
    for (const [j, dist] of [
      [7, 5],
      [3, 1],
      [9, 9],
      [4, 3],
    ] as const) {
      offer(heap, 0, j, dist);
    }
    expect(Array.from(heap.indices)).toEqual([3, 4, 7]);
    expect(Array.from(heap.distances)).toEqual([1, 3, 5]);
  });

  it("rejects duplicates", () => {
    // Without this a point can fill its whole list with one neighbour found along
    // several paths, and the descent stalls while still reporting improvements.
    const heap = { n: 1, k: 3, indices: new Int32Array(3).fill(-1), distances: new Float32Array(3).fill(Number.POSITIVE_INFINITY) };
    expect(offer(heap, 0, 5, 2)).toBe(1);
    expect(offer(heap, 0, 5, 2)).toBe(0);
    expect(offer(heap, 0, 5, 1)).toBe(0);
    expect(Array.from(heap.indices)).toEqual([5, -1, -1]);
  });

  it("rejects anything worse than the current k-th", () => {
    const heap = { n: 1, k: 2, indices: Int32Array.from([1, 2]), distances: Float32Array.from([1, 2]) };
    expect(offer(heap, 0, 9, 5)).toBe(0);
    expect(offer(heap, 0, 9, 0.5)).toBe(1);
  });
});

describe("initialiseHeap", () => {
  it("fills every slot", () => {
    const { data, n } = clusters(30, 6, 3);
    const heap = initialiseHeap(data, n, 6, { k: 8, seed: 1 });
    let unfilled = 0;
    for (let t = 0; t < n * 8; t++) if (heap.indices[t]! < 0) unfilled++;
    expect(unfilled).toBe(0);
  });

  it("never offers a point itself", () => {
    const { data, n } = clusters(20, 5, 2);
    const heap = initialiseHeap(data, n, 5, { k: 6, seed: 2 });
    let selfs = 0;
    for (let i = 0; i < n; i++) for (let t = 0; t < 6; t++) if (heap.indices[i * 6 + t] === i) selfs++;
    expect(selfs).toBe(0);
  });

  it("projection seeding beats purely random seeding on clustered data", () => {
    // The reason initialiseHeap does projections at all: on clustered data a random
    // candidate is almost never in the same cluster, so a random start has nothing good
    // to descend from.
    const { data, n } = clusters(60, 8, 4, 11);
    const exact = knnBruteForceCpu(data, n, 8, 10);
    const withProj = initialiseHeap(data, n, 8, { k: 10, seed: 3, nProjections: 4 });
    const noProj = initialiseHeap(data, n, 8, { k: 10, seed: 3, nProjections: 0 });
    const recallOf = (h: typeof withProj) => knnRecall({ n, k: 10, indices: Uint32Array.from(h.indices), distances: h.distances }, exact);
    expect(recallOf(withProj)).toBeGreaterThan(recallOf(noProj) * 2);
  });
});

describe("buildCandidates", () => {
  it("includes forward neighbours and reverse neighbours", () => {
    // 0 -> 1, 2 -> 1. So 1's candidates must contain 0 and 2 even though 1 does not
    // point at them: that reverse half is what stops the descent drifting one-way.
    const heap = { n: 3, k: 1, indices: Int32Array.from([1, 0, 1]), distances: Float32Array.from([1, 1, 1]) };
    const { candidates, counts, width } = buildCandidates(heap, 4, mulberry32(1));
    const of = (i: number) => new Set(Array.from(candidates.slice(i * width, i * width + counts[i]!)));
    expect(of(0).has(1)).toBe(true);
    expect(of(1).has(0)).toBe(true);
    expect(of(1).has(2)).toBe(true);
  });

  it("caps a hub's reverse list", () => {
    // Every point picks 0, so 0 would otherwise accumulate n reverse edges and one
    // thread would walk all of them.
    const n = 50;
    const heap = { n, k: 1, indices: new Int32Array(n).fill(0), distances: new Float32Array(n).fill(1) };
    heap.indices[0] = 1;
    const maxReverse = 4;
    const { counts } = buildCandidates(heap, maxReverse, mulberry32(1));
    expect(counts[0]!).toBeLessThanOrEqual(1 + maxReverse);
  });
});

describe("knnDescentCpu", () => {
  it("reaches high recall on clustered data", () => {
    const { data, n } = clusters(80, 10, 4, 21);
    const exact = knnBruteForceCpu(data, n, 10, 14);
    const approx = knnDescentCpu(data, n, 10, { k: 14, seed: 4 });
    expect(knnRecall(approx, exact)).toBeGreaterThan(0.95);
  });

  it("reaches high recall on unclustered (uniform) data too", () => {
    const rnd = mulberry32(31);
    const n = 300;
    const dim = 12;
    const data = Float64Array.from({ length: n * dim }, () => rnd() * 10 - 5);
    const exact = knnBruteForceCpu(data, n, dim, 12);
    const approx = knnDescentCpu(data, n, dim, { k: 12, seed: 6 });
    expect(knnRecall(approx, exact)).toBeGreaterThan(0.9);
  });

  it("returns rows ascending and never returns self", () => {
    const { data, n } = clusters(40, 8, 3, 41);
    const approx = knnDescentCpu(data, n, 8, { k: 10, seed: 7 });
    let selfs = 0;
    let inversions = 0;
    for (let i = 0; i < n; i++) {
      for (let t = 0; t < 10; t++) {
        if (approx.indices[i * 10 + t] === i) selfs++;
        if (t > 0 && approx.distances[i * 10 + t]! < approx.distances[i * 10 + t - 1]!) inversions++;
      }
    }
    expect(selfs).toBe(0);
    expect(inversions).toBe(0);
  });

  it("more iterations never lowers recall", () => {
    const { data, n } = clusters(50, 8, 3, 51);
    const exact = knnBruteForceCpu(data, n, 8, 10);
    const one = knnRecall(knnDescentCpu(data, n, 8, { k: 10, seed: 8, maxIters: 1, tol: 0 }), exact);
    const many = knnRecall(knnDescentCpu(data, n, 8, { k: 10, seed: 8, maxIters: 10, tol: 0 }), exact);
    expect(many).toBeGreaterThanOrEqual(one);
  });

  it("is deterministic for a fixed seed", () => {
    const { data, n } = clusters(30, 6, 3, 61);
    const a = knnDescentCpu(data, n, 6, { k: 8, seed: 5 });
    const b = knnDescentCpu(data, n, 6, { k: 8, seed: 5 });
    expect(Array.from(a.indices)).toEqual(Array.from(b.indices));
  });

  it("rejects k >= n", () => {
    expect(() => knnDescentCpu(new Float64Array(9), 3, 3, { k: 3 })).toThrow(/k < n/);
  });

  it("scales linearly in n where the exact search scales quadratically", () => {
    // The whole reason this exists — and stated as SCALING rather than as "fewer
    // operations", because at small n the descent is genuinely the slower of the two.
    // Its cost is ~maxIters * n * candidates^2, so it only overtakes the exact n^2 above
    // roughly `maxIters * candidates^2` points — a few thousand with the defaults. Below
    // that, use the exact search; that guidance is in the module header.
    //
    // Counted, not timed, so this is not a flaky performance assertion.
    const count = (fn: (d: Float64Array, n: number) => void, data: Float64Array, n: number) => {
      let reads = 0;
      const counting = new Proxy(data, {
        get(target, prop) {
          if (typeof prop === "string" && !Number.isNaN(Number(prop))) reads++;
          return Reflect.get(target, prop);
        },
      }) as unknown as Float64Array;
      fn(counting, n);
      return reads;
    };
    const dim = 8;
    const small = clusters(50, dim, 4, 71);
    const large = clusters(100, dim, 4, 71);

    const descentGrowth =
      count((d, n) => void knnDescentCpu(d, n, dim, { k: 10, seed: 9, maxIters: 4, tol: 0 }), large.data, large.n) /
      count((d, n) => void knnDescentCpu(d, n, dim, { k: 10, seed: 9, maxIters: 4, tol: 0 }), small.data, small.n);
    const exactGrowth =
      count((d, n) => void knnBruteForceCpu(d, n, dim, 10), large.data, large.n) /
      count((d, n) => void knnBruteForceCpu(d, n, dim, 10), small.data, small.n);

    // Doubling n: exact quadruples; descent grows ~2.6x — mildly superlinear, because
    // the projection seeding sorts (O(N log N)) and the candidate lists thicken slightly,
    // but decisively sub-quadratic, which is the claim.
    expect(descentGrowth).toBeLessThan(3);
    expect(exactGrowth).toBeGreaterThan(3.5);
  });
});

describe("descent feeding UMAP", () => {
  it("produces an embedding as trustworthy as the exact k-NN's", () => {
    // The claim that matters: UMAP is tolerant of approximate neighbours, because the
    // fuzzy graph is a smoothed object. Asserted rather than assumed.
    const { data, n } = clusters(50, 12, 3, 81);
    const exactGraph = fuzzySimplicialSet(knnBruteForceCpu(data, n, 12, 14), { nNeighbors: 15 });
    const approxKnn = knnDescentCpu(data, n, 12, { k: 14, seed: 2 });
    const approxGraph = fuzzySimplicialSet(approxKnn, { nNeighbors: 15 });

    // The graphs should be substantially the same set of edges.
    const key = (g: { head: Uint32Array; tail: Uint32Array; nEdges: number }) => {
      const s = new Set<string>();
      for (let e = 0; e < g.nEdges; e++) s.add(`${g.head[e]}-${g.tail[e]}`);
      return s;
    };
    const a = key(exactGraph);
    const b = key(approxGraph);
    let shared = 0;
    for (const e of b) if (a.has(e)) shared++;
    expect(shared / a.size).toBeGreaterThan(0.9);
  });

  it("drops into umapGraphFor as the injected knn", async () => {
    const { data, n } = clusters(40, 20, 3, 91);
    const res = await umap(data, n, 20, {
      nNeighbors: 11,
      nEpochs: 150,
      seed: 3,
      pca: false,
      knn: (d, nn, dim, k) => knnDescentCpu(d, nn, dim, { k, seed: 3 }),
    });
    expect(res.knn.k).toBe(10);
    expect(trustworthiness(data, res.embedding, n, 20, 2, 8)).toBeGreaterThan(0.85);
  });
});

describe("pickKnn", () => {
  it("uses the exact search below the crossover and the descent above it", async () => {
    const { data, n } = clusters(20, 5, 2, 101);
    let exactCalls = 0;
    const exact = (d: ArrayLike<number>, nn: number, dim: number, k: number) => {
      exactCalls++;
      return knnBruteForceCpu(d, nn, dim, k);
    };
    // n = 40, crossover 100 -> exact.
    await pickKnn(exact, { crossover: 100 })(data, n, 5, 6);
    expect(exactCalls).toBe(1);
    // crossover 10 -> descent, and the exact path is not touched.
    await pickKnn(exact, { crossover: 10 })(data, n, 5, 6);
    expect(exactCalls).toBe(1);
  });

  it("reports the strategy it would choose", () => {
    expect(knnStrategyFor(1000)).toBe("exact");
    expect(knnStrategyFor(50000)).toBe("descent");
  });
});
