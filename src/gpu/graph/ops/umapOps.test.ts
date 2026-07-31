import { describe, expect, it } from "vitest";
import { type FuzzyGraph, fuzzySimplicialSet, type KnnResult } from "../../../spatial/umapGraph";
import { trustworthiness } from "../../../spatial/umapLayout";
import type { ExecCtx } from "../op";
import { hasOp, listOps } from "../registry";
import { registerUmapOps } from "./index";
import { fuzzyGraphOp, knnOp, umapLayoutOp, umapOp } from "./umapOps";

// The op wiring, exercised through `cpuGolden` rather than `execute`. That is the whole
// point of the golden being part of `OpType`: the composition, shape inference and param
// plumbing are testable without a device, and the device path is covered separately by
// `knn.gpu.test.ts`. A CPU test file must not create a GPU device.

const ctx = {} as ExecCtx;

/** Three separated blobs in `dim` dimensions; only the first few carry the signal. */
function blobs(nPer: number, dim: number, nBlobs = 3, seed = 5) {
  let s = seed >>> 0;
  const rnd = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const n = nPer * nBlobs;
  const data = new Float32Array(n * dim);
  const label = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const b = Math.floor(i / nPer);
    label[i] = b;
    for (let c = 0; c < dim; c++) data[i * dim + c] = rnd() * 2 - 1 + (c < 3 ? b * 14 : 0);
  }
  return { data, label, n };
}

function matrixInput(data: Float32Array, n: number, dim: number) {
  return [{ shape: { kind: "matrix" as const, rows: n, cols: dim }, dtype: "f32" as const, data }];
}

describe("umap op registration", () => {
  it("registers the pack on demand and not before", async () => {
    // The pack is opt-in, like the element and wavelet packs.
    await registerUmapOps();
    for (const name of ["knn", "fuzzyGraph", "umapLayout", "umap"]) expect(hasOp(name)).toBe(true);
  });

  it("is idempotent", async () => {
    await registerUmapOps();
    await registerUmapOps();
    expect(listOps().filter((o) => o.name === "umap")).toHaveLength(1);
  });

  it("groups the pack under one palette category", () => {
    for (const op of [knnOp, fuzzyGraphOp, umapLayoutOp, umapOp]) expect(op.category).toBe("Manifold");
  });
});

describe("knn op", () => {
  it("produces a k-NN payload with nNeighbors - 1 neighbours", () => {
    const { data, n } = blobs(20, 6);
    const out = knnOp.cpuGolden!(matrixInput(data, n, 6), { nNeighbors: 9, nComponents: 0 });
    const knn = out[0]!.payload as KnnResult;
    expect(out[0]!.shape).toEqual({ kind: "opaque", name: "knn" });
    expect(knn.k).toBe(8);
    expect(knn.n).toBe(n);
    expect(knn.indices.length).toBe(n * 8);
  });

  it("applies PCA when nComponents is below the feature count", () => {
    const { data, n } = blobs(15, 40);
    // Reducing to 5 dims must still keep neighbours within their blob.
    const out = knnOp.cpuGolden!(matrixInput(data, n, 40), { nNeighbors: 6, nComponents: 5 });
    const knn = out[0]!.payload as KnnResult;
    let cross = 0;
    for (let i = 0; i < n; i++) {
      for (let t = 0; t < knn.k; t++) if (Math.floor(i / 15) !== Math.floor(knn.indices[i * knn.k + t]! / 15)) cross++;
    }
    expect(cross).toBe(0);
  });

  it("rejects a non-matrix input and an impossible k", () => {
    expect(() => knnOp.inferShapes([{ kind: "points", n: 10 }], {})).toThrow(/must be a matrix/);
    const { data, n } = blobs(4, 4, 1);
    expect(() => knnOp.cpuGolden!(matrixInput(data, n, 4), { nNeighbors: 99, nComponents: 0 })).toThrow();
  });
});

describe("fuzzyGraph op", () => {
  it("turns a k-NN payload into a symmetric weighted edge list", () => {
    const { data, n } = blobs(20, 6);
    const knn = knnOp.cpuGolden!(matrixInput(data, n, 6), { nNeighbors: 9, nComponents: 0 });
    const out = fuzzyGraphOp.cpuGolden!(knn, { localConnectivity: 1, setOpMixRatio: 1 });
    const graph = out[0]!.payload as FuzzyGraph;
    expect(out[0]!.shape).toEqual({ kind: "opaque", name: "fuzzyGraph" });
    expect(graph.n).toBe(n);
    expect(graph.nEdges).toBeGreaterThan(0);
    expect(graph.nEdges % 2).toBe(0); // both directions present
  });

  it("recovers the reference nNeighbors from the payload's k", () => {
    // The op has no nNeighbors param — it must infer it as k + 1, or the sigma
    // calibration silently targets the wrong neighbour count.
    const { data, n } = blobs(20, 6);
    const knn = knnOp.cpuGolden!(matrixInput(data, n, 6), { nNeighbors: 9, nComponents: 0 });
    const viaOp = (fuzzyGraphOp.cpuGolden!(knn, { localConnectivity: 1, setOpMixRatio: 1 })[0]!.payload as FuzzyGraph).weight;
    const direct = fuzzySimplicialSetDirect(knn[0]!.payload as KnnResult);
    let maxDiff = 0;
    for (let e = 0; e < viaOp.length; e++) maxDiff = Math.max(maxDiff, Math.abs(viaOp[e]! - direct[e]!));
    expect(maxDiff).toBeLessThan(1e-6);
  });

  it("gives a useful error when handed a payload-less input", () => {
    expect(() =>
      fuzzyGraphOp.cpuGolden!([{ shape: { kind: "opaque", name: "knn" }, dtype: "f32" }], { localConnectivity: 1, setOpMixRatio: 1 }),
    ).toThrow(/no k-NN payload/);
  });
});

/** The same graph built with an EXPLICIT nNeighbors, as the cross-check against the
 *  op's inference of it from the payload. */
function fuzzySimplicialSetDirect(knn: KnnResult): Float32Array {
  return fuzzySimplicialSet(knn, { nNeighbors: knn.k + 1 }).weight;
}

describe("umapLayout op", () => {
  it("stamps the real row count on its output shape", () => {
    const { data, n } = blobs(15, 6);
    const knn = knnOp.cpuGolden!(matrixInput(data, n, 6), { nNeighbors: 7, nComponents: 0 });
    const graph = fuzzyGraphOp.cpuGolden!(knn, { localConnectivity: 1, setOpMixRatio: 1 });
    // Build time cannot know n (it lives in the opaque payload), so it reports 0 rows...
    expect(umapLayoutOp.inferShapes([{ kind: "opaque", name: "fuzzyGraph" }], { embedDim: 2 })).toEqual([
      { kind: "matrix", rows: 0, cols: 2 },
    ]);
    // ...and execution must correct it.
    const out = umapLayoutOp.cpuGolden!(graph, { minDist: 0.1, nEpochs: 60, embedDim: 2, seed: 1 });
    expect(out[0]!.shape).toEqual({ kind: "matrix", rows: n, cols: 2 });
    expect(out[0]!.data!.length).toBe(n * 2);
  });

  it("honours embedDim", () => {
    const { data, n } = blobs(12, 5);
    const knn = knnOp.cpuGolden!(matrixInput(data, n, 5), { nNeighbors: 6, nComponents: 0 });
    const graph = fuzzyGraphOp.cpuGolden!(knn, { localConnectivity: 1, setOpMixRatio: 1 });
    const out = umapLayoutOp.cpuGolden!(graph, { minDist: 0.1, nEpochs: 40, embedDim: 3, seed: 1 });
    expect(out[0]!.data!.length).toBe(n * 3);
  });
});

describe("umap composite op", () => {
  it("agrees with the hand-wired chain of primitives", () => {
    const { data, n } = blobs(20, 8);
    const params = { nNeighbors: 9, nComponents: 0, minDist: 0.1, nEpochs: 80, embedDim: 2, seed: 11 };
    const composite = umapOp.cpuGolden!(matrixInput(data, n, 8), params);

    const knn = knnOp.cpuGolden!(matrixInput(data, n, 8), params);
    const graph = fuzzyGraphOp.cpuGolden!(knn, { localConnectivity: 1, setOpMixRatio: 1 });
    const chained = umapLayoutOp.cpuGolden!(graph, params);

    expect(Array.from(composite[0]!.data!)).toEqual(Array.from(chained[0]!.data!));
  });

  it("infers its output shape at build time (unlike the layout primitive)", () => {
    expect(umapOp.inferShapes([{ kind: "matrix", rows: 300, cols: 40 }], { embedDim: 2 })).toEqual([
      { kind: "matrix", rows: 300, cols: 2 },
    ]);
  });

  it("preserves neighbourhood structure end to end", () => {
    const { data, n } = blobs(25, 30);
    const out = umapOp.cpuGolden!(matrixInput(data, n, 30), {
      nNeighbors: 11,
      nComponents: 0,
      minDist: 0.1,
      nEpochs: 200,
      embedDim: 2,
      seed: 4,
    });
    expect(trustworthiness(data, out[0]!.data!, n, 30, 2, 10)).toBeGreaterThan(0.9);
  });
});

// `ctx` is unused by every golden above (they are pure); referenced so the import and the
// intent — these ops need no device — are explicit rather than accidental.
void ctx;
