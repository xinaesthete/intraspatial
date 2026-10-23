import { beforeAll, describe, expect, it } from "vitest";
import { getDevice, readBackBytes, writeView } from "./device";
import { Graph, pullData, registerBuiltinOps } from "./graph/index";
import { textureToBuffer } from "./graph/textureBridge";
import { adoptDevice } from "./interop/adoptDevice";
import { exclusiveScanGpu } from "./scan/prefixSum";
import { streamCompactGpu } from "./scan/streamCompact";
import { grayScottStepsGpu, seedGrayScott } from "./sim/reactionDiffusion";
import { boxKernel, convolveSeparableGpu, convolveSeparableResident } from "./spatial/convolveSeparable";
import { decimateGpu, decimateResident } from "./spatial/decimate";
import { fuzzyAdjacencyGpu } from "./spatial/fuzzyAdjacency";
import { fuzzyAdjacencyAdaptiveFromRhoGpu, fuzzyAdjacencyAdaptiveGpu } from "./spatial/fuzzyAdjacencyAdaptive";
import { getisOrdGpu } from "./spatial/getisOrd";
import { gramMatrixGpu } from "./spatial/gramMatrix";
import { buildGridIndexGpu } from "./spatial/gridIndex";
import { knnGpu } from "./spatial/knn";
import { kthNeighborDistanceGpu } from "./spatial/kthNeighborDistance";
import { morphologyGpu, morphologyResident } from "./spatial/morphology";
import { nearestNeighborDistancesGpu } from "./spatial/nnDistance";
import { splatDensityToTexture } from "./spatial/splatDensity";

// Every entry point that takes an optional `device`, called WITH it — what a deck layer, or the
// graph executor over `interop/adoptDevice`'s backend, does — against the same call without it.
//
// There is one device in this suite (a second Dawn instance is off limits), so these pin the
// plumbing: the parameter reaches the kernel, the per-device context and pools are built from it,
// and the answer is bit-identical to the default path. That state made for one device is never
// handed to another is pinned without a GPU, in `perDevice.test.ts`.
//
// Kept small, and every comparison is one aggregated assertion: per-element `expect` loops and
// heavy host work both kill the Dawn fork.

let device: GPUDevice;
beforeAll(async () => {
  device = await getDevice();
});

/** Largest |got - want| and largest |want|, so a test can insist the answer is not trivially 0. */
function compare(got: ArrayLike<number>, want: ArrayLike<number>): { diff: number; peak: number; lengths: [number, number] } {
  let diff = 0;
  let peak = 0;
  for (let i = 0; i < want.length; i++) {
    const w = want[i]!;
    const g = got[i]!;
    // Identical infinities (a k-NN pad) are a match, not a NaN.
    if (g !== w) diff = Math.max(diff, Math.abs(g - w));
    if (Number.isFinite(w)) peak = Math.max(peak, Math.abs(w));
  }
  return { diff, peak, lengths: [got.length, want.length] };
}

function expectSame(got: ArrayLike<number>, want: ArrayLike<number>): void {
  const c = compare(got, want);
  expect(c.lengths[0]).toBe(c.lengths[1]);
  expect(c.peak, "the default path produced nothing to compare").toBeGreaterThan(0);
  expect(c.diff).toBe(0);
}

// A small, deterministic cloud: a jittered lattice, so neighbour distances are distinct.
const N = 48;
const XS = Float32Array.from({ length: N }, (_, i) => 1 + (i % 8) * 1.5 + ((i * 7) % 5) * 0.11);
const YS = Float32Array.from({ length: N }, (_, i) => 1 + Math.floor(i / 8) * 1.7 + ((i * 3) % 7) * 0.09);
const XY = Float32Array.from({ length: 2 * N }, (_, t) => (t % 2 === 0 ? XS[t / 2]! : YS[(t - 1) / 2]!));

const W = 20;
const H = 16;
const GRID = Float32Array.from({ length: W * H }, (_, i) => ((i * 2654435761) % 97) / 97);

function storage(bytes: number): GPUBuffer {
  return device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
}

async function readF32(buf: GPUBuffer, n: number): Promise<Float32Array> {
  return new Float32Array(await readBackBytes(device, "explicitDevice:test", buf, 0, n * 4));
}

describe("scan and compaction", () => {
  it("exclusiveScanGpu", async () => {
    const values = Uint32Array.from({ length: 3000 }, (_, i) => (i * 31) % 11);
    const want = await exclusiveScanGpu(values);
    const got = await exclusiveScanGpu(values, { device });
    expectSame(got.scan, want.scan);
    expect(got.total).toBe(want.total);
  });

  it("streamCompactGpu", async () => {
    const mask = Uint32Array.from({ length: 1000 }, (_, i) => (i % 3 === 0 ? 1 : 0));
    const want = await streamCompactGpu(mask);
    const got = await streamCompactGpu(mask, { device });
    expectSame(got.indices, want.indices);
  });
});

describe("grid index and the queries that walk it", () => {
  it("buildGridIndexGpu", async () => {
    const want = await buildGridIndexGpu(XY, { cell: 2 });
    const got = await buildGridIndexGpu(XY, { cell: 2, device });
    expectSame(got.cellOffsets, want.cellOffsets);
    // Order within a cell is whatever the atomics did, so compare the ids as a set.
    expectSame(Int32Array.from(got.pointIds).sort(), Int32Array.from(want.pointIds).sort());
  });

  it("nearestNeighborDistancesGpu, indexed", async () => {
    const want = await nearestNeighborDistancesGpu(XS, YS, { cell: 3 });
    expectSame(await nearestNeighborDistancesGpu(XS, YS, { cell: 3, device }), want);
  });

  it("kthNeighborDistanceGpu, brute force and indexed", async () => {
    expectSame(await kthNeighborDistanceGpu(XS, YS, 4, { device }), await kthNeighborDistanceGpu(XS, YS, 4));
    expectSame(await kthNeighborDistanceGpu(XS, YS, 4, { cell: 3, device }), await kthNeighborDistanceGpu(XS, YS, 4, { cell: 3 }));
  });

  it("knnGpu, indexed", async () => {
    const want = await knnGpu(XY, { n: N, dim: 2, k: 5, cell: 3 });
    const got = await knnGpu(XY, { n: N, dim: 2, k: 5, cell: 3, device });
    expectSame(got.distances, want.distances);
    expectSame(got.indices, want.indices);
  });
});

describe("fuzzy adjacency", () => {
  it("fuzzyAdjacencyGpu", async () => {
    const want = await fuzzyAdjacencyGpu(XS, YS, { sigma: 1.5 });
    expectSame((await fuzzyAdjacencyGpu(XS, YS, { sigma: 1.5, device })).membership, want.membership);
  });

  it("fuzzyAdjacencyAdaptiveFromRhoGpu and fuzzyAdjacencyAdaptiveGpu", async () => {
    const rho = await kthNeighborDistanceGpu(XS, YS, 3);
    const want = await fuzzyAdjacencyAdaptiveFromRhoGpu(XS, YS, rho);
    expectSame((await fuzzyAdjacencyAdaptiveFromRhoGpu(XS, YS, rho, { device })).membership, want.membership);
    const wantK = await fuzzyAdjacencyAdaptiveGpu(XS, YS, { k: 3 });
    expectSame((await fuzzyAdjacencyAdaptiveGpu(XS, YS, { k: 3, device })).membership, wantK.membership);
  });
});

describe("grid kernels", () => {
  it("convolveSeparableGpu and convolveSeparableResident", async () => {
    const k = boxKernel(2);
    const want = await convolveSeparableGpu(GRID, W, H, k);
    expectSame(await convolveSeparableGpu(GRID, W, H, k, device), want);

    const src = storage(W * H * 4);
    const dst = storage(W * H * 4);
    writeView(device.queue, src, GRID);
    await convolveSeparableResident(src, dst, W, H, k, device);
    expectSame(await readF32(dst, W * H), want);
  });

  it("morphologyGpu and morphologyResident", async () => {
    const want = await morphologyGpu(GRID, W, H, 1, "open");
    expectSame(await morphologyGpu(GRID, W, H, 1, "open", device), want);

    const src = storage(W * H * 4);
    const dst = storage(W * H * 4);
    writeView(device.queue, src, GRID);
    await morphologyResident(src, dst, W, H, 1, "open", device);
    expectSame(await readF32(dst, W * H), want);
  });

  it("decimateGpu and decimateResident", async () => {
    const want = await decimateGpu(GRID, W, H, 3, "mean");
    expectSame(await decimateGpu(GRID, W, H, 3, "mean", device), want);

    const src = storage(W * H * 4);
    const dst = storage(want.length * 4);
    writeView(device.queue, src, GRID);
    await decimateResident(src, dst, W, H, 3, "mean", device);
    expectSame(await readF32(dst, want.length), want);
  });

  it("getisOrdGpu", async () => {
    const want = await getisOrdGpu(GRID, W, H, { radius: 2 });
    expectSame((await getisOrdGpu(GRID, W, H, { radius: 2, device })).z, want.z);
  });

  it("grayScottStepsGpu", async () => {
    const seed = seedGrayScott(24, 24);
    const want = await grayScottStepsGpu(seed, 20);
    const got = await grayScottStepsGpu(seed, 20, {}, device);
    expectSame(got.u, want.u);
    expectSame(got.v, want.v);
  });
});

describe("render-path entry points", () => {
  it("textureToBuffer, over a splat on the device it is handed", async () => {
    // The texture comes from an adopted backend's pool, as it would inside a graph run on a
    // renderer's device.
    const backend = adoptDevice(device, "explicitDevice:test");
    const pts = storage(XY.byteLength);
    writeView(device.queue, pts, XY);
    const tex = await backend.leaseTexture(W, H);
    await splatDensityToTexture(pts, N, tex.texture, { width: W, height: H, sigma: 1, bbox: [0, 0, 14, 14], device });

    const viaDefault = storage(W * H * 4);
    const viaExplicit = storage(W * H * 4);
    await textureToBuffer(tex, viaDefault);
    await textureToBuffer(tex, viaExplicit, device);
    expectSame(await readF32(viaExplicit, W * H), await readF32(viaDefault, W * H));
    backend.releaseTexture(tex);
  });

  it("gramMatrixGpu, with its rasters left on the device it is handed", async () => {
    const channels = [
      { label: "a", xs: XS, ys: YS },
      { label: "b", xs: YS, ys: XS },
    ];
    const params = { bbox: [0, 0, 14, 14] as const, width: 32, height: 32, radius: 2 };
    const want = await gramMatrixGpu(channels, params);
    const got = await gramMatrixGpu(channels, { ...params, device });
    expectSame(got.c, want.c);
    expectSame(got.corr, want.corr);
  });
});

describe("graph ops on the executor's backend device", () => {
  it("a resident chain run through an adopted backend matches the default backend", async () => {
    // Each op here hands `ctx.backend.getDevice()` to its kernel, and the executor's texture bridge
    // takes the same device. Through `adoptDevice` that is a renderer's device in real use; here it
    // is the suite's only one, so the answer must match the default backend's exactly.
    registerBuiltinOps();
    const build = () => {
      const g = new Graph();
      const pts = g.points(XS, YS);
      const density = g.op1("splatDensity", { points: pts }, { width: W, height: H, sigma: 1, radiusSigma: 4, bbox: [0, 0, 14, 14] });
      const smooth = g.op1("convolveSeparable", { grid: density }, { kernel: "box", radius: 1 });
      const opened = g.op1("morphology", { grid: smooth }, { op: "open", radius: 1 });
      const coarse = g.op1("decimate", { grid: opened }, { factor: 2, mode: "max" });
      const counts = g.op1("cellCounts", {
        buckets: g.op1("gridIndex", { points: pts }, { cell: 2, minX: 0, minY: 0, maxX: 14, maxY: 14 }),
      });
      return { g, coarse, counts };
    };
    const ctx = { backend: adoptDevice(device, "explicitDevice:graph") };

    const a = build();
    const b = build();
    expectSame(await pullData(b.g, b.coarse, { ctx }), await pullData(a.g, a.coarse));
    expectSame(await pullData(b.g, b.counts, { ctx }), await pullData(a.g, a.counts));
  });
});
