import { describe, expect, it } from "vitest";
import { fuzzySimplicialSet, knnBruteForceCpu } from "../../spatial/umapGraph";
import { mulberry32, optimizeLayout, trustworthiness } from "../../spatial/umapLayout";
import { GpuUmapLayout } from "./umapLayoutGpu";

// The GPU layout races by design (Hogwild; see the module header), so NOTHING here
// compares coordinates — not against the host implementation, and not against another
// GPU run. What must hold is that the embedding preserves the same neighbourhood
// structure, which is what `trustworthiness` measures and what a UMAP is actually for.
//
// Fixtures are built at module scope, before any device work.

function blobs(nPer: number, dim: number, nBlobs: number, seed: number) {
  const rnd = mulberry32(seed);
  const gauss = () => {
    const u = Math.max(rnd(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
  };
  const n = nPer * nBlobs;
  const data = new Float64Array(n * dim);
  const label = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const b = Math.floor(i / nPer);
    label[i] = b;
    for (let c = 0; c < dim; c++) data[i * dim + c] = gauss() + (c < 2 ? b * 18 : 0);
  }
  return { data, label, n };
}

const DIM = 6;
const NPER = 30;
const NBLOBS = 3;
const fixture = blobs(NPER, DIM, NBLOBS, 17);
const graph = fuzzySimplicialSet(knnBruteForceCpu(fixture.data, fixture.n, DIM, 10), { nNeighbors: 11 });
/** The host layout on the same graph — the quality bar the GPU has to reach, not a
 *  coordinate reference. */
const hostEmbedding = optimizeLayout(graph, { nEpochs: 300, seed: 5 });
const hostTrust = trustworthiness(fixture.data, hostEmbedding, fixture.n, DIM, 2, 8);

describe("GpuUmapLayout", () => {
  it("reaches host-comparable trustworthiness from a random init", async () => {
    const layout = await GpuUmapLayout.create(graph, { nEpochs: 300, seed: 5 });
    try {
      layout.step(300);
      const emb = await layout.read();
      let finite = true;
      for (let t = 0; t < emb.length; t++) if (!Number.isFinite(emb[t]!)) finite = false;
      expect(finite).toBe(true);

      const gpuTrust = trustworthiness(fixture.data, emb, fixture.n, DIM, 2, 8);
      expect(gpuTrust).toBeGreaterThan(0.9);
      // Allowed to be a little worse than the host — the races cost some precision — but
      // not materially worse. A large gap means the kernel is wrong, not merely racy.
      expect(gpuTrust).toBeGreaterThan(hostTrust - 0.05);
    } finally {
      layout.destroy();
    }
  });

  it("separates the blobs it was given", async () => {
    const layout = await GpuUmapLayout.create(graph, { nEpochs: 300, seed: 9 });
    try {
      layout.step(300);
      const emb = await layout.read();
      // Every point's nearest embedded neighbour should share its blob.
      let sameBlob = 0;
      for (let i = 0; i < fixture.n; i++) {
        let best = Number.POSITIVE_INFINITY;
        let bestJ = -1;
        for (let j = 0; j < fixture.n; j++) {
          if (j === i) continue;
          const dx = emb[i * 2]! - emb[j * 2]!;
          const dy = emb[i * 2 + 1]! - emb[j * 2 + 1]!;
          const d2 = dx * dx + dy * dy;
          if (d2 < best) {
            best = d2;
            bestJ = j;
          }
        }
        if (fixture.label[i] === fixture.label[bestJ]!) sameBlob++;
      }
      expect(sameBlob / fixture.n).toBeGreaterThan(0.95);
    } finally {
      layout.destroy();
    }
  });

  it("keeps the embedding resident — stepping twice continues, it does not restart", async () => {
    const layout = await GpuUmapLayout.create(graph, { nEpochs: 300, seed: 3 });
    try {
      layout.step(150);
      const half = await layout.read();
      expect(layout.epoch).toBe(150);
      layout.step(150);
      const full = await layout.read();
      expect(layout.epoch).toBe(300);

      // Continuing must improve on the halfway layout, and must not have thrown the
      // coordinates away (a restart would land somewhere unrelated).
      const halfTrust = trustworthiness(fixture.data, half, fixture.n, DIM, 2, 8);
      const fullTrust = trustworthiness(fixture.data, full, fixture.n, DIM, 2, 8);
      expect(fullTrust).toBeGreaterThanOrEqual(halfTrust - 0.02);

      let moved = 0;
      for (let t = 0; t < full.length; t++) moved += Math.abs(full[t]! - half[t]!);
      const scale = Math.max(...Array.from(full, Math.abs));
      // It moved (the second half did work) but stayed in the same neighbourhood of
      // configuration space (it did not restart).
      expect(moved).toBeGreaterThan(0);
      expect(moved / full.length).toBeLessThan(scale);
    } finally {
      layout.destroy();
    }
  });

  it("continues from a supplied embedding", async () => {
    // The animation path: hand it coordinates and it refines them rather than
    // re-initialising. With alpha low the layout should barely move.
    const layout = await GpuUmapLayout.create(graph, { nEpochs: 300, seed: 1, initialAlpha: 0.02 }, Float32Array.from(hostEmbedding));
    try {
      layout.step(5);
      const emb = await layout.read();
      let drift = 0;
      for (let t = 0; t < emb.length; t++) drift += Math.abs(emb[t]! - hostEmbedding[t]!);
      const span = Math.max(...Array.from(hostEmbedding, Math.abs));
      expect(drift / emb.length).toBeLessThan(span * 0.5);
      // And it is still a good layout, not a smeared one.
      expect(trustworthiness(fixture.data, emb, fixture.n, DIM, 2, 8)).toBeGreaterThan(hostTrust - 0.05);
    } finally {
      layout.destroy();
    }
  });

  it("reheat re-bases the schedule so a settled layout moves again", async () => {
    const layout = await GpuUmapLayout.create(graph, { nEpochs: 100, seed: 4 });
    try {
      layout.step(100);
      const settled = await layout.read();
      expect(layout.alphaAt(layout.epoch)).toBe(0);

      // Stepping a settled layout does nothing: no learning rate, and every edge's
      // next-sample epoch is past the horizon.
      layout.step(5);
      const stillSettled = await layout.read();
      let idle = 0;
      for (let t = 0; t < settled.length; t++) idle += Math.abs(stillSettled[t]! - settled[t]!);
      expect(idle).toBe(0);

      // Reheating restores both, so it optimises again.
      layout.reheat(0);
      expect(layout.alphaAt(layout.epoch)).toBeGreaterThan(0);
      layout.step(10);
      const reheated = await layout.read();
      let moved = 0;
      for (let t = 0; t < settled.length; t++) moved += Math.abs(reheated[t]! - settled[t]!);
      expect(moved).toBeGreaterThan(0);
    } finally {
      layout.destroy();
    }
  });

  it("supports 3-D and rejects out-of-range dims", async () => {
    const layout = await GpuUmapLayout.create(graph, { dim: 3, nEpochs: 300, seed: 6 });
    try {
      layout.step(300);
      const emb = await layout.read();
      expect(emb.length).toBe(fixture.n * 3);
      expect(trustworthiness(fixture.data, emb, fixture.n, DIM, 3, 8)).toBeGreaterThan(0.9);
    } finally {
      layout.destroy();
    }
    await expect(GpuUmapLayout.create(graph, { dim: 4 })).rejects.toThrow(/dim must be/);
  });
});
