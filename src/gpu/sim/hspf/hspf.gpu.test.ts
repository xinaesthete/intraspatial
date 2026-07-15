// Cross-check the fused WGSL kernel against the CPU-tested per-cell math (ADR-0011, decision 5):
// a small fixture, a host reference gather built from `gatherCell`/`ld`, compared to the kernel
// on a handful of cells. This is the one-time port cross-check, not a maintained whole-field
// golden. Kept lean (few cells, one step, one test) because Dawn-on-Node's exit teardown is
// timing-sensitive and heavy per-fork work trips its segfault.
import { describe, expect, it } from "vitest";
import { getDevice } from "../../device";
import { type HspfParams, type HspfScaffold, hspfStepsGpu } from "./kernel";
import { blendFitness, DEFAULT_FITNESS, gatherCell, ld, selectionWeights, sum4, type Vec4 } from "./math";
import type { Neighbourhood } from "./neighbourhood";

const W = 16;
const H = 12;
const N = W * H;

function fixtureScaffold(): HspfScaffold {
  const hbs = new Float32Array(N);
  const weights = new Float32Array(N);
  for (let row = 0; row < H; row++) {
    for (let col = 0; col < W; col++) {
      const i = row * W + col;
      hbs[i] = col < 2 ? -2 : 0.1 + 0.2 * (col / W); // ocean strip + HbS gradient
      weights[i] = 0.5 + 0.5 * (row / H);
    }
  }
  return { hbs, weights, width: W, height: H };
}

function fixturePfsa(scaffold: HspfScaffold): Float32Array {
  const out = new Float32Array(5 * N);
  for (let row = 0; row < H; row++) {
    for (let col = 0; col < W; col++) {
      const i = row * W + col;
      if ((scaffold.hbs[i] ?? -2) < 0) {
        for (let l = 0; l < 5; l++) out[l * N + i] = scaffold.hbs[i] ?? -2;
        continue;
      }
      const raw: [number, number, number, number] = [0.4 + 0.2 * Math.sin(col * 0.5), 0.2 + 0.1 * Math.cos(row * 0.5), 0.2, 0.15];
      const t = raw[0] + raw[1] + raw[2] + raw[3];
      for (let g = 0; g < 4; g++) out[g * N + i] = (raw[g] ?? 0) / t;
      const v: Vec4 = [out[0 * N + i] ?? 0, out[1 * N + i] ?? 0, out[2 * N + i] ?? 0, out[3 * N + i] ?? 0];
      out[4 * N + i] = ld(v);
    }
  }
  return out;
}

const NBHD: Neighbourhood = {
  data: Float32Array.from([0, 0, 1, 1, 0, 0.8, -1, 0, 0.8, 0, 2, 0.6, 3, -1, 0.5, -2, -3, 0.4]),
  count: 6,
};

/** Host reference for one interior cell — mirrors the kernel's gather using the tested math. */
function referenceCell(
  scaffold: HspfScaffold,
  nbhd: Neighbourhood,
  pfsa: Float32Array,
  params: HspfParams,
  col: number,
  row: number,
): Vec4 {
  const { width: w, height: h } = scaffold;
  const n = w * h;
  const fs = scaffold.hbs[row * w + col] ?? -2;
  const { a, s } = selectionWeights(fs);
  const fit = blendFitness(params.fitness ?? DEFAULT_FITNESS, a, s);
  const bites: { pf: Vec4; weight: number }[] = [];
  for (let k = 0; k < nbhd.count; k++) {
    const bx = col + (nbhd.data[k * 3] ?? 0);
    const by = row + (nbhd.data[k * 3 + 1] ?? 0);
    if (bx < 0 || bx >= w || by < 0 || by >= h) continue;
    const bidx = by * w + bx;
    if ((scaffold.hbs[bidx] ?? -2) < 0) continue;
    const weight = (nbhd.data[k * 3 + 2] ?? 0) * (scaffold.weights[bidx] ?? 0);
    bites.push({ pf: [pfsa[bidx] ?? 0, pfsa[n + bidx] ?? 0, pfsa[2 * n + bidx] ?? 0, pfsa[3 * n + bidx] ?? 0], weight });
  }
  return gatherCell(bites, fit, params.twoBiteRate ?? 0);
}

describe("HsPf fused kernel", () => {
  it("mirrors the CPU gather reference and stays a bounded distribution", async () => {
    const device = await getDevice();
    const scaffold = fixtureScaffold();
    const pfsa = fixturePfsa(scaffold);
    const params: HspfParams = { twoBiteRate: 0.3 };

    const gpu = await hspfStepsGpu(device, scaffold, NBHD, pfsa, 1, params);

    // Cross-check a spread of interior land cells against the host reference.
    for (const [col, row] of [
      [3, 3],
      [8, 6],
      [12, 2],
      [5, 9],
      [14, 10],
      [2, 5],
    ] as const) {
      const ref = referenceCell(scaffold, NBHD, pfsa, params, col, row);
      const i = row * W + col;
      for (let g = 0; g < 4; g++) expect(Math.abs((gpu[g * N + i] ?? 0) - (ref[g] ?? 0))).toBeLessThan(1e-4);
      expect(Math.abs((gpu[4 * N + i] ?? 0) - ld(ref))).toBeLessThan(1e-4);
    }

    // Direct invariants on the GPU output: ocean stays sentinel; land layers form a distribution.
    for (let i = 0; i < N; i++) {
      const fs = scaffold.hbs[i] ?? -2;
      if (fs < 0) {
        expect(gpu[i] ?? 0).toBeLessThan(0);
      } else {
        const v: Vec4 = [gpu[i] ?? 0, gpu[N + i] ?? 0, gpu[2 * N + i] ?? 0, gpu[3 * N + i] ?? 0];
        expect(Math.abs(sum4(v) - 1)).toBeLessThan(1e-3);
      }
    }
  });
});
