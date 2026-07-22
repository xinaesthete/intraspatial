import { beforeAll, describe, expect, it } from "vitest";
import { getDevice } from "../device";
import { splatDensityGpu } from "../spatial/splatDensity";
import { Graph, pullData } from "./index";

// ADR-0017 stages 2-3, numeric half: the converted ops must still agree with their references now
// that their edges no longer round-trip through the host. Lease/pool behaviour and the transfer
// counts live in resident.gpu.test.ts.
//
// Split across two files because Dawn-on-Node segfaults a fork once enough device churn
// accumulates, and it takes the not-yet-reported results down with it (see vitest.gpu.config.ts).
//
// Built on a plain host `grid` source rather than `splatDensity`: the subject is the two
// *converted* ops, and splat's own GPU and CPU paths differ by ~1.6e-2, which would swamp this.

const W = 16;
const H = 16;

/** A smooth, non-degenerate field — no flat regions, so a convolution actually changes it. */
function ramp(): Float32Array {
  const g = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) g[y * W + x] = Math.sin(x * 0.4) * Math.cos(y * 0.3) + 1.5;
  }
  return g;
}

/** Element-wise closeness, reported on the first mismatch rather than as hundreds of separate
 *  assertions (which is both slow and unreadable when it fails). */
function expectClose(got: Float32Array, want: Float32Array, digits: number): void {
  expect(got.length).toBe(want.length);
  for (let i = 0; i < got.length; i++) {
    if (Math.abs(got[i]! - want[i]!) > 0.5 * 10 ** -digits) {
      expect.fail(`index ${i}: got ${got[i]}, want ${want[i]} (tolerance 1e-${digits})`);
    }
  }
}

describe("Tier-2 resident edges — values", () => {
  beforeAll(async () => {
    await getDevice();
  });

  it("matches the CPU reference across a resident convolve -> threshold chain", async () => {
    const g = new Graph();
    const src = g.grid(ramp(), W, H);
    // Gaussian, not box: `boxKernel` is all ones, i.e. a local *sum*, which scales the field by
    // ~9 at radius 1 and drives the threshold into saturation (every cell 1.0 — a test that
    // passes while checking nothing). The gaussian is normalised, so values stay in range.
    const smooth = g.op1("convolveSeparable", { grid: src }, { kernel: "gaussian", radius: 2 });
    // Soft (logistic) threshold on purpose: it is continuous, so a small float difference stays
    // a small difference. A hard step would turn one into a full 0/1 flip at the boundary and
    // make this a coin toss rather than a numeric check.
    const mask = g.op1("threshold", { in: smooth }, { thresh: 1.5, soft: true, softness: 8 });

    const gpu = await pullData(g, mask);
    const cpu = await pullData(g, mask, { mode: "cpu" });

    expect(gpu.length).toBe(W * H);
    expectClose(gpu, cpu, 4);
    // Guard against passing on a degenerate (constant) field.
    expect(new Set(Array.from(gpu)).size).toBeGreaterThan(1);

    // --- and the same, with the value fanned out to two consumers ---
    //
    // Kept in this test rather than its own: `src` feeds two independent convolves whose
    // results are summed, so the producing buffer's refcount must reach zero only after BOTH
    // have read it. Release it after the first and the second reads a buffer the pool has
    // already handed out, and `sum` quietly stops being a + b.
    const g2 = new Graph();
    const src2 = g2.grid(ramp(), W, H);
    const a = g2.op1("convolveSeparable", { grid: src2 }, { kernel: "gaussian", radius: 1 });
    const b = g2.op1("convolveSeparable", { grid: src2 }, { kernel: "gaussian", radius: 3 });
    const sum = g2.op1("addGrids", { a, b });

    const gotSum = await pullData(g2, sum);
    expectClose(gotSum, await pullData(g2, sum, { mode: "cpu" }), 3);
    expect(new Set(Array.from(gotSum)).size).toBeGreaterThan(1);
  });

  it("splats identically resident and host-side", async () => {
    // The resident splat changed two things that could silently corrupt the grid: the vertex
    // stage now reads the graph's packed [x0,y0,...] points at stride 2 instead of a host-packed
    // stride-3 buffer, and the copy's 256-byte row padding is stripped by a compute pass instead
    // of a JS loop. Both are invisible to the download budget, which only counts transfers.
    //
    // Compared against `splatDensityGpu` — the host path, which has its own CPU-KDE golden test
    // — rather than against `cpuGolden`: same render, same texture, so the two should agree to
    // float precision, and any disagreement isolates exactly what this change touched. (Splat's
    // render path vs the analytic CPU KDE differ by ~1.6e-2 regardless, which would mask a real
    // de-pad bug.)
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < 40; i++) {
      xs.push(50 + (i % 7) * 0.6);
      ys.push(50 + Math.floor(i / 7) * 0.6);
    }
    // Width 24 is deliberate: 24*4 = 96 bytes per row, which is NOT 256-aligned, so the copy is
    // genuinely padded and the de-pad pass has real work to do. A 64-wide grid would be aligned
    // by luck and test nothing.
    const w = 24;
    const h = 24;
    const bbox: [number, number, number, number] = [0, 0, 100, 100];
    const opts = { width: w, height: h, sigma: 2, radiusSigma: 4, bbox };

    const g = new Graph();
    const resident = await pullData(g, g.op1("splatDensity", { points: g.points(xs, ys) }, opts));
    const host = (await splatDensityGpu(xs, ys, opts)).data;

    expect(resident.length).toBe(w * h);
    expectClose(resident, host, 6);
    // And it is a real field, not an all-zero grid that would match trivially.
    expect(resident.some((v) => v > 0.01)).toBe(true);
  });
});
