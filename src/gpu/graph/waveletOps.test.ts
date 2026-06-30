// Wavelet-domain ops through the graph (ADR-0006): the fdwt/idwt pair round-trips,
// and the headline `fdwt → thresholdDetail → idwt` chain denoises. CPU mode (fast).
import { describe, it, expect, beforeAll } from "vitest";
import { Graph, pull, registerWaveletOps } from "./index";

beforeAll(async () => {
  await registerWaveletOps();
});

const W = 32, H = 32, LEVELS = 3;

/** A smooth low-frequency bump — almost all of its energy is in the LL band, so its
 *  detail coefficients are tiny (the regime where wavelet denoising shines). */
function smoothBump(): Float32Array {
  const a = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      a[y * W + x] = 128 + 60 * Math.sin((Math.PI * x) / (W - 1)) * Math.sin((Math.PI * y) / (H - 1));
    }
  }
  return a;
}

/** Deterministic ±12 additive noise (hash-based — reproducible, no Math.random). */
function addNoise(clean: Float32Array): Float32Array {
  const a = Float32Array.from(clean);
  for (let i = 0; i < a.length; i++) a[i] = a[i]! + ((((i * 2654435761) >>> 0) % 25) - 12);
  return a;
}

const mse = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i]! - b[i]!) ** 2;
  return s / a.length;
};

describe("wavelet ops (graph, CPU)", () => {
  it("fdwt → idwt round-trips a grid exactly (5/3, lossless)", async () => {
    const clean = smoothBump();
    const g = new Graph();
    const src = g.grid(clean, W, H);
    const coeffs = g.op1("fdwt", { in: src }, { kernel: "5/3", levels: LEVELS });
    const back = g.op1("idwt", { coeffs }, { kernel: "5/3", levels: LEVELS });
    const out = await pull(g, back, { mode: "cpu" });
    let maxDiff = 0;
    for (let i = 0; i < clean.length; i++) maxDiff = Math.max(maxDiff, Math.abs(out.data![i]! - clean[i]!));
    expect(maxDiff).toBe(0);
  });

  it("fdwt → thresholdDetail → idwt denoises (closer to clean than the noisy input)", async () => {
    const clean = smoothBump();
    const noisy = addNoise(clean);
    const g = new Graph();
    const src = g.grid(noisy, W, H);
    const coeffs = g.op1("fdwt", { in: src }, { kernel: "5/3", levels: LEVELS });
    // Moderate soft shrinkage: enough to knock back the ±12 noise in the detail bands
    // without over-shrinking the smooth signal's own (small) detail coefficients.
    const shrunk = g.op1("thresholdDetail", { coeffs }, { levels: LEVELS, thresh: 6, soft: true });
    const denoised = g.op1("idwt", { coeffs: shrunk }, { kernel: "5/3", levels: LEVELS });
    const out = await pull(g, denoised, { mode: "cpu" });

    const before = mse(noisy, clean);
    const after = mse(out.data!, clean);
    expect(after).toBeLessThan(before); // the denoise actually helped
  });

  it("rejects non-grid input at build time", () => {
    const g = new Graph();
    const pts = g.points([0, 1, 2], [0, 1, 2]);
    expect(() => g.op1("fdwt", { in: pts })).toThrow(/grid/);
  });
});
