import { describe, expect, it } from "vitest";
import { QUARTIC, TOPHAT } from "../../spatial/kernels";
import type { CellCloud } from "../../spatial/tcm";
import { computeTcmKernel } from "../../spatial/tcmKernel";
import { computeTcmRender, kernelDensityGpu } from "./tcmRender";

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function cloud(n: number, seed: number): CellCloud {
  const rnd = mulberry32(seed);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(2 + rnd() * 16);
    ys.push(2 + rnd() * 16);
  }
  return { xs, ys };
}

// SCOPE NOTE, rewritten 2026-08-01. This file used to say the parity suite had to live in the
// BROWSER, because "this module reliably segfaults Dawn-on-Node's atexit walk once a test file has
// run a couple of render-plus-readback cycles". That was the Dawn Instance-lifetime bug in
// `src/gpu/device.ts` (fixed 2026-07-29, `4e326b0`) — 24 render-plus-readback cycles in this file
// now run clean, so the parity sweep is back below.
//
// Kept from the old note: mass conservation pins pass 1's normalisation and the world mapping,
// sign structure pins pass 2's nonlinearity. Both are cheap and fail loudly.

const BBOX = [0, 0, 20, 20] as const;

describe("computeTcmRender / kernelDensityGpu", () => {
  it("pass 1 conserves mass: the kernels are unit-mass, so ∫ρ̂ = n", async () => {
    // The strongest cheap check of pass 1. Every kernel integrates to 1, so summing the rendered
    // density over the grid and multiplying by the cell area must return the point COUNT — which
    // simultaneously pins the normalisation, the world-to-NDC mapping and the grid extent.
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < 40; i++) {
      xs.push(6 + (i % 8) * 1.1);
      ys.push(6 + Math.floor(i / 8) * 1.4);
    }
    const w = 64;
    const cellArea = ((BBOX[2] - BBOX[0]) / w) ** 2;
    const g = await kernelDensityGpu(xs, ys, { width: w, height: w, radius: 2.5, kernel: QUARTIC, bbox: BBOX });
    const mass = g.reduce((s, v) => s + v, 0) * cellArea;
    expect(mass).toBeGreaterThan(39.0); // all 40 points sit well inside the ROI, so nothing is clipped
    expect(mass).toBeLessThan(41.0);
  });

  it("Γ is positive where A sits inside B and negative where A is excluded", async () => {
    // Pass 2's whole job: the per-point mark fetch, the nonlinearity, and the signed weighted splat.
    const bx: number[] = [];
    const by: number[] = [];
    for (let i = 0; i < 7; i++)
      for (let j = 0; j < 7; j++) {
        bx.push(4 + i / 3);
        by.push(4 + j / 3);
      }
    const A: CellCloud = { xs: [5, 15], ys: [5, 15] };
    const grid = await computeTcmRender(
      A,
      { xs: bx, ys: by },
      {
        width: 40,
        height: 40,
        bbox: BBOX,
        radius: 5,
        sigma: 1,
        alpha: 5,
        kernel: TOPHAT,
        markWidth: 128,
        markHeight: 128,
      },
    );
    const at = (x: number, y: number) => grid[y * 40 + x]!;
    expect(at(10, 10)).toBeGreaterThan(0); // A at (5,5), buried in the B cluster
    expect(at(30, 30)).toBeLessThan(0); // A at (15,15), no B within the radius
  });

  // The sweep that used to be exiled to the browser. It compares the rendered field against the
  // continuous oracle at several resolutions, because the interesting claim is not one error
  // number — it is that the error is DISCRETISATION and therefore shrinks as the grid refines. A
  // formulation error would sit at roughly constant relative size instead.
  it("converges to the continuous oracle as the grid refines", async () => {
    const a = cloud(120, 0x51);
    const b = cloud(150, 0x9e);
    const errs: number[] = [];
    for (const w of [32, 64, 128]) {
      const p = { width: w, height: w, radius: 3, sigma: 1.5, alpha: 5, kernel: QUARTIC, bbox: BBOX } as const;
      const gpu = await computeTcmRender(a, b, p);
      const cpu = computeTcmKernel(a, b, p);
      let peak = 0;
      for (let i = 0; i < cpu.length; i++) peak = Math.max(peak, Math.abs(cpu[i]!));
      expect(peak, `${w}²: the oracle is all zeroes, so nothing was compared`).toBeGreaterThan(0);
      let relMax = 0;
      for (let i = 0; i < cpu.length; i++) relMax = Math.max(relMax, Math.abs(gpu[i]! - cpu[i]!) / peak);
      errs.push(relMax);
    }
    // measured 2026-08-01: 3.3e-2 → 5.6e-3 → 2.3e-3, monotone
    expect(errs[1]!).toBeLessThan(errs[0]!);
    expect(errs[2]!).toBeLessThan(errs[1]!);
    expect(errs[2]!).toBeLessThan(5e-3);
  });

  // TOPHAT is deliberately excluded from the convergence check above: it is a hard disk, so the
  // field has a step, and a MAX-norm error at a discontinuity does not vanish with resolution
  // (measured: 5.3e-2 → 2.1e-2 → 2.6e-2, i.e. it plateaus). Bounded is the honest assertion.
  it("stays bounded against the oracle for the discontinuous tophat mark", async () => {
    const a = cloud(120, 0x51);
    const b = cloud(150, 0x9e);
    const p = { width: 64, height: 64, radius: 3, sigma: 1.5, alpha: 5, kernel: TOPHAT, bbox: BBOX } as const;
    const gpu = await computeTcmRender(a, b, p);
    const cpu = computeTcmKernel(a, b, p);
    let peak = 0;
    for (let i = 0; i < cpu.length; i++) peak = Math.max(peak, Math.abs(cpu[i]!));
    let sq = 0;
    let relMax = 0;
    for (let i = 0; i < cpu.length; i++) {
      const e = Math.abs(gpu[i]! - cpu[i]!) / peak;
      relMax = Math.max(relMax, e);
      sq += e * e;
    }
    // bounds set from measurement, 2026-08-01: relMax 2.1e-2, RMS 6.7e-3 at 64²
    expect(relMax).toBeLessThan(5e-2);
    // RMS is the meaningful bound: it stays small where the max-norm cannot, because
    // only the cells straddling the step carry the large error
    expect(Math.sqrt(sq / cpu.length)).toBeLessThan(1.5e-2);
  });

  // The crash this file was shaped around: "a couple of render-plus-readback cycles" killed the
  // fork. Twelve, asserting each one, is a regression guard for the Instance-lifetime bug.
  it("survives twelve render-plus-readback cycles", async () => {
    for (let k = 0; k < 12; k++) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (let i = 0; i < 40; i++) {
        xs.push(6 + (i % 8) * 1.1);
        ys.push(6 + Math.floor(i / 8) * 1.4);
      }
      const w = 64;
      const cellArea = ((BBOX[2] - BBOX[0]) / w) ** 2;
      const g = await kernelDensityGpu(xs, ys, { width: w, height: w, radius: 2.5, kernel: QUARTIC, bbox: BBOX });
      const mass = g.reduce((s, v) => s + v, 0) * cellArea;
      expect(mass, `cycle ${k}`).toBeGreaterThan(39.0);
      expect(mass, `cycle ${k}`).toBeLessThan(41.0);
    }
  });
});
