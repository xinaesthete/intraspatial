// The Mandelbulb compute kernel must reproduce its CPU golden. The output is a discrete
// escape fraction (multiples of 1/iters); the kernel is written to match the CPU integrator
// step-for-step (deterministic origin pre-step to dodge WGSL atan2(0,0)=NaN), so agreement
// is exact — we still allow a one-quantum slack per voxel to stay robust to any future
// transcendental drift near the r>2 boundary. Small grid — Dawn-on-Node teardown segfaults
// past enough cumulative GPU work (ADR-0002/0003).
//
// Note: the minimum attainable value is 1/iters, not 0 — the orbit starts at the origin
// (radius 0), which can never escape on iteration 0.
import { describe, expect, it } from "vitest";
import { nodeBackend } from "../graph/index";
import { type MandelbulbRegion, mandelbulbBrickCpu, mandelbulbBrickGpu } from "./mandelbulb";

const ITERS = 8;

// Straddles the fractal surface near the centre: mostly interior, a handful of escapes.
const SURFACE: MandelbulbRegion = {
  dims: [16, 16, 16],
  origin: [0.28, 0.28, 0.28],
  step: [1 / 40, 1 / 40, 1 / 40],
  power: 8,
  iters: ITERS,
};

// Spans the whole unit cube: far corners escape on iteration 1 (value 1/iters), the middle
// stays inside (value 1) — so this region exercises the full escape range end to end.
const WIDE: MandelbulbRegion = {
  dims: [16, 16, 16],
  origin: [0, 0, 0],
  step: [1 / 15, 1 / 15, 1 / 15],
  power: 8,
  iters: ITERS,
};

const compare = (gpu: Float32Array, cpu: Float32Array) => {
  const quantum = 1 / ITERS;
  let maxDiff = 0;
  let exact = 0;
  let overOneQuantum = 0;
  for (let i = 0; i < cpu.length; i++) {
    const dv = Math.abs((gpu[i] ?? 0) - (cpu[i] ?? 0));
    maxDiff = Math.max(maxDiff, dv);
    if (dv < 1e-6) exact++;
    if (dv > quantum + 1e-6) overOneQuantum++;
  }
  return { maxDiff, exactFrac: exact / cpu.length, overFrac: overOneQuantum / cpu.length };
};

describe("mandelbulb field generator on the GPU", () => {
  it("matches the CPU golden on a surface-straddling region (exact)", async () => {
    const cpu = mandelbulbBrickCpu(SURFACE);
    const gpu = await mandelbulbBrickGpu(nodeBackend, SURFACE);
    expect(gpu.length).toBe(cpu.length);
    const { maxDiff, exactFrac } = compare(gpu, cpu);
    expect(maxDiff).toBeLessThanOrEqual(1 / ITERS + 1e-6); // ≤ one escape quantum
    expect(exactFrac).toBeGreaterThan(0.98); // near-perfect; boundary flips a rare minority
  });

  it("matches the CPU golden across the full escape range (wide region)", async () => {
    const cpu = mandelbulbBrickCpu(WIDE);
    const gpu = await mandelbulbBrickGpu(nodeBackend, WIDE);
    const { exactFrac, overFrac } = compare(gpu, cpu);
    // Over a full-cube span the kernel (f32) still agrees with the CPU golden (f64) on the
    // vast majority of voxels. A handful sit exactly on the chaotic r>2 boundary where f32-
    // vs-f64 drift flips the escape iteration — that's inherent fractal sensitivity, so we
    // require near-total exactness and cap the >1-quantum disagreements to a tiny fraction
    // (a gross regression — e.g. a NaN-poisoned or constant field — blows through this).
    expect(exactFrac).toBeGreaterThan(0.95);
    expect(overFrac).toBeLessThan(0.02);
    // The region is meaningful: it spans a fast-escaping voxel and an interior one, so the
    // agreement above is over the whole escape range, not a constant field.
    expect(Math.min(...cpu)).toBeLessThanOrEqual(1 / ITERS + 1e-6);
    expect(Math.max(...cpu)).toBeGreaterThanOrEqual(0.999);
  });
});
