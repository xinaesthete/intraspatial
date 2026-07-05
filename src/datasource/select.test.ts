// Golden tests for the pure `Select` heuristic (ADR-0008 §4): frustum culling,
// nearest-point Nyquist level pick, the receding-resolution gradient across an
// oblique plane, and the degrade-to-fit budget policy. CPU-only (no GPU, no I/O).
import { describe, expect, it } from "vitest";
import type { Camera } from "./math";
import { select, selectWithinBudget } from "./select";
import { axisAlignedMultiscale } from "./syntheticLoader";
import type { Multiscale, SelectedChunk } from "./types";

// A small viewport height exaggerates the projected pixel pitch so several pyramid
// levels are exercised within a modest camera range.
const baseCam = { up: [0, 1, 0] as const, fovY: Math.PI / 3, aspect: 1, near: 0.1, far: 3000, viewportHeightPx: 100 };

const volume = (): Multiscale =>
  axisAlignedMultiscale({ voxelDims0: [64, 64, 64], chunkShape: [16, 16, 16], levelCount: 4, voxelSizeWorld: 1 });

const levels = (chunks: readonly SelectedChunk[]): number[] => chunks.map((c) => c.id.level);
const meanLevel = (chunks: readonly SelectedChunk[]): number => chunks.reduce((s, c) => s + c.id.level, 0) / Math.max(1, chunks.length);

describe("select — validity", () => {
  it("names only in-range chunks and reports a consistent budget", () => {
    const ms = volume();
    const cam: Camera = { ...baseCam, eye: [0, 0, -120], forward: [0, 0, 1] };
    const sel = select(ms, cam, { q: 1 });

    expect(sel.chunks.length).toBeGreaterThan(0);
    for (const c of sel.chunks) {
      expect(c.id.level).toBeGreaterThanOrEqual(0);
      expect(c.id.level).toBeLessThan(ms.levelCount);
      expect(c.id.x).toBeGreaterThanOrEqual(0);
      expect(c.nearestDepth).toBeGreaterThanOrEqual(cam.near);
    }
    const summed = sel.chunks.reduce((s, c) => s + c.approxBytes, 0);
    expect(sel.totalApproxBytes).toBe(summed);
    expect(sel.countByLevel.reduce((s, n) => s + n, 0)).toBe(sel.chunks.length);
  });
});

describe("select — distance drives resolution", () => {
  it("picks full-resolution up close and only coarse levels far away", () => {
    const ms = volume();
    const near: Camera = { ...baseCam, eye: [0, 0, -100], forward: [0, 0, 1] };
    const far: Camera = { ...baseCam, eye: [0, 0, -800], forward: [0, 0, 1] };

    const sn = select(ms, near, { q: 1 });
    const sf = select(ms, far, { q: 1 });

    // Up close: some chunks are at level 0 (finest). Far away: none are.
    expect(levels(sn.chunks)).toContain(0);
    expect(levels(sf.chunks)).not.toContain(0);
    // Coarser far selection fetches strictly less data.
    expect(sn.totalApproxBytes).toBeGreaterThan(sf.totalApproxBytes);
  });

  it("a larger detail budget q pulls in finer levels (more bytes)", () => {
    const ms = volume();
    const cam: Camera = { ...baseCam, eye: [0, 0, -400], forward: [0, 0, 1] };
    const coarse = select(ms, cam, { q: 0.5 });
    const fine = select(ms, cam, { q: 4 });
    expect(fine.totalApproxBytes).toBeGreaterThan(coarse.totalApproxBytes);
    expect(meanLevel(fine.chunks)).toBeLessThan(meanLevel(coarse.chunks));
  });
});

describe("select — oblique-plane resolution gradient (the money shot)", () => {
  it("nearer chunks of a tilted plane get finer levels than farther ones", () => {
    // A large plane so a grazing view spans several octaves of depth (near edge
    // ~20 units away, far edge ~1000), giving a genuine multi-level gradient.
    const ms = axisAlignedMultiscale({ voxelDims0: [1024, 1024, 1], chunkShape: [64, 64, 1], levelCount: 6, voxelSizeWorld: 1 });
    // Grazing view across a plane in z≈0: low-y edge is near, high-y edge is far.
    const cam: Camera = { ...baseCam, eye: [0, -500, 60], forward: [0, 0.99, -0.14] };
    const sel = select(ms, cam, { q: 1 });
    expect(sel.chunks.length).toBeGreaterThan(8);

    const depths = sel.chunks.map((c) => c.nearestDepth).sort((a, b) => a - b);
    const median = depths[Math.floor(depths.length / 2)] ?? 0;
    const nearHalf = sel.chunks.filter((c) => c.nearestDepth <= median);
    const farHalf = sel.chunks.filter((c) => c.nearestDepth > median);

    expect(nearHalf.length).toBeGreaterThan(0);
    expect(farHalf.length).toBeGreaterThan(0);
    expect(meanLevel(nearHalf)).toBeLessThan(meanLevel(farHalf));
  });
});

describe("selectWithinBudget — degrade to fit, then fail honestly", () => {
  const ms = volume();
  const near: Camera = { ...baseCam, eye: [0, 0, -100], forward: [0, 0, 1] };

  it("returns the full selection when it fits", () => {
    const generous = select(ms, near, { q: 1 }).totalApproxBytes + 1;
    const r = selectWithinBudget(ms, near, generous, { q: 1 });
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.value.totalApproxBytes).toBeLessThanOrEqual(generous);
  });

  it("coarsens globally to fit under a tight ceiling", () => {
    const r = selectWithinBudget(ms, near, 50_000, { q: 1 });
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.value.totalApproxBytes).toBeLessThanOrEqual(50_000);
      // It had to degrade below the full-resolution selection.
      expect(r.value.totalApproxBytes).toBeLessThan(select(ms, near, { q: 1 }).totalApproxBytes);
    }
  });

  it("returns Err('out of memory') when even the coarsest selection can't fit", () => {
    const r = selectWithinBudget(ms, near, 100, { q: 1 });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toBe("out of memory");
  });
});
