// The SyntheticLoader: deterministic, in-memory, bounded output. Proves getChunk
// returns correctly-sized, in-range, reproducible tiles (ADR-0008 Milestone 1).
import { describe, expect, it } from "vitest";
import { chunkCounts, chunkVoxelExtent } from "./multiscale";
import { mandelbrotField, mandelbulbField, syntheticPlane, syntheticVolume } from "./syntheticLoader";

const inUnit = (data: Float32Array): boolean => data.every((v) => Number.isFinite(v) && v >= 0 && v <= 1);

describe("synthetic fields", () => {
  it("mandelbulb centre is interior (never escapes → 1)", () => {
    expect(mandelbulbField([0.5, 0.5, 0.5])).toBe(1);
  });
  it("mandelbrot origin is interior, a corner escapes", () => {
    expect(mandelbrotField([0.7, 0.5, 0])).toBe(1); // c ≈ 0 → interior
    expect(mandelbrotField([0.995, 0.99, 0])).toBeLessThan(0.5); // far outside → escapes fast
  });
});

describe("syntheticLoader.getChunk", () => {
  it("returns a full interior chunk of the requested dims, in range", async () => {
    const { loader } = syntheticPlane({ width: 256, height: 256, chunk: 64, levelCount: 3 });
    const tile = await loader.getChunk({ level: 0, x: 1, y: 1, z: 0 });
    expect(tile.dims).toEqual([64, 64, 1]);
    expect(tile.data.length).toBe(64 * 64);
    expect(inUnit(tile.data)).toBe(true);
  });

  it("clamps border chunks to the level extent", async () => {
    // 100 voxels, chunk 64 → chunk index 1 holds the remaining 36.
    const { ms, loader } = syntheticPlane({ width: 100, height: 100, chunk: 64, levelCount: 2 });
    const extent = chunkVoxelExtent(ms, { level: 0, x: 1, y: 1, z: 0 });
    expect(extent).toEqual([36, 36, 1]);
    const tile = await loader.getChunk({ level: 0, x: 1, y: 1, z: 0 });
    expect(tile.dims).toEqual([36, 36, 1]);
    expect(tile.data.length).toBe(36 * 36);
  });

  it("is deterministic (same id → identical bytes)", async () => {
    const { loader } = syntheticVolume({ size: 64, chunk: 16, levelCount: 3 });
    const a = await loader.getChunk({ level: 1, x: 0, y: 1, z: 0 });
    const b = await loader.getChunk({ level: 1, x: 0, y: 1, z: 0 });
    expect(Array.from(a.data)).toEqual(Array.from(b.data));
    expect(inUnit(a.data)).toBe(true);
  });

  it("coarser levels have fewer chunks over the same field", async () => {
    const { ms } = syntheticVolume({ size: 64, chunk: 16, levelCount: 3 });
    const c0 = chunkCounts(ms, 0);
    const c2 = chunkCounts(ms, 2);
    expect(c0[0]).toBeGreaterThan(c2[0]);
  });
});
