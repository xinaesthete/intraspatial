import { describe, expect, it, vi } from "vite-plus/test";

// The engine by its source: under test the workspace package has no built dist/ to resolve to.
vi.mock("intraspatial/color", () => import("../../../src/color/index"));

const { categoricalColors, packPalette } = await import("./palette");

describe("palette", () => {
  it("packs colours as rgba in 0..1, alpha defaulting to opaque", () => {
    const packed = packPalette([
      [255, 0, 0],
      [0, 51, 0, 0],
    ]);
    expect(Array.from(packed)).toEqual([1, 0, 0, 1, 0, 0.2, 0, 0].map((v) => Math.fround(v)));
  });

  it("never packs an empty palette: the shader always has a stop to read", () => {
    expect(Array.from(packPalette([]))).toEqual([1, 1, 1, 1]);
  });

  it("gives distinct in-gamut colours, one per category", () => {
    const colors = categoricalColors(6);
    expect(colors).toHaveLength(6);
    expect(new Set(colors.map((c) => c.join(","))).size).toBe(6);
    const channels = colors.flat();
    expect(Math.min(...channels)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...channels)).toBeLessThanOrEqual(255);
  });
});
