// Build-time checks of the zonal ops: no device needed.
import { describe, expect, it } from "vite-plus/test";
import { polygonDomain } from "#src/spatial/domain";
import { polygonRingBuffers } from "#src/spatial/zonal";
import { Graph, polygonRingSources } from "../index";

describe("zonal op build", () => {
  it("shapes rows by regions, refuses mismatched grids and channel-bearing inputs", () => {
    const g = new Graph();
    const labels = g.grid([1, 2, 0, 1], 2, 2);
    const field = g.grid([1, 2, 3, 4], 2, 2);
    const outs = g.op("zonalHistogram", { labels, field }, { regions: 3, bins: 8 });
    expect(outs).toHaveLength(5);
    expect(() => g.op("zonalHistogram", { labels: g.grid([1, 2, 3], 3, 1), field }, { regions: 3 })).toThrow(/labels are 3x1/);
    const wide = g.source({
      shape: { kind: "grid", width: 2, height: 2 },
      dtype: "f32",
      data: new Float32Array(8),
      axes: [{ name: "c", type: "channel", length: 2 }],
    });
    expect(() => g.op("zonalHistogram", { labels, field: wide }, { regions: 3 })).toThrow(/single-channel/);
    const vec = g.source({
      shape: { kind: "grid", width: 2, height: 2 },
      dtype: "f32",
      element: { kind: "vec", n: 2 },
      data: new Float32Array(8),
    });
    expect(() => g.op("zonalHistogram", { labels, field: vec }, { regions: 3 })).toThrow(/must be scalar/);
  });

  it("rasterisePolygons takes ring-buffer matrices and gives a grid of the asked size", () => {
    const g = new Graph();
    const rings = polygonRingBuffers(
      [
        polygonDomain([
          [0, 0],
          [3, 0],
          [0, 3],
        ]),
      ],
      { width: 4, height: 4 },
    );
    const [labels] = g.op("rasterisePolygons", polygonRingSources(g, rings), { width: 4, height: 4 });
    expect(labels).toBeDefined();
    expect(() => g.op("rasterisePolygons", { edges: g.grid([0], 1, 1), polys: g.grid([0], 1, 1) })).toThrow(/matrix/);
    const { polys } = polygonRingSources(g, rings);
    const twoPlanes = g.source({
      shape: { kind: "matrix", rows: rings.edges.length / 4, cols: 4 },
      dtype: "f32",
      data: new Float32Array(rings.edges.length * 2),
      axes: [{ name: "c", type: "channel", length: 2 }],
    });
    expect(() => g.op("rasterisePolygons", { edges: twoPlanes, polys }, { width: 4, height: 4 })).toThrow(/single-channel/);
  });
});
