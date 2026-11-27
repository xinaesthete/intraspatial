// Build-time checks of the `histogram` ops: no device needed.
import { describe, expect, it } from "vite-plus/test";
import { Graph } from "../index";

describe("histogram op build", () => {
  it("refuses a channel-bearing grid, which the GPU and golden would bin differently", () => {
    const g = new Graph();
    const field = g.source({
      shape: { kind: "grid", width: 2, height: 2 },
      dtype: "f32",
      data: new Float32Array(8),
      axes: [{ name: "c", type: "channel", length: 2 }],
    });
    expect(() => g.op("histogram", { field })).toThrow(/single-channel/);
  });

  it("accepts a singleton channel axis on both ops", () => {
    const g = new Graph();
    const one = (data: Float32Array) =>
      g.source({ shape: { kind: "grid", width: 2, height: 2 }, dtype: "f32", data, axes: [{ name: "c", type: "channel", length: 1 }] });
    const field = one(new Float32Array([1, 2, 3, 4]));
    const mask = one(new Float32Array([1, 0, 1, 0]));
    expect(g.op("histogram", { field })).toHaveLength(5);
    expect(g.op("histogramMasked", { field, mask })).toHaveLength(5);
    const wide = g.source({
      shape: { kind: "grid", width: 2, height: 2 },
      dtype: "f32",
      data: new Float32Array(8),
      axes: [{ name: "c", type: "channel", length: 2 }],
    });
    expect(() => g.op("histogramMasked", { field, mask: wide })).toThrow(/single-channel/);
  });

  it("refuses a fixed range that collapses in f32", () => {
    const g = new Graph();
    const field = g.grid([1, 2, 3, 4], 2, 2);
    expect(() => g.op("histogram", { field }, { range: "fixed", lo: 1, hi: 1 + 2 ** -25 })).toThrow(/hi > lo in f32/);
    expect(g.op("histogram", { field }, { range: "fixed", lo: 0, hi: 1 })).toHaveLength(5);
  });
});
