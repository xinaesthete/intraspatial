// Build-time behaviour of the `decimate` graph op: output shape and DERIVED placement (ADR-0018).
// CPU-only on purpose — shape/placement inference runs at graph build, no device needed.
import { describe, expect, it } from "vitest";
import type { Affine3 } from "../../../coords";
import { applyAffine } from "../../../datasource/math";
import type { ResolvedPlacement } from "../handle";
import { Graph } from "../index";
import { decimatedPlacement } from "./decimate";

// A rotated, anisotropic, translated placement — not axis-aligned, so a wrong convention
// (centre vs corner, scaling the origin, scaling z) would show up.
const wfa: Affine3 = {
  origin: [100, -40, 7],
  axes: [
    [0.6, 0.8, 0],
    [-1.6, 1.2, 0],
    [0, 0, 3],
  ],
};
const placed: ResolvedPlacement = { system: "tissue", worldFromArray: wfa };

function gridSource(g: Graph, w: number, h: number, placement?: ResolvedPlacement) {
  return g.source({ shape: { kind: "grid", width: w, height: h }, dtype: "f32", data: new Float32Array(w * h), placement });
}

describe("decimate graph op (build time)", () => {
  it("output shape is ceil(w/f) x ceil(h/f)", () => {
    const g = new Graph();
    const out = g.op1("decimate", { grid: gridSource(g, 37, 23) }, { factor: 4, mode: "mean" });
    expect(out.shape).toEqual({ kind: "grid", width: 10, height: 6 });
    expect(out.placement).toBeUndefined(); // array space in ⇒ array space out
  });

  it("rejects factor outside 2..64 at build time", () => {
    const g = new Graph();
    expect(() => g.op1("decimate", { grid: gridSource(g, 8, 8) }, { factor: 1, mode: "mean" })).toThrow(/factor/);
    expect(() => g.op1("decimate", { grid: gridSource(g, 8, 8) }, { factor: 65, mode: "mean" })).toThrow(/factor/);
  });

  it("derived placement maps output cell (I,J) to the world centroid of its input block", () => {
    const f = 3;
    const g = new Graph();
    const out = g.op1("decimate", { grid: gridSource(g, 12, 9, placed) }, { factor: f, mode: "mean" });
    const pl = out.placement;
    expect(pl).toBeDefined();
    expect(pl!.system).toBe("tissue");
    expect(pl).toEqual(decimatedPlacement(placed, f));
    // Origin and z axis untouched; in-plane axes scaled by f.
    expect(pl!.worldFromArray.origin).toEqual(wfa.origin);
    expect(pl!.worldFromArray.axes[2]).toEqual(wfa.axes[2]);
    for (const [I, J] of [
      [0, 0],
      [3, 2],
      [1, 1],
    ]) {
      // Corner-indexed cells: centre of cell (i,j) is array point (i+½, j+½).
      const outCentre = applyAffine(pl!.worldFromArray, [I! + 0.5, J! + 0.5, 0]);
      const centroid: [number, number, number] = [0, 0, 0];
      for (let dy = 0; dy < f; dy++)
        for (let dx = 0; dx < f; dx++) {
          const c = applyAffine(wfa, [I! * f + dx + 0.5, J! * f + dy + 0.5, 0]);
          centroid[0] += c[0] / (f * f);
          centroid[1] += c[1] / (f * f);
          centroid[2] += c[2] / (f * f);
        }
      for (let k = 0; k < 3; k++) expect(outCentre[k]).toBeCloseTo(centroid[k]!, 9);
    }
    // And the output grid's outer corner coincides with the input's.
    expect(applyAffine(pl!.worldFromArray, [0, 0, 0])).toEqual(applyAffine(wfa, [0, 0, 0]));
  });
});
