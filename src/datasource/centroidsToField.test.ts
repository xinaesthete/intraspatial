// B2a — `centroidsToField` converter (docs/stream-b-bridge-plan.md). Pure, no zarr / sd.js / GPU.
import { describe, expect, it } from "vitest";
import { Graph } from "../gpu/graph/graph";
import type { FieldProvenance, ResolvedPlacement } from "../gpu/graph/handle";
import { unpackPoints } from "../gpu/graph/handle";
import { centroidsToField } from "./centroidsToField";

describe("centroidsToField — xs/ys → FieldValue(points)", () => {
  it("packs parallel arrays into interleaved [x0,y0,x1,y1,...]", () => {
    const xs = [10, 20, 30];
    const ys = [11, 21, 31];
    const fv = centroidsToField(xs, ys);
    expect(fv.shape).toEqual({ kind: "points", n: 3 });
    expect(fv.dtype).toBe("f32");
    expect(Array.from(fv.data as Float32Array)).toEqual([10, 11, 20, 21, 30, 31]);
    // round-trips through the shared unpacker the ops use
    const back = unpackPoints(fv);
    expect(back.n).toBe(3);
    expect(back.xs).toEqual(xs);
    expect(back.ys).toEqual(ys);
    // absent opts ⇒ today's behaviour: no facets stamped
    expect(fv.placement).toBeUndefined();
    expect(fv.provenance).toBeUndefined();
  });

  it("stamps the resolved placement + provenance it is handed (additive)", () => {
    const placement: ResolvedPlacement = {
      system: "Leap034",
      worldFromArray: {
        origin: [0, 0, 0],
        axes: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
      },
    };
    const provenance: FieldProvenance = { region: "Leap034_imc_cell_shapes", instanceKey: "cell_id", cellTypeId: 6 };
    const fv = centroidsToField([1], [2], { placement, provenance });
    expect(fv.placement).toBe(placement);
    expect(fv.provenance).toBe(provenance);
    expect(fv.provenance?.cellTypeId).toBe(6);
  });

  it("accepts an empty cloud (n === 0) — a cell type with no cells is valid", () => {
    const fv = centroidsToField([], []);
    expect(fv.shape).toEqual({ kind: "points", n: 0 });
    expect((fv.data as Float32Array).length).toBe(0);
  });

  it("throws on an xs/ys length mismatch", () => {
    expect(() => centroidsToField([1, 2], [1])).toThrow(/length mismatch/);
  });

  it("feeds a graph points source that carries the placement + provenance facets onto the handle", () => {
    const placement: ResolvedPlacement = {
      system: "Leap034",
      worldFromArray: {
        origin: [0, 0, 0],
        axes: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
      },
    };
    const provenance: FieldProvenance = { region: "R", instanceKey: "cell_id", cellTypeId: 3 };
    const g = new Graph();
    const src = g.source(centroidsToField([5, 6], [7, 8], { placement, provenance }), "cellType");
    expect(src.shape).toEqual({ kind: "points", n: 2 });
    expect(src.placement).toBe(placement);
    expect(src.provenance).toBe(provenance);
  });
});
