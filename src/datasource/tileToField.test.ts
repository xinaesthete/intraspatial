// B1a — `tileToField` converter (docs/stream-b-bridge-plan.md). Pure, no sd.js / no GPU device.
import { describe, expect, it } from "vitest";
import { Graph } from "../gpu/graph/graph";
import type { FieldRole, ResolvedPlacement, TensorAxis } from "../gpu/graph/handle";
import { sourceFromTile, tileToField } from "./tileToField";
import type { ChunkId, Tile } from "./types";

const scalarTile = (id: ChunkId, w: number, h: number, fill: (x: number, y: number) => number): Tile => {
  const data = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[y * w + x] = fill(x, y);
  return { id, dims: [w, h, 1], element: { kind: "scalar" }, dtype: "f32", data };
};

const id0: ChunkId = { level: 0, x: 0, y: 0, z: 0 };

describe("tileToField — Tile → FieldValue(grid)", () => {
  it("maps a scalar plane tile to a grid, carrying data/dtype/element by reference", () => {
    const tile = scalarTile(id0, 4, 3, (x, y) => x + y * 10);
    const fv = tileToField(tile);
    expect(fv.shape).toEqual({ kind: "grid", width: 4, height: 3 });
    expect(fv.dtype).toBe("f32");
    expect(fv.element).toEqual({ kind: "scalar" });
    expect(fv.data).toBe(tile.data); // carried, not copied
    // absent opts ⇒ today's behaviour: no facets stamped
    expect(fv.placement).toBeUndefined();
    expect(fv.axes).toBeUndefined();
    expect(fv.role).toBeUndefined();
  });

  it("stamps the resolved placement / axes / role it is handed (additive)", () => {
    const placement: ResolvedPlacement = {
      system: "global",
      worldFromArray: {
        origin: [1, 2, 0],
        axes: [
          [2, 0, 0],
          [0, 2, 0],
          [0, 0, 1],
        ],
      },
    };
    const axes: TensorAxis[] = [{ name: "c", type: "channel", length: 1, entries: [{ label: "DAPI" }] }];
    const role: FieldRole = { kind: "intensity" };
    const fv = tileToField(
      scalarTile(id0, 2, 2, () => 1),
      { placement, axes, role },
    );
    expect(fv.placement).toBe(placement);
    expect(fv.axes).toBe(axes);
    expect(fv.role).toBe(role);
  });

  it("extracts one interleaved lane from a vec tile as a scalar grid", () => {
    // A 2x2 interleaved vec3 tile: lane c at cell i holds (10*i + c).
    const w = 2,
      h = 2,
      lanes = 3;
    const data = new Float32Array(w * h * lanes);
    for (let i = 0; i < w * h; i++) for (let c = 0; c < lanes; c++) data[i * lanes + c] = 10 * i + c;
    const tile: Tile = { id: id0, dims: [w, h, 1], element: { kind: "vec", n: 3 }, dtype: "f32", data };
    const fv = tileToField(tile, { channel: 1 });
    expect(fv.shape).toEqual({ kind: "grid", width: 2, height: 2 });
    expect(fv.element).toEqual({ kind: "scalar" });
    expect(Array.from(fv.data as Float32Array)).toEqual([1, 11, 21, 31]);
  });

  it("rejects a volume tile, an out-of-range channel, and a length mismatch", () => {
    expect(() => tileToField({ id: id0, dims: [2, 2, 2], element: { kind: "scalar" }, dtype: "f32", data: new Float32Array(8) })).toThrow(
      /plane/,
    );
    const vec: Tile = { id: id0, dims: [1, 1, 1], element: { kind: "vec", n: 2 }, dtype: "f32", data: new Float32Array([0, 1]) };
    expect(() => tileToField(vec, { channel: 2 })).toThrow(/channel/);
    expect(() => tileToField({ id: id0, dims: [4, 4, 1], element: { kind: "scalar" }, dtype: "f32", data: new Float32Array(3) })).toThrow(
      /length/,
    );
  });

  it("sourceFromTile adds a source node whose handle carries the stamped facets", () => {
    const g = new Graph();
    const placement: ResolvedPlacement = {
      system: "global",
      worldFromArray: {
        origin: [0, 0, 0],
        axes: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
      },
    };
    const handle = sourceFromTile(
      g,
      scalarTile(id0, 8, 8, (x, y) => (x * y) % 5),
      { placement },
    );
    expect(handle.shape).toEqual({ kind: "grid", width: 8, height: 8 });
    expect(handle.placement).toBe(placement);
  });
});
