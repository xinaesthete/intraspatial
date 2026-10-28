import { describe, expect, it } from "vite-plus/test";
import { connectedRegion } from "./wandRegion";

/** A field from a picture: "." is far (9), "#" is near (0). */
const field = (rows: string[]) => ({
  d: Float32Array.from(rows.join("").split(""), (c) => (c === "#" ? 0 : 9)),
  width: rows[0]!.length,
  height: rows.length,
});
const picture = (region: Uint8Array, width: number) =>
  Array.from({ length: region.length / width }, (_, r) =>
    Array.from(region.subarray(r * width, (r + 1) * width), (v) => (v ? "#" : ".")).join(""),
  );

describe("connectedRegion", () => {
  it("keeps only the patch the seed is in, joined through shared edges", () => {
    const { d, width, height } = field([
      "##..#", //
      ".#..#",
      ".##..",
      "...#.", // touches the seed's patch only at a corner
    ]);
    expect(picture(connectedRegion(d, width, height, { col: 0.2, row: 0.4 }, 1), width)).toEqual(["##...", ".#...", ".##..", "....."]);
  });

  it("is empty when the seed pixel itself is out of range", () => {
    const { d, width, height } = field(["#.#"]);
    expect(Array.from(connectedRegion(d, width, height, { col: 1, row: 0 }, 1))).toEqual([0, 0, 0]);
  });

  it("grows with the tolerance", () => {
    const d = Float32Array.from([0, 1, 2, 3, 4]);
    expect(Array.from(connectedRegion(d, 5, 1, { col: 0, row: 0 }, 2.5))).toEqual([1, 1, 1, 0, 0]);
  });
});
