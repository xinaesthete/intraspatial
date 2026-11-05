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

  it("is empty when no pixel around the seed is in range", () => {
    const { d, width, height } = field(["#.#"]);
    expect(Array.from(connectedRegion(d, width, height, { col: 1, row: 0 }, 1))).toEqual([0, 0, 0]);
  });

  it("starts from the nearest pixel, either side of the half-pixel line", () => {
    const { d, width, height } = field(["#..", "...", "..#"]);
    // Rounding picks the patch: flooring both coordinates would pick the top-left one every time.
    expect(picture(connectedRegion(d, width, height, { col: 1.6, row: 1.6 }, 1), width)).toEqual(["...", "...", "..#"]);
    expect(picture(connectedRegion(d, width, height, { col: 0.4, row: 0.4 }, 1), width)).toEqual(["#..", "...", "..."]);
  });

  it("starts from an in-range corner of the seed's cell when the nearest pixel is out of range", () => {
    const { d, width, height } = field(["#.", ".."]);
    expect(picture(connectedRegion(d, width, height, { col: 0.6, row: 0.6 }, 1), width)).toEqual(["#.", ".."]);
  });

  it("grows with the tolerance, the tolerance itself included", () => {
    const d = Float32Array.from([0, 1, 2, 3, 4]);
    expect(Array.from(connectedRegion(d, 5, 1, { col: 0, row: 0 }, 2.5))).toEqual([1, 1, 1, 0, 0]);
    // A neighbour, and a seed, exactly at the tolerance are in range.
    expect(Array.from(connectedRegion(d, 5, 1, { col: 0, row: 0 }, 2))).toEqual([1, 1, 1, 0, 0]);
    expect(Array.from(connectedRegion(d, 5, 1, { col: 2, row: 0 }, 2))).toEqual([1, 1, 1, 0, 0]);
  });
});
