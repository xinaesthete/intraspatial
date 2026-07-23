import { describe, expect, it } from "vitest";
import { getisOrdGpu } from "../gpu/spatial/getisOrd";

describe("probe", () => {
  it("getisOrd LCG pattern 48x40", async () => {
    const w = 48, h = 40;
    const grid = Float32Array.from({ length: w * h }, (_, i) => ((i * 1103515245 + 12345) % 100) / 100);
    const { z } = await getisOrdGpu(grid, w, h, { radius: 2 });
    expect(z.length).toBe(w * h);
  });
});
