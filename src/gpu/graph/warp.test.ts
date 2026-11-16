import { describe, expect, it } from "vite-plus/test";
import { type WarpSpec, warpIsLinear, warpMap, warpUnmap, warpValid } from "./warp";

/** ControlSpec(...).map at 0, 0.25, 0.5, 0.8, 1, as sclang 3.13 computes it (the values
 *  inorganic's live-av-core warp tests check against). */
const SCLANG: [string, WarpSpec, number[]][] = [
  ["lin", { min: -1, max: 1 }, [-1, -0.5, 0, 0.6, 1]],
  ["exp", { min: 50, max: 3000, warp: "exp" }, [50, 139.15788418569, 387.29833462074, 1322.7903093326, 3000]],
  ["amp", { min: 0, max: 1, warp: "amp" }, [0, 0.0625, 0.25, 0.64, 1]],
  ["db", { min: -60, max: 12, warp: "db" }, [-60, -12.049734200985, -0.034656901014391, 8.1248266537023, 12]],
  ["curve 4", { min: 0, max: 10, warp: 4 }, [0, 0.32058603280085, 1.1920292202212, 4.3905489615886, 10]],
  ["curve -2", { min: 1, max: 100, warp: -2 }, [1, 46.050369158418, 73.37479928437, 92.379055010233, 100]],
  ["step", { min: 1, max: 64, step: 1 }, [1, 17, 33, 51, 64]],
  ["exp, negative", { min: -0.01, max: -10, warp: "exp" }, [-0.01, -0.056234132519035, -0.31622776601684, -2.5118864315096, -10]],
  ["amp, downwards", { min: 1, max: 0, warp: "amp" }, [1, 0.5625, 0.25, 0.04, 0]],
];
const POSITIONS = [0, 0.25, 0.5, 0.8, 1];

describe("warps, as SuperCollider's", () => {
  it.each(SCLANG)("%s maps as ControlSpec.map does", (_name, spec, values) => {
    expect(POSITIONS.map((u) => warpMap(u, spec))).toEqual(values.map((v) => expect.closeTo(v, 9)));
  });

  it.each(SCLANG.filter(([n]) => n !== "step"))("%s unmaps back", (_name, spec) => {
    for (const u of POSITIONS) expect(warpUnmap(warpMap(u, spec), spec)).toBeCloseTo(u, 9);
  });

  it("rounds to the step both ways, as sclang (unmap(17) of 1..64 is 16/63)", () => {
    expect(warpUnmap(17.2, { min: 1, max: 64, step: 1 })).toBeCloseTo(16 / 63, 12);
  });

  it("clips positions and values to the range", () => {
    const spec: WarpSpec = { min: 50, max: 3000, warp: "exp" };
    expect([warpMap(-1, spec), warpMap(2, spec)]).toEqual([50, 3000]);
    expect([warpUnmap(10, spec), warpUnmap(1e6, spec)]).toEqual([0, 1]);
  });

  it("an exp warp over Hz is a linear range of MIDI pitch", () => {
    const ftom = (hz: number) => 69 + 12 * Math.log2(hz / 440);
    const hz: WarpSpec = { min: 55, max: 1760, warp: "exp" };
    const midi: WarpSpec = { min: ftom(55), max: ftom(1760) };
    for (const u of POSITIONS) expect(ftom(warpMap(u, hz))).toBeCloseTo(warpMap(u, midi), 9);
  });

  it("validates ranges an exp warp cannot traverse", () => {
    expect(warpValid("exp", 50, 3000)).toBe(true);
    expect(warpValid("exp", -0.01, -10)).toBe(true);
    expect(warpValid("exp", 0, 1)).toBe(false);
    expect(warpValid("exp", -1, 1)).toBe(false);
    expect(warpValid("amp", 0, 1)).toBe(true);
    expect(warpValid(3, -1, 1)).toBe(true);
    expect(warpValid(1000, 0, 1)).toBe(false); // exp(1000) overflows: the map would be NaN
    expect(warpValid(Number.NaN, 0, 1)).toBe(false);
    expect(warpValid("cos" as never, 0, 1)).toBe(false);
    expect([warpIsLinear(undefined), warpIsLinear("lin"), warpIsLinear(0), warpIsLinear("exp"), warpIsLinear(2)]).toEqual([
      true,
      true,
      true,
      false,
      false,
    ]);
  });
});
