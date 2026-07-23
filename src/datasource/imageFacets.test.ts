// B1b (pure half) — `imageFacets` mapping (docs/stream-b-bridge-plan.md). Pure, no sd.js / no GPU.
import { describe, expect, it } from "vitest";
import type { Affine3 } from "../coords";
import { channelAxisFrom, placementFromMatrix, type ResolvedChannel, rgbToHex } from "./imageFacets";

describe("placementFromMatrix — resolved Affine3 → ResolvedPlacement", () => {
  it("wraps a matrix as a global placement", () => {
    const m: Affine3 = { origin: [3, 4, 0], axes: [[2, 0, 0], [0, 2, 0], [0, 0, 1]] };
    const p = placementFromMatrix(m);
    expect(p).toEqual({ system: "global", worldFromArray: m });
    expect(p?.worldFromArray).toBe(m); // resolved value carried, not re-composed
  });

  it("does NOT fabricate identity when the store carries no transform (absent ⇒ absent)", () => {
    expect(placementFromMatrix(undefined)).toBeUndefined();
  });

  it("honours a custom system name", () => {
    const m: Affine3 = { origin: [0, 0, 0], axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] };
    expect(placementFromMatrix(m, "microns")?.system).toBe("microns");
  });
});

describe("rgbToHex", () => {
  it("maps rgb floats to uppercase RRGGBB with clamping", () => {
    expect(rgbToHex([0, 0, 0])).toBe("000000");
    expect(rgbToHex([1, 1, 1])).toBe("FFFFFF");
    expect(rgbToHex([0, 0, 1])).toBe("0000FF");
    expect(rgbToHex([2, -1, 0.5])).toBe("FF0080"); // clamped to [0,1]
  });
});

describe("channelAxisFrom — resolved channels → channel TensorAxis (ADR-0015 fork B)", () => {
  it("builds a channel axis with per-index entries (label/color/window/active)", () => {
    const channels: ResolvedChannel[] = [
      { label: "DAPI", color: [0, 0, 1], contrastLimits: [0.1, 0.8], visible: true },
      { label: "CD20", color: [1, 0, 0], contrastLimits: [0, 0.5], visible: false },
    ];
    const axis = channelAxisFrom(channels);
    expect(axis).toBeDefined();
    expect(axis?.name).toBe("c");
    expect(axis?.type).toBe("channel");
    expect(axis?.length).toBe(2);
    expect(axis?.entries?.[0]).toEqual({ label: "DAPI", color: "0000FF", window: { min: 0, max: 1, start: 0.1, end: 0.8 }, active: true });
    expect(axis?.entries?.[1]).toEqual({ label: "CD20", color: "FF0000", window: { min: 0, max: 1, start: 0, end: 0.5 }, active: false });
  });

  it("returns undefined for no channels (⇒ no axis)", () => {
    expect(channelAxisFrom([])).toBeUndefined();
  });

  it("defaults active to true when visibility is unspecified", () => {
    const axis = channelAxisFrom([{ label: "x", color: [1, 1, 1], contrastLimits: [0, 1] }]);
    expect(axis?.entries?.[0]?.active).toBe(true);
  });
});
