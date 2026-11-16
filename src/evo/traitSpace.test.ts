import { describe, expect, it, vi } from "vite-plus/test";
import type { ParamSpec } from "../gpu/graph/op";
import { neutralSpecimen } from "./specimen";
import { paramsToSpecimen, specimenToParams, traitSpaceFromParams, withLocked } from "./traitSpace";

// A representative schema exercising every ParamType + the unbounded fallback.
const SPECS: ParamSpec[] = [
  { name: "attract", type: "number", default: 0.5, min: 0, max: 2 },
  { name: "count", type: "int", default: 8, min: 1, max: 32 },
  { name: "mode", type: "enum", default: "swirl", options: ["swirl", "orbit", "swim"] },
  { name: "collide", type: "bool", default: true },
  { name: "dt", type: "number", default: 1 }, // unbounded ⇒ fixed
];

describe("traitSpaceFromParams", () => {
  it("classifies each ParamSpec into the right trait kind", () => {
    const space = traitSpaceFromParams(SPECS);
    const byName = Object.fromEntries(space.traits.map((t) => [t.paramName, t]));
    expect(byName.attract?.kind).toBe("number");
    expect(byName.count?.kind).toBe("number");
    expect(byName.count?.isInt).toBe(true);
    expect(byName.mode?.kind).toBe("number");
    expect(byName.mode?.enumOptions).toEqual(["swirl", "orbit", "swim"]);
    expect(byName.collide?.kind).toBe("enable");
    expect(byName.dt?.kind).toBe("fixed");
    // pos array holds attract, count, mode (3 number traits); enable holds collide (1).
    expect(space.numCount).toBe(3);
    expect(space.enableCount).toBe(1);
  });

  it("round-trips params → specimen → params (ints/enums re-quantised, fixed passed through)", () => {
    const space = traitSpaceFromParams(SPECS);
    const params = { attract: 1.25, count: 20, mode: "orbit", collide: false, dt: 1 };
    const sp = paramsToSpecimen(space, params, 1);
    const back = specimenToParams(space, sp);
    expect(back.attract).toBeCloseTo(1.25, 10);
    expect(back.count).toBe(20);
    expect(back.mode).toBe("orbit");
    expect(back.collide).toBe(false);
    expect(back.dt).toBe(1); // fixed passthrough
  });

  it("decodes a neutral specimen to the centre of every range, all influences on", () => {
    const space = traitSpaceFromParams(SPECS);
    const p = specimenToParams(space, neutralSpecimen(space, 1));
    expect(p.attract).toBeCloseTo(1, 10); // centre of [0,2]
    expect(p.collide).toBe(true);
  });

  it("clamps out-of-range params into the trait's box", () => {
    const space = traitSpaceFromParams(SPECS);
    const sp = paramsToSpecimen(space, { attract: 99, count: -5, mode: "swirl", collide: true, dt: 1 }, 1);
    const back = specimenToParams(space, sp);
    expect(back.attract).toBe(2); // clamped to max
    expect(back.count).toBe(1); // clamped to min
  });

  it("withLocked marks the named traits (and all fixed traits) locked", () => {
    const space = withLocked(traitSpaceFromParams(SPECS), ["attract"]);
    const byName = Object.fromEntries(space.traits.map((t) => [t.paramName, t]));
    expect(byName.attract?.locked).toBe(true);
    expect(byName.count?.locked).toBe(false);
    expect(byName.dt?.locked).toBe(true); // fixed traits are always locked
  });
});

describe("warped traits", () => {
  const FREQ: ParamSpec = { name: "freq", type: "number", default: 500, min: 50, max: 3000, warp: "exp" };
  const GEO_MEAN = Math.sqrt(50 * 3000); // 387.298…

  it('leaves a param with no warp (or warp "lin") linear, as before', () => {
    const space = traitSpaceFromParams([
      { name: "a", type: "number", default: 1, min: 0, max: 2 },
      { name: "b", type: "number", default: 1, min: 0, max: 2, warp: "lin" },
    ]);
    expect(space.traits.map((t) => t.warp)).toEqual([undefined, undefined]);
    const p = specimenToParams(space, { ...neutralSpecimen(space, 1), pos: Float64Array.from([0.25, 0.75]) });
    expect([p.a, p.b]).toEqual([0.5, 1.5]);
  });

  it("reads an exp param at coord 0.5 as the geometric mean of min and max", () => {
    const space = traitSpaceFromParams([FREQ]);
    expect(space.traits[0]?.warp).toBe("exp");
    expect(specimenToParams(space, neutralSpecimen(space, 1)).freq).toBeCloseTo(GEO_MEAN, 9);
    // and evenly in ratio: each quarter of the coord range is the same factor (60^¼)
    const at = (u: number) => specimenToParams(space, { ...neutralSpecimen(space, 1), pos: Float64Array.from([u]) }).freq as number;
    expect(at(0.25) / at(0)).toBeCloseTo(at(1) / at(0.75), 9);
  });

  it("round-trips params → specimen → params through each warp", () => {
    const specs: ParamSpec[] = [
      FREQ,
      { name: "amp", type: "number", default: 0.1, min: 0, max: 1, warp: "amp" },
      { name: "gain", type: "number", default: -6, min: -60, max: 12, warp: "db" },
      { name: "curve", type: "number", default: 2, min: 0, max: 10, warp: 4 },
      { name: "neg", type: "number", default: -1, min: -10, max: -0.01, warp: "exp" },
      { name: "voices", type: "int", default: 4, min: 1, max: 64, warp: "exp" },
    ];
    const space = traitSpaceFromParams(specs);
    for (const params of [
      { freq: 440, amp: 0.3, gain: -12.5, curve: 7.25, neg: -0.5, voices: 17 },
      { freq: 50, amp: 0, gain: -60, curve: 0, neg: -10, voices: 1 },
      { freq: 3000, amp: 1, gain: 12, curve: 10, neg: -0.01, voices: 64 },
    ]) {
      const back = specimenToParams(space, paramsToSpecimen(space, params, 1));
      for (const [k, v] of Object.entries(params)) expect(back[k]).toBeCloseTo(v, 9);
    }
  });

  it("clamps out-of-range params into a warped trait's box", () => {
    const space = traitSpaceFromParams([FREQ]);
    expect(specimenToParams(space, paramsToSpecimen(space, { freq: 1e6 }, 1)).freq).toBeCloseTo(3000, 9);
    expect(specimenToParams(space, paramsToSpecimen(space, { freq: 1 }, 1)).freq).toBeCloseTo(50, 9);
  });

  it("falls back to linear, with a warning, for an exp warp over a range touching zero", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const space = traitSpaceFromParams([{ name: "x", type: "number", default: 0.5, min: 0, max: 1, warp: "exp" }]);
      expect(warn).toHaveBeenCalledOnce();
      expect(space.traits[0]?.kind).toBe("number");
      expect(space.traits[0]?.warp).toBeUndefined();
      expect(specimenToParams(space, neutralSpecimen(space, 1)).x).toBe(0.5);
    } finally {
      warn.mockRestore();
    }
  });
});

describe("a curve too steep to evaluate", () => {
  it("falls back to linear, with a warning", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const space = traitSpaceFromParams([{ name: "x", type: "number", default: 5, min: 0, max: 10, warp: 1000 }]);
      expect(warn).toHaveBeenCalledOnce();
      expect(space.traits[0]?.warp).toBeUndefined();
      const p = specimenToParams(space, { ...neutralSpecimen(space, 1), pos: Float64Array.from([0.25]) });
      expect(p.x).toBe(2.5);
    } finally {
      warn.mockRestore();
    }
  });
});

describe("delta → deltaNorm", () => {
  it("is absent without a delta", () => {
    expect(traitSpaceFromParams(SPECS).traits.map((t) => t.deltaNorm)).toEqual([undefined, undefined, undefined, undefined, undefined]);
  });

  it("is delta / (max - min) for a linear param", () => {
    const space = traitSpaceFromParams([
      { name: "a", type: "number", default: 0.5, min: 0, max: 2, delta: 0.1 },
      { name: "n", type: "int", default: 8, min: 1, max: 33, delta: 4 },
    ]);
    expect(space.traits.map((t) => t.deltaNorm)).toEqual([0.05, 0.125]);
  });

  it("ignores a delta on an enum, with a warning (its options have no distance between them)", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const space = traitSpaceFromParams([{ name: "e", type: "enum", default: "a", options: ["a", "b", "c"], delta: 1 }]);
      expect(warn).toHaveBeenCalledOnce();
      expect(space.traits[0]?.deltaNorm).toBeUndefined();
    } finally {
      warn.mockRestore();
    }
  });

  it("is the coord distance from the default to default + delta for a warped param", () => {
    const space = traitSpaceFromParams([{ name: "freq", type: "number", default: 500, min: 50, max: 3000, warp: "exp", delta: 50 }]);
    expect(space.traits[0]?.deltaNorm).toBeCloseTo(Math.log(1.1) / Math.log(60), 12);
  });

  it("measures downwards when default + delta would leave the range", () => {
    const space = traitSpaceFromParams([{ name: "freq", type: "number", default: 2900, min: 50, max: 3000, warp: "exp", delta: 290 }]);
    expect(space.traits[0]?.deltaNorm).toBeCloseTo(Math.log(2900 / 2610) / Math.log(60), 12);
  });

  it("ignores a negative or non-finite delta, with a warning", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const space = traitSpaceFromParams([
        { name: "a", type: "number", default: 0.5, min: 0, max: 1, delta: -1 },
        { name: "b", type: "number", default: 0.5, min: 0, max: 1, delta: Number.NaN },
      ]);
      expect(warn).toHaveBeenCalledTimes(2);
      expect(space.traits.map((t) => t.deltaNorm)).toEqual([undefined, undefined]);
    } finally {
      warn.mockRestore();
    }
  });
});
