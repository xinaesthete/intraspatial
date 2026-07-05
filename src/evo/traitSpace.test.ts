import { describe, expect, it } from "vitest";
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
    expect(byName.attract!.kind).toBe("number");
    expect(byName.count!.kind).toBe("number");
    expect(byName.count!.isInt).toBe(true);
    expect(byName.mode!.kind).toBe("number");
    expect(byName.mode!.enumOptions).toEqual(["swirl", "orbit", "swim"]);
    expect(byName.collide!.kind).toBe("enable");
    expect(byName.dt!.kind).toBe("fixed");
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
    expect(byName.attract!.locked).toBe(true);
    expect(byName.count!.locked).toBe(false);
    expect(byName.dt!.locked).toBe(true); // fixed traits are always locked
  });
});
