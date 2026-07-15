// The parameter seam (ADR-0011, decision 7): schema well-formedness, the Params→typed-config
// mapping, the filter-then-apply helpers, and — the point of the seam — that HsPf's ParamSpecs
// drop straight into the src/evo Mutator (round-trip + freeze-via-lock-complement).
import { describe, expect, it } from "vitest";
import { paramsToSpecimen, specimenToParams, traitSpaceFromParams, withLocked } from "../../../evo/traitSpace";
import { DEFAULT_FITNESS } from "./math";
import { complementNames, defaultHspfParams, filterSpecs, HSPF_PARAM_SPECS, toHspfConfig } from "./params";

const close = (a: number, b: number, eps = 1e-6) => Math.abs(a - b) < eps;

describe("HsPf param schema", () => {
  it("declares 17 dotted-path params with finite defaults", () => {
    expect(HSPF_PARAM_SPECS).toHaveLength(17); // 8 fitness + 4 spread + 1 dynamics + 4 init
    for (const s of HSPF_PARAM_SPECS) {
      expect(s.name).toContain(".");
      expect(Number.isFinite(s.default as number)).toBe(true);
    }
    expect(new Set(HSPF_PARAM_SPECS.map((s) => s.name)).size).toBe(17); // unique names
  });
});

describe("toHspfConfig", () => {
  it("maps the default genome to the original's typed configuration", () => {
    const cfg = toHspfConfig(defaultHspfParams());
    expect(cfg.hspf.twoBiteRate).toBe(0);
    for (let i = 0; i < 4; i++) {
      expect(close(cfg.hspf.fitness?.A[i] ?? 0, DEFAULT_FITNESS.A[i] ?? 0)).toBe(true);
      expect(close(cfg.hspf.fitness?.S[i] ?? 0, DEFAULT_FITNESS.S[i] ?? 0)).toBe(true);
    }
    expect(cfg.init).toEqual([0.9, 0, 0, 0.1]);
    expect(cfg.neighbourhood).toEqual({ mapWidthInKm: 10000, maxDistanceInKm: 2000, concentration: 6, count: 1000 });
  });
});

describe("filter-then-apply helpers", () => {
  it("filters by tag, path-prefix, and name-substring in declaration order", () => {
    expect(filterSpecs(HSPF_PARAM_SPECS, { tag: "fitness" })).toHaveLength(8);
    expect(filterSpecs(HSPF_PARAM_SPECS, { pathPrefix: "fitness.A" }).map((s) => s.name)).toEqual([
      "fitness.A.mm",
      "fitness.A.mp",
      "fitness.A.pm",
      "fitness.A.pp",
    ]);
    expect(filterSpecs(HSPF_PARAM_SPECS, { pathPrefix: "spread" })).toHaveLength(4);
    expect(filterSpecs(HSPF_PARAM_SPECS, { nameIncludes: "init." })).toHaveLength(4);
    expect(filterSpecs(HSPF_PARAM_SPECS)).toHaveLength(17); // empty filter = everything
  });
  it("complementNames returns exactly the unselected names", () => {
    const fitness = filterSpecs(HSPF_PARAM_SPECS, { tag: "fitness" });
    const rest = complementNames(HSPF_PARAM_SPECS, fitness);
    expect(rest).toHaveLength(9);
    for (const n of rest) expect(n.startsWith("fitness.")).toBe(false);
  });
});

describe("evo/Mutator integration", () => {
  it("builds a trait space with one trait per param", () => {
    const space = traitSpaceFromParams(HSPF_PARAM_SPECS);
    expect(space.traits).toHaveLength(17);
    expect(space.byParam["spread.concentration"]).toBeDefined();
  });
  it("round-trips the default genome through a specimen", () => {
    const space = traitSpaceFromParams(HSPF_PARAM_SPECS);
    const defaults = defaultHspfParams();
    const back = specimenToParams(space, paramsToSpecimen(space, defaults, 7));
    for (const name of ["spread.concentration", "dynamics.twoBiteRate", "init.mm", "fitness.S.mm"]) {
      expect(close(back[name] as number, defaults[name] as number, 1e-3)).toBe(true);
    }
  });
  it("freeze = filter → lock complement: only the filtered set stays mutable", () => {
    const base = traitSpaceFromParams(HSPF_PARAM_SPECS);
    const fitness = filterSpecs(HSPF_PARAM_SPECS, { tag: "fitness" });
    const locked = withLocked(base, complementNames(HSPF_PARAM_SPECS, fitness));
    for (const t of locked.traits) {
      expect(t.locked).toBe(!t.paramName.startsWith("fitness."));
    }
  });
});
