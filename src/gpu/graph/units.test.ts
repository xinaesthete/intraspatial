import { describe, expect, it } from "vite-plus/test";
import type { ResolvedPlacement } from "./handle";
import { convertUnit, fixedUnit, type ResolvedUnit, resolveUnit, spatialUnits, type UnitResolver } from "./units";

const placed = (cell: number, cellY = cell): ResolvedPlacement => ({
  system: "global",
  worldFromArray: {
    origin: [10, 20, 0],
    axes: [
      [cell, 0, 0],
      [0, cellY, 0],
      [0, 0, 1],
    ],
  },
});

describe("resolveUnit", () => {
  it("is undefined for undeclared units", () => {
    expect(resolveUnit(undefined, {}, [spatialUnits])).toBeUndefined();
  });

  it("treats a name no resolver claims as a fixed unit, converting only into itself", () => {
    const hz = resolveUnit("Hz", {}, [spatialUnits])!;
    expect(hz.label).toBe("Hz");
    expect(convertUnit(440, hz, fixedUnit("Hz"))).toBe(440);
    expect(convertUnit(440, hz, fixedUnit("s"))).toBeUndefined();
  });
});

describe("spatialUnits", () => {
  const resolve = (unit: string, ctx: Parameters<typeof spatialUnits.resolve>[1]) => resolveUnit(unit, ctx, [spatialUnits])!;

  it("says what a length in cells is in µm (ADR-0018: '2 cells = 0.8 µm')", () => {
    const cells = resolve("cells", { placement: placed(0.4), micrometresPerUnit: 1 });
    expect(cells.base).toBe("µm");
    expect(cells.toBase(2)).toBeCloseTo(0.8, 12);
  });

  it("converts world lengths into cells against the placement, with or without a physical scale", () => {
    for (const ctx of [{ placement: placed(0.5) }, { placement: placed(0.5), micrometresPerUnit: 1000 }]) {
      expect(convertUnit(3, resolve("world", ctx), resolve("cells", ctx))).toBeCloseTo(6, 12);
      expect(convertUnit(6, resolve("cells", ctx), resolve("world", ctx))).toBeCloseTo(3, 12);
    }
  });

  it("keeps an unknown scale distinct from 1", () => {
    const world = resolve("world", { placement: placed(1) });
    expect(world.base).toBe("world:global");
    expect(convertUnit(1, world, fixedUnit("µm"))).toBeUndefined();
    expect(convertUnit(1, resolve("world", { placement: placed(1), micrometresPerUnit: 1 }), fixedUnit("µm"))).toBe(1);
  });

  it("leaves cells unconvertible when the field is unplaced or its cells aren't square", () => {
    for (const ctx of [{}, { placement: placed(0.5, 0.25) }]) {
      expect(resolve("cells", ctx).base).toBe("cells");
      expect(convertUnit(1, resolve("world", ctx), resolve("cells", ctx))).toBeUndefined();
    }
  });
});

describe("a resolver from another domain", () => {
  // Transposition by scale degree: relative to a scale, not linear, and with no inverse (a
  // chromatic note has no degree). The same interface carries it.
  interface ScaleContext {
    scale?: readonly number[]; // semitones of each degree within an octave
  }
  const degrees: UnitResolver<ScaleContext> = {
    units: ["degrees"],
    resolve(_unit, { scale }) {
      if (!scale?.length) return undefined;
      const at = (d: number) => {
        const oct = Math.floor(d / scale.length);
        return 12 * oct + scale[d - oct * scale.length]!;
      };
      return { label: "degrees", base: "semitones", toBase: (d) => at(Math.round(d)) };
    },
  };
  const major = [0, 2, 4, 5, 7, 9, 11];

  it("composes with other resolvers through one context", () => {
    const ctx = { scale: major, placement: placed(1) };
    const resolvers = [degrees, spatialUnits];
    const third = resolveUnit("degrees", ctx, resolvers)!;
    expect(convertUnit(2, third, fixedUnit("semitones"))).toBe(4);
    expect(convertUnit(7, third, fixedUnit("semitones"))).toBe(12);
    expect(convertUnit(-1, third, fixedUnit("semitones"))).toBe(-1);
    expect(resolveUnit("cells", ctx, resolvers)?.base).toBe("world:global");
  });

  it("can't be converted into, and doesn't resolve without its context", () => {
    const third: ResolvedUnit = resolveUnit("degrees", { scale: major }, [degrees])!;
    expect(convertUnit(4, fixedUnit("semitones"), third)).toBeUndefined();
    expect(resolveUnit("degrees", {}, [degrees])).toBeUndefined();
  });
});
