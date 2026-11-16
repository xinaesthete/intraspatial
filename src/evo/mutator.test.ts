/** biome-ignore-all lint/style/noNonNullAssertion: lots of noise otherwise */
import { describe, expect, it } from "vite-plus/test";
import type { ParamSpec } from "../gpu/graph/op";
import { advance, breed, marry, mutate, steer, toward } from "./mutator";
import { mulberry32 } from "./rng";
import { neutralSpecimen, randomSpecimen } from "./specimen";
import { specimenToParams, traitSpaceFromParams, withLocked } from "./traitSpace";

const SPECS: ParamSpec[] = [
  { name: "attract", type: "number", default: 0.5, min: 0, max: 2 },
  { name: "swirl", type: "number", default: 0.5, min: 0, max: 2 },
  { name: "collide", type: "bool", default: true },
];
const space = traitSpaceFromParams(SPECS);

describe("mutate", () => {
  it("is deterministic in its rng", () => {
    const sp = neutralSpecimen(space, 7);
    const a = mutate(space, sp, 0.2, mulberry32(42));
    const b = mutate(space, sp, 0.2, mulberry32(42));
    expect(Array.from(a.pos)).toEqual(Array.from(b.pos));
    expect(a.seed).toBe(b.seed);
  });

  it("keeps every trait in [0,1] however wild the rate", () => {
    let sp = randomSpecimen(space, 3);
    const rng = mulberry32(99);
    for (let i = 0; i < 200; i++) sp = mutate(space, sp, 1.5, rng);
    for (const v of sp.pos) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("never moves a locked trait", () => {
    const locked = withLocked(space, ["attract"]);
    const sp = neutralSpecimen(locked, 1);
    const m = mutate(locked, sp, 0.5, mulberry32(5));
    expect(m.pos[0]).toBe(sp.pos[0]); // attract is trait slot 0, locked
    expect(m.pos[1]).not.toBe(sp.pos[1]); // swirl moved
  });
});

describe("marry", () => {
  it("places each trait between its parents", () => {
    const a = randomSpecimen(space, 11);
    const b = randomSpecimen(space, 22);
    const child = marry(space, a, b, mulberry32(1));
    for (let i = 0; i < space.numCount; i++) {
      const lo = Math.min(a.pos[i]!, b.pos[i]!);
      const hi = Math.max(a.pos[i]!, b.pos[i]!);
      expect(child.pos[i]!).toBeGreaterThanOrEqual(lo - 1e-12);
      expect(child.pos[i]!).toBeLessThanOrEqual(hi + 1e-12);
    }
  });

  it("inherits velocity between the parents (heritable momentum)", () => {
    const a = randomSpecimen(space, 11);
    a.vel[0] = 0.4;
    const b = randomSpecimen(space, 22);
    b.vel[0] = 0.8;
    const child = marry(space, a, b, mulberry32(2));
    expect(child.vel[0]!).toBeGreaterThanOrEqual(0.4 - 1e-12);
    expect(child.vel[0]!).toBeLessThanOrEqual(0.8 + 1e-12);
  });

  it("takes locked traits from the primary parent a", () => {
    const locked = withLocked(space, ["attract"]);
    const a = randomSpecimen(locked, 11);
    const b = randomSpecimen(locked, 22);
    const child = marry(locked, a, b, mulberry32(3));
    expect(child.pos[0]).toBe(a.pos[0]);
  });
});

describe("steer + advance (the playhead)", () => {
  it("steer adds an impulse to velocity, leaving position put", () => {
    const sp = neutralSpecimen(space, 1);
    const dir = Float64Array.from([1, 0]);
    const s = steer(space, sp, dir, 0.1);
    expect(s.vel[0]!).toBeCloseTo(0.1, 12);
    expect(s.vel[1]).toBe(0);
    expect(s.pos[0]).toBe(sp.pos[0]); // unchanged until advance
  });

  it("advance moves position by velocity and decays it", () => {
    const sp = neutralSpecimen(space, 1);
    sp.vel[0] = 0.1;
    const a = advance(space, sp, 1, 0.9);
    expect(a.pos[0]!).toBeCloseTo(0.6, 12);
    expect(a.vel[0]!).toBeCloseTo(0.09, 12);
  });

  it("reflects off the [0,1] walls instead of sticking", () => {
    const sp = neutralSpecimen(space, 1);
    sp.pos[0] = 0.95;
    sp.vel[0] = 0.2; // would overshoot 1
    const a = advance(space, sp, 1, 1);
    expect(a.pos[0]!).toBeLessThanOrEqual(1);
    expect(a.vel[0]!).toBeLessThan(0); // velocity reflected
  });

  it("toward points from a to b", () => {
    const a = neutralSpecimen(space, 1);
    const b = neutralSpecimen(space, 1);
    b.pos[0] = 0.9;
    const d = toward(a, b);
    expect(d[0]!).toBeCloseTo(0.4, 12); // 0.9 - 0.5
  });
});

describe("breed", () => {
  it("produces a replayable generation", () => {
    const parents = [randomSpecimen(space, 1), randomSpecimen(space, 2)];
    const g1 = breed(space, parents, 9, { rate: 0.2, rng: mulberry32(7) });
    const g2 = breed(space, parents, 9, { rate: 0.2, rng: mulberry32(7) });
    expect(g1.map((sp) => sp.seed)).toEqual(g2.map((sp) => sp.seed));
    expect(g1.map((sp) => Array.from(sp.pos))).toEqual(g2.map((sp) => Array.from(sp.pos)));
  });

  it("keeps the elite as element 0 when asked", () => {
    const parent = randomSpecimen(space, 5);
    const gen = breed(space, [parent], 9, { rate: 0.3, rng: mulberry32(7), keepElite: true });
    expect(Array.from(gen[0]!.pos)).toEqual(Array.from(parent.pos));
    expect(gen).toHaveLength(9);
  });

  it("makes random individuals when there are no parents", () => {
    const gen = breed(space, [], 9, { rate: 0.2, rng: mulberry32(7) });
    expect(gen).toHaveLength(9);
    // not all identical
    const first = Array.from(gen[0]!.pos).join();
    expect(gen.some((sp) => Array.from(sp.pos).join() !== first)).toBe(true);
  });
});

describe("mutate through warps and deltas", () => {
  const median = (xs: number[]) => {
    const s = [...xs].sort((a, b) => a - b);
    return (s[(s.length - 1) >> 1]! + s[s.length >> 1]!) / 2;
  };
  const sd = (xs: number[]) => {
    const m = xs.reduce((a, b) => a + b, 0) / xs.length;
    return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
  };
  /** `n` independent mutations of `sp`, each param's value read out. */
  const samples = (sp: ParamSpec[], name: string, rate: number, n = 4000) => {
    const sp0 = traitSpaceFromParams(sp);
    const start = neutralSpecimen(sp0, 1);
    const rng = mulberry32(2024);
    return Array.from({ length: n }, () => specimenToParams(sp0, mutate(sp0, start, rate, rng))[name] as number);
  };

  it("leaves a param without delta exactly as before (σ = rate)", () => {
    // deltaNorm 1 is the identity: a delta of the whole range must give bit-identical mutations
    const plain = traitSpaceFromParams([{ name: "a", type: "number", default: 0.5, min: 0, max: 2 }]);
    const full = traitSpaceFromParams([{ name: "a", type: "number", default: 0.5, min: 0, max: 2, delta: 2 }]);
    const sp = neutralSpecimen(plain, 3);
    expect(Array.from(mutate(full, sp, 0.3, mulberry32(9)).pos)).toEqual(Array.from(mutate(plain, sp, 0.3, mulberry32(9)).pos));
  });

  it("mutates an exp param from its midpoint around the geometric mean, not the arithmetic", () => {
    const freq: ParamSpec = { name: "freq", type: "number", default: 500, min: 50, max: 3000, warp: "exp" };
    const xs = samples([freq], "freq", 0.15);
    const geo = Math.sqrt(50 * 3000); // 387
    expect(median(xs) / geo).toBeGreaterThan(0.95);
    expect(median(xs) / geo).toBeLessThan(1.05);
    // even in pitch: as many mutations land below the geometric mean as above, where linearly
    // most of them would sit above 1 kHz
    expect(xs.filter((x) => x > 1000).length / xs.length).toBeLessThan(0.3);
    const linear = samples([{ ...freq, warp: undefined }], "freq", 0.15);
    expect(median(linear)).toBeCloseTo(1525, -2);
  });

  it("scales the spread by delta", () => {
    const spec = (delta: number): ParamSpec[] => [{ name: "a", type: "number", default: 50, min: 0, max: 100, delta }];
    const small = sd(samples(spec(1), "a", 1));
    const big = sd(samples(spec(5), "a", 1));
    expect(small).toBeCloseTo(1, 1); // rate 1 ⇒ one mutation moves about one delta
    expect(big / small).toBeCloseTo(5, 1);
    // and `rate` still multiplies it
    expect(sd(samples(spec(5), "a", 0.5)) / big).toBeCloseTo(0.5, 1);
  });

  it("gives a warped param a delta-sized step in ratio", () => {
    // delta 50 Hz at a default of 500 Hz ⇒ steps of about ×1.1 anywhere in the range
    const freq: ParamSpec[] = [{ name: "freq", type: "number", default: 500, min: 50, max: 3000, warp: "exp", delta: 50 }];
    const logs = samples(freq, "freq", 1).map((f) => Math.log(f));
    expect(sd(logs) / Math.log(1.1)).toBeCloseTo(1, 1);
  });
});
