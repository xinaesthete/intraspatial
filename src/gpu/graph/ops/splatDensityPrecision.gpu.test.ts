import { describe, expect, it } from "vitest";
import { nodeBackend } from "../backend.node";
import { Graph, pull, pullResident, registerBuiltinOps } from "../index";

// What does accumulating a density field in f16 actually cost?
//
// The field is a render target the splat blends into additively, so half precision is not a
// uniform rounding of the answer: each ADD rounds, so the error grows with the number of
// overlapping contributions in a texel, and the top of the range saturates at 65504. These tests
// put numbers on both ends rather than asserting a tolerance someone guessed, because the whole
// decision — "is f16 good enough for this field?" — is a question about those numbers.
//
// House rules: aggregated assertions, small grids.

registerBuiltinOps();

function cloud(n: number, seed: number, spread = 1) {
  let a = seed;
  const rnd = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(50 + (rnd() - 0.5) * 100 * spread);
    ys.push(50 + (rnd() - 0.5) * 100 * spread);
  }
  return { xs, ys };
}

const PARAMS = { width: 64, height: 64, sigma: 3, radiusSigma: 4, bbox: [0, 0, 100, 100] };

async function densityAt(precision: "f32" | "f16", n: number, seed: number, spread = 1) {
  const g = new Graph();
  const { xs, ys } = cloud(n, seed, spread);
  const field = g.op1("splatDensity", { points: g.points(xs, ys) }, { ...PARAMS, precision });
  return (await pull(g, field)).data as Float32Array;
}

/** Max relative error against the f32 field, measured only where the f32 field is meaningfully
 *  non-zero — a relative error on a texel holding ~1e-30 says nothing about the picture. */
function compare(f16: Float32Array, f32: Float32Array) {
  let peak = 0;
  for (const v of f32) if (v > peak) peak = v;
  const floor = peak * 1e-3;
  let maxRel = 0;
  let sum16 = 0;
  let sum32 = 0;
  for (let i = 0; i < f32.length; i++) {
    sum16 += f16[i]!;
    sum32 += f32[i]!;
    if (f32[i]! < floor) continue;
    const rel = Math.abs(f16[i]! - f32[i]!) / f32[i]!;
    if (rel > maxRel) maxRel = rel;
  }
  return { maxRel, massRatio: sum16 / sum32, peak };
}

describe("splatDensity precision", () => {
  it("costs one rounding, not a compounding one — error is flat as overlap deepens", async () => {
    // THE property, not a tolerance: because the field accumulates in f32 and is narrowed once
    // on store, the error does not grow with how many splats land in a texel. Blending straight
    // into f16 fails exactly here — measured, that version lost 9% of the field's mass at 6400
    // points and 75% at 25000 (see docs/field-precision.md), while these stay flat.
    const light = compare(await densityAt("f16", 400, 0xbeef), await densityAt("f32", 400, 0xbeef));
    const heavy = compare(await densityAt("f16", 6400, 0xbeef, 0.5), await densityAt("f32", 6400, 0xbeef, 0.5));

    // Something actually accumulated — an all-zero field would pass any error bound.
    expect(light.peak).toBeGreaterThan(0);
    // The heavy case must be deep enough to have broken the naive version.
    expect(heavy.peak).toBeGreaterThan(20 * light.peak);

    // ~2 ulp of f16 (its relative resolution is 2^-11 ≈ 0.049%), at both depths.
    expect(light.maxRel).toBeLessThan(2e-3);
    expect(heavy.maxRel).toBeLessThan(2e-3);
    // And no systematic drift: the integral the statistic uses is preserved either way.
    expect(Math.abs(light.massRatio - 1)).toBeLessThan(1e-3);
    expect(Math.abs(heavy.massRatio - 1)).toBeLessThan(1e-3);
  });

  it("returns an f32 field either way, so no consumer can tell", async () => {
    const f16 = await densityAt("f16", 200, 3);
    expect(f16).toBeInstanceOf(Float32Array);
    expect(f16.length).toBe(PARAMS.width * PARAMS.height);
  });

  it("is unaffected by a larger field computed before it (the shared accumulator)", async () => {
    // The f16 path renders into one reused f32 accumulator. Compute a BIG field first so the
    // accumulator is oversized, then a small one: without a viewport the small field would be
    // splatted across the whole target and cropped — a plausible-looking wrong answer.
    // Sizes chosen to land INSIDE the registry's reuse window (64*64 <= 2*48*48), so the small
    // field really is handed the bigger target. Outside it the registry reallocates exactly and
    // the viewport would never be exercised.
    const big = { ...PARAMS, width: 64, height: 64 };
    const small = { ...PARAMS, width: 48, height: 48 };
    const { xs, ys } = cloud(500, 0x5eed);
    const run = async (p: Record<string, unknown>, precision: "f32" | "f16") => {
      const g = new Graph();
      return (await pull(g, g.op1("splatDensity", { points: g.points(xs, ys) }, { ...p, precision }))).data as Float32Array;
    };

    await run(big, "f16"); // grows the accumulator past what the next call needs
    const after = await run(small, "f16");
    const want = await run(small, "f32");

    let maxRel = 0;
    let peak = 0;
    for (const v of want) peak = Math.max(peak, v);
    for (let i = 0; i < want.length; i++) {
      if (want[i]! < peak * 1e-3) continue;
      maxRel = Math.max(maxRel, Math.abs(after[i]! - want[i]!) / want[i]!);
    }
    expect(peak).toBeGreaterThan(0);
    expect(maxRel).toBeLessThan(2e-3);
  });

  it("the op's field really is an r16float texture, and costs half the bytes", async () => {
    const g = new Graph();
    const { xs, ys } = cloud(50, 11);
    const resident = await pullResident(g, g.op1("splatDensity", { points: g.points(xs, ys) }, { ...PARAMS, precision: "f16" }));
    expect(resident.texture?.format).toBe("r16float");
    // The value still calls itself f32: precision is a storage decision, and the bridge widens it.
    expect(resident.dtype).toBe("f32");

    // The saving, measured on the pool rather than inferred: two fresh sizes the pool has not
    // seen, so each lease allocates and `bytes` moves by exactly what it created.
    const before = nodeBackend.poolStats().bytes;
    const a = await nodeBackend.leaseTexture(97, 73, "r32float");
    const mid = nodeBackend.poolStats().bytes;
    const b = await nodeBackend.leaseTexture(97, 73, "r16float");
    const after = nodeBackend.poolStats().bytes;
    nodeBackend.releaseTexture(a);
    nodeBackend.releaseTexture(b);

    expect(mid - before).toBe(97 * 73 * 4);
    expect(after - mid).toBe(97 * 73 * 2);
  });
});
