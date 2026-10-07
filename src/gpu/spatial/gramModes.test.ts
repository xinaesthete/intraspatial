import { describe, expect, it } from "vitest";
import { type ModeSource, modeBasis, modeParams } from "./gramModes";

// Two channels with corr [[1, .5], [.5, 1]]: modes (1, 1)/√2 with λ = 1.5 and (1, −1)/√2 with λ = .5.
const s = Math.SQRT1_2;
const vectors = new Float64Array([s, s, s, -s]);
const res: ModeSource = {
  labels: ["a", "b"],
  corr: new Float64Array([1, 0.5, 0.5, 1]),
  resident: { mean: new Float64Array([1, 2]), sd: new Float64Array([1, 2]) },
};

describe("modeParams with a basis", () => {
  it("is unchanged when the basis is the result's own", () => {
    const own = modeParams(res, { vectors });
    const based = modeParams(res, { vectors, basis: modeBasis(res) });
    expect(Array.from(based.chan)).toEqual(Array.from(own.chan));
    expect(based.sigmas).toEqual(own.sigmas);
    expect(own.sigmas[0]).toBeCloseTo(Math.sqrt(1.5), 12); // σ_k = √λ_k
  });

  it("standardises and scales with the basis, not the result", () => {
    const basis = { labels: ["a", "b"], mean: [10, 20], sd: [4, 5], corr: [1, 0, 0, 1] };
    const p = modeParams(res, { vectors, basis });
    expect(Array.from(p.chan.filter((_, i) => i % 5 < 2))).toEqual(Array.from(Float32Array.of(10, 0.25, 20, 0.2))); // mean, 1/sd; f32 for the GPU
    expect(p.sigmas[0]).toBeCloseTo(1, 12); // vᵀ·I·v for a unit vector
    expect(p.sigmas[1]).toBeCloseTo(1, 12);
  });

  it("refuses a basis for other channels", () => {
    expect(() => modeParams(res, { vectors, basis: { ...modeBasis(res), labels: ["a", "c"] } })).toThrow(/mode basis/);
  });

  it("copies, so the basis survives the result's arrays being reused", () => {
    const mean = new Float64Array([1, 2]);
    const b = modeBasis({ ...res, resident: { ...res.resident, mean } });
    mean[0] = 99;
    expect(b.mean[0]).toBe(1);
  });
});
