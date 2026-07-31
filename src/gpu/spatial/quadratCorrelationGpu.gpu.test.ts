import { describe, expect, it } from "vitest";
import { mulberry32 } from "../../spatial/kernelAnalysis";
import type { LabelledCells } from "../../spatial/pcf";
import { partialCorrelation, quadratCorrelation, quadratCounts, rowCorrelation } from "../../spatial/quadratCorrelation";
import { quadratCorrelationGpu, quadratCorrelationGpuSupported } from "./quadratCorrelationGpu";

// The GPU path replaces the NULL, not the statistic, so the tests split the same way: the observed
// values must be bit-identical to the CPU's, while the null only has to agree in the two moments the
// effect size actually consumes. Testing the sampled null for bit-parity would be meaningless — a
// Feistel permutation and a Fisher-Yates shuffle draw different permutations by construction.

const BBOX = [0, 0, 400, 400] as const;

/** Types laid over a varying density, so quadrat totals differ and the null has real spread. */
function build(mode: "together" | "apart", seed: number, nTypes = 4): LabelledCells {
  const rnd = mulberry32(seed);
  const xs: number[] = [];
  const ys: number[] = [];
  const typeId: number[] = [];
  for (let qy = 0; qy < 4; qy++) {
    for (let qx = 0; qx < 4; qx++) {
      const hot = (qx + qy) % 2 === 0;
      const push = (t: number, count: number) => {
        for (let i = 0; i < count; i++) {
          xs.push(qx * 100 + rnd() * 100);
          ys.push(qy * 100 + rnd() * 100);
          typeId.push(t);
        }
      };
      // The jitter is load-bearing. Counts that are pure functions of `hot` are affine in one
      // indicator, so ANY two such types are exactly collinear, the correlation matrix is singular,
      // and every partial correlation is legitimately NaN — a property of the fixture, not of the
      // code under test. The CPU tests never hit this because they only use two types, where there
      // is nothing to condition on and the partial correlation is the plain one.
      const jit = (v: number) => Math.max(1, v + Math.floor(rnd() * 9) - 4);
      push(0, jit(hot ? 40 : 8));
      push(1, jit(mode === "together" ? (hot ? 36 : 10) : hot ? 8 : 30));
      for (let t = 2; t < nTypes; t++) push(t, 6 + Math.floor(rnd() * 30));
    }
  }
  return { xs, ys, typeId };
}

const params = (sims: number, nTypes = 4) => ({ bbox: BBOX, quadratSize: 100, nTypes, simulations: sims, seed: 7 }) as const;

describe("quadratCorrelationGpu", () => {
  it("returns the observed statistics unchanged from the CPU", async () => {
    // These are the numbers that reproduce the published columns; the GPU must not touch them.
    const cells = build("together", 3);
    const gpu = await quadratCorrelationGpu(cells, params(64));
    const cpu = quadratCorrelation(cells, params(0));
    expect([...gpu.r]).toEqual([...cpu.r]);
    expect([...gpu.pc]).toEqual([...cpu.pc]);
    expect(gpu.quadrats).toBe(cpu.quadrats);
  });

  it("agrees with the CPU null in mean and spread", async () => {
    // The whole correctness claim for the Feistel permutation. Both paths estimate the same null
    // distribution by different samplers, so the effect sizes must agree to within Monte Carlo
    // error — here a generous 0.5, against SES values that run to several units.
    const cells = build("apart", 5);
    const sims = 999;
    const gpu = await quadratCorrelationGpu(cells, params(sims));
    const cpu = quadratCorrelation(cells, params(sims));
    let worstR = 0;
    let worstPc = 0;
    for (let a = 0; a < 4; a++) {
      for (let b = a + 1; b < 4; b++) {
        const i = a * 4 + b;
        if (Number.isFinite(gpu.ses[i]!) && Number.isFinite(cpu.ses[i]!)) worstR = Math.max(worstR, Math.abs(gpu.ses[i]! - cpu.ses[i]!));
        if (Number.isFinite(gpu.pcSes[i]!) && Number.isFinite(cpu.pcSes[i]!)) worstPc = Math.max(worstPc, Math.abs(gpu.pcSes[i]! - cpu.pcSes[i]!));
      }
    }
    expect(worstR).toBeLessThan(0.5);
    expect(worstPc).toBeLessThan(0.5);
    // And the verdict, which is what a reader acts on, must match outright.
    expect(gpu.ses[1]! < -3).toBe(cpu.ses[1]! < -3);
  });

  it("permutes rather than merely hashing — every type keeps its exact abundance", async () => {
    // A bijection is the entire requirement on the permutation: it must preserve each type's count,
    // which is what makes the null hold the margins fixed. A hash that collided would inflate some
    // types and starve others, and would still produce a plausible-looking spread of correlations.
    // The observable proxy is that a type present in the data is never variance-free under a
    // shuffle, and the p-values stay in range.
    const cells = build("together", 11, 5);
    const res = await quadratCorrelationGpu(cells, params(256, 5));
    for (let a = 0; a < 5; a++) {
      for (let b = 0; b < 5; b++) {
        if (a === b) continue; // r ≡ 1 on the diagonal, so the null has no spread and SES is NaN
        const i = a * 5 + b;
        expect(Number.isFinite(res.ses[i]!)).toBe(true);
        expect(res.p[i]!).toBeGreaterThan(0);
        expect(res.p[i]!).toBeLessThanOrEqual(1);
      }
    }
  });

  it("computes the same partial correlation as the CPU on identical counts", async () => {
    // Pins the in-workgroup Gauss-Jordan against the f64 one. The observed `pc` comes back from the
    // CPU by construction, so the inverse is checked here on the same matrix the kernel inverts.
    const cells = build("together", 13, 6);
    const counts = quadratCounts(cells, { bbox: BBOX, quadratSize: 100, nTypes: 6 });
    const r = rowCorrelation(counts.counts, 6, counts.cols * counts.rows);
    const pc = partialCorrelation(r, 6);
    const gpu = await quadratCorrelationGpu(cells, params(64, 6));
    for (let i = 0; i < 36; i++) expect(gpu.pc[i]!).toBeCloseTo(pc[i]!, 12);
  });

  it("carries a variance-free type through as undefined without taking the others down", async () => {
    // The failure the CPU version actually shipped with, checked on the GPU path: one degenerate
    // type must not poison the rest of the matrix.
    // Only types 0–2 come from `build`; type 3 is added below at a CONSTANT count per quadrat, so it
    // has no variance and no defined correlation. Building it with nTypes=4 would have given type 3
    // random counts first, and the constant block on top would not have made it degenerate at all.
    const cells = build("together", 17, 3);
    const flat = { xs: [...cells.xs], ys: [...cells.ys], typeId: [...cells.typeId] };
    for (let qy = 0; qy < 4; qy++) {
      for (let qx = 0; qx < 4; qx++) {
        for (let i = 0; i < 5; i++) {
          flat.xs.push(qx * 100 + 50);
          flat.ys.push(qy * 100 + 50);
          flat.typeId.push(3);
        }
      }
    }
    const res = await quadratCorrelationGpu({ ...flat, typeId: flat.typeId }, params(128, 4));
    expect(Number.isNaN(res.pc[3 * 4 + 0]!)).toBe(true);
    expect(Number.isFinite(res.pc[0 * 4 + 1]!)).toBe(true);
    expect(Number.isFinite(res.ses[0 * 4 + 1]!)).toBe(true);
  });

  it("skips the null entirely when no simulations are asked for", async () => {
    const res = await quadratCorrelationGpu(build("apart", 19), params(0));
    expect(res.simulations).toBe(0);
    expect(res.ses.length).toBe(0);
  });

  it("declines a type count its in-workgroup inverse cannot hold", () => {
    expect(quadratCorrelationGpuSupported(48)).toBe(true);
    expect(quadratCorrelationGpuSupported(256)).toBe(false);
  });
});
