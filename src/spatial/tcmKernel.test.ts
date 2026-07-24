import { describe, expect, it } from "vitest";
import { EPANECHNIKOV, equivalentRadius, GAUSSIAN, KERNELS, kernelLabel, QUARTIC, TOPHAT, TRIWEIGHT } from "./kernels";
import { type CellCloud, computeTcmReference, crossMarks, type TcmParams } from "./tcm";
import { kernelMarks, type TcmKernelParams, tcmKernelField } from "./tcmKernel";

function rng(seed: number) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rnd = rng(0x5eed);
const cloud = (n: number, cx: number, cy: number, sp: number): CellCloud => {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(cx + (rnd() - 0.5) * sp);
    ys.push(cy + (rnd() - 0.5) * sp);
  }
  return { xs, ys };
};

const A = cloud(150, 10, 10, 18);
const B = cloud(260, 8, 12, 14);
const base: TcmParams = { width: 48, height: 48, bbox: [0, 0, 20, 20], radius: 3, sigma: 1.5, alpha: 5 };

describe("kernelMarks", () => {
  it("with the top-hat it IS eq 9 — the generalisation costs nothing", () => {
    const paper = crossMarks(A, B, base);
    const kernelForm = kernelMarks(A, B, { ...base, kernel: TOPHAT });
    let maxAbs = 0;
    for (let i = 0; i < paper.length; i++) maxAbs = Math.max(maxAbs, Math.abs(paper[i]! - kernelForm[i]!));
    expect(maxAbs).toBeLessThan(1e-12);
  });

  it("every kernel puts the mark on the same CSR scale (mean ≈ the same, 1 = CSR)", () => {
    // Unit-mass kernels ⇒ m is a density ratio however you smooth. At MATCHED scale the mean mark
    // over the same cells should agree closely; without `equivalentRadius` it would not.
    const means = KERNELS.map((k) => {
      const m = kernelMarks(A, B, { ...base, kernel: k, radius: equivalentRadius(k, base.radius) });
      return m.reduce((s, v) => s + v, 0) / m.length;
    });
    const lo = Math.min(...means);
    const hi = Math.max(...means);
    expect(lo).toBeGreaterThan(0.5);
    expect(hi / lo).toBeLessThan(1.1); // same statistic, differently smoothed
  });

  it("smoother kernels give a less jumpy mark as the radius is nudged", () => {
    // The tangible cost of the hard disk: a 2% change in r moves the top-hat's marks far more than
    // a smooth kernel's, because B cells cross the boundary discontinuously. Measured at matched
    // scale, relative to each kernel's own mark magnitude.
    const wobble = (k: (typeof KERNELS)[number]) => {
      const r = equivalentRadius(k, base.radius);
      const m0 = kernelMarks(A, B, { ...base, kernel: k, radius: r });
      const m1 = kernelMarks(A, B, { ...base, kernel: k, radius: r * 1.02 });
      let num = 0;
      let den = 0;
      for (let i = 0; i < m0.length; i++) {
        num += (m1[i]! - m0[i]!) ** 2;
        den += m0[i]! ** 2;
      }
      return Math.sqrt(num / Math.max(den, 1e-30));
    };
    const tophat = wobble(TOPHAT);
    for (const k of [EPANECHNIKOV, QUARTIC, TRIWEIGHT, GAUSSIAN]) {
      expect(wobble(k), kernelLabel(k)).toBeLessThan(tophat);
    }
  });
});

describe("tcmKernelField", () => {
  it("reproduces the paper's Γ up to the support convention", () => {
    // Same statistic; the only difference is that this splat uses a world box of radiusSigma·σ
    // where computeTcmReference uses a cell-quantised ±kr box. At radiusSigma = 6 the render's box
    // strictly contains the reference's, so the gap is the reference's own tail truncation.
    const ref = computeTcmReference(A, B, base);
    const p: TcmKernelParams = { ...base, kernel: TOPHAT, radiusSigma: 6 };
    const { gamma } = tcmKernelField(A, B, p);
    let peak = 0;
    let maxAbs = 0;
    for (let i = 0; i < ref.length; i++) {
      peak = Math.max(peak, Math.abs(ref[i]!));
      maxAbs = Math.max(maxAbs, Math.abs(gamma[i]! - ref[i]!));
    }
    expect(peak).toBeGreaterThan(0.5);
    expect(maxAbs / peak).toBeLessThan(0.02); // ~e^-4.5, the tail the reference drops
  });

  it("keeps the sign structure of the paper's map under every kernel", () => {
    // Whatever the kernel, Γ must stay positive where A sits inside B and negative where it does
    // not — otherwise the smoothing has changed the reading, not just the noise.
    const bx: number[] = [];
    const by: number[] = [];
    for (let i = 0; i < 7; i++)
      for (let j = 0; j < 7; j++) {
        bx.push(4 + i / 3);
        by.push(4 + j / 3);
      }
    const pts: CellCloud = { xs: [5, 15], ys: [5, 15] };
    for (const k of KERNELS) {
      const p: TcmKernelParams = {
        width: 20,
        height: 20,
        bbox: [0, 0, 20, 20],
        radius: equivalentRadius(k, 5),
        sigma: 1,
        alpha: 5,
        kernel: k,
      };
      const { gamma } = tcmKernelField(pts, { xs: bx, ys: by }, p);
      expect(gamma[5 * 20 + 5]!, kernelLabel(k)).toBeGreaterThan(0);
      expect(gamma[15 * 20 + 15]!, kernelLabel(k)).toBeLessThan(0);
    }
  });

  it("exposes the marks behind the raster", () => {
    const { marks } = tcmKernelField(A, B, { ...base, kernel: QUARTIC });
    expect(marks.length).toBe(A.xs.length);
    expect(marks.every((v) => v >= -1 && v <= 1)).toBe(true);
  });
});
