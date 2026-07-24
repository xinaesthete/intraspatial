import { describe, expect, it } from "vitest";
import { auc, type KernelScore, makeAssociationScene, scoreKernels } from "./kernelAnalysis";
import { EPANECHNIKOV, kernelLabel, TOPHAT } from "./kernels";

const byLabel = (scores: readonly KernelScore[], k: Parameters<typeof kernelLabel>[0]) => scores.find((s) => s.label === kernelLabel(k))!;
const smooth = (scores: readonly KernelScore[]) => scores.filter((s) => s.label !== kernelLabel(TOPHAT));

describe("auc", () => {
  it("is 1 for perfect separation, 0.5 for none, and splits ties evenly", () => {
    expect(auc([3, 4, 5], [0, 1, 2])).toBe(1);
    expect(auc([0, 1, 2], [3, 4, 5])).toBe(0);
    expect(auc([1, 1, 1], [1, 1, 1])).toBe(0.5); // all tied — no information, and it says so
    expect(auc([2, 2], [1, 2])).toBeCloseTo(0.75, 10); // one clear win, one tie
  });
});

describe("kernel comparison on a scene with known association", () => {
  // These are measurements, not preferences. The numbers below are what the harness actually
  // reports (seeded, so they are stable); the assertions are deliberately loose enough to be about
  // the EFFECT, not the digits.
  const strong = scoreKernels(makeAssociationScene());
  const weak = scoreKernels(makeAssociationScene({ bPerPatch: 26, bBackground: 900 }));
  const sparse = scoreKernels(makeAssociationScene({ bPerPatch: 12, bBackground: 120 }));

  it("matches every kernel to the same spatial scale, which is what makes the rest fair", () => {
    const target = byLabel(strong, TOPHAT).moment2;
    for (const s of strong) expect(s.moment2, s.label).toBeCloseTo(target, 6);
    // Smoother kernels reach that scale with a WIDER support — the comparison trap, made visible.
    for (const s of smooth(strong)) expect(s.radius, s.label).toBeGreaterThan(byLabel(strong, TOPHAT).radius);
  });

  it("the top-hat's mark is massively DISCRETISED — its largest practical cost", () => {
    // The mark is a count over ρ_B·πr², so it takes only a handful of distinct values and most
    // cells cannot be ordered at all. Measured: 93% of A cells tied on the strong scene, 99.8% when
    // B is sparse. Every smooth kernel is at or near zero.
    expect(byLabel(strong, TOPHAT).tied).toBeGreaterThan(0.9);
    expect(byLabel(sparse, TOPHAT).tied).toBeGreaterThan(0.95);
    for (const s of smooth(strong)) expect(s.tied, s.label).toBeLessThan(0.01);
    for (const s of smooth(sparse)) expect(s.tied, s.label).toBeLessThan(0.1);
  });

  it("the top-hat is far more sensitive to the (arbitrary) radius choice", () => {
    // A 2% nudge in r. Measured 2.6% / 4.0% / 6.5% for the top-hat across the three scenes, against
    // ~1.0-1.6% for every smooth kernel — so 2-5x more of the answer depends on a free parameter.
    for (const scores of [strong, weak, sparse]) {
      const hat = byLabel(scores, TOPHAT).radiusSensitivity;
      for (const s of smooth(scores)) expect(s.radiusSensitivity, s.label).toBeLessThan(hat * 0.75);
    }
  });

  it("...and less robust to positional error in the cells themselves", () => {
    // Jitter of 5% of the radius — i.e. segmentation/centroid noise, a property of real data rather
    // than of the numerics. Measured 5.6% / 7.2% / 11.4% top-hat vs ~4.0-6.5% smooth.
    for (const scores of [strong, weak, sparse]) {
      const hat = byLabel(scores, TOPHAT).jitterSensitivity;
      for (const s of smooth(scores)) expect(s.jitterSensitivity, s.label).toBeLessThan(hat);
    }
  });

  it("but discrimination barely moves — the honest negative result", () => {
    // Smoothing does NOT buy separating power on an easy scene: every kernel lands within 0.0005
    // AUC of every other. Worth stating plainly, because the temptation is to claim it does.
    const aucs = strong.map((s) => s.auc);
    expect(Math.max(...aucs) - Math.min(...aucs)).toBeLessThan(0.005);
  });

  it("...except where it matters: subtle association, where the smooth kernels do pull ahead", () => {
    // Weak-contrast patches are the regime you actually care about. Measured: top-hat 0.9548 vs
    // 0.960-0.961 for the smooth kernels — small, but consistent and in the same direction.
    const hat = byLabel(weak, TOPHAT).auc;
    for (const s of smooth(weak)) expect(s.auc, s.label).toBeGreaterThan(hat);
  });

  it("reproduces the textbook AMISE ordering at matched scale", () => {
    // R(K)·μ₂ is dimensionless and scale-invariant; Epanechnikov minimises it, as theory says.
    const eff = strong.map((s) => ({ label: s.label, v: s.roughness * s.moment2 }));
    const best = eff.reduce((a, b) => (b.v < a.v ? b : a));
    expect(best.label).toBe(kernelLabel(EPANECHNIKOV));
    expect(byLabel(strong, TOPHAT).roughness * byLabel(strong, TOPHAT).moment2).toBeGreaterThan(best.v);
  });
});
