import { describe, expect, it } from "vite-plus/test";
import { type PlanInput, planField, planServes } from "./fieldPlan";

const base: PlanInput = {
  visible: [0, 0, 1000, 1000],
  pixels: [1000, 1000],
  data: [0, 0, 1000, 1000],
  sigma: 10,
  radiusSigma: 4,
  cellsPerSigma: 3,
  pixelsPerCell: 1,
  margin: 0.25,
  maxCells: 4_000_000,
};

describe("planField", () => {
  it("uses about a screen pixel per cell when zoomed out", () => {
    const plan = planField({ ...base, sigma: 1 })!;
    expect(plan.cell).toBeCloseTo(1);
  });

  it("never goes finer than sigma / cellsPerSigma when zoomed in", () => {
    // 10 data units across 1000 px is 0.01 per pixel, far below sigma/3.
    const plan = planField({ ...base, visible: [500, 500, 510, 510] })!;
    expect(plan.cell).toBeCloseTo(10 / 3);
  });

  it("clips the grid to where the field can be non-zero", () => {
    // Visible region far wider than the data: the grid stops at data ± 4σ (snapped outward).
    const plan = planField({ ...base, visible: [-5000, -5000, 6000, 6000], pixels: [1100, 1100] })!;
    expect(plan.bbox[0]).toBeGreaterThanOrEqual(-40 - plan.cell);
    expect(plan.bbox[2]).toBeLessThanOrEqual(1040 + plan.cell);
  });

  it("returns null when no part of the field is on screen", () => {
    expect(planField({ ...base, visible: [5000, 5000, 6000, 6000] })).toBeNull();
  });

  it("tiles the bbox exactly with cells on a lattice fixed to the origin", () => {
    const plan = planField({ ...base, visible: [123.4, 56.7, 789, 654.3] })!;
    expect((plan.bbox[2] - plan.bbox[0]) / plan.cell).toBeCloseTo(plan.width);
    expect((plan.bbox[3] - plan.bbox[1]) / plan.cell).toBeCloseTo(plan.height);
    expect(Math.abs(plan.bbox[0] / plan.cell - Math.round(plan.bbox[0] / plan.cell))).toBeLessThan(1e-9);
  });

  it("grows the cell to stay within maxCells", () => {
    const plan = planField({ ...base, sigma: 0.1, maxCells: 10_000 })!;
    expect(plan.width * plan.height).toBeLessThanOrEqual(10_000);
  });
});

describe("planServes", () => {
  it("reuses a plan while the view pans inside its margin", () => {
    const view = { ...base, visible: [200, 200, 400, 400] as PlanInput["visible"], pixels: [200, 200] as PlanInput["pixels"] };
    const plan = planField(view)!;
    expect(planServes(plan, { ...view, visible: [230, 220, 430, 420] })).toBe(true);
  });

  it("re-plans once the view leaves the margin", () => {
    const view = { ...base, visible: [200, 200, 400, 400] as PlanInput["visible"], pixels: [200, 200] as PlanInput["pixels"] };
    const plan = planField(view)!;
    expect(planServes(plan, { ...view, visible: [300, 200, 500, 400] })).toBe(false);
  });

  it("re-plans when zoom moves the wanted cell by more than √2", () => {
    const view = { ...base, sigma: 0.5 };
    const plan = planField(view)!;
    // Zoom in 2× about the centre: the wanted cell halves.
    expect(planServes(plan, { ...view, visible: [250, 250, 750, 750] })).toBe(false);
    // A 1.2× zoom stays within √2.
    expect(planServes(plan, { ...view, visible: [42, 42, 958, 958] })).toBe(true);
  });
});
