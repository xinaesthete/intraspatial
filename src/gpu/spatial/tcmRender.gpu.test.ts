import { describe, expect, it } from "vitest";
import { QUARTIC, TOPHAT } from "../../spatial/kernels";
import type { CellCloud } from "../../spatial/tcm";
import { computeTcmRender, kernelDensityGpu } from "./tcmRender";

// SCOPE NOTE. These are smoke tests with teeth, not the parity suite. The full comparison against
// the continuous oracle (`src/spatial/tcmKernel.ts`) runs in the BROWSER, on the cell-stats demo
// page, because this module reliably segfaults Dawn-on-Node's atexit walk once a test file has run
// a couple of render-plus-readback cycles — the assertions pass, then the fork dies before vitest
// can flush the results. Measured in-process before moving them out: relMax vs the oracle was
// 1.8e-4 for a quartic mark at 48², i.e. the parity is real, only the harness is not.
//
// What survives here is chosen to be cheap and to fail loudly if the formulation is wrong at all:
// mass conservation pins pass 1's normalisation, sign structure pins pass 2's nonlinearity.

const BBOX = [0, 0, 20, 20] as const;

describe("computeTcmRender / kernelDensityGpu", () => {
  it("pass 1 conserves mass: the kernels are unit-mass, so ∫ρ̂ = n", async () => {
    // The strongest cheap check of pass 1. Every kernel integrates to 1, so summing the rendered
    // density over the grid and multiplying by the cell area must return the point COUNT — which
    // simultaneously pins the normalisation, the world-to-NDC mapping and the grid extent.
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < 40; i++) {
      xs.push(6 + (i % 8) * 1.1);
      ys.push(6 + Math.floor(i / 8) * 1.4);
    }
    const w = 64;
    const cellArea = ((BBOX[2] - BBOX[0]) / w) ** 2;
    const g = await kernelDensityGpu(xs, ys, { width: w, height: w, radius: 2.5, kernel: QUARTIC, bbox: BBOX });
    const mass = g.reduce((s, v) => s + v, 0) * cellArea;
    expect(mass).toBeGreaterThan(39.0); // all 40 points sit well inside the ROI, so nothing is clipped
    expect(mass).toBeLessThan(41.0);
  });

  it("Γ is positive where A sits inside B and negative where A is excluded", async () => {
    // Pass 2's whole job: the per-point mark fetch, the nonlinearity, and the signed weighted splat.
    const bx: number[] = [];
    const by: number[] = [];
    for (let i = 0; i < 7; i++)
      for (let j = 0; j < 7; j++) {
        bx.push(4 + i / 3);
        by.push(4 + j / 3);
      }
    const A: CellCloud = { xs: [5, 15], ys: [5, 15] };
    const grid = await computeTcmRender(
      A,
      { xs: bx, ys: by },
      {
        width: 40,
        height: 40,
        bbox: BBOX,
        radius: 5,
        sigma: 1,
        alpha: 5,
        kernel: TOPHAT,
        markWidth: 128,
        markHeight: 128,
      },
    );
    const at = (x: number, y: number) => grid[y * 40 + x]!;
    expect(at(10, 10)).toBeGreaterThan(0); // A at (5,5), buried in the B cluster
    expect(at(30, 30)).toBeLessThan(0); // A at (15,15), no B within the radius
  });
});
