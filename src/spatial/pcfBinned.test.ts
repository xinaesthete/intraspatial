import { describe, expect, it } from "vitest";
import { mulberry32 } from "./kernelAnalysis";
import { crossPCF, crossPCFMatrix, crossPCFMatrixBinned, type LabelledCells } from "./pcf";

// `crossPCFMatrixBinned` sits between two functions that already exist and are already tested, and
// the cheapest way to trust it is to pin it to both: collapse its bins and it must be
// `crossPCFMatrix`; slice out one type pair and it must be `crossPCF`. The edge-correction tests
// then check the thing neither of those covers, against the property the correction exists to
// restore — that a uniform (CSR) cloud reads g = 1 everywhere, including next to the boundary.

function uniformCloud(n: number, w: number, h: number, nTypes: number, seed: number): LabelledCells {
  const rnd = mulberry32(seed);
  const xs: number[] = [];
  const ys: number[] = [];
  const typeId: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(rnd() * w);
    ys.push(rnd() * h);
    typeId.push(Math.floor(rnd() * nTypes));
  }
  return { xs, ys, typeId };
}

function subset(c: LabelledCells, t: number) {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < c.xs.length; i++) {
    if (c.typeId[i] === t) {
      xs.push(c.xs[i]!);
      ys.push(c.ys[i]!);
    }
  }
  return { xs, ys };
}

describe("crossPCFMatrixBinned", () => {
  const bbox = [0, 0, 400, 300] as const;
  const cells = uniformCloud(3000, 400, 300, 3, 0xc0ffee);

  it("reduces to crossPCFMatrix with a single bin", () => {
    const radius = 25;
    const binned = crossPCFMatrixBinned(cells, { bbox, rMax: radius, nBins: 1, nTypes: 3 });
    const disk = crossPCFMatrix(cells, { bbox, radius });
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) {
        expect(binned.g[a * 3 * 1 + b * 1]!).toBeCloseTo(disk.g[a * 3 + b]!, 10);
      }
    }
  });

  it("agrees with per-pair crossPCF, bin for bin", () => {
    const rMax = 60;
    const nBins = 6;
    const binned = crossPCFMatrixBinned(cells, { bbox, rMax, nBins, nTypes: 3 });
    for (const [a, b] of [
      [0, 1],
      [1, 0],
      [2, 2],
    ] as const) {
      const pair = crossPCF(subset(cells, a), subset(cells, b), { bbox, rMax, nBins });
      for (let k = 0; k < nBins; k++) {
        // The a == b case is the one that can differ: crossPCF takes two independent clouds and so
        // counts every self-pair, while the batched pass excludes j === i. Compare only off-diagonal
        // exactly, and allow the diagonal its N self-pairs in bin 0.
        if (a === b && k === 0) continue;
        expect(binned.g[a * 3 * nBins + b * nBins + k]!).toBeCloseTo(pair.g[k]!, 9);
      }
    }
  });

  it("counts every ordered in-range pair exactly once", () => {
    const rMax = 40;
    const nBins = 4;
    const r = crossPCFMatrixBinned(cells, { bbox, rMax, nBins, nTypes: 3 });
    let total = 0;
    for (const v of r.pairs) total += v;
    // Brute force, same predicate.
    let brute = 0;
    for (let i = 0; i < cells.xs.length; i++) {
      for (let j = 0; j < cells.xs.length; j++) {
        if (i === j) continue;
        const dx = cells.xs[j]! - cells.xs[i]!;
        const dy = cells.ys[j]! - cells.ys[i]!;
        if (dx * dx + dy * dy < rMax * rMax) brute++;
      }
    }
    expect(total).toBe(brute);
  });

  it("agrees with edge-corrected crossPCF pair by pair", () => {
    // Ties the single-pair path to the batched one under the correction too — the batched one is
    // what the 74,567-comparison parity sweep validated, so this is how that result reaches the
    // function the g(r) panel actually calls.
    const rMax = 60;
    const nBins = 6;
    const binned = crossPCFMatrixBinned(cells, { bbox, rMax, nBins, nTypes: 3, edgeCorrected: true });
    for (const [a, b] of [
      [0, 1],
      [2, 0],
    ] as const) {
      const pair = crossPCF(subset(cells, a), subset(cells, b), { bbox, rMax, nBins, edgeCorrected: true });
      for (let k = 0; k < nBins; k++) {
        expect(binned.g[a * 3 * nBins + b * nBins + k]!).toBeCloseTo(pair.g[k]!, 9);
      }
    }
  });

  it("keeps a fixed type axis when nTypes is given", () => {
    // Type 3 has no cells anywhere; it must still occupy a row and a column.
    const r = crossPCFMatrixBinned(cells, { bbox, rMax: 30, nBins: 2, nTypes: 5 });
    expect(r.nTypes).toBe(5);
    expect(r.counts[3]).toBe(0);
    expect(r.counts[4]).toBe(0);
    expect(r.g.length).toBe(5 * 5 * 2);
  });

  it("rejects a type id outside the declared axis", () => {
    expect(() => crossPCFMatrixBinned(cells, { bbox, rMax: 10, nBins: 1, nTypes: 2 })).toThrow(/outside/);
  });

  it("honours an explicit roiArea for ρ_B", () => {
    const base = crossPCFMatrixBinned(cells, { bbox, rMax: 30, nBins: 2, nTypes: 3 });
    const doubled = crossPCFMatrixBinned(cells, { bbox, rMax: 30, nBins: 2, nTypes: 3, roiArea: 2 * 400 * 300 });
    // ρ_B halves, so g doubles.
    for (let i = 0; i < base.g.length; i++) expect(doubled.g[i]!).toBeCloseTo(2 * base.g[i]!, 9);
  });
});

describe("crossPCFMatrixBinned — edge correction", () => {
  it("is a no-op when every anchor is further than rMax from the boundary", () => {
    // All cells in the middle third of a large ROI, rMax small enough that no annulus is cut.
    const rnd = mulberry32(7);
    const xs: number[] = [];
    const ys: number[] = [];
    const typeId: number[] = [];
    for (let i = 0; i < 800; i++) {
      xs.push(300 + rnd() * 400);
      ys.push(300 + rnd() * 400);
      typeId.push(i % 2);
    }
    const cells = { xs, ys, typeId };
    const bbox = [0, 0, 1000, 1000] as const;
    const p = { bbox, rMax: 50, nBins: 5, nTypes: 2 } as const;
    const plain = crossPCFMatrixBinned(cells, p);
    const corrected = crossPCFMatrixBinned(cells, { ...p, edgeCorrected: true });
    for (let i = 0; i < plain.g.length; i++) expect(corrected.g[i]!).toBeCloseTo(plain.g[i]!, 12);
  });

  it("restores g = 1 for a CSR cloud, where the uncorrected estimator reads low", () => {
    // A small ROI on purpose: the bias scales as perimeter·r/area, so it is only visible when the
    // boundary is a meaningful fraction of the domain. 200×200 with rMax 30 puts most cells within
    // reach of an edge.
    const cells = uniformCloud(6000, 200, 200, 1, 0xbeef);
    const bbox = [0, 0, 200, 200] as const;
    const p = { bbox, rMax: 30, nBins: 3, nTypes: 1 } as const;
    const plain = crossPCFMatrixBinned(cells, p);
    const corrected = crossPCFMatrixBinned(cells, { ...p, edgeCorrected: true });

    for (let k = 0; k < 3; k++) {
      // Uncorrected: biased low everywhere, and the correction must move every bin upward.
      expect(plain.g[k]!).toBeLessThan(0.97);
      expect(corrected.g[k]!).toBeGreaterThan(plain.g[k]!);
      // Corrected: unbiased to within sampling noise.
      expect(corrected.g[k]!).toBeGreaterThan(0.98);
      expect(corrected.g[k]!).toBeLessThan(1.02);
    }
    // The uncorrected bias grows with radius — more of a wider annulus falls outside the ROI — which
    // is the signature that distinguishes this from a constant scale error. Measured here:
    // 0.957 → 0.899 → 0.843, against a corrected 1.001 → 0.996 → 0.996.
    expect(plain.g[1]!).toBeLessThan(plain.g[0]!);
    expect(plain.g[2]!).toBeLessThan(plain.g[1]!);
    expect(plain.g[0]! - plain.g[2]!).toBeGreaterThan(0.05);
  });

  it("bias grows with the perimeter-to-area ratio", () => {
    // Same density and rMax, two ROI sizes. The smaller ROI must be further from 1.
    const mk = (side: number, n: number) => {
      const cells = uniformCloud(n, side, side, 1, 0x1234);
      const bbox = [0, 0, side, side] as const;
      return crossPCFMatrixBinned(cells, { bbox, rMax: 20, nBins: 1, nTypes: 1 }).g[0]!;
    };
    const small = mk(150, 2250); // density 0.1
    const large = mk(600, 36000);
    expect(1 - small).toBeGreaterThan(1 - large);
    expect(1 - large).toBeGreaterThan(0);
  });
});
