import { describe, expect, it } from "vite-plus/test";
import { domainAnnulusAreasInto, domainArea, domainContains, inverseAnnulusAreas, type MaskDomain, maskDomain, rectDomain } from "./domain";
import { annulusAreasInto, diskRectArea } from "./edgeCorrection";
import { mulberry32 } from "./kernelAnalysis";
import { crossPCF, crossPCFBinMembership, crossPCFMatrixBinned, type LabelledCells } from "./pcf";

// A mask domain is checked three ways, each against something independent of it:
//   - a pixel-aligned rectangle mask must reproduce `edgeCorrection.ts` to rounding, since the pixel
//     union then IS the rectangle — this pins the bin-edge walk;
//   - a tilted rectangle and a disk, whose clipped areas have closed forms, must stay within the
//     staircase error measured in docs/stats-on-arbitrary-domains.md (sources 2 and 3);
//   - end to end, a tilted ROI must give the same g as the exact rectangle estimator run in the
//     ROI's own frame, where rotation leaves every distance unchanged.

/** Pixel-centre rasterisation, row 0 at the top — the convention `MaskDomain` documents. */
function rasterise(
  bbox: readonly [number, number, number, number],
  width: number,
  height: number,
  inside: (x: number, y: number) => boolean,
): MaskDomain {
  const [minX, minY, maxX, maxY] = bbox;
  const pw = (maxX - minX) / width;
  const ph = (maxY - minY) / height;
  const data = new Uint8Array(width * height);
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) data[j * width + i] = inside(minX + (i + 0.5) * pw, maxY - (j + 0.5) * ph) ? 1 : 0;
  }
  return maskDomain(bbox, width, height, data);
}

/** |disk(r) ∩ disk(R)| for centres `d` apart. */
function lens(r: number, R: number, d: number): number {
  if (r <= 0 || d >= r + R) return 0;
  if (d <= Math.abs(R - r)) return Math.PI * Math.min(r, R) ** 2;
  const a = r * r * Math.acos((d * d + r * r - R * R) / (2 * d * r));
  const b = R * R * Math.acos((d * d + R * R - r * r) / (2 * d * R));
  return a + b - 0.5 * Math.sqrt((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R));
}

/** A square of side `side` centred on the origin, rotated by `deg`: membership, world → frame. */
function tiltedSquare(side: number, deg: number) {
  const t = (deg * Math.PI) / 180;
  const c = Math.cos(t);
  const s = Math.sin(t);
  const toFrame = (x: number, y: number): [number, number] => [c * x + s * y, -s * x + c * y];
  const fromFrame = (u: number, v: number): [number, number] => [c * u - s * v, s * u + c * v];
  const half = side / 2;
  const inside = (x: number, y: number) => {
    const [u, v] = toFrame(x, y);
    return Math.abs(u) <= half && Math.abs(v) <= half;
  };
  return { half, inside, toFrame, fromFrame };
}

describe("rectDomain / maskDomain", () => {
  it("refuses shapes that would be read as a different region", () => {
    expect(() => rectDomain([0, 0, 0, 10])).toThrow(/positive extent/);
    expect(() => maskDomain([0, 0, 10, 10], 4, 4, new Uint8Array(15))).toThrow(/15 pixels/);
    expect(() => maskDomain([0, 0, 10, 10], 2.5, 4, new Uint8Array(10))).toThrow(/raster shape/);
  });

  it("reads row 0 as the top, and measures the area of its set pixels", () => {
    // 4×2 raster over [0,0,8,4]: pixels are 2×2. Only the top-left pixel is set.
    const m = maskDomain([0, 0, 8, 4], 4, 2, Uint8Array.from([1, 0, 0, 0, 0, 0, 0, 0]));
    expect(domainArea(m)).toBe(4);
    expect(domainContains(m, 1, 3)).toBe(true);
    expect(domainContains(m, 1, 1)).toBe(false);
    expect(domainContains(m, 3, 3)).toBe(false);
    expect(domainContains(m, -0.1, 3)).toBe(false);
    // Max edges are closed: a point on maxX / maxY of a set pixel is inside.
    const full = maskDomain([0, 0, 8, 4], 4, 2, new Uint8Array(8).fill(1));
    expect(domainContains(full, 8, 0)).toBe(true);
    expect(domainContains(full, 0, 4)).toBe(true);
  });
});

describe("domainAnnulusAreasInto on a mask", () => {
  it.each([
    ["square pixels", 100, 80],
    ["non-square pixels", 50, 160],
  ])("reproduces the exact rectangle clip when the mask is a pixel-aligned rectangle (%s)", (_, width, height) => {
    // Raster over [0,0,100,80]; set exactly [10,20]–[70,60], which falls on pixel edges for both
    // pixel shapes, so the pixel union is the rectangle and the two must agree to rounding.
    const rect = [10, 20, 70, 60] as const;
    const mask = rasterise([0, 0, 100, 80], width, height, (x, y) => x > rect[0] && x < rect[2] && y > rect[1] && y < rect[3]);
    expect(domainArea(mask)).toBeCloseTo(60 * 40, 9);
    const nBins = 4;
    const dr = 6;
    const got = new Float64Array(nBins);
    const want = new Float64Array(nBins);
    let worst = 0;
    // Anchors across the whole rectangle, its corners and a little outside it.
    for (let x = 5; x <= 75; x += 2.3) {
      for (let y = 15; y <= 65; y += 1.7) {
        domainAnnulusAreasInto(got, mask, x, y, dr, nBins);
        annulusAreasInto(want, x, y, dr, nBins, ...rect);
        for (let k = 0; k < nBins; k++) worst = Math.max(worst, Math.abs(got[k]! - want[k]!));
      }
    }
    const outerArea = Math.PI * (nBins * dr) ** 2;
    expect(worst / outerArea).toBeLessThan(1e-9);
  });

  it("takes the full-annulus fast path only where it is exact", () => {
    const mask = rasterise([0, 0, 200, 200], 200, 200, (x, y) => Math.hypot(x - 100, y - 100) < 90);
    const weights = inverseAnnulusAreas(mask, 30, 3);
    const areas = new Float64Array(3);
    let worst = 0;
    for (const [x, y] of [
      [100, 100], // deep inside: fast path
      [130, 110], // boundary 30-odd away: near the threshold
      [170, 100], // cut
      [100, 15], // cut hard
    ] as const) {
      const inv = weights(x, y);
      domainAnnulusAreasInto(areas, mask, x, y, 10, 3);
      for (let k = 0; k < 3; k++) worst = Math.max(worst, Math.abs(inv[k]! * areas[k]! - 1));
    }
    expect(worst).toBeLessThan(1e-9);
  });

  it("does not take the fast path past a single outside pixel", () => {
    // A full raster with one hole: the only thing that can cut the anchor's disk is that pixel.
    const mask = rasterise([0, 0, 100, 100], 100, 100, (x, y) => !(x > 50 && x < 51 && y > 50 && y < 51));
    const inv = inverseAnnulusAreas(mask, 30, 3)(56, 45);
    const areas = new Float64Array(3);
    domainAnnulusAreasInto(areas, mask, 56, 45, 10, 3);
    expect(Math.abs(inv[0]! * areas[0]! - 1)).toBeLessThan(1e-9);
    expect(areas[0]).toBeLessThan(Math.PI * 100 - 0.99);
  });

  // Tolerances: about twice the RMS and max measured over 1500 anchors at 1 µm pixels by
  // `pnpm bench:domain-mask --all` (docs/stats-on-arbitrary-domains.md, source 2), bins
  // [0,10) [10,20) [20,30).
  const cases = [
    {
      name: "a square tilted 30°",
      ...(() => {
        const sq = tiltedSquare(800, 30);
        return {
          inside: sq.inside,
          dist: (x: number, y: number) => {
            const [u, v] = sq.toFrame(x, y);
            return Math.min(sq.half - Math.abs(u), sq.half - Math.abs(v));
          },
          exact: (x: number, y: number, r: number) => {
            const [u, v] = sq.toFrame(x, y);
            return diskRectArea(u, v, r, -sq.half, -sq.half, sq.half, sq.half);
          },
        };
      })(),
      extent: 600,
      rms: [0.0025, 0.0015, 0.001],
      max: [0.012, 0.007, 0.004],
    },
    {
      name: "a disk of radius 100",
      inside: (x: number, y: number) => Math.hypot(x, y) <= 100,
      dist: (x: number, y: number) => 100 - Math.hypot(x, y),
      exact: (x: number, y: number, r: number) => lens(r, 100, Math.hypot(x, y)),
      extent: 140,
      rms: [0.006, 0.0035, 0.003],
      max: [0.04, 0.015, 0.012],
    },
  ];

  it.each(cases)("stays within the measured staircase error on $name", (c) => {
    // A grid phase that is not a round number, so no edge sits on a pixel boundary by luck.
    const o = 0.37;
    const side = 2 * c.extent;
    const mask = rasterise([-c.extent + o, -c.extent + o, c.extent + o, c.extent + o], side, side, c.inside);
    const rnd = mulberry32(0xd0a1);
    const N = 300;
    const sum = [0, 0, 0];
    const sum2 = [0, 0, 0];
    const max = [0, 0, 0];
    const got = new Float64Array(3);
    for (let n = 0; n < N;) {
      const x = (rnd() * 2 - 1) * c.extent;
      const y = (rnd() * 2 - 1) * c.extent;
      if (!c.inside(x, y) || c.dist(x, y) >= 30) continue;
      n++;
      domainAnnulusAreasInto(got, mask, x, y, 10, 3);
      for (let k = 0; k < 3; k++) {
        const want = c.exact(x, y, 10 * (k + 1)) - c.exact(x, y, 10 * k);
        const e = (got[k]! - want) / want;
        sum[k]! += e;
        sum2[k]! += e * e;
        max[k] = Math.max(max[k]!, Math.abs(e));
      }
    }
    const report = [0, 1, 2].map((k) => ({
      mean: Math.abs(sum[k]! / N) < 0.001,
      rms: Math.sqrt(sum2[k]! / N) < c.rms[k]!,
      max: max[k]! < c.max[k]!,
    }));
    expect(report).toEqual([0, 1, 2].map(() => ({ mean: true, rms: true, max: true })));
  });
});

describe("crossPCF / crossPCFMatrixBinned with a domain", () => {
  /** CSR cells of `nTypes` types inside `inside`, drawn over `[-extent, extent]²`. */
  function csrIn(inside: (x: number, y: number) => boolean, extent: number, n: number, nTypes: number, seed: number): LabelledCells {
    const rnd = mulberry32(seed);
    const xs: number[] = [];
    const ys: number[] = [];
    const typeId: number[] = [];
    while (xs.length < n) {
      const x = (rnd() * 2 - 1) * extent;
      const y = (rnd() * 2 - 1) * extent;
      if (!inside(x, y)) continue;
      xs.push(x);
      ys.push(y);
      typeId.push(xs.length % nTypes);
    }
    return { xs, ys, typeId };
  }

  it("matches the exact estimator on a tilted ROI, run in the ROI's own frame", () => {
    // ≈ the 648,000 µm² covid ROI, tilted 30°, 0.002 cells/µm² per type — the bench's source-3 case.
    const sq = tiltedSquare(805, 30);
    const cells = csrIn(sq.inside, 570, 2600, 2, 0x7a11);
    const mask = rasterise([-570.37, -570.37, 569.63, 569.63], 1140, 1140, sq.inside);
    const masked = crossPCFMatrixBinned(cells, { bbox: mask.bbox, domain: mask, rMax: 30, nBins: 3, nTypes: 2, edgeCorrected: true });

    // Rotating into the square's frame changes no distance, and there the ROI is a rectangle.
    const framed = { ...cells, xs: [] as number[], ys: [] as number[] };
    for (let i = 0; i < cells.xs.length; i++) {
      const [u, v] = sq.toFrame(cells.xs[i]!, cells.ys[i]!);
      framed.xs.push(u);
      framed.ys.push(v);
    }
    const exact = crossPCFMatrixBinned(framed, {
      bbox: [-sq.half, -sq.half, sq.half, sq.half],
      rMax: 30,
      nBins: 3,
      nTypes: 2,
      edgeCorrected: true,
    });

    // Measured over 10 seeds the ratio sits within 0.02%; a single seed also carries the few
    // staircase-band cells the mask drops, hence 0.5%.
    const ratios = Array.from(masked.g, (g, i) => g / exact.g[i]!);
    expect(ratios.every((q) => Math.abs(q - 1) < 0.005)).toBe(true);
    expect(masked.outside).toBeLessThan(10);
  });

  it("restores g ≈ 1 on a disk ROI, where the bounding-box clip reads low", () => {
    const inside = (x: number, y: number) => Math.hypot(x, y) <= 100;
    const cells = csrIn(inside, 100, 6000, 1, 0xd15c);
    const mask = rasterise([-110.37, -110.37, 109.63, 109.63], 220, 220, inside);
    const p = { bbox: mask.bbox, rMax: 30, nBins: 3, nTypes: 1, edgeCorrected: true } as const;
    const masked = crossPCFMatrixBinned(cells, { ...p, domain: mask });
    // Today's best for a round ROI: the true area, clipped to the bounding box.
    const boxed = crossPCFMatrixBinned(cells, { ...p, bbox: [-100, -100, 100, 100], roiArea: Math.PI * 100 * 100 });
    for (let k = 0; k < 3; k++) {
      expect(masked.g[k]!).toBeGreaterThan(0.98);
      expect(masked.g[k]!).toBeLessThan(1.02);
    }
    expect(boxed.g[2]!).toBeLessThan(0.95);
  });

  it("drops cells outside the domain from both populations, and counts them", () => {
    // Left half of [0,100]² is the domain. Every cell on the right is dropped.
    const mask = rasterise([0, 0, 100, 100], 100, 100, (x) => x < 50);
    const rnd = mulberry32(3);
    const mk = (n: number) => {
      const xs: number[] = [];
      const ys: number[] = [];
      for (let i = 0; i < n; i++) {
        xs.push(rnd() * 100);
        ys.push(rnd() * 100);
      }
      return { xs, ys };
    };
    const a = mk(400);
    const b = mk(400);
    const right = (c: { xs: number[] }) => c.xs.filter((x) => x >= 50).length;
    const p = { bbox: [0, 0, 100, 100] as const, domain: mask, rMax: 12, nBins: 3, edgeCorrected: true };
    const res = crossPCF(a, b, p);
    expect(res.outside).toBe(right(a) + right(b));

    // The same pair through the matrix form, as types 0 and 1, must be the same number.
    const cells: LabelledCells = { xs: [...a.xs, ...b.xs], ys: [...a.ys, ...b.ys], typeId: [...a.xs.map(() => 0), ...b.xs.map(() => 1)] };
    const mat = crossPCFMatrixBinned(cells, { ...p, nTypes: 2 });
    expect(mat.outside).toBe(res.outside);
    const g01 = Array.from(mat.g.subarray(3, 6));
    expect(g01.every((g, k) => Math.abs(g / res.g[k]! - 1) < 1e-9)).toBe(true);

    // And the highlight never lights a cell the curve did not use.
    const mem = crossPCFBinMembership(a, b, p);
    const litOutside = a.xs.some((x, i) => x >= 50 && mem.maskA[i] !== 0) || b.xs.some((x, j) => x >= 50 && mem.maskB[j] !== 0);
    expect(litOutside).toBe(false);
  });

  it("drops nothing and reports nothing without a domain", () => {
    const cells = csrIn(() => true, 50, 200, 2, 9);
    const res = crossPCFMatrixBinned(cells, { bbox: [-50, -50, 50, 50], rMax: 10, nBins: 2, nTypes: 2 });
    expect(res.outside).toBeUndefined();
  });
});
