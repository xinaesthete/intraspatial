import { describe, expect, it } from "vite-plus/test";
import {
  domainAnnulusAreasInto,
  domainArea,
  domainContains,
  inverseAnnulusAreas,
  type MaskDomain,
  maskDomain,
  maskWhere,
  polygonDomain,
  rectDomain,
} from "./domain";
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
    // …and does light the ones it did: a cell inside with a partner in range.
    expect(a.xs.some((x, i) => x < 50 && mem.maskA[i] !== 0)).toBe(true);
    expect(b.xs.some((x, j) => x < 50 && mem.maskB[j] !== 0)).toBe(true);
  });

  it("drops nothing and reports nothing without a domain", () => {
    const cells = csrIn(() => true, 50, 200, 2, 9);
    const res = crossPCFMatrixBinned(cells, { bbox: [-50, -50, 50, 50], rMax: 10, nBins: 2, nTypes: 2 });
    expect(res.outside).toBeUndefined();
  });
});

describe("maskWhere — masks from fields", () => {
  it("keeps the field's raster and bbox, inside where the predicate holds", () => {
    const field = { data: Float32Array.from([0.1, 0.9, 0.5, 0.7]), width: 2, height: 2, bbox: [0, 0, 2, 2] as const };
    const m = maskWhere(field, (v) => v > 0.6);
    expect(Array.from(m.data)).toEqual([0, 1, 0, 1]);
    expect(m.bbox).toBe(field.bbox);
    expect(domainArea(m)).toBe(2);
    expect(m.derivedFrom).toBeUndefined();
    expect(() => maskWhere({ ...field, width: 3 }, () => true)).toThrow(/4 values/);
  });

  it("flags exactly the pairs whose both members built the mask", () => {
    const cells = csrIn(() => true, 50, 300, 3, 5);
    const field = { data: new Float32Array(100).fill(1), width: 10, height: 10, bbox: [-50, -50, 50, 50] as const };
    const mask = maskWhere(field, () => true, { derivedFrom: [0, 2] });
    const res = crossPCFMatrixBinned(cells, { bbox: mask.bbox, domain: mask, rMax: 10, nBins: 2, nTypes: 3 });
    expect(Array.from(res.confounded ?? [])).toEqual([1, 0, 1, 0, 0, 0, 1, 0, 1]);
    // A drawn mask (no derivedFrom) and a rect flag nothing.
    expect(
      crossPCFMatrixBinned(cells, { bbox: mask.bbox, domain: maskWhere(field, () => true), rMax: 10, nBins: 2, nTypes: 3 }).confounded,
    ).toBeUndefined();
  });
});

// A polygon domain is clipped in closed form, so every check is to rounding against an exact answer:
// an axis-aligned rectangle against `edgeCorrection.ts`; a rotated one against the same, in its own
// frame; a hole against the difference of two rectangles.
describe("polygonDomain", () => {
  const box = (x0: number, y0: number, x1: number, y1: number): [number, number][] => [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1],
  ];
  const anchors: [number, number][] = [
    [50, 40], // deep inside
    [3, 40], // near the left edge
    [2, 1], // in a corner
    [97, 78], // the opposite corner
    [-5, 40], // just outside
    [130, 130], // far outside
  ];
  const areas = (d: Parameters<typeof domainAnnulusAreasInto>[1], x: number, y: number, dr: number, n: number) => {
    const out = new Float64Array(n);
    domainAnnulusAreasInto(out, d, x, y, dr, n);
    return out;
  };

  it("refuses rings that are not polygons", () => {
    expect(() => polygonDomain([])).toThrow(/no rings/);
    expect(() =>
      polygonDomain([
        [0, 0],
        [1, 0],
        [0, 0],
      ]),
    ).toThrow(/fewer than 3/);
    expect(() =>
      polygonDomain([
        [0, 0],
        [1, 1],
        [2, 2],
      ]),
    ).toThrow(/zero area/);
    expect(() =>
      polygonDomain([
        [0, 0],
        [1, Number.NaN],
        [0, 1],
      ]),
    ).toThrow(/not finite/);
  });

  it("reproduces the rectangle's annulus areas to rounding, at any anchor", () => {
    const poly = polygonDomain(box(0, 0, 100, 80));
    const rect = rectDomain([0, 0, 100, 80]);
    expect(domainArea(poly)).toBeCloseTo(8000, 9);
    for (const [x, y] of anchors) {
      const a = areas(poly, x, y, 7, 6);
      const b = areas(rect, x, y, 7, 6);
      expect(Math.max(...a.map((v, k) => Math.abs(v - b[k]!)))).toBeLessThan(1e-9);
    }
  });

  it("is exact on a rotated rectangle: the same areas as the axis-aligned one in its own frame", () => {
    const sq = tiltedSquare(80, 23);
    const poly = polygonDomain([sq.fromFrame(-40, -40), sq.fromFrame(40, -40), sq.fromFrame(40, 40), sq.fromFrame(-40, 40)]);
    const rect = rectDomain([-40, -40, 40, 40]);
    for (const [u, v] of [
      [0, 0],
      [37, 5],
      [38, -39],
      [-45, 0],
    ] as const) {
      const [x, y] = sq.fromFrame(u, v);
      const a = areas(poly, x, y, 6, 5);
      const b = areas(rect, u, v, 6, 5);
      expect(Math.max(...a.map((q, k) => Math.abs(q - b[k]!)))).toBeLessThan(1e-9);
      expect(domainContains(poly, x, y)).toBe(Math.abs(u) <= 40 && Math.abs(v) <= 40);
    }
  });

  it("reads a nested ring as a hole, whatever the rings' orientation or closing vertex", () => {
    const outer = box(0, 0, 100, 100);
    const hole = box(30, 30, 70, 60);
    const closed = (r: [number, number][]) => [...r, r[0]!];
    const variants = [
      polygonDomain([outer, hole]),
      polygonDomain([[...outer].reverse(), [...hole].reverse()]),
      polygonDomain([closed(hole), closed(outer)]),
    ];
    const big = rectDomain([0, 0, 100, 100]);
    const small = rectDomain([30, 30, 70, 60]);
    for (const d of variants) {
      expect(domainArea(d)).toBeCloseTo(10000 - 1200, 9);
      expect(domainContains(d, 50, 45)).toBe(false);
      expect(domainContains(d, 10, 45)).toBe(true);
      for (const [x, y] of [
        [50, 45],
        [25, 45],
        [5, 5],
      ] as const) {
        const a = areas(d, x, y, 5, 6);
        const outerA = areas(big, x, y, 5, 6);
        const holeA = areas(small, x, y, 5, 6);
        expect(Math.max(...a.map((q, k) => Math.abs(q - (outerA[k]! - holeA[k]!))))).toBeLessThan(1e-9);
      }
    }
  });

  it("takes the full-annulus fast path only where nothing is clipped", () => {
    const d = polygonDomain([box(0, 0, 100, 100), box(40, 40, 60, 60)]);
    const inv = inverseAnnulusAreas(d, 12, 4);
    const rnd = mulberry32(0x9a7);
    for (let t = 0; t < 300; t++) {
      const x = rnd() * 100;
      const y = rnd() * 100;
      const want = areas(d, x, y, 3, 4).map((a) => (a > 0 ? 1 / a : 0));
      const got = inv(x, y);
      expect(Math.max(...want.map((w, k) => Math.abs(w - got[k]!) / Math.max(w, 1e-12)))).toBeLessThan(1e-9);
    }
  });

  it("finds only the near edges on a many-vertex gate, exactly", () => {
    // A rotated rectangle with a rectangular hole, each side cut into many short collinear edges:
    // still exactly rectangle − rectangle in its own frame, but now the edge grid and the crossing
    // bands decide which of 360 edges each anchor sees.
    const sq = tiltedSquare(1, 17);
    const ring = (x0: number, y0: number, x1: number, y1: number, perSide: number): [number, number][] => {
      const corners: [number, number][] = [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
      ];
      const out: [number, number][] = [];
      for (let c = 0; c < 4; c++) {
        const [ax, ay] = corners[c]!;
        const [bx, by] = corners[(c + 1) % 4]!;
        for (let i = 0; i < perSide; i++) out.push(sq.fromFrame(ax + ((bx - ax) * i) / perSide, ay + ((by - ay) * i) / perSide));
      }
      return out;
    };
    const d = polygonDomain([ring(-100, -80, 100, 80, 60), ring(10, -30, 50, 20, 30)]);
    const outer = rectDomain([-100, -80, 100, 80]);
    const hole = rectDomain([10, -30, 50, 20]);
    const rnd = mulberry32(0x6a7e);
    let worst = 0;
    for (let t = 0; t < 200; t++) {
      const u = (rnd() * 2 - 1) * 110;
      const v = (rnd() * 2 - 1) * 90;
      const [x, y] = sq.fromFrame(u, v);
      // Random points land on an edge with probability zero, so the open sets decide.
      const inOuter = Math.abs(u) < 100 && Math.abs(v) < 80;
      const inHole = u > 10 && u < 50 && v > -30 && v < 20;
      expect(domainContains(d, x, y)).toBe(inOuter && !inHole);
      const a = areas(d, x, y, 4, 5);
      const o = areas(outer, u, v, 4, 5);
      const h = areas(hole, u, v, 4, 5);
      for (let k = 0; k < 5; k++) worst = Math.max(worst, Math.abs(a[k]! - (o[k]! - h[k]!)));
    }
    expect(worst).toBeLessThan(1e-9);
  });

  it("restores g ≈ 1 on a round ROI given as a 96-gon, with no staircase", () => {
    const ring: [number, number][] = Array.from({ length: 96 }, (_, i) => [
      100 * Math.cos((2 * Math.PI * i) / 96),
      100 * Math.sin((2 * Math.PI * i) / 96),
    ]);
    const d = polygonDomain(ring);
    const cells = csrIn((x, y) => domainContains(d, x, y), 100, 6000, 1, 0xd15c);
    const res = crossPCFMatrixBinned(cells, { bbox: d.bbox, domain: d, rMax: 30, nBins: 3, nTypes: 1, edgeCorrected: true });
    expect(res.outside).toBe(0);
    for (let k = 0; k < 3; k++) {
      expect(res.g[k]!).toBeGreaterThan(0.98);
      expect(res.g[k]!).toBeLessThan(1.02);
    }
  });

  it("matches the exact rectangle estimator on a tilted ROI, run in the ROI's own frame", () => {
    const sq = tiltedSquare(805, 30);
    const h = sq.half;
    const d = polygonDomain([sq.fromFrame(-h, -h), sq.fromFrame(h, -h), sq.fromFrame(h, h), sq.fromFrame(-h, h)]);
    const cells = csrIn(sq.inside, 570, 2600, 2, 0x7a11);
    const p = { rMax: 30, nBins: 3, nTypes: 2, edgeCorrected: true } as const;
    const poly = crossPCFMatrixBinned(cells, { ...p, bbox: d.bbox, domain: d });
    const framed = { ...cells, xs: [] as number[], ys: [] as number[] };
    for (let i = 0; i < cells.xs.length; i++) {
      const [u, v] = sq.toFrame(cells.xs[i]!, cells.ys[i]!);
      framed.xs.push(u);
      framed.ys.push(v);
    }
    const exact = crossPCFMatrixBinned(framed, { ...p, bbox: [-h, -h, h, h] });
    // The mask version needs 0.5% for its staircase; the polygon has none.
    expect(poly.outside).toBe(0);
    expect(Array.from(poly.g, (g, i) => Math.abs(g / exact.g[i]! - 1)).every((e) => e < 1e-9)).toBe(true);
  });
});
