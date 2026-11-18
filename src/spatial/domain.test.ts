import { describe, expect, it } from "vite-plus/test";
import {
  domainAnnulusAreasInto,
  domainArea,
  domainContains,
  type FillRule,
  inverseAnnulusAreas,
  type MaskDomain,
  maskDomain,
  maskWhere,
  normaliseRings,
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

  it("rejects self-crossing and mutually crossing rings, naming the edges", () => {
    const figureEight: [number, number][] = [
      [0, 0],
      [10, 10],
      [10, 0],
      [0, 10],
    ];
    expect(() => polygonDomain(figureEight)).toThrow(/edges cross \(ring 0 edge \d and ring 0 edge \d/);
    expect(() => polygonDomain([box(0, 0, 10, 10), box(5, 5, 15, 15)])).toThrow(/ring 0 edge \d and ring 1 edge \d/);
    // Collinear overlap of two rings' edges is a crossing too.
    expect(() => polygonDomain([box(0, 0, 10, 10), box(5, 0, 15, 10)])).toThrow(/edges cross/);
  });

  it("accepts rings that only touch at a vertex, and nested or disjoint rings", () => {
    // Two squares sharing a single corner point.
    expect(() => polygonDomain([box(0, 0, 10, 10), box(10, 10, 20, 20)])).not.toThrow();
    // A pinched ring touching itself at a vertex (two triangles joined at (5, 5)).
    expect(() =>
      polygonDomain([
        [0, 0],
        [5, 5],
        [10, 0],
        [10, 10],
        [5, 5],
        [0, 10],
      ]),
    ).not.toThrow();
    expect(() => polygonDomain([box(0, 0, 100, 100), box(40, 40, 60, 60)])).not.toThrow();
    expect(() => polygonDomain([box(0, 0, 10, 10), box(20, 0, 30, 10)])).not.toThrow();
  });

  it("reads a hole as a hole when its vertices and edge midpoints all touch the shell", () => {
    const hole: [number, number][] = [
      [8, 0],
      [0, 8],
      [0, -8],
    ];
    const shell: [number, number][] = [
      [8, 0],
      [7, 3],
      [4, 4],
      [3, 7],
      [0, 8],
      [-2, 4],
      [0, 0],
      [-2, -4],
      [0, -8],
      [3, -7],
      [4, -4],
      [7, -3],
    ];
    const d = polygonDomain([shell, hole]);
    expect(domainArea(d)).toBeCloseTo(32, 9);
  });

  it("keeps the area of a ring far from the origin", () => {
    const n = 2 ** 27;
    expect(domainArea(polygonDomain(box(n, n, n + 1, n + 1)))).toBe(1);
    expect(() =>
      polygonDomain(
        [
          [n, n + 1],
          [n + 1, n + 2],
          [n + 5, n + 6],
        ],
        { fillRule: "evenodd" },
      ),
    ).toThrow(/zero area/);
  });

  it("can be mapped over shapes directly", () => {
    const ds = [box(0, 0, 1, 1), box(0, 0, 2, 2)].map(polygonDomain);
    expect(ds.map(domainArea)).toEqual([1, 4]);
  });

  it("reads a hole whose every vertex touches the outer ring as a hole", () => {
    // Each of the triangle's vertices lies on the square, so none of them can say which side of
    // the square the triangle is on; an edge midpoint can.
    const hole: [number, number][] = [
      [0, 5],
      [5, 0],
      [10, 5],
    ];
    for (const h of [hole, [...hole].reverse()]) {
      const d = polygonDomain([box(0, 0, 10, 10), h]);
      expect(domainArea(d)).toBeCloseTo(75, 9);
      expect(domainContains(d, 5, 3)).toBe(false);
      expect(domainContains(d, 5, 8)).toBe(true);
    }
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

describe("normaliseRings / polygonDomain with a fillRule", () => {
  type Pt = [number, number];
  const box = (x0: number, y0: number, x1: number, y1: number): Pt[] => [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1],
  ];
  const figureEight: Pt[] = [
    [0, 0],
    [10, 10],
    [10, 0],
    [0, 10],
  ];

  /** Winding number of the raw input rings at a point, by Sunday's rule: the reference. */
  const winding = (rings: readonly Pt[][], x: number, y: number) => {
    let w = 0;
    for (const ring of rings) {
      for (let i = 0; i < ring.length; i++) {
        const [ax, ay] = ring[i]!;
        const [bx, by] = ring[(i + 1) % ring.length]!;
        const side = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
        if (ay <= y) {
          if (by > y && side > 0) w++;
        } else if (by <= y && side < 0) w--;
      }
    }
    return w;
  };
  const filled = (rule: FillRule, w: number) => (rule === "evenodd" ? Math.abs(w) % 2 === 1 : w !== 0);

  it("leaves polygonDomain strict when no rule is given", () => {
    expect(() => polygonDomain(figureEight)).toThrow(/edges cross/);
  });

  it("splits a figure-eight into two rings touching at the crossing, under either rule", () => {
    for (const rule of ["evenodd", "nonzero"] as const) {
      const rings = normaliseRings(figureEight, rule);
      expect(rings).toHaveLength(2);
      for (const r of rings) expect(r.some(([x, y]) => x === 5 && y === 5)).toBe(true);
      const d = polygonDomain(figureEight, { fillRule: rule });
      expect(domainArea(d)).toBeCloseTo(50, 9);
      expect(domainArea(polygonDomain(rings))).toBeCloseTo(50, 9);
      expect(domainContains(d, 2, 5)).toBe(true);
      expect(domainContains(d, 5, 2)).toBe(false);
    }
  });

  it("counts a loop drawn twice as outside under evenodd, and as one loop under nonzero", () => {
    const twice = [...box(0, 0, 10, 10), ...box(0, 0, 10, 10)];
    expect(() => normaliseRings(twice, "evenodd")).toThrow(/normaliseRings: the rings enclose no area under the evenodd fill rule/);
    expect(() => polygonDomain(twice, { fillRule: "evenodd" })).toThrow(/polygonDomain: the rings enclose no area/);
    expect(domainArea(polygonDomain(twice, { fillRule: "nonzero" }))).toBeCloseTo(100, 9);
  });

  it("merges overlaps under nonzero and cuts them out under evenodd, collinear edges included", () => {
    const crossing = [box(0, 0, 10, 10), box(5, 5, 15, 15)];
    expect(normaliseRings(crossing, "nonzero")).toHaveLength(1);
    expect(domainArea(polygonDomain(crossing, { fillRule: "nonzero" }))).toBeCloseTo(175, 9);
    expect(domainArea(polygonDomain(crossing, { fillRule: "evenodd" }))).toBeCloseTo(150, 9);
    const sharingEdges = [box(0, 0, 10, 10), box(5, 0, 15, 10)];
    expect(domainArea(polygonDomain(sharingEdges, { fillRule: "nonzero" }))).toBeCloseTo(150, 9);
    const evenodd = polygonDomain(sharingEdges, { fillRule: "evenodd" });
    expect(domainArea(evenodd)).toBeCloseTo(100, 9);
    expect(domainContains(evenodd, 7, 5)).toBe(false);
  });

  it("reads holes by orientation under nonzero and by nesting under evenodd", () => {
    const outer = box(0, 0, 100, 100);
    const sameWay = box(30, 30, 70, 60);
    const opposite = [...sameWay].reverse();
    expect(domainArea(polygonDomain([outer, sameWay], { fillRule: "evenodd" }))).toBeCloseTo(8800, 9);
    expect(domainArea(polygonDomain([outer, opposite], { fillRule: "evenodd" }))).toBeCloseTo(8800, 9);
    expect(domainArea(polygonDomain([outer, opposite], { fillRule: "nonzero" }))).toBeCloseTo(8800, 9);
    expect(domainArea(polygonDomain([outer, sameWay], { fillRule: "nonzero" }))).toBeCloseTo(10000, 9);
  });

  it("splits at a vertex resting on another ring's edge", () => {
    // The triangle's apex sits on the square's top edge and its base pokes out below.
    const rings = [
      box(0, 0, 10, 10),
      [
        [5, 10],
        [2, -5],
        [8, -5],
      ] as Pt[],
    ];
    for (const rule of ["evenodd", "nonzero"] as const) {
      const d = polygonDomain(rings, { fillRule: rule });
      for (const [x, y] of [
        [5, 8],
        [1, 8],
        [5, -3],
        [9, 1],
      ] as const) {
        expect(domainContains(d, x, y)).toBe(filled(rule, winding(rings, x, y)));
      }
    }
  });

  it("does not read two nearly collinear pieces of one line, end to end, as crossing", () => {
    // Found by a stress run: ring 1's last edge is cut into pieces whose orientation tests round
    // to opposite signs (about 1e-14) though the pieces lie end to end along the line.
    const rings: Pt[][] = [
      [
        [47.13645833544433, 1.2458717450499535],
        [31.42607077024877, 4.398097191005945],
        [85.61779940500855, 35.1692825788632],
        [34.559293603524566, 54.121679067611694],
        [4.316083365119994, 8.6185562890023],
      ],
      [
        [23.424670309759676, 55.76175753958523],
        [77.87560825236142, 71.65884026326239],
        [88.83472031448036, 40.65263450611383],
        [43.11143602244556, 17.354330141097307],
        [8.055235515348613, 60.02604081295431],
        [99.32845791336149, 1.1558493366464972],
        [20.16342217102647, 13.01232606638223],
      ],
      [
        [47.12962748017162, 73.5728369327262],
        [3.2537611667066813, 23.406186304055154],
        [26.933959568850696, 70.11317301075906],
        [86.21231366414577, 2.5026755640283227],
      ],
    ];
    for (const rule of ["evenodd", "nonzero"] as const) expect(() => polygonDomain(normaliseRings(rings, rule))).not.toThrow();
  });

  it("returns simple rings when a hole touches its shell, splitting the walk there", () => {
    const rings = normaliseRings(
      [
        box(0, 0, 10, 10),
        [
          [5, 0],
          [7, 4],
          [3, 4],
        ] as Pt[],
      ],
      "evenodd",
    );
    expect(rings).toHaveLength(2);
    for (const r of rings) expect(new Set(r.map(([x, y]) => `${x},${y}`)).size).toBe(r.length);
    const d = polygonDomain(rings);
    expect(domainArea(d)).toBeCloseTo(92, 9);
    expect(domainContains(d, 5, 2)).toBe(false);
  });

  it("keeps a small ring that lies far from the others", () => {
    const rings = [box(0, 0, 1, 1), box(1e11, 0, 1e11 + 100, 100)];
    for (const rule of ["evenodd", "nonzero"] as const) {
      const d = polygonDomain(rings, { fillRule: rule });
      expect(domainContains(d, 0.5, 0.5)).toBe(true);
      expect(domainArea(d)).toBeCloseTo(10001, 3);
    }
  });

  it("visits each crossing once however many grid cells the two edges share", () => {
    // 200 figure-eights drawn over each other: every diagonal crosses every opposite one at the
    // centre, across a grid of cells, which used to record the crossing once per shared cell.
    const ring: Pt[] = [];
    for (let i = 0; i < 200; i++) ring.push([0, 0], [100, 100], [100, 0], [0, 100]);
    const d = polygonDomain(ring, { fillRule: "nonzero" });
    expect(domainArea(d)).toBeCloseTo(5000, 6);
  });

  it("seeds many separate components without scanning every piece for each", () => {
    const squares = Array.from({ length: 3000 }, (_, i) =>
      box(3 * (i % 60), 3 * Math.floor(i / 60), 3 * (i % 60) + 2, 3 * Math.floor(i / 60) + 2),
    );
    const d = polygonDomain(squares, { fillRule: "nonzero" });
    expect(domainArea(d)).toBeCloseTo(12000, 6);
  });

  it("still throws on bad input, with polygonDomain's messages", () => {
    for (const rule of ["evenodd", "nonzero"] as const) {
      expect(() => polygonDomain([], { fillRule: rule })).toThrow(/polygonDomain: no rings/);
      expect(() =>
        polygonDomain(
          [
            [0, 0],
            [1, 0],
            [0, 0],
          ],
          { fillRule: rule },
        ),
      ).toThrow(/polygonDomain: ring 0 has 2 distinct vertices, fewer than 3/);
      expect(() =>
        polygonDomain(
          [
            [0, 0],
            [1, Number.NaN],
            [0, 1],
          ],
          { fillRule: rule },
        ),
      ).toThrow(/polygonDomain: ring 0 vertex 1 is not finite/);
      expect(() =>
        polygonDomain(
          [
            [0, 0],
            [1, 1],
            [2, 2],
          ],
          { fillRule: rule },
        ),
      ).toThrow(/polygonDomain: a ring has zero area/);
      // Zero signed area that is not a figure-eight: out along a path and back.
      expect(() =>
        polygonDomain(
          [
            [0, 0],
            [4, 1],
            [8, 0],
            [4, 1],
          ],
          { fillRule: rule },
        ),
      ).toThrow(/a ring has zero area/);
      // A degenerate ring alongside a good one is still an error, not silently dropped.
      expect(() =>
        normaliseRings(
          [
            box(0, 0, 10, 10),
            [
              [20, 20],
              [21, 21],
              [22, 22],
            ],
          ],
          rule,
        ),
      ).toThrow(/normaliseRings: a ring has zero area/);
    }
  });

  it("leaves the input untouched and returns fresh, open rings", () => {
    const input = figureEight.map((p): Pt => [p[0], p[1]]);
    const copy = JSON.stringify(input);
    const rings = normaliseRings(input, "evenodd");
    expect(JSON.stringify(input)).toBe(copy);
    for (const r of rings) {
      expect(r[0]).not.toEqual(r[r.length - 1]);
      for (const p of r) expect(input).not.toContain(p);
    }
  });

  it("covers exactly the region the raw rings cover under each rule (property)", () => {
    const rnd = mulberry32(0xf111);
    const randomRing = (): Pt[] => {
      const n = 3 + Math.floor(rnd() * 8);
      const cx = rnd() * 60 + 20;
      const cy = rnd() * 60 + 20;
      // Half the rings are free scribbles (crossing freely), half are star-shaped with jitter.
      if (rnd() < 0.5) return Array.from({ length: n }, (): Pt => [cx + (rnd() - 0.5) * 60, cy + (rnd() - 0.5) * 60]);
      const sgn = rnd() < 0.5 ? 1 : -1;
      return Array.from({ length: n }, (_, i): Pt => {
        const a = (sgn * 2 * Math.PI * (i + rnd() * 0.5)) / n;
        const r = 5 + rnd() * 30;
        return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
      });
    };
    let mismatches = 0;
    let checked = 0;
    for (let trial = 0; trial < 300; trial++) {
      const rings = Array.from({ length: 1 + Math.floor(rnd() * 3) }, randomRing);
      for (const rule of ["evenodd", "nonzero"] as const) {
        // Random rings in general position always enclose area under either rule (empty regions
        // are tested deterministically above), so this must not throw.
        const d = polygonDomain(rings, { fillRule: rule });
        // The normalised rings also pass the strict constructor and give the same domain.
        const strict = polygonDomain(normaliseRings(rings, rule));
        expect(domainArea(strict)).toBeCloseTo(domainArea(d), 6);
        let hits = 0;
        for (let i = 0; i < 200; i++) {
          // The generator reaches [-15, 115]; sample past it on every side.
          const x = rnd() * 140 - 20;
          const y = rnd() * 140 - 20;
          const want = filled(rule, winding(rings, x, y));
          if (want) hits++;
          checked++;
          if (domainContains(d, x, y) !== want || domainContains(strict, x, y) !== want) mismatches++;
        }
        // Area against a grid count of the raw rings. Only boundary cells can disagree, so the
        // error is at most perimeter × cell size; orientation is pinned by the strict comparison.
        let inGrid = 0;
        const g = 140;
        const h = 140 / g;
        for (let j = 0; j < g; j++) {
          for (let i = 0; i < g; i++) if (filled(rule, winding(rings, (i + 0.5) * h - 20, (j + 0.5) * h - 20))) inGrid++;
        }
        let perimeter = 0;
        for (const r of rings)
          r.forEach(([x, y], i) => (perimeter += Math.hypot(r[(i + 1) % r.length]![0] - x, r[(i + 1) % r.length]![1] - y)));
        expect(Math.abs(domainArea(d) - inGrid * h * h)).toBeLessThan(perimeter * h);
        expect(hits === 0 || domainArea(d) > 0).toBe(true);
      }
    }
    // Random points fall on an edge with probability zero, so every sample must agree.
    expect(checked).toBeGreaterThan(50000);
    expect(mismatches).toBe(0);
  });
});
