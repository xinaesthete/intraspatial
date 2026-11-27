import { describe, expect, it } from "vite-plus/test";
import { type PolygonDomain, domainContains, polygonDomain } from "./domain";
import { histogramCpu } from "./histogram";
import {
  ZONAL_SLOT,
  ZONAL_TOTALS,
  polygonLayers,
  polygonRingBuffers,
  rasterisePolygonsCpu,
  zonalCollectCpu,
  zonalExactPlan,
  zonalExactQuantiles,
  zonalHistogramCpu,
  zonalMerge,
  zonalReferenceQuantile,
  zonalTable,
} from "./zonal";

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32;
}

const W = 40;
const H = 30;

/** Three regions: a blob, a ring-shaped region with a hole, a small corner; 0 elsewhere. */
function shapes() {
  return [
    polygonDomain([
      [2.3, 3.1],
      [17.8, 2.2],
      [14.4, 20.6],
      [4.1, 15.9],
    ]),
    polygonDomain([
      [
        [20.2, 4.4],
        [37.7, 4.1],
        [38.1, 26.3],
        [21.6, 27.2],
      ],
      [
        [25.1, 10.3],
        [32.9, 10.7],
        [31.2, 20.4],
      ],
    ]),
    polygonDomain([
      [0.2, 22.1],
      [6.9, 22.6],
      [3.3, 29.4],
    ]),
  ];
}

describe("polygon rasterising", () => {
  it("labels pixel centres by domainContains, highest id on overlap, through a frame", () => {
    const polys = shapes();
    const labels = rasterisePolygonsCpu(polys, { width: W, height: H });
    let inside = 0;
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const want = polys.reduce((best, p, k) => (domainContains(p, i + 0.5, j + 0.5) ? k + 1 : best), 0);
        if (labels[j * W + i] !== want) throw new Error(`pixel ${i},${j}: ${labels[j * W + i]} != ${want}`);
        if (want) inside++;
      }
    }
    expect(inside).toBeGreaterThan(300);
    // A 2× coarser raster offset by (10, 4): pixel (i, j)'s centre is (10 + 2i + 1, 4 + 2j + 1).
    const coarse = rasterisePolygonsCpu(polys, { width: 15, height: 12, originX: 10, originY: 4, scaleX: 2, scaleY: 2 });
    expect(coarse[3 * 15 + 7]).toBe(polys.findIndex((p) => domainContains(p, 10 + 15, 4 + 7)) + 1);
    // Overlap: a square over everything with a lower id loses, with a higher id wins.
    const big = polygonDomain([
      [0, 0],
      [40, 0],
      [40, 30],
      [0, 30],
    ]);
    expect(rasterisePolygonsCpu([big, ...polys], { width: W, height: H }).every((v, q) => v === (labels[q] ? labels[q]! + 1 : 1))).toBe(
      true,
    );
    expect(rasterisePolygonsCpu([...polys, big], { width: W, height: H }).every((v) => v === 4)).toBe(true);
  });

  it("ring buffers in the pixel frame give the same parity as the exact geometry", () => {
    const polys = shapes();
    const frame = { width: 15, height: 12, originX: 10, originY: 4, scaleX: 2, scaleY: 2 };
    const rb = polygonRingBuffers(polys, frame);
    expect(rb.count).toBe(2); // the corner triangle is outside this raster
    const labels = rasterisePolygonsCpu(polys, frame);
    for (let q = 0; q < frame.width * frame.height; q++) {
      const x = (q % frame.width) + 0.5;
      const y = Math.floor(q / frame.width) + 0.5;
      let best = 0;
      for (let p = 0; p < rb.count; p++) {
        const o = p * 8;
        let inside = false;
        for (let e = rb.polys[o]!; e < rb.polys[o]! + rb.polys[o + 1]!; e++) {
          const [x0, y0, x1, y1] = [0, 1, 2, 3].map((k) => rb.edges[e * 4 + k]!) as [number, number, number, number];
          if (y0 > y !== y1 > y && x < ((x1 - x0) * (y - y0)) / (y1 - y0) + x0) inside = !inside;
        }
        if (inside) best = Math.max(best, rb.polys[o + 2]!);
      }
      if (best !== labels[q]) throw new Error(`pixel ${q}: ${best} != ${labels[q]}`);
    }
  });

  it("splits a half-stride patch grid into four disjoint layers; touching boxes share one", () => {
    const patch = (x: number, y: number) =>
      polygonDomain([
        [x, y],
        [x + 8, y],
        [x + 8, y + 8],
        [x, y + 8],
      ]);
    const grid: PolygonDomain[] = [];
    for (let y = 0; y + 8 <= H; y += 4) for (let x = 0; x + 8 <= W; x += 4) grid.push(patch(x, y));
    const layers = polygonLayers(grid);
    expect(layers).toHaveLength(4);
    expect(layers.flat().sort((a, b) => a - b)).toEqual(grid.map((_, k) => k));
    for (const layer of layers) {
      const per = layer.map((k) => rasterisePolygonsCpu([grid[k]!], { width: W, height: H }));
      const covered = new Uint8Array(W * H);
      for (const l of per) l.forEach((v, q) => (covered[q]! += v ? 1 : 0));
      expect(covered.every((c) => c <= 1)).toBe(true);
    }
    // Every pixel of the inner area is in exactly 4 patches across the layers.
    expect(polygonLayers([patch(0, 0), patch(8, 0), patch(0, 8)])).toHaveLength(1);
  });
});

/** Labels 0..3 (3 regions), two channels with NaN, zeros, negatives and a support mask. */
function scene(seed: number) {
  const rnd = lcg(seed);
  const n = W * H;
  const labels = new Float32Array(n);
  const a = new Float32Array(n);
  const b = new Float32Array(n);
  const sup = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    labels[i] = Math.floor(((i % W) / W) * 4); // vertical bands 0,1,2,3
    a[i] = Math.exp(rnd() * 3 - 1) * (rnd() < 0.1 ? 0 : 1);
    b[i] = rnd() * 10 - 3;
    sup[i] = rnd() < 0.8 ? 1 : 0;
  }
  a[5] = Number.NaN;
  b[17] = Infinity;
  labels[W * H - 1] = 9; // above `regions`: ignored
  return { labels, a, b, sup };
}

describe("labelled histogram golden", () => {
  it("each row is R1's histogram of the region's values; totals partition the region", () => {
    const { labels, a, b, sup } = scene(1);
    const opts = { regions: 3, bins: 32, channels: [{ valid: "positive" as const }, { range: [-1, 5] as [number, number] }] };
    const h = zonalHistogramCpu(labels, [{ values: a, support: sup }, { values: b }], opts);
    expect(h.counts.length).toBe(3 * 2 * 32);
    for (let g = 0; g < 3; g++) {
      for (let c = 0; c < 2; c++) {
        const vals: number[] = [];
        let pixels = 0;
        for (let i = 0; i < labels.length; i++) {
          if (labels[i] !== g + 1) continue;
          pixels++;
          const v = c === 0 ? a[i]! : b[i]!;
          if (c === 0 && (sup[i] === 0 || !(v > 0))) continue;
          if (Number.isFinite(v)) vals.push(v);
        }
        const range: [number, number] = c === 0 ? [h.edges[0]!, h.edges[1]!] : [-1, 5];
        const ref = histogramCpu(vals, { bins: 32, range });
        const r = g * 2 + c;
        expect(Array.from(h.counts.subarray(r * 32, r * 32 + 32))).toEqual(Array.from(ref.counts));
        const t = Array.from(h.totals.subarray(r * ZONAL_TOTALS, (r + 1) * ZONAL_TOTALS));
        expect(t[ZONAL_SLOT.pixels]).toBe(pixels);
        expect(t.slice(0, 6).reduce((s, x) => s + x, 0)).toBe(pixels);
        expect(t.slice(0, 3)).toEqual(Array.from(ref.totals.subarray(0, 3)));
        expect(h.moments[r * 4 + 3]).toBeCloseTo(ref.moments[3]!, 10);
      }
    }
    // Auto edges of channel 0 are the in-region positive supported min/max.
    let mn = Infinity;
    let mx = -Infinity;
    labels.forEach((l, i) => {
      if (l >= 1 && l <= 3 && sup[i] && a[i]! > 0) {
        mn = Math.min(mn, a[i]!);
        mx = Math.max(mx, a[i]!);
      }
    });
    expect([h.edges[0], h.edges[1]]).toEqual([Math.fround(mn), Math.fround(mx)]);
  });

  it("merging two halves equals the whole", () => {
    const { labels, a } = scene(2);
    const opts = { regions: 3, bins: 16, channels: [{ range: [0, 20] as [number, number] }] };
    const half = labels.length / 2;
    const whole = zonalHistogramCpu(labels, [{ values: a }], opts);
    const m = zonalMerge(
      zonalHistogramCpu(labels.subarray(0, half), [{ values: a.subarray(0, half) }], opts),
      zonalHistogramCpu(labels.subarray(half), [{ values: a.subarray(half) }], opts),
    );
    expect(Array.from(m.counts)).toEqual(Array.from(whole.counts));
    expect(Array.from(m.totals)).toEqual(Array.from(whole.totals));
    for (let i = 0; i < m.moments.length; i++) expect(m.moments[i]).toBeCloseTo(whole.moments[i]!, 9);
  });
});

describe("exact quantiles", () => {
  it("match numpy-linear on every row, including tiny and censored ones", () => {
    const { labels, a, b, sup } = scene(3);
    labels.fill(0, 0, W * H);
    for (let i = 0; i < W * H; i++) labels[i] = i < 1 ? 1 : i < 3 ? 2 : i < 400 ? 3 : 4; // n = 1, 2, 397, rest
    const data = [{ values: a, support: sup }, { values: b }];
    const opts = { regions: 4, bins: 8, channels: [{ valid: "positive" as const }, { range: [0, 4] as [number, number] }] };
    const h = zonalHistogramCpu(labels, data, opts);
    const qs = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1];
    const plan = zonalExactPlan(h, qs);
    const exact = zonalExactQuantiles(h, plan, zonalCollectCpu(labels, data, opts, h, plan));
    for (let g = 0; g < 4; g++) {
      for (let c = 0; c < 2; c++) {
        const vals: number[] = [];
        labels.forEach((l, i) => {
          if (l !== g + 1) return;
          const v = Math.fround(c === 0 ? a[i]! : b[i]!);
          if (c === 0 && (sup[i] === 0 || !(v > 0))) return;
          if (Number.isFinite(v)) vals.push(v);
        });
        const sorted = Float32Array.from(vals);
        qs.forEach((q, j) => {
          const got = exact[(g * 2 + c) * qs.length + j]!;
          if (vals.length === 0) return expect(got).toBeNaN();
          const want = zonalReferenceQuantile(sorted, q);
          // Channel 1 is fixed-range [0, 4]: ranks outside it are censored to the edge.
          const p = q * (vals.length - 1);
          const lo = Math.floor(p);
          const sortedV = [...sorted].sort((x, y) => x - y);
          const censored =
            c === 1 && (sortedV[lo]! < 0 || sortedV[lo]! > 4 || (lo + 1 < vals.length && (sortedV[lo + 1]! < 0 || sortedV[lo + 1]! > 4)));
          if (!censored) expect(got).toBeCloseTo(want, 12);
        });
      }
    }
    // The plan only collects target bins: far fewer values than the region holds.
    expect(plan.total).toBeLessThan(W * H);
  });

  it("builds a regions × features table, binned within a bin width and exact on request", () => {
    const { labels, a } = scene(4);
    const opts = { regions: 3, bins: 64, channels: [{}] };
    const h = zonalHistogramCpu(labels, [{ values: a }], opts);
    const qs = [0.1, 0.25, 0.5, 0.75, 0.9];
    const plan = zonalExactPlan(h, qs);
    const exact = zonalExactQuantiles(h, plan, zonalCollectCpu(labels, [{ values: a }], opts, h, plan));
    const binned = zonalTable(h, { names: ["a"], quantiles: qs });
    const ex = zonalTable(h, { names: ["a"], quantiles: qs, exact });
    expect(Array.from(binned.labels)).toEqual([1, 2, 3]);
    const w = binned.columns.get("a_binWidth")!;
    for (let g = 0; g < 3; g++) {
      expect(Math.abs(binned.columns.get("a_median")![g]! - ex.columns.get("a_median")![g]!)).toBeLessThanOrEqual(w[g]!);
      expect(ex.columns.get("a_p50")![g]).toBe(ex.columns.get("a_median")![g]);
      expect(ex.columns.get("a_validFraction")![g]).toBeGreaterThan(0.95);
    }
    expect([...binned.columns.keys()]).toContain("a_p10");
  });
});
