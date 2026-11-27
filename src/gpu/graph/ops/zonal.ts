// Tier-2 graph nodes for zonal statistics (R3; docs/zonal-stats.md):
//
// - `rasterisePolygons`: ring buffers (`polygonRingBuffers`, as two matrix sources) → a u32 label
//   grid. Polygon k's label at every pixel whose centre it contains; the highest label on overlap.
// - `zonalHistogram`: labels + one field → R1's five outputs with one row per region (label L is
//   row L − 1). Single channel, like `histogram`: a labels × channels table is one node per channel
//   sharing the labels, or `zonalStatsTiled` / `ZonalAccumulator` outside a graph.
//
// One image per node. Tiled whole-slide input does not fit a single graph value; it goes through
// `zonalStatsTiled`, which drives the same kernels tile by tile.
import type { RasterPlane } from "#src/gpu/spatial/histogram";
import { denseRasterPlane } from "#src/gpu/spatial/histogram";
import { ZonalAccumulator, rasterisePolygonBuffers, zonalByteSizes } from "#src/gpu/spatial/zonal";
import { HISTOGRAM_MOMENTS, histogramBins, histogramFixedRange } from "#src/spatial/histogram";
import { POLY_STRIDE, type PolygonRingBuffers, ZONAL_TOTALS, type ZonalOptions, zonalHistogramCpu } from "#src/spatial/zonal";
import type { Graph } from "../graph";
import type { FieldValue, GpuField, ResidentBuffer, Shape } from "../handle";
import type { OpType, Params, PortSpec } from "../op";
import { RASTER_PLANE, type RasterPlanePayload } from "./histogram";

function gridOf(s: Shape, who: string): { width: number; height: number } {
  if (s.kind !== "grid") throw new Error(`${who}: input must be a grid`);
  return { width: s.width, height: s.height };
}

function matrixOf(s: Shape, cols: number, who: string): number {
  if (s.kind !== "matrix" || s.cols !== cols) throw new Error(`${who}: expected a matrix with ${cols} columns`);
  return s.rows;
}

/** The two sources `rasterisePolygons` takes: `g.op("rasterisePolygons", polygonRingSources(g,
 *  polygonRingBuffers(polys, frame)), { width, height })`. */
export function polygonRingSources(g: Graph, rings: PolygonRingBuffers): { edges: GpuField; polys: GpuField } {
  const matrix = (data: Float32Array, cols: number): FieldValue => ({
    shape: { kind: "matrix", rows: data.length / cols, cols },
    dtype: "f32",
    data,
  });
  return { edges: g.source(matrix(rings.edges, 4), "polygonEdges"), polys: g.source(matrix(rings.polys, POLY_STRIDE), "polygons") };
}

export const rasterisePolygonsOp: OpType = {
  name: "rasterisePolygons",
  label: "Rasterise polygons (labels)",
  category: "Statistics",
  describe: "Polygons → a u32 label grid: each pixel gets the highest label of the polygons containing its centre.",
  help: {
    detail:
      "Inputs are `polygonRingBuffers(polygons, frame)` as matrices (edges E×4, polygons P×8) in the output's " +
      "pixel frame, so pixel (i, j)'s centre is (i + ½, j + ½). Even–odd inside test per pixel against the edges " +
      "of every polygon whose box holds the centre; 0 where no polygon does.",
  },
  inputs: [
    { name: "edges", kind: "matrix", dtype: "f32" },
    { name: "polys", kind: "matrix", dtype: "f32" },
  ],
  outputs: [{ name: "labels", kind: "grid", dtype: "u32" }],
  params: [
    { name: "width", type: "int", default: 256, min: 1, max: 65536 },
    { name: "height", type: "int", default: 256, min: 1, max: 65536 },
  ],
  inferShapes(inShapes, params) {
    matrixOf(inShapes[0]!, 4, "rasterisePolygons");
    matrixOf(inShapes[1]!, POLY_STRIDE, "rasterisePolygons");
    return [{ kind: "grid", width: params.width as number, height: params.height as number }];
  },
  inferPlacement: () => [undefined],
  resident: true,
  async execute(ctx, ins, params) {
    const [e, p] = ins as [FieldValue, FieldValue];
    if (!e.buffer || !p.buffer) throw new Error("rasterisePolygons: resident op received a non-resident input");
    const width = params.width as number;
    const height = params.height as number;
    const count = matrixOf(p.shape, POLY_STRIDE, "rasterisePolygons");
    const rows = matrixOf(e.shape, 4, "rasterisePolygons");
    const out = await ctx.backend.lease(width * height * 4);
    try {
      await rasterisePolygonBuffers(
        { buffer: e.buffer.buffer, offset: e.buffer.offset ?? 0, size: rows * 16 },
        { buffer: p.buffer.buffer, offset: p.buffer.offset ?? 0, size: count * POLY_STRIDE * 4 },
        count,
        { buffer: out.buffer, width, height },
        await ctx.backend.getDevice(),
      );
    } catch (err) {
      ctx.backend.release(out);
      throw err;
    }
    return [{ shape: { kind: "grid", width, height }, dtype: "u32", buffer: out }];
  },
  cpuGolden(ins, params) {
    // The same crossing test in f64 over the same f32 buffers: equal to the GPU except for a
    // centre within rounding of an edge.
    const edges = ins[0]!.data!;
    const polys = ins[1]!.data!;
    const width = params.width as number;
    const height = params.height as number;
    const out = new Uint32Array(width * height);
    for (let q = 0; q < out.length; q++) {
      const x = (q % width) + 0.5;
      const y = Math.floor(q / width) + 0.5;
      let best = 0;
      for (let o = 0; o < polys.length; o += POLY_STRIDE) {
        const id = polys[o + 2]!;
        if (!(id > best && x >= polys[o + 4]! && x <= polys[o + 6]! && y >= polys[o + 5]! && y <= polys[o + 7]!)) continue;
        let inside = false;
        for (let k = polys[o]!; k < polys[o]! + polys[o + 1]!; k++) {
          const [x0, y0, x1, y1] = [edges[k * 4]!, edges[k * 4 + 1]!, edges[k * 4 + 2]!, edges[k * 4 + 3]!];
          if (y0 > y !== y1 > y && x < ((x1 - x0) * (y - y0)) / (y1 - y0) + x0) inside = !inside;
        }
        if (inside) best = id;
      }
      out[q] = best;
    }
    return [{ shape: { kind: "grid", width, height }, dtype: "u32", data: out }];
  },
};

const OUTPUTS: PortSpec[] = [
  { name: "counts", kind: "matrix", dtype: "u32" },
  { name: "cdf", kind: "matrix", dtype: "u32" },
  { name: "edges", kind: "matrix", dtype: "f32" },
  { name: "totals", kind: "matrix", dtype: "u32" },
  { name: "moments", kind: "matrix", dtype: "f32" },
];

function zonalOpts(params: Params): ZonalOptions {
  const bins = histogramBins({ bins: params.bins as number });
  const valid = params.valid === "positive" ? "positive" : "finite";
  const range = params.range === "fixed" ? histogramFixedRange([params.lo as number, params.hi as number]) : undefined;
  return { regions: params.regions as number, bins, channels: [{ range, valid }] };
}

function outShapes(regions: number, bins: number): Shape[] {
  return [
    { kind: "matrix", rows: regions, cols: bins },
    { kind: "matrix", rows: regions, cols: bins },
    { kind: "matrix", rows: 1, cols: 2 },
    { kind: "matrix", rows: regions, cols: ZONAL_TOTALS },
    { kind: "matrix", rows: regions, cols: HISTOGRAM_MOMENTS },
  ];
}

function planeOf(v: FieldValue, width: number, height: number): RasterPlane {
  const payload = v.payload as RasterPlanePayload | undefined;
  if (payload?.kind === RASTER_PLANE) return payload.plane;
  if (!v.buffer) throw new Error("zonalHistogram: resident op received a non-resident field");
  if (v.dtype !== "f32") throw new Error(`zonalHistogram: the field must be f32 (got ${v.dtype})`);
  return denseRasterPlane(v.buffer.buffer, width, height, v.buffer.offset ?? 0);
}

export const zonalHistogramOp: OpType = {
  name: "zonalHistogram",
  label: "Zonal histogram (per region)",
  category: "Statistics",
  describe: "Value histogram per labelled region: row L−1 for label L, R1's counts/CDF/edges/totals/moments.",
  help: {
    detail:
      "Labels 1..regions select rows; 0 and larger labels are ignored. `totals` is [inRange, under, over, " +
      "nonFinite, excluded (≤ 0 under valid = positive), unsupported, pixels, 0] per region; `moments` " +
      "[min, max, sum, mean] of the region's values. Edges are shared by all regions (auto: the values' " +
      "min..max over every region). Quantiles via `zonalRow` + `histogramQuantile`, or the exact pass.",
  },
  inputs: [
    { name: "labels", kind: "grid" },
    { name: "field", kind: "grid", dtype: "f32" },
  ],
  outputs: OUTPUTS,
  params: [
    { name: "regions", type: "int", default: 1, min: 1, max: 2 ** 24, describe: "Labels 1..regions are rows." },
    { name: "bins", type: "int", default: 256, min: 1, max: 65536, describe: "Bin count." },
    { name: "range", type: "enum", default: "auto", options: ["auto", "fixed"] },
    { name: "lo", type: "number", default: 0 },
    { name: "hi", type: "number", default: 1 },
    { name: "valid", type: "enum", default: "finite", options: ["finite", "positive"], describe: "positive: v ≤ 0 is background." },
  ],
  inferShapes(inShapes, params) {
    const l = gridOf(inShapes[0]!, "zonalHistogram");
    const f = gridOf(inShapes[1]!, "zonalHistogram");
    if (l.width !== f.width || l.height !== f.height)
      throw new Error(`zonalHistogram: labels are ${l.width}x${l.height}, field ${f.width}x${f.height}`);
    const o = zonalOpts(params);
    return outShapes(o.regions, o.bins!);
  },
  inferPlacement: () => OUTPUTS.map(() => undefined),
  inferAxes(inAxes) {
    const planes = (axes: (typeof inAxes)[number]) => (axes ?? []).reduce((n, ax) => n * ax.length, 1);
    if (inAxes.some((axes) => planes(axes) !== 1))
      throw new Error("zonalHistogram: inputs must be single-channel (open axes of total length 1)");
    return OUTPUTS.map(() => undefined);
  },
  inferRole: () => OUTPUTS.map(() => undefined),
  resident: true,
  slices: true,
  async execute(ctx, ins, params) {
    const o = zonalOpts(params);
    const [lv, fv] = ins as [FieldValue, FieldValue];
    const { width, height } = gridOf(fv.shape, "zonalHistogram");
    if (!lv.buffer) throw new Error("zonalHistogram: the labels must be a resident buffer");
    const device = await ctx.backend.getDevice();
    const acc = await ZonalAccumulator.create(o, device);
    const sizes = zonalByteSizes(o.bins!, o.regions, 1);
    const leased: ResidentBuffer[] = [];
    try {
      const tile = {
        labels: {
          buffer: lv.buffer.buffer,
          byteOffset: lv.buffer.offset ?? 0,
          width,
          height,
          dtype: lv.dtype === "f32" ? ("f32" as const) : ("u32" as const),
        },
        channels: [{ plane: planeOf(fv, width, height) }],
      };
      if (acc.needsRange) {
        acc.rangeTile(tile);
        acc.finalizeEdges();
      }
      acc.binTile(tile);
      for (const key of ["counts", "cdf", "edges", "totals", "moments"] as const) leased.push(await ctx.backend.lease(sizes[key]));
      const [counts, cdf, edges, totals, moments] = leased.map((b) => b.buffer) as [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer];
      acc.finish({ counts, cdf, edges, totals, moments });
    } catch (err) {
      for (const b of leased) ctx.backend.release(b);
      throw err;
    } finally {
      acc.release();
    }
    const shapes = outShapes(o.regions, o.bins!);
    return leased.map((buffer, i) => ({ shape: shapes[i]!, dtype: OUTPUTS[i]!.dtype!, buffer }));
  },
  cpuGolden(ins, params) {
    const o = zonalOpts(params);
    const labels = ins[0]!.data;
    const values = ins[1]!.data;
    if (!labels || !values) throw new Error("zonalHistogram: cpuGolden needs host data (a raster-plane source has none)");
    const h = zonalHistogramCpu(labels, [{ values }], o);
    const shapes = outShapes(o.regions, o.bins!);
    const outs = [h.counts, h.cdf, h.edges, h.totals, Float32Array.from(h.moments)];
    return outs.map((d, i) => ({ shape: shapes[i]!, dtype: OUTPUTS[i]!.dtype!, data: d }));
  },
};
