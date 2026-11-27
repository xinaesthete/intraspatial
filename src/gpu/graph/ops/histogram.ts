// Tier-2 graph nodes wrapping `histogramResident`: a raster → value histogram, edges, CDF, totals
// and moments, all left on the device. R1 of the region/raster statistics plan
// (docs/histogram-op.md); the base for per-mode gram histograms, the wand's tolerance histogram
// and zonal (per-region) stats.
//
// Two nodes because a port cannot be optional here: `histogram` (one row) and `histogramMasked`
// (row 0 inside the mask, row 1 outside, one dispatch, one shared set of edges).
//
// Outputs are `matrix` values, `rows × bins` for the counts and CDF, so a later labelled (zonal)
// reduction is the same op with more rows (`labels × channels`) rather than a new value shape.
// Every output is its own lease (one lease per port, op.ts).
//
// The input is a grid — an f32 field, bound by range, so a packed slice works as is — or one plane
// of `gramMatrixGpu`'s `ResidentRasters`, wrapped by `rasterPlaneValue` as a grid-shaped value
// carrying the plane as its payload (the executor passes payloads through to a resident op
// untouched). A raster plane has no host data, so its `cpuGolden` refuses it: test such a plane
// against `histogramCpu` over its decoded texels instead.
import { type RasterPlane, denseRasterPlane, histogramByteSizes, histogramResident } from "#src/gpu/spatial/histogram";
import { HISTOGRAM_MOMENTS, HISTOGRAM_TOTALS, histogramBins, histogramCpu } from "#src/spatial/histogram";
import type { FieldValue, Shape } from "../handle";
import type { OpType, ParamSpec, Params, PortSpec } from "../op";

/** The payload tag a raster-plane source carries. */
export const RASTER_PLANE = "rasterPlane";

export interface RasterPlanePayload {
  readonly kind: typeof RASTER_PLANE;
  readonly plane: RasterPlane;
}

/** A graph source value for one GPU raster plane (e.g. `residentRasterPlane(res.resident, k, w,
 *  h)`): `graph.source(rasterPlaneValue(plane))`. The plane's buffer stays the caller's; its
 *  lifetime rules (a `ResidentRasters` is valid until the next `gramMatrixGpu`) still apply. */
export function rasterPlaneValue(plane: RasterPlane): FieldValue {
  const payload: RasterPlanePayload = { kind: RASTER_PLANE, plane };
  return { shape: { kind: "grid", width: plane.width, height: plane.height }, dtype: "f32", payload };
}

const PARAMS: ParamSpec[] = [
  { name: "bins", type: "int", default: 256, min: 1, max: 65536, describe: "Bin count." },
  { name: "range", type: "enum", default: "auto", options: ["auto", "fixed"], describe: "auto: finite min..max; fixed: lo..hi." },
  { name: "lo", type: "number", default: 0, describe: "Lower edge when range = fixed." },
  { name: "hi", type: "number", default: 1, describe: "Upper edge when range = fixed (> lo)." },
];

const OUTPUTS: PortSpec[] = [
  { name: "counts", kind: "matrix", dtype: "u32" },
  { name: "cdf", kind: "matrix", dtype: "u32" },
  { name: "edges", kind: "matrix", dtype: "f32" },
  { name: "totals", kind: "matrix", dtype: "u32" },
  { name: "moments", kind: "matrix", dtype: "f32" },
];

function optsOf(params: Params): { bins: number; range?: [number, number] } {
  const bins = histogramBins({ bins: params.bins as number });
  if (params.range !== "fixed") return { bins };
  const lo = params.lo as number;
  const hi = params.hi as number;
  if (!(hi > lo)) throw new Error(`histogram: fixed range needs hi > lo (got [${lo}, ${hi}])`);
  return { bins, range: [lo, hi] };
}

function gridOf(s: Shape, who: string): { width: number; height: number } {
  if (s.kind !== "grid") throw new Error(`${who}: input must be a grid`);
  return { width: s.width, height: s.height };
}

function outShapes(bins: number, rows: number): Shape[] {
  return [
    { kind: "matrix", rows, cols: bins },
    { kind: "matrix", rows, cols: bins },
    { kind: "matrix", rows: 1, cols: 2 },
    { kind: "matrix", rows, cols: HISTOGRAM_TOTALS },
    { kind: "matrix", rows, cols: HISTOGRAM_MOMENTS },
  ];
}

function planeOf(v: FieldValue, who: string): RasterPlane {
  const { width, height } = gridOf(v.shape, who);
  const payload = v.payload as RasterPlanePayload | undefined;
  if (payload?.kind === RASTER_PLANE) return payload.plane;
  if (!v.buffer) throw new Error(`${who}: resident op received a non-resident input`);
  if (v.dtype !== "f32") throw new Error(`${who}: the field must be f32 (got ${v.dtype})`);
  return denseRasterPlane(v.buffer.buffer, width, height, v.buffer.offset ?? 0);
}

function makeOp(masked: boolean): OpType {
  const name = masked ? "histogramMasked" : "histogram";
  const rows = masked ? 2 : 1;
  const inputs: PortSpec[] = [{ name: "field", kind: "grid", dtype: "f32" }];
  if (masked) inputs.push({ name: "mask", kind: "grid" });
  return {
    name,
    label: masked ? "Histogram (inside / outside mask)" : "Histogram",
    category: "Statistics",
    describe: masked
      ? "Value histogram split by a mask: row 0 inside (non-zero), row 1 outside, shared edges."
      : "Value histogram of a field: counts, CDF, edges, totals and min/max/sum/mean, on the GPU.",
    help: {
      detail:
        "Bins finite values into `bins` equal-width bins over the finite min..max (or a fixed lo..hi, " +
        "with values outside counted as under/over). NaN and ±Inf are skipped and counted. `totals` is " +
        "[inRange, under, over, nonFinite] per row; `moments` is [min, max, sum, mean]. Quantiles read " +
        "off the CDF (`histogramSummary`) are within one bin width of exact.",
      math: "b(v) = \\left\\lfloor \\frac{(v-\\mathrm{lo})\\,B}{\\mathrm{hi}-\\mathrm{lo}} \\right\\rfloor",
    },
    inputs,
    outputs: OUTPUTS,
    params: PARAMS,
    inferShapes(inShapes, params) {
      const g = gridOf(inShapes[0]!, name);
      if (masked) {
        const m = gridOf(inShapes[1]!, name);
        if (m.width !== g.width || m.height !== g.height)
          throw new Error(`${name}: mask is ${m.width}x${m.height}, field ${g.width}x${g.height}`);
      }
      return outShapes(optsOf(params).bins, rows);
    },
    // Not spatial: a histogram has no placement, axes or label role.
    inferPlacement: () => OUTPUTS.map(() => undefined),
    inferAxes: () => OUTPUTS.map(() => undefined),
    inferRole: () => OUTPUTS.map(() => undefined),
    resident: true,
    slices: true,
    async execute(ctx, ins, params) {
      const opts = optsOf(params);
      const plane = planeOf(ins[0]!, name);
      let mask: { buffer: GPUBuffer; byteOffset: number; dtype: "u32" | "f32" } | undefined;
      if (masked) {
        const m = ins[1]!;
        if (!m.buffer) throw new Error(`${name}: the mask must be a resident buffer`);
        mask = { buffer: m.buffer.buffer, byteOffset: m.buffer.offset ?? 0, dtype: m.dtype === "f32" ? "f32" : "u32" };
      }
      const sizes = histogramByteSizes(opts.bins, rows);
      const counts = await ctx.backend.lease(sizes.counts);
      const cdf = await ctx.backend.lease(sizes.cdf);
      const edges = await ctx.backend.lease(sizes.edges);
      const totals = await ctx.backend.lease(sizes.totals);
      const moments = await ctx.backend.lease(sizes.moments);
      await histogramResident(
        plane,
        { counts: counts.buffer, cdf: cdf.buffer, edges: edges.buffer, totals: totals.buffer, moments: moments.buffer },
        { ...opts, mask },
        await ctx.backend.getDevice(),
      );
      const shapes = outShapes(opts.bins, rows);
      return [counts, cdf, edges, totals, moments].map((buffer, i) => ({ shape: shapes[i]!, dtype: OUTPUTS[i]!.dtype!, buffer }));
    },
    cpuGolden(ins, params) {
      const opts = optsOf(params);
      const data = ins[0]!.data;
      if (!data) throw new Error(`${name}: cpuGolden needs host data (a raster-plane source has none)`);
      const h = histogramCpu(data, masked ? { ...opts, mask: ins[1]!.data! } : opts);
      const shapes = outShapes(opts.bins, rows);
      const outs = [h.counts, h.cdf, new Float32Array([h.lo, h.hi]), h.totals, Float32Array.from(h.moments)];
      return outs.map((d, i) => ({ shape: shapes[i]!, dtype: OUTPUTS[i]!.dtype!, data: d }));
    },
  };
}

export const histogramOp: OpType = makeOp(false);
export const histogramMaskedOp: OpType = makeOp(true);
