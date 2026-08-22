// Filtering a table as a subgraph — ADR-0005's `support` in its dense encoding, built
// out of ordinary fields rather than a new facet.
//
// ## A selection is a field, so the filter graph already exists
//
// ADR-0005 argues a filter is "just a subgraph that produces a `support` field". Taken
// literally that needs a new facet on every value and a propagation rule on every op.
// Taken one step less literally it needs nothing at all: a **soft mask is one weight per
// row in [0,1]**, which is exactly a `points{n}` f32 field. The composer already draws
// DAGs of those, the memo already invalidates precisely the dependents of a changed
// param, and the executor already schedules them. So the ops here are small, and the
// interesting content is which operators they are — not the plumbing.
//
// `docs/mdv-dimension-vs-support-facet.md` §6 is the argument for that; this file is it,
// running.
//
// ## The operators are min / max / 1−a, and that is a correction
//
// ADR-0005's body specifies `AND = a·b`, `OR = max(a,b)`, `NOT = 1−a`. Those three do not
// belong together — a t-norm and a t-conorm are De Morgan dual under `1−a` only when
// `S(a,b) = 1 − T(1−a, 1−b)`, and the ADR took its AND from the product pair and its OR
// from the Gödel pair. Two consequences:
//
//   De Morgan fails    1−(1−a)(1−b) = 0.92  vs  max(a,b) = 0.8   at a=0.8, b=0.6
//   AND is not idempotent   a·a = 0.64      vs  a = 0.8
//
// The second is the one that matters *here*, because it only appears once a list becomes a
// graph. A diamond — two branches derived from one upstream gate, recombined downstream —
// is the normal shape of a gating tree, and under the product the shared ancestor is
// multiplied in twice. The result then depends on the graph's topology rather than on the
// set being described, and it does so silently: the mask just gets dimmer. A flat
// conjunction (MDV's Dimension list) cannot contain a diamond, which is why nobody has had
// to face this before.
//
// So: **min / max / 1−a**, which are idempotent, De Morgan-consistent, associative, and
// the cheapest of the three. They live in `fieldArithmetic.ts` beside the other pointwise
// ops, because that is what they are. Set difference is `min(a, 1−b)`.
//
// Both defects are invisible for hard 0/1 masks (`a·b == min(a,b)` there), which is why
// they survived review — and soft membership is the entire reason the byte was widened to
// a weight.
//
// ## Hard is the boxcar case of soft, not a separate thing
//
// `softness = 0` gives a crisp in/out test; anything above it gives a ramp at each edge.
// That is the "windowing, not quadrats" position the spatial front already took, and it is
// why a brush should be a window with a falloff rather than a crisp box. It also makes the
// mask usable as the weight vector for a weighted-permutation null, which a 0/1 byte
// cannot be.
import { gaussianKdeField } from "../../../spatial/scalarField";
import type { FieldValue, Shape } from "../handle";
import { numCells } from "../handle";
import type { OpType, Params } from "../op";

/** A per-row value column. Distinct from a *points* field of the same shape, which carries
 *  two interleaved lanes per row — see `requireColumn`. */
const COLUMN_KIND = "points" as const;

/**
 * Read a value field as one scalar per row.
 *
 * `points{n}` is ambiguous in this codebase: a centroid cloud declares `points{n}` and
 * carries 2n floats (x,y interleaved), while a column declares `points{n}` and carries n.
 * `shapesEqual` cannot tell them apart, so connecting a cloud where a column belongs
 * type-checks and then reads coordinates as values — a wrong answer that looks like data.
 * Checking the length is the cheapest place to catch it, and the error can name the cause.
 */
function requireColumn(v: FieldValue, op: string, port: string): ArrayLike<number> {
  const n = numCells(v.shape);
  const data = v.data;
  if (!data) throw new Error(`${op}: '${port}' has no host data`);
  if (data.length === 2 * n && n > 0) {
    throw new Error(
      `${op}: '${port}' carries ${data.length} values for ${n} rows — that is an (x,y) point cloud, not a column. ` +
        `Connect a column output (or a mask) here, not the cloud itself.`,
    );
  }
  if (data.length !== n) throw new Error(`${op}: '${port}' has ${data.length} values for ${n} rows`);
  return data;
}

/** A ramp from 0 at `edge - w` to 1 at `edge + w`, smooth at both ends (smoothstep), for
 *  `w > 0`. Chosen over a logistic because it reaches exactly 0 and exactly 1 at finite
 *  distance: a brush that never quite excludes anything is not a selection, and a logistic
 *  tail leaves every row faintly present, which then leaks into every weighted reduction
 *  downstream.
 *
 *  The hard case is NOT this function with `w = 0`. Doing it that way needs the rising edge
 *  to be `>=` and the falling edge `>`, or the row sitting exactly on `hi` is dropped from
 *  what the op documents as a closed interval — so the boundary is handled at the call site
 *  where both bounds are in view. */
function ramp(x: number, edge: number, w: number): number {
  const t = (x - (edge - w)) / (2 * w);
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  return t * t * (3 - 2 * t);
}

function maskShape(inputs: Shape[]): Shape[] {
  return [inputs[0]!];
}

/**
 * `maskRange` — the leaf of a filter graph: a value column becomes a membership weight.
 *
 * `lo`/`hi` bound the window; `softness` is the half-width of the ramp at each edge, in the
 * column's own units, and 0 gives the boxcar. An unbounded side is expressed by pushing the
 * bound out, not by a separate op, so a one-sided filter is the same node.
 */
export const maskRangeOp: OpType = {
  name: "maskRange",
  label: "Mask: range",
  describe: "Select rows whose value lies in [lo, hi]; softness ramps the edges (0 = hard).",
  category: "Filter",
  help: {
    detail:
      "One weight per row in [0,1] — ADR-0005's soft `support`, as an ordinary field. " +
      "A hard filter is the boxcar case (softness 0); above that each edge becomes a smoothstep ramp of " +
      "half-width `softness`, so a brush is a window with a falloff rather than a crisp box. " +
      "Combine several with Min / Max / Invert (AND / OR / NOT).",
    math: "m(x)=\\mathrm{ramp}(x;\\mathrm{lo})\\,\\bigl(1-\\mathrm{ramp}(x;\\mathrm{hi})\\bigr)",
  },
  inputs: [{ name: "value", kind: COLUMN_KIND }],
  outputs: [{ name: "mask", kind: COLUMN_KIND, dtype: "f32" }],
  params: [
    { name: "lo", type: "number", default: 0, min: -1e6, max: 1e6, step: 0.1, describe: "lower bound" },
    { name: "hi", type: "number", default: 1, min: -1e6, max: 1e6, step: 0.1, describe: "upper bound" },
    { name: "softness", type: "number", default: 0, min: 0, max: 100, step: 0.01, describe: "edge ramp half-width (0 = hard)" },
  ],
  inferShapes: maskShape,
  execute: async (_ctx, inputs, params) => [maskRange(inputs[0]!, params)],
  cpuGolden: (inputs, params) => [maskRange(inputs[0]!, params)],
};

function maskRange(v: FieldValue, params: Params): FieldValue {
  const data = requireColumn(v, "maskRange", "value");
  const lo = params.lo as number;
  const hi = params.hi as number;
  const w = params.softness as number;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const x = data[i]!;
    // Hard: the closed interval, both ends inclusive, exactly as the op documents it.
    // Soft: rising at lo, falling at hi; their product is the window, and a row sitting on
    // either bound gets 1/2, which is what a ramp centred on the bound should give.
    out[i] = w <= 0 ? (x >= lo && x <= hi ? 1 : 0) : ramp(x, lo, w) * (1 - ramp(x, hi, w));
  }
  return { shape: v.shape, dtype: "f32", data: out };
}

/**
 * `maskEquals` — categorical selection, the other leaf.
 *
 * MDV's `filterCategories` and the `cell_type_id` split are both this. `tolerance` exists
 * because a category code that arrived through an f32 field is not exactly an integer, and
 * an `===` against it silently selects nothing.
 */
export const maskEqualsOp: OpType = {
  name: "maskEquals",
  label: "Mask: equals",
  describe: "Select rows whose value equals a category code.",
  category: "Filter",
  help: {
    detail:
      "Categorical selection — MDV's `filterCategories`, and the same thing `cellTable`'s per-`cell_type_id` " +
      "split does eagerly at load time. As a mask it costs no copy and the grouping column can be changed " +
      "without re-reading the table.",
  },
  inputs: [{ name: "value", kind: COLUMN_KIND }],
  outputs: [{ name: "mask", kind: COLUMN_KIND, dtype: "f32" }],
  params: [
    { name: "code", type: "number", default: 0, min: -1e6, max: 1e6, step: 1, describe: "category code to select" },
    { name: "tolerance", type: "number", default: 0.5, min: 0, max: 1, step: 0.01, describe: "match window (codes arrive as f32)" },
  ],
  inferShapes: maskShape,
  execute: async (_ctx, inputs, params) => [maskEquals(inputs[0]!, params)],
  cpuGolden: (inputs, params) => [maskEquals(inputs[0]!, params)],
};

function maskEquals(v: FieldValue, params: Params): FieldValue {
  const data = requireColumn(v, "maskEquals", "value");
  const code = params.code as number;
  const tol = params.tolerance as number;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = Math.abs(data[i]! - code) <= tol ? 1 : 0;
  return { shape: v.shape, dtype: "f32", data: out };
}

/**
 * `maskCount` — how many rows the selection keeps, and how much weight.
 *
 * MDV's `filterSize`, which every chart reads. Under a DAG it is a reduction per node
 * rather than a maintained counter; the design note flags that as an open question because
 * on the host it is a readback, and readback is the step this repo keeps measuring as the
 * expensive one. At playground scale it is free, and having the number on screen is what
 * makes a soft mask legible at all — `count` and `weight` diverge exactly when the mask
 * stops being 0/1.
 */
export const maskCountOp: OpType = {
  name: "maskCount",
  label: "Mask: count",
  describe: "Rows kept (weight > 1/2) and total weight — MDV's filterSize, as a reduction.",
  category: "Filter",
  inputs: [{ name: "mask", kind: COLUMN_KIND }],
  outputs: [
    { name: "count", kind: "scalar", dtype: "f32" },
    { name: "weight", kind: "scalar", dtype: "f32" },
  ],
  params: [],
  inferShapes: () => [{ kind: "scalar" }, { kind: "scalar" }],
  execute: async (_ctx, inputs) => maskCount(inputs[0]!),
  cpuGolden: (inputs) => maskCount(inputs[0]!),
};

function maskCount(v: FieldValue): FieldValue[] {
  const data = requireColumn(v, "maskCount", "mask");
  let count = 0;
  let weight = 0;
  for (let i = 0; i < data.length; i++) {
    weight += data[i]!;
    if (data[i]! > 0.5) count++;
  }
  return [
    { shape: { kind: "scalar" }, dtype: "f32", data: Float32Array.of(count) },
    { shape: { kind: "scalar" }, dtype: "f32", data: Float32Array.of(weight) },
  ];
}

/**
 * `maskedDensity` — the sink that makes a mask visible: a KDE weighted by the selection.
 *
 * This is literally the call MDV's filtering note asks for in its §B1 —
 * `splatDensityGpu(xs, ys, {weights: passingMask})` — "a live filtered density field with
 * zero extra marshalling". Both the GPU splat and the CPU KDE already took `weights`; only
 * the graph op did not expose it.
 *
 * Deliberately NOT the resident `splatDensity` op with an extra port. That one renders from
 * a stride-2 GPU buffer and would need the mask zipped into a stride-3 one by a kernel that
 * does not exist yet. This is the Tier-1 form, which is what the composer's host-resident
 * values want anyway, and it leaves the resident path untouched.
 *
 * A soft mask is doing real work here rather than decorating: a hard 0/1 filter makes the
 * density jump as the brush crosses a point, and a ramped one moves it continuously.
 */
export const maskedDensityOp: OpType = {
  name: "maskedDensity",
  label: "Masked KDE",
  describe: "Gaussian density of a point cloud, weighted by a selection mask.",
  category: "Filter",
  help: {
    detail:
      "The selection made visible. Each point contributes its mask weight, so an excluded row adds " +
      "nothing and a partly-selected one adds part of a kernel. Connect a points cloud and the mask " +
      "built from that same table — they must be the same length, which is what makes the mask a " +
      "property of the rows rather than of the geometry.",
  },
  inputs: [
    { name: "points", kind: "points" },
    { name: "mask", kind: COLUMN_KIND },
  ],
  outputs: [{ name: "density", kind: "grid", dtype: "f32" }],
  params: [
    { name: "width", type: "int", default: 64, min: 8, max: 512, describe: "grid width" },
    { name: "height", type: "int", default: 64, min: 8, max: 512, describe: "grid height" },
    { name: "sigma", type: "number", default: 2, min: 0.1, max: 40, step: 0.1, describe: "bandwidth (world units)" },
    { name: "radiusSigma", type: "number", default: 4, min: 1, max: 8, step: 0.5 },
  ],
  inferShapes(_inputs, params) {
    return [{ kind: "grid", width: params.width as number, height: params.height as number }];
  },
  execute: async (_ctx, inputs, params) => [maskedDensity(inputs[0]!, inputs[1]!, params)],
  cpuGolden: (inputs, params) => [maskedDensity(inputs[0]!, inputs[1]!, params)],
};

function maskedDensity(pts: FieldValue, mask: FieldValue, params: Params): FieldValue {
  const n = numCells(pts.shape);
  const xy = pts.data;
  if (!xy) throw new Error("maskedDensity: 'points' has no host data");
  if (xy.length !== 2 * n) {
    throw new Error(`maskedDensity: 'points' carries ${xy.length} values for ${n} rows — expected an interleaved (x,y) cloud`);
  }
  const weights = requireColumn(mask, "maskedDensity", "mask");
  if (weights.length !== n) {
    throw new Error(
      `maskedDensity: mask has ${weights.length} weights for ${n} points. A mask belongs to the rows of one table — ` +
        `build it from a column of the same table as this cloud.`,
    );
  }
  const xs = new Float32Array(n);
  const ys = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = xy[2 * i]!;
    ys[i] = xy[2 * i + 1]!;
  }
  const width = params.width as number;
  const height = params.height as number;
  const field = gaussianKdeField(xs, ys, {
    width,
    height,
    sigma: params.sigma as number,
    radiusSigma: params.radiusSigma as number,
    weights,
  });
  return { shape: { kind: "grid", width, height }, dtype: "f32", data: field.data };
}

/** The filter pack, for the composer's registrar. */
export const FILTER_OPS: OpType[] = [maskRangeOp, maskEqualsOp, maskCountOp, maskedDensityOp];
