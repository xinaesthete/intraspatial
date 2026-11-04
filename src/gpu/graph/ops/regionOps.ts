// Region ops: the distance transform, the flood fill from a seed, and the distance to the mixture at a
// seed, as graph nodes (CPU, Tier-1). With the existing Gaussian, Laplacian, threshold and mask
// arithmetic they compose the wand's patch (docs/wand-metric-and-regions.md):
//
//   d = mixtureDistance(u) → convolveSeparable (σ) → laplacian → threshold ≥ 0 → floodFrom (seed in)
//   opening by a disc of radius ρ: distanceTransform (border) > ρ, then distanceTransform of its
//   complement ≤ ρ, then floodFrom the seed.
//
// An opt-in pack (REGION_OPS), like the filter ops, so the always-loaded registry does not grow.
import { distanceTransform, floodFrom, mixtureDistance } from "#src/spatial/regions";
import { channelAxis, type FieldValue, type Shape } from "../handle";
import type { OpType, Params } from "../op";

function gridOf(s: Shape, name: string): { width: number; height: number } {
  if (s.kind !== "grid") throw new Error(`${name}: input must be a grid`);
  return { width: s.width, height: s.height };
}

const grid = (shape: Shape, data: Float32Array): FieldValue => ({ shape, dtype: "f32", data });

/** A host-only op whose `execute` is its reference. */
function hostOp(op: Omit<OpType, "execute" | "cpuGolden">, body: (inputs: FieldValue[], params: Params) => FieldValue[]): OpType {
  return { ...op, execute: async (_ctx, inputs, params) => body(inputs, params), cpuGolden: body };
}

const SEED_PARAMS: OpType["params"] = [
  { name: "col", type: "number", default: 0, step: 0.25, describe: "seed column (pixel centres at integers)" },
  { name: "row", type: "number", default: 0, step: 0.25, describe: "seed row (pixel centres at integers)" },
];

export const distanceTransformOp: OpType = hostOp(
  {
    name: "distanceTransform",
    label: "Distance transform",
    describe: "Each pixel's Euclidean distance to the nearest pixel below 0.5.",
    category: "Regions",
    help: {
      detail:
        "Exact (Felzenszwalb–Huttenlocher), in pixels between pixel centres. With `border`, the ring just outside the raster counts as below 0.5, so a region is also measured from the raster's edge. Thresholding it at ρ erodes a mask by a disc of radius ρ.",
      math: String.raw`D(x) = \min_{y:\ m(y) < 1/2} \lVert x - y \rVert`,
    },
    inputs: [{ name: "in", kind: "grid" }],
    outputs: [{ name: "out", kind: "grid", dtype: "f32" }],
    params: [{ name: "border", type: "bool", default: false, describe: "the outside of the raster counts as below 0.5" }],
    inferShapes: (inputs) => [inputs[0]!],
  },
  (inputs, params) => {
    const v = inputs[0]!;
    const { width, height } = gridOf(v.shape, "distanceTransform");
    return [grid(v.shape, distanceTransform(v.data!, width, height, !!params.border))];
  },
);

export const floodFromOp: OpType = hostOp(
  {
    name: "floodFrom",
    label: "Flood from seed",
    describe: "The 4-connected piece of a mask (≥ 0.5) that holds the seed, as 0/1.",
    category: "Regions",
    inputs: [{ name: "in", kind: "grid" }],
    outputs: [{ name: "out", kind: "grid", dtype: "f32" }],
    params: [...SEED_PARAMS, { name: "seedIn", type: "bool", default: false, describe: "the seed is in even when the mask is not" }],
    inferShapes: (inputs) => [inputs[0]!],
  },
  (inputs, params) => {
    const v = inputs[0]!;
    const { width, height } = gridOf(v.shape, "floodFrom");
    return [grid(v.shape, floodFrom(v.data!, width, height, params.col as number, params.row as number, !!params.seedIn))];
  },
);

export const mixtureDistanceOp: OpType = hostOp(
  {
    name: "mixtureDistance",
    label: "Distance to the seed's mixture",
    describe: "Euclidean distance over the channel axis from every pixel to the (interpolated) value at the seed.",
    category: "Regions",
    help: {
      detail:
        "Over a grid with a channel axis (planar). In a whitened metric (channels u = A·z) this is the wand's distance d. The reference is interpolated bilinearly at the seed, so it moves smoothly as the seed does.",
      math: String.raw`d(x) = \lVert u(x) - u(s) \rVert`,
    },
    inputs: [{ name: "in", kind: "grid" }],
    outputs: [{ name: "out", kind: "grid", dtype: "f32" }],
    params: SEED_PARAMS,
    inferShapes: (inputs) => [inputs[0]!],
    inferAxes: () => [undefined],
  },
  (inputs, params) => {
    const v = inputs[0]!;
    const { width, height } = gridOf(v.shape, "mixtureDistance");
    const K = channelAxis(v)?.length ?? 1;
    return [grid(v.shape, mixtureDistance(v.data!, K, width, height, params.col as number, params.row as number))];
  },
);

/** The region pack, for the composer's registrar and the wand. */
export const REGION_OPS: OpType[] = [distanceTransformOp, floodFromOp, mixtureDistanceOp];
