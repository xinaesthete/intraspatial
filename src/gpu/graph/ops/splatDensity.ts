// Tier-1 graph node wrapping `splatDensityGpu` (points -> KDE density grid). The
// legacy function owns its own pipeline/pool/readback; this adapter only marshals
// the graph's `FieldValue`s in and out (download/upload at the boundary). A hot
// path would later promote this to a GPU-resident (Tier-2) op.
import { splatDensityGpu } from "../../spatial/splatDensity";
import type { FieldValue } from "../handle";
import type { OpType, Params } from "../op";
import { param } from "../op";
import { gaussianKdeField } from "../../../spatial/scalarField";

function unpackXY(v: FieldValue): { xs: number[]; ys: number[] } {
  const d = v.data!;
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < d.length; i += 2) {
    xs.push(d[i]!);
    ys.push(d[i + 1]!);
  }
  return { xs, ys };
}

function bboxParam(params: Params): [number, number, number, number] | undefined {
  const b = params.bbox as number[] | undefined;
  return b && b.length === 4 ? [b[0]!, b[1]!, b[2]!, b[3]!] : undefined;
}

// World box for the grid. An explicit `bbox` wins; otherwise we use the points'
// own bounds plus a FIXED fractional margin. Crucially the margin does NOT depend on
// sigma — `splatDensityGpu`'s own default pads by sigma·radiusSigma, which grows the
// world box as fast as the kernel, so raising sigma zoomed out instead of blurring.
// A fixed box means sigma visibly controls the blur.
function resolveBbox(xs: number[], ys: number[], params: Params): [number, number, number, number] {
  const explicit = bboxParam(params);
  if (explicit) return explicit;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i]!, y = ys[i]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (!Number.isFinite(minX)) return [0, 0, 1, 1];
  const pad = 0.12 * Math.max(maxX - minX, maxY - minY, 1) + 1e-6;
  return [minX - pad, minY - pad, maxX + pad, maxY + pad];
}

export const splatDensityOp: OpType = {
  name: "splatDensity",
  label: "KDE splat",
  describe: "Rasterise a point cloud into a Gaussian density grid (points -> grid).",
  inputs: [{ name: "points", kind: "points" }],
  outputs: [{ name: "density", kind: "grid", dtype: "f32" }],
  params: [
    { name: "width", type: "int", default: 64, min: 8, max: 512, describe: "grid width" },
    { name: "height", type: "int", default: 64, min: 8, max: 512, describe: "grid height" },
    { name: "sigma", type: "number", default: 2, min: 0.1, max: 40, step: 0.1, describe: "bandwidth (world units)" },
    { name: "radiusSigma", type: "number", default: 4, min: 1, max: 8, step: 0.5 },
  ],
  inferShapes(_inputs, params) {
    return [{ kind: "grid", width: param<number>(params, this.params[0]!), height: param<number>(params, this.params[1]!) }];
  },
  async execute(_ctx, inputs, params) {
    const { xs, ys } = unpackXY(inputs[0]!);
    const width = params.width as number, height = params.height as number;
    const field = await splatDensityGpu(xs, ys, {
      width,
      height,
      sigma: params.sigma as number,
      radiusSigma: params.radiusSigma as number,
      bbox: resolveBbox(xs, ys, params),
    });
    return [{ shape: { kind: "grid", width, height }, dtype: "f32", data: field.data }];
  },
  cpuGolden(inputs, params) {
    const { xs, ys } = unpackXY(inputs[0]!);
    const width = params.width as number, height = params.height as number;
    const field = gaussianKdeField(xs, ys, {
      width,
      height,
      sigma: params.sigma as number,
      radiusSigma: params.radiusSigma as number,
      bbox: resolveBbox(xs, ys, params),
    });
    return [{ shape: { kind: "grid", width, height }, dtype: "f32", data: field.data }];
  },
};
