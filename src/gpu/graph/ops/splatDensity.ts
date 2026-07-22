// Tier-2 graph node wrapping `splatDensityResident` (points -> KDE density grid).
//
// This is the source end of invariant 4: the graph's packed `[x0,y0,...]` points value is bound
// straight into the splat's vertex stage (no host repacking), and the density grid it produces
// stays on the GPU. With this converted, a splat -> convolve -> threshold chain performs no
// transfers at all beyond the source upload and whatever the sink genuinely consumes.
//
// The one thing still resolved host-side is the default `bbox`, which needs the points' own
// bounds. See `resolveBbox` below.

import { gaussianKdeField } from "../../../spatial/scalarField";
import { splatDensityResident } from "../../spatial/splatDensity";
import type { FieldValue, Shape } from "../handle";
import type { OpType, Params } from "../op";
import { param } from "../op";

function unpackXY(v: FieldValue): { xs: number[]; ys: number[] } {
  const d = v.data!;
  const xs: number[] = [],
    ys: number[] = [];
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
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i]!,
      y = ys[i]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (!Number.isFinite(minX)) return [0, 0, 1, 1];
  const pad = 0.12 * Math.max(maxX - minX, maxY - minY, 1) + 1e-6;
  return [minX - pad, minY - pad, maxX + pad, maxY + pad];
}

function pointCount(s: Shape): number {
  if (s.kind !== "points") throw new Error("splatDensity: input must be points");
  return s.n;
}

/** The world box for the resident path.
 *
 *  An explicit `bbox` param wins and needs nothing from the host. Otherwise we fall back to the
 *  points' own bounds, which requires their values — and that is the one place this op is still
 *  host-coupled. It works because the executor keeps `data` alongside `buffer` when it uploads a
 *  host value, so a points *source* still has its array here. A points value produced by some
 *  upstream resident op would not, and we throw rather than silently downloading it: reading the
 *  points back is exactly the transfer this path exists to remove. The fix when that case arrives
 *  is a GPU min/max reduction, not a readback. */
function residentBbox(v: FieldValue, params: Params): [number, number, number, number] {
  const explicit = bboxParam(params);
  if (explicit) return explicit;
  if (!v.data) {
    throw new Error(
      "splatDensity: resident points need an explicit `bbox` param — deriving it from the points " +
        "would require reading them back (ADR-0017 invariant 4). Pass bbox, or add a GPU bounds reduction.",
    );
  }
  const { xs, ys } = unpackXY(v);
  return resolveBbox(xs, ys, params);
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
  resident: true,
  async execute(ctx, inputs, params) {
    const inField = inputs[0]!;
    const src = inField.buffer;
    if (!src) throw new Error("splatDensity: resident op received a non-resident input");
    const width = params.width as number,
      height = params.height as number;

    const dst = await ctx.backend.lease(width * height * 4);
    await splatDensityResident(src.buffer, pointCount(inField.shape), dst.buffer, {
      width,
      height,
      sigma: params.sigma as number,
      radiusSigma: params.radiusSigma as number,
      bbox: residentBbox(inField, params),
    });
    return [{ shape: { kind: "grid", width, height }, dtype: "f32", buffer: dst }];
  },
  cpuGolden(inputs, params) {
    const { xs, ys } = unpackXY(inputs[0]!);
    const width = params.width as number,
      height = params.height as number;
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
