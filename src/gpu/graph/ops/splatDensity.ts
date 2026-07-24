// Tier-2 graph node wrapping `splatDensityResident` (points -> KDE density grid).
//
// This is the source end of invariant 4: the graph's packed `[x0,y0,...]` points value is bound
// straight into the splat's vertex stage (no host repacking), and the density grid it produces
// stays on the GPU. With this converted, a splat -> convolve -> threshold chain performs no
// transfers at all beyond the source upload and whatever the sink genuinely consumes.
//
// The one thing still resolved host-side is the default `bbox`, which needs the points' own
// bounds. See `resolveBbox` below.

import type { Affine3 } from "../../../coords";
import { gaussianKdeField } from "../../../spatial/scalarField";
import { splatDensityToTexture } from "../../spatial/splatDensity";
import type { FieldValue, ResolvedPlacement, Shape } from "../handle";
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

/** The output grid's array→world placement (ADR-0018): the axis-aligned scale+translate that
 *  maps grid cell index (i, j) to its world position across `bbox` at `width`×`height`.
 *
 *    world = bbox.min + (i, j) · cellSize,   cellSize = (span / resolution)
 *
 *  i.e. cell (0,0) sits at (minX, minY) and one cell step moves one `cellSize` in world. This
 *  records the world→cell relation on the output so a placed point cloud's coordinate system does
 *  not vanish at the points→grid boundary. z is left identity (this is a 2-D splat). */
function gridWorldFromArray(bbox: [number, number, number, number], width: number, height: number): Affine3 {
  const [minX, minY, maxX, maxY] = bbox;
  const cellX = (maxX - minX) / width;
  const cellY = (maxY - minY) / height;
  return {
    origin: [minX, minY, 0],
    axes: [
      [cellX, 0, 0],
      [0, cellY, 0],
      [0, 0, 1],
    ],
  };
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
  // A placement-constructing source (ADR-0018): the splat happens in the points' *own* system —
  // no transform is applied to the points here — so the output grid carries that same `system`,
  // with a fresh `worldFromArray` mapping grid cell → world across the region box. This overrides
  // the pass-through default (which would wrongly copy the points' matrix onto the grid); the
  // executor then stamps it because `execute` leaves `placement` unset, so this concrete placement
  // wins. Two boundary cases, both deliberate:
  //   - Unplaced points ⇒ `undefined`: array space in ⇒ array space out. Do NOT fabricate an
  //     identity placement (array-space and placed-at-identity are distinct states, ADR-0018).
  //   - Placed points but no explicit `bbox`: the world box is only known at execute time from the
  //     points' own bounds, which build-time inference cannot see, so no concrete placement is
  //     recorded. `bbox` is the region selector (until vector-`ParamType`, slice 4); supply it to
  //     get a placed output grid.
  inferPlacement(inputs, params) {
    const pl = inputs[0];
    if (pl === undefined) return [undefined];
    const bbox = bboxParam(params);
    if (bbox === undefined) return [undefined];
    const width = param<number>(params, this.params[0]!);
    const height = param<number>(params, this.params[1]!);
    return [{ system: pl.system, worldFromArray: gridWorldFromArray(bbox, width, height) } satisfies ResolvedPlacement];
  },
  resident: true,
  // Output stays a TEXTURE. The additive splat renders into one anyway, so handing that straight to
  // the executor costs nothing, and a consumer that also renders — a paint to canvas, another pass
  // — then reads it in place. The `copyTextureToBuffer` + de-pad this used to do unconditionally is
  // now the executor's bridge, paid only when a buffer-binding op actually consumes the value.
  async execute(ctx, inputs, params) {
    const inField = inputs[0]!;
    const src = inField.buffer;
    if (!src) throw new Error("splatDensity: resident op received a non-resident input");
    const width = params.width as number,
      height = params.height as number;

    const dst = await ctx.backend.leaseTexture(width, height);
    await splatDensityToTexture(src.buffer, pointCount(inField.shape), dst.texture, {
      width,
      height,
      sigma: params.sigma as number,
      radiusSigma: params.radiusSigma as number,
      bbox: residentBbox(inField, params),
    });
    return [{ shape: { kind: "grid", width, height }, dtype: "f32", texture: dst }];
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
