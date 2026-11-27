// Tier-2 graph node wrapping `patchSupportResident`: support pixels of a mask grid inside each of
// n rectangles. R4 of the region/raster statistics plan (docs/patch-grid.md).
//
// The rectangles are a `matrix` input, `n × 4` u32 rows `[x0, y0, x1, y1]` (`rectsArray` of a
// `patchGrid`'s patches). The patch grid itself is not a node: how many patches it keeps depends
// on the mask's contents, and a node's output shape must be known at build (`inferShapes`). So the
// layout runs on the host (`patchGrid`, fast over row runs) and this node re-measures support on
// a resident mask — a threshold or morphology output, say — without bringing it back.
//
// A grid is f32 (support = finite and > 0) or u32 (support = non-zero). Output: `n × 1` u32.
import { patchSupportResident } from "#src/gpu/spatial/patchSupport";
import type { FieldValue, Shape } from "../handle";
import type { OpType } from "../op";

function gridOf(s: Shape): { width: number; height: number } {
  if (s.kind !== "grid") throw new Error("patchSupport: mask must be a grid");
  return { width: s.width, height: s.height };
}

function rectCount(s: Shape): number {
  if (s.kind !== "matrix" || s.cols !== 4) throw new Error("patchSupport: rects must be an n × 4 matrix [x0, y0, x1, y1]");
  return s.rows;
}

function rectsData(v: FieldValue, width: number, height: number): Uint32Array {
  const d = v.data;
  if (!d) throw new Error("patchSupport: cpuGolden needs host rectangles");
  const out = Uint32Array.from(d);
  for (let i = 0; i < out.length; i += 4)
    if (out[i + 2]! > width || out[i + 3]! > height || out[i]! > out[i + 2]! || out[i + 1]! > out[i + 3]!)
      throw new Error(`patchSupport: rectangle ${i / 4} leaves the canvas or is inverted`);
  return out;
}

/** Support pixels of a dense row-major mask in each `[x0, y0, x1, y1]` rectangle: the golden. */
export function patchSupportCpu(mask: ArrayLike<number>, width: number, dtype: "f32" | "u32", rects: Uint32Array): Uint32Array {
  const on = dtype === "f32" ? (v: number) => v > 0 && v !== Infinity : (v: number) => v !== 0;
  const out = new Uint32Array(rects.length / 4);
  for (let i = 0; i < out.length; i++) {
    let s = 0;
    for (let y = rects[i * 4 + 1]!; y < rects[i * 4 + 3]!; y++)
      for (let x = rects[i * 4]!; x < rects[i * 4 + 2]!; x++) if (on(mask[y * width + x]!)) s++;
    out[i] = s;
  }
  return out;
}

export const patchSupportOp: OpType = {
  name: "patchSupport",
  label: "Patch support (pixels per rectangle)",
  category: "Statistics",
  describe: "Count a mask's support pixels inside each rectangle (n × 4 [x0, y0, x1, y1]) with one prefix scan.",
  help: {
    detail:
      "Support is finite and > 0 for an f32 mask, non-zero for u32. A flat exclusive scan over the mask gives each row's " +
      "prefix, so a rectangle sums one difference per row. Divide by the rectangle's area for a support fraction. " +
      "Rectangles come from `patchGrid` (`rectsArray`).",
    math: "c_i = \\sum_{y=y_0}^{y_1-1} P[yW + x_1] - P[yW + x_0]",
  },
  inputs: [
    { name: "mask", kind: "grid" },
    { name: "rects", kind: "matrix", dtype: "u32" },
  ],
  outputs: [{ name: "counts", kind: "matrix", dtype: "u32" }],
  params: [],
  inferShapes(inShapes) {
    gridOf(inShapes[0]!);
    return [{ kind: "matrix", rows: rectCount(inShapes[1]!), cols: 1 }];
  },
  inferPlacement: () => [undefined],
  inferAxes(inAxes) {
    const planes = (inAxes[0] ?? []).reduce((n, ax) => n * ax.length, 1);
    if (planes !== 1) throw new Error("patchSupport: the mask must be single-channel");
    return [undefined];
  },
  inferRole: () => [undefined],
  resident: true,
  async execute(ctx, ins) {
    const [mask, rects] = ins as [FieldValue, FieldValue];
    const { width, height } = gridOf(mask.shape);
    const nRects = rectCount(rects.shape);
    if (!mask.buffer || !rects.buffer) throw new Error("patchSupport: resident op received a non-resident input");
    const counts = await ctx.backend.lease(Math.max(nRects, 1) * 4);
    try {
      await patchSupportResident(
        {
          mask: mask.buffer.buffer,
          maskOffset: mask.buffer.offset ?? 0,
          encoding: mask.dtype === "f32" ? "f32" : "u32",
          width,
          height,
          rects: rects.buffer.buffer,
          rectsOffset: rects.buffer.offset ?? 0,
          nRects,
          counts: counts.buffer,
        },
        await ctx.backend.getDevice(),
      );
    } catch (err) {
      ctx.backend.release(counts);
      throw err;
    }
    return [{ shape: { kind: "matrix", rows: nRects, cols: 1 }, dtype: "u32", buffer: counts }];
  },
  cpuGolden(ins) {
    const [mask, rects] = ins as [FieldValue, FieldValue];
    const { width, height } = gridOf(mask.shape);
    if (!mask.data) throw new Error("patchSupport: cpuGolden needs host mask data");
    const data = patchSupportCpu(mask.data, width, mask.dtype === "f32" ? "f32" : "u32", rectsData(rects, width, height));
    return [{ shape: { kind: "matrix", rows: rectCount(rects.shape), cols: 1 }, dtype: "u32", data }];
  },
};
