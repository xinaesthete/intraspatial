// Cell centroids → Gram channels, and the window the Gram is measured over. Pure host code; the page
// and its hooks only decide when to call it.

import type { SpatialData } from "@spatialdata/core";
import { type ChannelCloud, channelsFromExpression } from "../../../src/spatial/gram";
import { type CellTable, readCellTable } from "../datasource/cellTable";
import { readVarColumns, type VarCatalog } from "../datasource/varMatrix";

export const MAX_CHANNELS = 32;

export type Bbox = [number, number, number, number];

/** Where the channels come from: the categories of an `obs` column, or chosen vars of `X`. A cell type
 *  is the one-hot case of a per-cell mark, so both take the identical path from here on. */
export type ChannelSpec =
  | { readonly kind: "obs"; readonly column: string }
  | { readonly kind: "x"; readonly vars: readonly number[]; readonly matrix: string };

/**
 * Apply the table's array→world placement to a cloud, on the host.
 *
 * `gramMatrixGpu` has **no placement facet**: it splats raw `xs`/`ys` straight against the `bbox`,
 * so points and bbox must already be in one space. (`splatDensity` does carry a placement and
 * applies it on the GPU per ADR-0018 — the Gram splat does not, and conflating the two put the whole
 * tissue in a corner covering 1/scale of each axis.) World is the right space to land in: the NGFF
 * scale is real information, every length the statistic takes is a world length, and it is the
 * frame sd.js draws the table's coordinate system in.
 */
function placeCloud(t: CellTable, xs: readonly number[], ys: readonly number[]): { xs: number[]; ys: number[] } {
  const m = t.placement.worldFromArray;
  const n = xs.length;
  const wx = new Array<number>(n);
  const wy = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    wx[i] = m.origin[0] + m.axes[0][0] * xs[i]! + m.axes[1][0] * ys[i]!;
    wy[i] = m.origin[1] + m.axes[0][1] * xs[i]! + m.axes[1][1] * ys[i]!;
  }
  return { xs: wx, ys: wy };
}

/** TIGHT bounds — deliberately unpadded. A cosmetic margin would be indistinguishable from tissue
 *  that happens to be empty, and would silently swallow the window inset: pad by 2%, inset by a
 *  radius smaller than that, and the "inset" window still contains every cell, so the apron comes
 *  out empty while looking like it worked. */
function boundsOf(xs: readonly number[], ys: readonly number[]): Bbox {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    if (xs[i]! < minX) minX = xs[i]!;
    if (xs[i]! > maxX) maxX = xs[i]!;
    if (ys[i]! < minY) minY = ys[i]!;
    if (ys[i]! > maxY) maxY = ys[i]!;
  }
  if (!Number.isFinite(minX)) return [0, 0, 1, 1];
  return [minX, minY, maxX, maxY];
}

export function tableBounds(t: CellTable): Bbox {
  const w = placeCloud(t, t.rowOrder?.xs ?? [], t.rowOrder?.ys ?? []);
  return boundsOf(w.xs, w.ys);
}

export type WindowMode = "apron" | "full" | "half" | "quarter";

export const WINDOW_MODES: readonly { readonly value: WindowMode; readonly label: string }[] = [
  { value: "apron", label: "inset by range — apron populated" },
  { value: "full", label: "full extent — edge-biased" },
  { value: "half", label: "centre 50%" },
  { value: "quarter", label: "centre 25%" },
];

/**
 * The analysis window — a sub-rectangle of the data extent, with the cells outside it as the apron.
 *
 * Insetting is not cosmetic. `gramMatrixGpu` splats every cell it is handed but counts mass only
 * inside the `bbox`, so a window strictly inside the data gets an edge-corrected `g`; a window set
 * to the data's own extent has nothing outside to correct with and carries the full deficit
 * (0.97 at r=5 down to 0.82 at r=25 under CSR — `gram.test.ts`). An inset of one splat radius is
 * all it takes, which is why that is the default.
 */
export function windowBbox(full: Bbox, radius: number, mode: WindowMode): { bbox: Bbox; label: string } {
  const [x0, y0, x1, y1] = full;
  const spanX = x1 - x0;
  const spanY = y1 - y0;
  const shrink = (fx: number, fy: number, label: string) => ({
    bbox: [x0 + spanX * fx, y0 + spanY * fy, x1 - spanX * fx, y1 - spanY * fy] as Bbox,
    label,
  });
  switch (mode) {
    case "full":
      return { bbox: full, label: "full extent (no apron)" };
    case "half":
      return shrink(0.25, 0.25, "centre 50%");
    case "quarter":
      return shrink(0.375, 0.375, "centre 25%");
    default: {
      // Inset by the splat radius on each side — the exact width of the halo the window's own
      // pixels can reach. Clamped so a huge range on a small tissue cannot invert the rectangle.
      const dx = Math.min(radius, spanX * 0.4);
      const dy = Math.min(radius, spanY * 0.4);
      return { bbox: [x0 + dx, y0 + dy, x1 - dx, y1 - dy], label: "inset by range" };
    }
  }
}

/** Raster dims on an AREA budget, so a non-square window is not stretched and world cells stay square. */
export function viewDims(b: Bbox, target: number): { width: number; height: number } {
  const w = Math.max(b[2] - b[0], 1e-9);
  const h = Math.max(b[3] - b[1], 1e-9);
  const scale = Math.sqrt((target * target) / (w * h));
  return { width: Math.max(16, Math.round(w * scale)), height: Math.max(16, Math.round(h * scale)) };
}

/** Build the channel set for a spec, in world coordinates, with a one-line note of what was read. */
export async function buildCellChannels(
  sdata: SpatialData,
  t: CellTable,
  spec: ChannelSpec,
  catalog: VarCatalog | undefined,
): Promise<{ channels: ChannelCloud[]; note: string }> {
  if (spec.kind === "obs") {
    const typed = await readCellTable(sdata, { table: t.tableName, typeColumn: spec.column });
    const channels = typed.types
      .filter((ty) => ty.n > 0)
      .sort((a, b) => b.n - a.n)
      .slice(0, MAX_CHANNELS)
      .map((ty) => ({ label: ty.label ?? `type ${ty.id}`, ...placeCloud(t, ty.xs, ty.ys) }));
    const dropped = typed.types.length - channels.length;
    return {
      channels,
      note: `${channels.length} categories of '${spec.column}'${dropped > 0 ? ` (largest ${MAX_CHANNELS}; ${dropped} dropped)` : ""}`,
    };
  }
  if (!catalog) throw new Error("no X catalogue for this table");
  if (spec.vars.length < 2) throw new Error("pick at least two vars — a 1×1 matrix has nothing to correlate");
  if (!t.rowOrder) throw new Error("this source has no table row order to join X against");
  const names = spec.vars.map((v) => catalog.names[v] ?? `var${v}`);
  const cols = await readVarColumns(sdata, { table: t.tableName, vars: [...spec.vars], matrix: spec.matrix, names });
  if (cols.nCells !== t.rowOrder.xs.length) {
    throw new Error(`X has ${cols.nCells} rows but the table has ${t.rowOrder.xs.length} centroids`);
  }
  const thin = cols.stats.filter((s) => s.nonZero < 20).map((s) => `${s.label} (${s.nonZero})`);
  const w = placeCloud(t, t.rowOrder.xs, t.rowOrder.ys);
  return {
    channels: channelsFromExpression(w.xs, w.ys, cols.values, cols.names),
    note: `${cols.names.length} vars from ${cols.matrix}${thin.length ? ` · sparse: ${thin.join(", ")}` : ""}`,
  };
}
