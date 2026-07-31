// MDV project → `CellTable`, so the cell-stats front runs on a real MDV project's cells.
//
// The reader (`src/datasource/mdvStore.ts`) is generic and tested; this is the playground-side glue
// that turns it into the same `CellTable` the zarr and CSV paths produce. Everything downstream —
// splats, TCM, cross-PCF, the N-way matrix — is unchanged, because it only ever sees a `CellTable`.
//
// ONE THING IS GENUINELY NEW HERE, and it is not glue: **an MDV cell table holds many regions**.
// The covid project is 545,400 cells across 32 ROIs, and every statistic in the paper is computed
// *per ROI* — ρ_B is a density over one ROI, the edge correction is against one ROI's boundary. So
// this loader takes a region and returns that region's cells, and the UI grows a region picker.
// Loading all 32 at once would not be a bigger dataset, it would be a wrong one: the cross-PCF of
// 32 disjoint tissue sections pooled into one bounding box measures the gaps between sections.
//
// UNITS COME FROM THE PROJECT. MDV's `regions` block states `scale` + `scale_unit`, which the store
// resolves to µm-per-unit (`micrometresPerUnit`). For covid that is exactly 1 µm per unit, so the
// paper's r = 100 µm and σ = 50 µm are literal. Where a project does not say, `units.micrometres`
// stays undefined and the UI shows unitless — see `cellTable.TableUnits`, and `docs/cell-stats.md`
// §6 on why "unknown" must not collapse to 1.

import type { Affine3 } from "../../../src/datasource";
import { centroidsToField } from "../../../src/datasource";
import {
  type MdvDatasourceInfo,
  type MdvStore,
  micrometresPerUnit,
  type RegionCells,
  readRegionCells,
} from "../../../src/datasource/mdvStore";
import type { Graph } from "../../../src/gpu/graph/graph";
import type { FieldProvenance, ResolvedPlacement } from "../../../src/gpu/graph/handle";
import type { CellTable, CellTypeCloud } from "./cellTable";

export { MdvStore } from "../../../src/datasource/mdvStore";

const IDENTITY_AFFINE: Affine3 = {
  origin: [0, 0, 0],
  axes: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
};

/** Above this many categories a column is an identifier, not a cell type — same reasoning (and
 *  number) as `cellTable.MAX_CELL_TYPES`: the N-way matrix is N×N. */
const MAX_CELL_TYPES = 512;

export interface MdvTypeColumn {
  readonly name: string;
  readonly nCategories: number;
  readonly categories: readonly string[];
  /** Heuristic plausibility as THE cell-type column; higher is better, negative disqualifies. */
  readonly score: number;
}

export interface MdvTableInfo {
  readonly name: string;
  readonly rows: number;
  /** Categorical columns that could plausibly be the type axis, best first. */
  readonly typeColumns: readonly MdvTypeColumn[];
  /** The column MDV itself names as the region splitter, when the table declares one. */
  readonly regionColumn?: string;
  /** Region names, from the region column's categories. */
  readonly regions: readonly string[];
  /** Positional columns MDV declares, `[x, y]`. */
  readonly positionFields?: readonly string[];
  /** Resolved µm per world unit, or undefined when the project does not say. */
  readonly micrometres?: number;
  readonly unitRaw?: string;
  /** Best-guess type column, when one scored positively. */
  readonly suggested?: string;
}

/** Names that, bare, are a table's PRIMARY cell-type axis rather than a variant of it. */
const CANONICAL_TYPE_NAMES = new Set(["annotation", "annotations", "celltype", "cell_type", "celltypes", "cell_types", "type", "types"]);

/**
 * Rank a categorical as a candidate cell-type axis.
 *
 * The bonus for an unqualified name is what makes this useful rather than merely plausible, and the
 * covid project is why. It offers `annotations` (50 types — the axis its own published spatial
 * statistics are computed on) alongside `broad_celltypes` and `global_annotations` (5 each) and
 * `myeloid_` / `structural_` / `lymphocyte_annotations` (16–19, each a lineage subset that is
 * unassigned outside its lineage). All six match the same keyword and all six are plausible; only
 * one is the annotation the rest are derived FROM. A qualifier in the name is the signal — a column
 * called `broad_x` is announcing that it is the coarsened variant of some `x`.
 */
function scoreTypeColumn(name: string, nCategories: number, regionColumn?: string): number {
  if (name === regionColumn) return -100; // the region splitter is never the type axis
  if (nCategories < 2 || nCategories > MAX_CELL_TYPES) return -50;
  const n = name.toLowerCase();
  let s = 0;
  if (/(^|_)annotation/.test(n)) s += 10;
  if (/cell.?type|celltype/.test(n)) s += 10;
  if (/cluster/.test(n)) s += 6;
  if (/(^|_)type($|_)/.test(n)) s += 4;
  if (CANONICAL_TYPE_NAMES.has(n)) s += 8; // unqualified — the primary axis, not a variant of it
  if (/broad|coarse|global|fine|major|minor/.test(n)) s -= 3; // explicitly a re-grained variant
  if (/sample|roi|condition|disease|state|batch|slide|region/.test(n)) s -= 12;
  // A handful of categories is a condition; tens is a cell-type annotation.
  if (nCategories >= 5 && nCategories <= 200) s += 3;
  if (nCategories <= 4) s -= 4;
  return s;
}

/** Inspect a datasource: what could be the type axis, what splits it into regions, what the scale
 *  is. Reads metadata only — no column data — so it is cheap on a 545,400-row table. */
export function inspectMdvTable(ds: MdvDatasourceInfo): MdvTableInfo {
  const regionColumn = ds.regions?.region_field;
  const cats = ds.columns.filter((c) => c.values && c.values.length > 0);
  const typeColumns: MdvTypeColumn[] = cats
    .map((c) => ({
      name: c.field,
      nCategories: c.values!.length,
      categories: c.values!,
      score: scoreTypeColumn(c.field, c.values!.length, regionColumn),
    }))
    .sort((a, b) => b.score - a.score || a.nCategories - b.nCategories);
  const regionCol = cats.find((c) => c.field === regionColumn);
  const best = typeColumns[0];
  return {
    name: ds.name,
    rows: ds.rows,
    typeColumns,
    ...(regionColumn ? { regionColumn } : {}),
    regions: regionCol ? [...regionCol.values!].sort() : [],
    ...(ds.regions?.position_fields ? { positionFields: ds.regions.position_fields } : {}),
    ...(micrometresPerUnit(ds.regions) !== undefined ? { micrometres: micrometresPerUnit(ds.regions) } : {}),
    ...(ds.regions?.scale_unit ? { unitRaw: ds.regions.scale_unit } : {}),
    ...(best && best.score > 0 ? { suggested: best.name } : {}),
  };
}

/** Datasources that carry spatial cells — the ones worth offering as a source. */
export function spatialDatasources(store: MdvStore): MdvDatasourceInfo[] {
  return store.datasources.filter((d) => d.regions?.position_fields && d.regions.position_fields.length >= 2);
}

export interface MdvReadOptions {
  readonly datasource: string;
  readonly typeColumn: string;
  /** Which region to load. Required — see the header on why "all of them" is not an option. */
  readonly region: string;
  /** Drop these type labels (case-insensitive). Defaults to `["NA"]`: MDV annotation columns carry
   *  an explicit unassigned category, and it is not a cell type — pooling it would give the matrix a
   *  row and column of "cells we could not call". */
  readonly dropTypes?: readonly string[];
  /** For the HUD. */
  readonly label?: string;
}

/** Read one region of an MDV cell table as a `CellTable`. */
export async function readMdvCellTable(store: MdvStore, opts: MdvReadOptions): Promise<CellTable> {
  const ds = store.datasource(opts.datasource);
  const info = inspectMdvTable(ds);
  if (!info.regionColumn) throw new Error(`'${opts.datasource}' declares no region column — nothing to split ROIs on`);
  const typeCol = ds.columns.find((c) => c.field === opts.typeColumn);
  if (!typeCol?.values) throw new Error(`'${opts.typeColumn}' is not a categorical column of '${opts.datasource}'`);

  const drop = new Set((opts.dropTypes ?? ["NA"]).map((s) => s.toLowerCase()));
  const typeLabels = typeCol.values.filter((v) => !drop.has(v.toLowerCase()));
  if (typeLabels.length < 1) throw new Error(`'${opts.typeColumn}' has no usable categories`);
  if (typeLabels.length > MAX_CELL_TYPES) {
    throw new Error(`'${opts.typeColumn}' has ${typeLabels.length} categories — that is an identifier, not a cell type`);
  }

  const [xField, yField] = info.positionFields ?? ["x", "y"];
  const byRegion = await readRegionCells(store, opts.datasource, {
    xField,
    yField,
    typeField: opts.typeColumn,
    regionField: info.regionColumn,
    typeLabels,
  });
  const cells = byRegion.get(opts.region);
  if (!cells) throw new Error(`region '${opts.region}' has no cells in '${opts.datasource}'`);

  return regionToCellTable(cells, {
    system: opts.region,
    tableName: opts.datasource,
    typeColumn: opts.typeColumn,
    label: opts.label ?? `${opts.datasource} · ${opts.region}`,
    micrometres: info.micrometres,
    unitRaw: info.unitRaw,
  });
}

interface ToTableOptions {
  system: string;
  tableName: string;
  typeColumn: string;
  label: string;
  micrometres?: number;
  unitRaw?: string;
}

/** Group one region's cells by type into the `CellTable` shape. Types with no cells in this region
 *  are DROPPED rather than carried as empty clouds — the pickers and the N-way matrix are per-ROI,
 *  and 20 empty rows would be 20 rows of nothing to hover. */
function regionToCellTable(cells: RegionCells, opts: ToTableOptions): CellTable {
  const placement: ResolvedPlacement = { system: opts.system, worldFromArray: IDENTITY_AFFINE };
  const K = cells.typeLabels.length;
  const xs: number[][] = Array.from({ length: K }, () => []);
  const ys: number[][] = Array.from({ length: K }, () => []);
  for (let i = 0; i < cells.xs.length; i++) {
    const t = cells.typeId[i]!;
    xs[t]!.push(cells.xs[i]!);
    ys[t]!.push(cells.ys[i]!);
  }

  const types: CellTypeCloud[] = [];
  for (let id = 0; id < K; id++) {
    const n = xs[id]!.length;
    if (n === 0) continue;
    const label = cells.typeLabels[id]!;
    const cx = xs[id]!;
    const cy = ys[id]!;
    const provenance: FieldProvenance = { region: opts.system, instanceKey: "row", cellTypeId: id };
    types.push({
      id,
      label,
      n,
      xs: cx,
      ys: cy,
      source: (g: Graph) => g.source(centroidsToField(cx, cy, { placement, provenance }), `cellType:${label}`),
    });
  }

  const total = types.reduce((s, t) => s + t.n, 0);
  const unit = opts.micrometres !== undefined ? `${opts.micrometres} µm/unit` : "unitless";
  return {
    types,
    placement,
    provenance: { region: opts.system, instanceKey: "row" },
    tableName: opts.tableName,
    typeColumn: opts.typeColumn,
    units: {
      ...(opts.unitRaw ? { raw: opts.unitRaw } : {}),
      ...(opts.micrometres !== undefined ? { micrometres: opts.micrometres } : {}),
      via: opts.system,
    },
    totalCells: total,
    system: opts.system,
    label: `${opts.label} · ${opts.typeColumn} · ${total} cells · ${types.length} types · ${unit}`,
    // No `rowOrder`. It is defined as the join key back to the table's OTHER columns, indexed by
    // table row — and these cells are one region's, with the unassigned category dropped, so their
    // order is neither. Supplying it would silently mis-join any expression weight read later.
    // Restoring it means carrying the original row indices through `readRegionCells`.
  };
}
