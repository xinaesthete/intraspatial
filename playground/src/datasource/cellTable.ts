// B2b — the zarr-facing half of the cell-centroid ingestion (docs/stream-b-bridge-plan.md).
//
// Reads a SpatialData **regions table** (AnnData, zarr v3) and turns it into per-`cell_type_id`
// centroid clouds, each a graph points source built through the dep-free B2a converter
// (`centroidsToField`). Heavy zarr access + all transform/placement reasoning live here on the
// playground side (ADR-0015 ownership boundary); `src/datasource/centroidsToField.ts` receives only
// resolved values.
//
//   openExtraConsolidated(url) → tree.tables[name]   (zarrextra's own zarrita; consolidated
//                                                      metadata lives at the store root)
//     obsm/spatial   [N,2] float64 → per-cell (x,y) centroids
//     obs/cell_type_id  [N] int64  → per-cell integer type id
//
// ADR-0018: a points cloud is keyed per `(table, region, cell_type)` — we group by `cell_type_id`
// into SEPARATE clouds and NEVER merge them. The centroids are carried, not host-transformed: the
// placement rides along on each cloud and is applied on the GPU (splatDensity).

import type { Affine3 } from "../../../src/datasource";
import { centroidsToField } from "../../../src/datasource";
import type { Graph } from "../../../src/gpu/graph/graph";
import type { FieldProvenance, GpuField, ResolvedPlacement } from "../../../src/gpu/graph/handle";
import { type Affine2, IDENTITY2, type NgffAxis, resolveNgffXY } from "../../../src/spatial/ngffTransform";

/** Default target table on the Leap034 store. */
export const DEFAULT_CELL_TABLE = "Leap034_imc_cells";

/** A single cell type's centroid cloud, ready to add to a graph as a points source (via B2a). */
export interface CellTypeCloud {
  /** The type id this cloud was split on. For a numeric column that is the value itself; for an
   *  AnnData categorical it is the CODE, and `label` carries the human name. */
  readonly id: number;
  /** The category name, when the type column is a categorical (so it carries names). Absent for a
   *  bare integer column, where the store simply does not say what the numbers mean. */
  readonly label?: string;
  /** Number of cells of this type. */
  readonly n: number;
  /** This type's centroids, parallel arrays (for the scatter render; the GPU path packs them). */
  readonly xs: readonly number[];
  readonly ys: readonly number[];
  /** Build this cloud as a graph points source, carrying the shared placement + per-type
   *  provenance. Deferred so one read can seed many graphs / re-runs. */
  readonly source: (g: Graph) => GpuField;
}

export interface CellTable {
  /** Per-`cell_type_id` clouds, ascending by id. Never one merged cloud (ADR-0018). */
  readonly types: CellTypeCloud[];
  /** The shared array→world placement for every cloud (same table ⇒ same system). */
  readonly placement: ResolvedPlacement;
  /** The table's region + instance key (NGFF `region` / `instance_key`), for provenance/HUD. */
  readonly provenance: { region: string; instanceKey: string };
  /** Which table was read, and which `obs` column the types were split on. */
  readonly tableName: string;
  readonly typeColumn: string;
  /** Total cells read across all types. */
  readonly totalCells: number;
  /** The coordinate-system name the placement targets (mirrors `placement.system`, for the HUD). */
  readonly system: string;
  /** Human label for the HUD. */
  readonly label: string;
  /** What the store says about physical scale. See `resolveTableSpace`. */
  readonly units: TableUnits;
}

/** The physical-scale story for a table, deduced where the metadata states it and left explicitly
 *  UNKNOWN where it does not. Every statistic here takes a length (the paper: a 100 µm radius, a
 *  50 µm bandwidth), so the difference between "1 world unit is 1 µm" and "we have no idea" has to
 *  survive all the way to the UI rather than being papered over with a default of 1. */
export interface TableUnits {
  /** The unit string the store gave, verbatim — e.g. "micrometer", or SpatialData's placeholder
   *  "unit". Absent when no axis metadata was found at all. */
  readonly raw?: string;
  /** Micrometres per world unit, when `raw` names a physical length. `undefined` means the store
   *  did not say — NOT that the scale is 1. */
  readonly micrometres?: number;
  /** Which spatial element the transform and unit came from (the table's annotated `region`). */
  readonly via?: string;
  /** Transform types present that the resolver does not implement; non-empty means the placement
   *  is an approximation. */
  readonly unsupported?: string[];
}

// ---- structural zarr shapes (zarrita is zarrextra's dep, not ours — no value import) ---------------

interface ZarrChunk {
  data: ArrayLike<number | bigint>;
  shape: number[];
  stride: number[];
}
interface ZarrArrayLike {
  readonly shape: number[];
  readonly chunks: number[];
  readonly dtype: string;
  getChunk(coords: number[]): Promise<ZarrChunk>;
}
interface LazyZarrArrayLike {
  get(): Promise<ZarrArrayLike>;
}

/** Descend a zarrextra ZarrTree node to a named lazy-array leaf by path, e.g. `["obsm","spatial"]`.
 *  A leaf is identified by exposing a `get()`; group nodes are plain objects. */
function leafAt(node: unknown, path: string[]): LazyZarrArrayLike {
  let cur: unknown = node;
  for (const key of path) {
    if (!cur || typeof cur !== "object") throw new Error(`cellTable: path .${path.join(".")} — '${key}' parent is not a group`);
    cur = (cur as Record<string, unknown>)[key];
  }
  if (!cur || typeof (cur as LazyZarrArrayLike).get !== "function") {
    throw new Error(`cellTable: no zarr array at .${path.join(".")}`);
  }
  return cur as LazyZarrArrayLike;
}

/** Is this tree node a zarr array (as opposed to a group)? Leaves expose `get()`. */
const isLeaf = (v: unknown): v is LazyZarrArrayLike => !!v && typeof v === "object" && typeof (v as LazyZarrArrayLike).get === "function";

/** An AnnData categorical column is a GROUP holding `codes` (int) + `categories` (string).
 *  Detected structurally rather than by the `encoding-type` attribute: the structure is what we
 *  actually need to read, and it does not depend on how the writer spelled its metadata. */
const isCategorical = (v: unknown): boolean =>
  !!v && typeof v === "object" && isLeaf((v as Record<string, unknown>).codes) && isLeaf((v as Record<string, unknown>).categories);

// ---- discovery -------------------------------------------------------------------------------

export interface ColumnInfo {
  readonly name: string;
  readonly kind: "categorical" | "numeric";
  /** Category names, for a categorical (they are small, so we read them during discovery — this is
   *  what lets the UI show real cell-type names rather than bare integers). Only read for columns
   *  still in contention. */
  readonly categories?: string[];
  /** How many categories, for a categorical — known from the array's shape without reading it. */
  readonly nCategories?: number;
  /** Heuristic plausibility as THE cell-type column; higher is better, negative disqualifies. */
  readonly score: number;
}

export interface TableInfo {
  readonly name: string;
  /** Row count, from `obsm/spatial` where present. */
  readonly nRows: number;
  /** Whether the table has the `obsm/spatial` centroids this demo needs. */
  readonly hasCentroids: boolean;
  /** Candidate type columns, best first. */
  readonly columns: ColumnInfo[];
  /** Highest-scoring column, if any scored positively. */
  readonly suggested?: string;
  /** Populated instead of the above when the table could not be inspected. */
  readonly error?: string;
}

/** How plausible is `name` as the cell-type column?
 *
 *  Deliberately a transparent heuristic rather than anything clever: it only picks a DEFAULT, and
 *  the UI always shows the full list so a wrong guess costs one click. Categoricals get a bonus
 *  because they carry names, which is the difference between "type 10 → type 3" and something a
 *  biologist can read. */
function scoreColumn(name: string, kind: "categorical" | "numeric", nCategories?: number): number {
  if (name === "_index") return -100;
  // A column with fewer than two categories partitions nothing — it cannot be the cell type
  // however promising its name. (Leap034's `annot_region` is exactly this: one category, and it
  // out-scored `cell_type_id` until this rule existed.)
  if (kind === "categorical" && nCategories !== undefined && nCategories < 2) return -100;
  let score = kind === "categorical" ? 5 : 0;
  const isCellType = /cell.?type|celltype|phenotype/i.test(name);
  if (isCellType) score += 20;
  else if (/cluster|annot|label|class|subset|population|lineage/i.test(name)) score += 10;
  // `cell_type_id` is a cell-type column that happens to end in `_id`; only penalise names where
  // the `_id` is the whole story.
  if (!isCellType && /(^|_)(id|idx|index)$/i.test(name)) score -= 8;
  if (/area|perim|diam|x$|y$|coord|intensity|mean|median|sum|count|dna|nuclei/i.test(name)) score -= 5;
  return score;
}

function describeColumns(obsNode: unknown): Array<{ name: string; kind: "categorical" | "numeric"; node: unknown }> {
  if (!obsNode || typeof obsNode !== "object") return [];
  const out: Array<{ name: string; kind: "categorical" | "numeric"; node: unknown }> = [];
  for (const [name, node] of Object.entries(obsNode as Record<string, unknown>)) {
    if (isCategorical(node)) out.push({ name, kind: "categorical", node });
    else if (isLeaf(node)) out.push({ name, kind: "numeric", node });
  }
  return out;
}

/** Read a 1-D zarr array of strings (AnnData writes `categories` as vlen-utf8). */
async function readStrings1D(arr: ZarrArrayLike): Promise<string[]> {
  const n = arr.shape[0] ?? 0;
  const cn = arr.chunks[0] ?? (n || 1);
  const out = new Array<string>(n);
  for (let ci = 0; ci < Math.ceil(n / cn); ci++) {
    const chunk = (await arr.getChunk([ci])) as unknown as { data: ArrayLike<unknown>; stride: number[] };
    const base = ci * cn;
    const extent = Math.min(cn, n - base);
    const stride = chunk.stride[0] ?? 1;
    for (let i = 0; i < extent; i++) out[base + i] = String(chunk.data[i * stride]);
  }
  return out;
}

/**
 * Inspect a SpatialData store: which tables are there, and which `obs` column in each one plausibly
 * holds the cell type?
 *
 * This exists because the cell-type column is NOT standardised — `cell_type_id`, `cell_type`,
 * `phenotype`, `cluster`, `annot_*` are all in the wild, sometimes as bare integers whose meaning
 * lives only in a collaborator's head. So rather than hard-code a name and fail on the next store,
 * we enumerate and rank, auto-select when there is no ambiguity, and let the user override.
 *
 * Tables that fail to inspect are reported rather than dropped: a store with one broken table and
 * one good one should still be usable, and the reason should be visible.
 */
export async function listCellTables(url: string): Promise<TableInfo[]> {
  const zx = await import("zarrextra");
  const opened = await zx.openExtraConsolidated(url);
  const { tree } = zx.unwrap(opened) as { tree: { tables?: Record<string, unknown> } };
  const tables = tree.tables ?? {};

  return Promise.all(
    Object.entries(tables).map(async ([name, node]): Promise<TableInfo> => {
      try {
        let nRows = 0;
        let hasCentroids = false;
        try {
          const spatial = await leafAt(node, ["obsm", "spatial"]).get();
          nRows = spatial.shape[0] ?? 0;
          hasCentroids = (spatial.shape[1] ?? 0) >= 2;
        } catch {
          hasCentroids = false;
        }
        const described = describeColumns((node as Record<string, unknown>).obs);
        const columns: ColumnInfo[] = await Promise.all(
          described.map(async ({ name: col, kind, node: colNode }) => {
            // The category COUNT comes from the array's shape (metadata we need anyway) and feeds
            // the score; the names themselves are only read for columns still in contention.
            let catArr: ZarrArrayLike | undefined;
            let nCategories: number | undefined;
            if (kind === "categorical") {
              try {
                catArr = await leafAt(colNode, ["categories"]).get();
                nCategories = catArr.shape[0] ?? 0;
              } catch {
                catArr = undefined;
              }
            }
            const score = scoreColumn(col, kind, nCategories);
            let categories: string[] | undefined;
            if (catArr && score > 0) {
              try {
                categories = await readStrings1D(catArr);
              } catch {
                categories = undefined;
              }
            }
            return { name: col, kind, categories, nCategories, score };
          }),
        );
        columns.sort((a, b) => b.score - a.score || a.name.localeCompare(b.name));
        return { name, nRows, hasCentroids, columns, suggested: columns.find((c) => c.score > 0)?.name };
      } catch (err) {
        return { name, nRows: 0, hasCentroids: false, columns: [], error: err instanceof Error ? err.message : String(err) };
      }
    }),
  );
}

/** Lift a resolved 2-D mapping into the graph's `Affine3` (z left identity — this is a 2-D front).
 *  `axes[0]` is the image of x̂ and `axes[1]` of ŷ, matching `splatDensity`'s grid placement. */
function affine3From(m: Affine2): Affine3 {
  return {
    origin: [m.tx, m.ty, 0],
    axes: [
      [m.a, m.b, 0],
      [m.c, m.d, 0],
      [0, 0, 1],
    ],
  };
}

const IDENTITY_AFFINE: Affine3 = {
  origin: [0, 0, 0],
  axes: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
};

/** Read a full 1-D zarr array into a plain `number[]`, reassembling across its chunk grid.
 *  Handles int64 (`BigInt64Array` → `Number`) as well as float dtypes. */
async function readArray1D(arr: ZarrArrayLike): Promise<number[]> {
  const n = arr.shape[0] ?? 0;
  const cn = arr.chunks[0] ?? (n || 1);
  const out = new Array<number>(n);
  const nChunks = Math.ceil(n / cn);
  for (let ci = 0; ci < nChunks; ci++) {
    const chunk = await arr.getChunk([ci]);
    const base = ci * cn;
    const extent = Math.min(cn, n - base);
    const stride = chunk.stride[0] ?? 1;
    for (let i = 0; i < extent; i++) out[base + i] = Number(chunk.data[i * stride]!);
  }
  return out;
}

/** Read a full `[N,2]` zarr array into parallel `xs`/`ys`, reassembling across the (possibly
 *  column-chunked) grid. The Leap034 store chunks this `[24875,1]`, so column 0 (x) and column 1
 *  (y) land in separate chunks — the generic grid walk below handles that. */
async function readCentroids2D(arr: ZarrArrayLike): Promise<{ xs: number[]; ys: number[] }> {
  const rows = arr.shape[0] ?? 0;
  const cols = arr.shape[1] ?? 0;
  if (cols < 2) throw new Error(`cellTable: obsm/spatial has ${cols} columns, expected >= 2 (x,y)`);
  const cr = arr.chunks[0] ?? (rows || 1);
  const cc = arr.chunks[1] ?? (cols || 1);
  const xs = new Array<number>(rows);
  const ys = new Array<number>(rows);
  const nR = Math.ceil(rows / cr);
  // Only the chunks covering columns 0 and 1 matter.
  const colChunks = new Set([Math.floor(0 / cc), Math.floor(1 / cc)]);
  for (let ir = 0; ir < nR; ir++) {
    const rBase = ir * cr;
    const rExtent = Math.min(cr, rows - rBase);
    for (const ic of colChunks) {
      const chunk = await arr.getChunk([ir, ic]);
      const sR = chunk.stride[0] ?? 1;
      const sC = chunk.stride[1] ?? 1;
      const cBase = ic * cc;
      for (const col of [0, 1]) {
        if (Math.floor(col / cc) !== ic) continue;
        const localC = col - cBase;
        const target = col === 0 ? xs : ys;
        for (let i = 0; i < rExtent; i++) target[rBase + i] = Number(chunk.data[i * sR + localC * sC]!);
      }
    }
  }
  return { xs, ys };
}

/** Read the group attributes (NGFF `region` / `instance_key`) off the table node's zarrextra attrs.
 *  Symbol-keyed in the tree; falls back to sensible Leap034 defaults if absent. */
function readTableAttrs(tableNode: unknown): { region: string; instanceKey: string } {
  const attrs = collectAttrs(tableNode);
  const region = typeof attrs?.region === "string" ? attrs.region : "";
  const instanceKey = typeof attrs?.instance_key === "string" ? attrs.instance_key : "";
  return { region, instanceKey };
}

/** zarrextra stashes a node's zarr.json attributes under a Symbol key; pull the first symbol-keyed
 *  object that looks like NGFF regions-table attrs. */
function collectAttrs(node: unknown): Record<string, unknown> | undefined {
  return symbolAttrs(node, (v) => "region" in v || "instance_key" in v);
}

/** The generic form: the first symbol-keyed plain object on a tree node satisfying `want`. */
function symbolAttrs(node: unknown, want: (v: Record<string, unknown>) => boolean): Record<string, unknown> | undefined {
  if (!node || typeof node !== "object") return undefined;
  for (const sym of Object.getOwnPropertySymbols(node)) {
    const v = (node as Record<symbol, unknown>)[sym];
    if (v && typeof v === "object" && !Array.isArray(v) && want(v as Record<string, unknown>)) {
      return v as Record<string, unknown>;
    }
  }
  return undefined;
}

/**
 * Resolve a table's physical space: find the spatial element it annotates, read that element's
 * `coordinateTransformations`, and take the unit off the output axes.
 *
 * A table is not itself a spatial element and carries no transform — SpatialData does not relate
 * table coordinates to a coordinate system directly. What it does carry is `region`: the name of
 * the element the rows annotate. That element (shapes, labels or an image) IS placed, and
 * `obsm/spatial` centroids live in its space, so its transform is the table's transform. This is
 * the annotation-walk that `MDV/python/mdvtools/spatial/conversion.py` performs on the Python side.
 *
 * Returns identity with `raw: undefined` when nothing can be found — which is a real answer
 * ("unplaced, scale unknown") and deliberately distinct from a resolved identity in µm.
 */
function resolveTableSpace(
  tree: Record<string, unknown>,
  region: string,
  target: string | undefined,
): { affine: Affine2; units: TableUnits } {
  for (const kind of ["shapes", "labels", "points", "images"] as const) {
    const group = tree[kind];
    const node = group && typeof group === "object" ? (group as Record<string, unknown>)[region] : undefined;
    if (!node) continue;
    // Shapes/points put `coordinateTransformations` at the top of their attrs; images and labels
    // nest theirs inside the OME multiscales block, alongside the axes the transforms index.
    const flat = symbolAttrs(node, (v) => "coordinateTransformations" in v);
    const ome = symbolAttrs(node, (v) => "ome" in v || "multiscales" in v);
    let cts: unknown = flat?.coordinateTransformations;
    let axes: NgffAxis[] | undefined;
    if (!cts && ome) {
      const block = (ome.ome as Record<string, unknown> | undefined) ?? ome;
      const ms = (block.multiscales as Array<Record<string, unknown>> | undefined)?.[0];
      cts = ms?.coordinateTransformations;
      axes = ms?.axes as NgffAxis[] | undefined;
    }
    const got = resolveNgffXY(cts, { target, axes });
    if (!got) continue;
    return {
      affine: got.affine,
      units: {
        raw: got.unit,
        micrometres: got.micrometres,
        via: `${kind}/${region}`,
        unsupported: got.unsupported.length ? got.unsupported : undefined,
      },
    };
  }
  return { affine: IDENTITY2, units: {} };
}

/**
 * Read a SpatialData regions table and group its centroids into per-`cell_type_id` clouds.
 *
 * Placement and units come from the element the table ANNOTATES (`resolveTableSpace`): a table is
 * not a spatial element and carries no transform of its own, but its `region` names one that does.
 * The centroids are carried, not host-transformed — the placement rides along on each cloud and is
 * applied on the GPU.
 *
 * ADR-0018: never merges the clouds — one points source per `cell_type_id`, each carrying the
 * shared placement + its own provenance `{ region, instanceKey, cellTypeId }`.
 */
export async function readCellTable(url: string, opts: { table?: string; typeColumn?: string; system?: string } = {}): Promise<CellTable> {
  const tableName = opts.table ?? DEFAULT_CELL_TABLE;
  const typeColumn = opts.typeColumn ?? "cell_type_id";
  const zx = await import("zarrextra");
  const opened = await zx.openExtraConsolidated(url);
  const { tree } = zx.unwrap(opened) as { tree: { tables?: Record<string, unknown> } };
  const tableNode = tree.tables?.[tableName];
  if (!tableNode) {
    const have = Object.keys(tree.tables ?? {}).join(", ") || "(none)";
    throw new Error(`cellTable: no table '${tableName}' in store; available: ${have}`);
  }

  const spatialArr = await leafAt(tableNode, ["obsm", "spatial"]).get();

  // The type column is either a plain integer array or an AnnData categorical (a group of
  // `codes` + `categories`). The categorical case is the one worth having: it comes with NAMES.
  const obsNode = (tableNode as Record<string, unknown>).obs;
  const colNode = obsNode && typeof obsNode === "object" ? (obsNode as Record<string, unknown>)[typeColumn] : undefined;
  if (!colNode) {
    const have = describeColumns(obsNode)
      .map((c) => c.name)
      .join(", ");
    throw new Error(`cellTable: no obs column '${typeColumn}' in '${tableName}'; available: ${have || "(none)"}`);
  }
  let categories: string[] | undefined;
  let idArr: ZarrArrayLike;
  if (isCategorical(colNode)) {
    idArr = await leafAt(colNode, ["codes"]).get();
    categories = await readStrings1D(await leafAt(colNode, ["categories"]).get());
  } else {
    idArr = await leafAt(obsNode, [typeColumn]).get();
  }

  const rows = spatialArr.shape[0] ?? 0;
  if ((idArr.shape[0] ?? 0) !== rows) {
    throw new Error(`cellTable: obsm/spatial rows (${rows}) != obs/${typeColumn} length (${idArr.shape[0]})`);
  }

  const { xs, ys } = await readCentroids2D(spatialArr);
  const ids = await readArray1D(idArr);

  const attrs = readTableAttrs(tableNode);
  const region = attrs.region || opts.system || "";
  const instanceKey = attrs.instanceKey || "cell_id";
  // The target coordinate system: the store's own declared one unless the caller overrides.
  const rootAttrs = symbolAttrs(tree, (v) => "coordinate_system" in v || "spatialdata_attrs" in v);
  const system = opts.system ?? (typeof rootAttrs?.coordinate_system === "string" ? rootAttrs.coordinate_system : "Leap034");

  const space = resolveTableSpace(tree as Record<string, unknown>, region, system);
  const placement: ResolvedPlacement = { system, worldFromArray: affine3From(space.affine) };

  // Group by cell_type_id into separate clouds (ADR-0018 — never one merged cloud).
  const byId = new Map<number, { xs: number[]; ys: number[] }>();
  for (let i = 0; i < rows; i++) {
    const id = ids[i]!;
    let bucket = byId.get(id);
    if (!bucket) {
      bucket = { xs: [], ys: [] };
      byId.set(id, bucket);
    }
    bucket.xs.push(xs[i]!);
    bucket.ys.push(ys[i]!);
  }

  const types: CellTypeCloud[] = [...byId.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([id, cloud]) => {
      const provenance: FieldProvenance = { region, instanceKey, cellTypeId: id };
      // A categorical code of -1 is AnnData's NaN; anything past the end is a broken store. Both
      // fall back to the bare number rather than inventing a name.
      const label = categories?.[id];
      return {
        id,
        label,
        n: cloud.xs.length,
        xs: cloud.xs,
        ys: cloud.ys,
        source: (g: Graph) => g.source(centroidsToField(cloud.xs, cloud.ys, { placement, provenance }), `cellType:${label ?? id}`),
      };
    });

  return {
    types,
    placement,
    provenance: { region, instanceKey },
    tableName,
    typeColumn,
    totalCells: rows,
    system,
    units: space.units,
    label: `${tableName} · ${typeColumn} · ${rows} cells · ${types.length} types`,
  };
}

/** A dep-free synthetic 2-type cloud that flows through the identical path, so the demo works with
 *  no store reachable. Two gaussian blobs, distinct `cell_type_id`s, placed in a fixture system. */
export function syntheticCellTable(opts: { perType?: number; system?: string } = {}): CellTable {
  const perType = opts.perType ?? 800;
  const system = opts.system ?? "fixture";
  const placement: ResolvedPlacement = { system, worldFromArray: IDENTITY_AFFINE };
  // Deterministic PRNG so the fixture is stable.
  let seed = 12345;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  const gauss = () => (rnd() + rnd() + rnd() + rnd() - 2) / 2; // ~N(0, ~0.29)
  const blob = (cx: number, cy: number, spread: number) => {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < perType; i++) {
      xs.push(cx + gauss() * spread);
      ys.push(cy + gauss() * spread);
    }
    return { xs, ys };
  };
  const clouds: Array<{ id: number; blob: { xs: number[]; ys: number[] } }> = [
    { id: 1, blob: blob(30, 30, 12) },
    { id: 2, blob: blob(75, 70, 10) },
  ];
  const types: CellTypeCloud[] = clouds.map(({ id, blob: cloud }) => {
    const provenance: FieldProvenance = { region: "synthetic", instanceKey: "cell_id", cellTypeId: id };
    return {
      id,
      n: cloud.xs.length,
      xs: cloud.xs,
      ys: cloud.ys,
      source: (g: Graph) => g.source(centroidsToField(cloud.xs, cloud.ys, { placement, provenance }), `cellType:${id}`),
    };
  });
  return {
    types,
    placement,
    provenance: { region: "synthetic", instanceKey: "cell_id" },
    totalCells: perType * clouds.length,
    system,
    tableName: "synthetic",
    typeColumn: "cell_type_id",
    units: {},
    label: `synthetic · ${perType * clouds.length} cells · 2 types`,
  };
}
