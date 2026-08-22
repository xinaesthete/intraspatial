// MDV projects as a columnar zarr store — reader half.
//
// MDV's own on-disk form is a single HDF5 file, one contiguous uncompressed dataset per column
// (`datafile.h5`), with the *meaning* of those columns held separately in `datasources.json`:
// display names, MDV datatypes, and — for every text column — the `values` list that the stored
// integer codes index into. The bytes are already columnar; only the metadata split makes it
// awkward to read from a browser.
//
// So the translation to zarr is close to mechanical, and this module fixes the layout both halves
// agree on (`scripts/mdv-h5-to-zarr.ts` writes it, this reads it):
//
//   <store>.zarr/
//     zarr.json                 group, attrs.mdv = { version, datasources: [{ name, key, rows }] }
//     <ds-key>/
//       zarr.json               group, attrs.mdv = { name, rows, columns: [MdvColumn...] }
//       <col-key>/              array [rows], attrs.mdv = { field, name, datatype, values? }
//
// TWO THINGS ARE DELIBERATE, and both are the reason a `key` exists alongside a `field`.
//
// 1. **Field names are not path-safe.** Real MDV columns include `Cell Type 1`, `%contacts`,
//    `Network(%)`. Those survive a POSIX filesystem, but a zarr store is also read over HTTP, where
//    `%c` in a key is a percent-escape and `Network(%)` is simply a broken URL. Keys are sanitised
//    to `[A-Za-z0-9._-]` and the true field name rides in the attrs, so a store written once reads
//    back identically from disk or from a server.
//
// 2. **Categoricals stay coded.** A text column is stored as its uint8/uint16 codes plus a `values`
//    list, exactly as MDV holds it — not expanded to strings. 545,400 cell annotations are 545 kB
//    as codes and ~12 MB as strings, and the code is what the statistics actually want: a dense
//    small integer per cell.
//
// The reader never consults `datasources.json` — everything it needs was copied into the store's
// attrs at conversion time. That is the point of converting: one artefact, self-describing, and
// readable from the browser with no server.

import * as zarr from "zarrita";

/** Store-format version written into the root attrs. Bumped when the layout changes. */
export const MDV_ZARR_VERSION = 1;

/** One column's metadata, as carried in the array's `attrs.mdv`. */
export interface MdvColumn {
  /** MDV field name — the key used in `datasources.json` and in view configs. May contain spaces
   *  and punctuation (`Cell Type 1`, `%contacts`). */
  readonly field: string;
  /** Path-safe zarr array key. Derived from `field`; unique within the datasource. */
  readonly key: string;
  /** Human display name (often equal to `field`, but e.g. `MH_SES` → "Quadrat Correlation …"). */
  readonly name: string;
  /** MDV datatype: `double` | `integer` | `text` | `multitext` | `unique`. */
  readonly datatype: string;
  /** Category labels for a text column; the stored value is an index into this array. */
  readonly values?: readonly string[];
}

/** MDV's `regions` block for a datasource: which columns are positional, which splits the table into
 *  regions, and — the part that matters for any statistic taking a length — the physical scale. */
export interface MdvRegions {
  /** Positional column names, `[x, y]`. */
  readonly position_fields?: readonly string[];
  /** The column that splits rows into regions (covid: `sample_id`). */
  readonly region_field?: string;
  /** Physical size of one world unit, in `scale_unit`s. */
  readonly scale?: number;
  /** Unit `scale` is expressed in — `mm`, `um`, … */
  readonly scale_unit?: string;
  readonly default_color?: string;
  /** Per-region metadata, including the region's images and their pixel dimensions. */
  readonly all_regions?: Readonly<Record<string, unknown>>;
}

/** One datasource (an MDV "table") — `cells`, `samples`, `spatial_stats`, … */
export interface MdvDatasourceInfo {
  readonly name: string;
  readonly key: string;
  readonly rows: number;
  readonly columns: readonly MdvColumn[];
  /** Present on a spatial table; absent on a plain one. */
  readonly regions?: MdvRegions;
}

/**
 * Micrometres per world unit, from MDV's `regions.scale` + `scale_unit`.
 *
 * `undefined` when the metadata does not name a physical length — which is NOT the same as 1, and
 * must stay distinguishable all the way to the UI. Every statistic here takes a length (a 100 µm
 * radius, a 50 µm bandwidth), so a silent default of 1 would turn "we don't know the scale" into a
 * confidently wrong number. The covid project says `scale: 0.001, scale_unit: "mm"`, i.e. exactly
 * 1 µm per unit.
 */
export function micrometresPerUnit(regions: MdvRegions | undefined): number | undefined {
  if (!regions || typeof regions.scale !== "number" || !Number.isFinite(regions.scale)) return undefined;
  const per: Record<string, number> = { m: 1e6, mm: 1e3, um: 1, µm: 1, micrometer: 1, micrometre: 1, micron: 1, nm: 1e-3 };
  const factor = per[(regions.scale_unit ?? "").trim().toLowerCase()];
  return factor === undefined ? undefined : regions.scale * factor;
}

interface RootAttrs {
  readonly version: number;
  readonly datasources: readonly { name: string; key: string; rows: number }[];
}

/** A text column read back as codes + labels. `codes[i]` indexes `labels`. */
export interface CategoricalColumn {
  readonly codes: Uint8Array | Uint16Array | Int32Array;
  readonly labels: readonly string[];
}

/** Cells labelled by an integer type id, in one region — the shape `src/spatial/pcf.ts` consumes. */
export interface RegionCells {
  readonly region: string;
  readonly xs: Float64Array;
  readonly ys: Float64Array;
  /** Per-cell type code, index-aligned to xs/ys. Indexes `typeLabels`. */
  readonly typeId: Int32Array;
  readonly typeLabels: readonly string[];
}

/** Path-safe key for a field name. Anything outside `[A-Za-z0-9._-]` becomes `_`; a leading digit
 *  or a collision is disambiguated by the caller (see `assignKeys`). */
export function sanitiseKey(field: string): string {
  const s = field.replace(/[^A-Za-z0-9._-]/g, "_");
  return s.length === 0 ? "_" : s;
}

/** Assign unique path-safe keys to a list of field names, preserving order. Collisions after
 *  sanitising (`Network(%)` and `Network__` would both become `Network___`) get a `__2`, `__3`
 *  suffix — deterministic, so a re-conversion produces the same store. */
export function assignKeys(fields: readonly string[]): string[] {
  const used = new Set<string>();
  return fields.map((f) => {
    const base = sanitiseKey(f);
    let key = base;
    let n = 2;
    while (used.has(key)) key = `${base}__${n++}`;
    used.add(key);
    return key;
  });
}

/** An opened MDV zarr store. Metadata is read once at open; column data is read on demand. */
export class MdvStore {
  private constructor(
    private readonly loc: zarr.Location<never>,
    readonly datasources: readonly MdvDatasourceInfo[],
  ) {}

  static async open(location: string): Promise<MdvStore> {
    const loc = await openZarrLocation(location);
    const root = await zarr.open(loc, { kind: "group" });
    const attrs = (root.attrs as { mdv?: RootAttrs }).mdv;
    if (!attrs) throw new Error(`${location} is not an MDV zarr store (no root 'mdv' attrs)`);
    if (attrs.version !== MDV_ZARR_VERSION) {
      throw new Error(`MDV zarr store version ${attrs.version}, expected ${MDV_ZARR_VERSION} — re-run scripts/mdv-h5-to-zarr.ts`);
    }
    const dss: MdvDatasourceInfo[] = [];
    for (const d of attrs.datasources) {
      const g = await zarr.open(loc.resolve(d.key), { kind: "group" });
      const meta = (g.attrs as { mdv?: { name: string; rows: number; columns: MdvColumn[]; regions?: MdvRegions } }).mdv;
      if (!meta) throw new Error(`datasource '${d.name}' has no 'mdv' attrs`);
      dss.push({ name: meta.name, key: d.key, rows: meta.rows, columns: meta.columns, ...(meta.regions ? { regions: meta.regions } : {}) });
    }
    return new MdvStore(loc, dss);
  }

  datasource(name: string): MdvDatasourceInfo {
    const d = this.datasources.find((x) => x.name === name);
    if (!d) throw new Error(`no datasource '${name}' — have ${this.datasources.map((x) => x.name).join(", ")}`);
    return d;
  }

  column(dsName: string, field: string): MdvColumn {
    const ds = this.datasource(dsName);
    const c = ds.columns.find((x) => x.field === field);
    if (!c) throw new Error(`datasource '${dsName}' has no column '${field}'`);
    return c;
  }

  /** Read a whole column as its stored typed array. */
  async read(dsName: string, field: string): Promise<zarr.TypedArray<zarr.NumberDataType>> {
    const ds = this.datasource(dsName);
    const col = this.column(dsName, field);
    const arr = await zarr.open(this.loc.resolve(`${ds.key}/${col.key}`), { kind: "array" });
    const chunk = await zarr.get(arr as never, null);
    return (chunk as { data: zarr.TypedArray<zarr.NumberDataType> }).data;
  }

  /** Read a column as f64, whatever it is stored as. Convenience for numeric work. */
  async readF64(dsName: string, field: string): Promise<Float64Array> {
    const data = await this.read(dsName, field);
    return data instanceof Float64Array ? data : Float64Array.from(data as ArrayLike<number>);
  }

  /** Read a text column as codes + labels. Throws if the column carries no `values` list. */
  async readCategorical(dsName: string, field: string): Promise<CategoricalColumn> {
    const col = this.column(dsName, field);
    if (!col.values) throw new Error(`column '${field}' of '${dsName}' is not categorical (no values list)`);
    const codes = await this.read(dsName, field);
    if (!(codes instanceof Uint8Array || codes instanceof Uint16Array || codes instanceof Int32Array)) {
      throw new Error(`column '${field}' of '${dsName}' has values but is stored as ${codes.constructor.name}`);
    }
    return { codes, labels: col.values };
  }
}

/** http(s) or a local path. Mirrors `annDataIo.openStore` — the node import stays dynamic so a
 *  browser bundle never pulls `node:fs` in. */
async function openZarrLocation(location: string): Promise<zarr.Location<never>> {
  if (/^https?:\/\//.test(location)) {
    return zarr.root(new zarr.FetchStore(location)) as unknown as zarr.Location<never>;
  }
  const mod = (await import("@zarrita/storage/fs")) as { default: new (path: string) => never };
  return zarr.root(new mod.default(location));
}

export interface RegionCellsOptions {
  /** Positional columns. Default `x` / `y`. */
  readonly xField?: string;
  readonly yField?: string;
  /** The categorical column that splits cells into types. */
  readonly typeField: string;
  /** The categorical column that splits cells into regions (MDV's `regions.region_field`). */
  readonly regionField: string;
  /** Restrict the type axis to these labels, in this order. Labels not present in the type column
   *  are an error; cells whose type is absent from the list are DROPPED.
   *
   *  Comparing against a precomputed MDV stats table needs exactly this: that table's cell-type
   *  axis is its own categorical, whose order differs from the cell table's and which omits some
   *  labels (`NA`). Passing its `values` list here puts both on one axis. */
  readonly typeLabels?: readonly string[];
  /** Compare type labels case-insensitively. The covid project needs it: the cell table says
   *  `Blood vessels` and the stats table says `Blood Vessels`, and nothing else differs. */
  readonly caseInsensitiveTypes?: boolean;
}

/** Split a cell table into per-region labelled point clouds.
 *
 *  One pass over the coordinate and code columns, so the cost is the read, not the split. */
export async function readRegionCells(store: MdvStore, dsName: string, opts: RegionCellsOptions): Promise<Map<string, RegionCells>> {
  const xs = await store.readF64(dsName, opts.xField ?? "x");
  const ys = await store.readF64(dsName, opts.yField ?? "y");
  const type = await store.readCategorical(dsName, opts.typeField);
  const region = await store.readCategorical(dsName, opts.regionField);
  const n = xs.length;
  if (ys.length !== n || type.codes.length !== n || region.codes.length !== n) {
    throw new Error(`readRegionCells: column lengths disagree in '${dsName}'`);
  }

  const typeLabels = opts.typeLabels ? [...opts.typeLabels] : [...type.labels];
  const norm = (s: string) => (opts.caseInsensitiveTypes ? s.toLowerCase() : s);
  const wanted = new Map(typeLabels.map((l, i) => [norm(l), i]));
  // Source type code → target type index, or -1 to drop the cell.
  const remap = new Int32Array(type.labels.length);
  for (let i = 0; i < type.labels.length; i++) remap[i] = wanted.get(norm(type.labels[i]!)) ?? -1;

  // Count first so each region's arrays are allocated exactly once.
  const nRegions = region.labels.length;
  const kept = new Int32Array(nRegions);
  for (let i = 0; i < n; i++) if (remap[type.codes[i]!]! >= 0) kept[region.codes[i]!]!++;

  const out = new Map<string, RegionCells>();
  const cursor = new Int32Array(nRegions);
  const build = region.labels.map((label, r) => {
    const m = kept[r]!;
    const rc = { region: label, xs: new Float64Array(m), ys: new Float64Array(m), typeId: new Int32Array(m), typeLabels };
    if (m > 0) out.set(label, rc);
    return rc;
  });
  for (let i = 0; i < n; i++) {
    const t = remap[type.codes[i]!]!;
    if (t < 0) continue;
    const r = region.codes[i]!;
    const rc = build[r]!;
    const k = cursor[r]!++;
    rc.xs[k] = xs[i]!;
    rc.ys[k] = ys[i]!;
    rc.typeId[k] = t;
  }
  return out;
}
