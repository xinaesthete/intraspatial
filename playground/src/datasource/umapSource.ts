// Where the UMAP page gets its feature matrix: a synthetic manifold, or a real AnnData
// table out of a SpatialData store.
//
// Both produce the same `UmapDataset`, so nothing downstream — the graph build, the
// optimiser, the programme toggles, the colour channels — knows which it is holding. That
// is the point of the module: the interesting question is whether the behaviour you learn
// to read on the synthetic shapes still reads the same way on real data, and it can only
// be asked if the two go through identical code.
//
// **The real path is a selection, not a full load** (ADR-0005). A dense `[nCells, nGenes]`
// matrix at 100k cells and 2000 genes is 1.6 GB as f64, so the gene count is capped and
// the cap is the honest limit rather than a slow path. What the cap does NOT bound is the
// zarr read: a CSR matrix scatters each gene across every cell's row, so pulling 20 genes
// costs the same full scan as pulling 2000 (`selectionCost` reports it).

import { pcaGpu } from "../../../src/gpu/spatial/pcaGpu";
import { type PcaResult, pca } from "../../../src/spatial/pca";
import { expressManifold, type FeatureBlock, type Manifold, makeManifold } from "../../../src/spatial/syntheticManifolds";
import {
  type ColumnInfo,
  isLeaf,
  leafAt,
  listCellTables,
  openSpatialData,
  readStrings1D,
  type TableInfo,
  type ZarrArrayLike,
} from "./cellTable";
import { listVars, readVarColumns, selectionCost, type VarCatalog } from "./varMatrix";

/** One feature matrix plus everything the page needs to describe and colour it. */
export interface UmapDataset {
  readonly kind: "synthetic" | "store";
  readonly title: string;
  readonly n: number;
  readonly dim: number;
  /** Row-major `[n, dim]` — already reduced, so the graph build runs with `pca: false`. */
  readonly values: Float32Array;
  /** Toggleable groups of columns: latent axes for a manifold, principal components for a
   *  real table. Both answer the same question — "what happens if this signal is not
   *  available to the manifold?". */
  readonly blocks: FeatureBlock[];
  readonly label: Uint8Array;
  readonly labelNames: string[];
  readonly truth?: Float32Array;
  readonly truthName?: string;
  readonly truthCyclic?: boolean;
  /** For a generator, what a faithful embedding should look like. */
  readonly expect?: string;
  /** Provenance, shown under the title. */
  readonly note: string;
}

// --- synthetic ------------------------------------------------------------------------

export interface SyntheticOptions {
  readonly n: number;
  readonly seed?: number;
  readonly genesPerAxis?: number;
  /** Override the generator's own default noise. */
  readonly noise?: number;
}

export function syntheticDataset(key: string, opts: SyntheticOptions): UmapDataset & { manifold: Manifold } {
  const m = makeManifold(key, opts.n, opts.seed ?? 11);
  const e = expressManifold(m, { genesPerAxis: opts.genesPerAxis, noise: opts.noise, seed: opts.seed });
  return {
    kind: "synthetic",
    title: key,
    manifold: m,
    n: m.n,
    dim: e.dim,
    values: e.values,
    blocks: e.blocks,
    label: m.label,
    labelNames: m.labelNames,
    truth: m.truth,
    truthName: m.truthName,
    truthCyclic: m.truthCyclic,
    expect: m.expect,
    note: `${m.n.toLocaleString()} cells · ${e.dim} genes · ${m.nLatent} programmes · noise ${e.noise}× extent`,
  };
}

// --- real store -----------------------------------------------------------------------

/** What a store offers, so the UI can be populated before anything expensive is read. */
export interface StoreCatalog {
  readonly tables: TableInfo[];
  readonly vars: Record<string, VarCatalog>;
}

/** Enumerate tables and their expression matrices. Cheap: attributes and the `var` index,
 *  never the matrix itself. */
export async function inspectStore(url: string): Promise<StoreCatalog> {
  const sdata = await openSpatialData(url);
  const tables = await listCellTables(sdata);
  const vars: Record<string, VarCatalog> = {};
  for (const t of tables) vars[t.name] = await listVars(sdata, t.name);
  return { tables, vars };
}

export interface StoreLoadOptions {
  readonly table: string;
  readonly matrix?: string;
  /** Hard cap on genes read into memory. The dense result is `nCells * maxGenes` f64. */
  readonly maxGenes?: number;
  /**
   * Cap on cells kept, by uniform random subsample.
   *
   * This used to exist because the k-NN was quadratic: an exact GPU search over the whole
   * 162,254-cell Xenium table took over two minutes in the browser. With NN-descent on the
   * device that search is seconds, so the cap is now about the READ and the memory it
   * costs — a dense `[nCells, nGenes]` block of f64 — rather than about the search.
   *
   * A uniform sample preserves the manifold's shape and loses only the rarest populations,
   * which is the conventional trade for an interactive view.
   */
  readonly maxCells?: number;
  /** Keep this many highest-variance genes of those read. */
  readonly nHvg?: number;
  /** Principal components retained — these become the programme toggles. */
  readonly nComponents?: number;
  /** `log1p` before PCA. On by default: raw counts are heavy-tailed enough that a handful
   *  of high-expressing genes otherwise dominate every component. */
  readonly log1p?: boolean;
  /** An `obs` column to colour by. */
  readonly labelColumn?: string;
  /** Progress, so a multi-second read is not a frozen page. */
  readonly onProgress?: (message: string) => void;
}

/** `maxCells` 100000 is set by the dense read, not by the search: at 30 components the
 *  device k-NN handles that in a couple of seconds (§3), while the matrix it comes from is
 *  hundreds of MB. Raise it if the store is narrow. */
const DEFAULTS = { maxGenes: 400, nHvg: 300, nComponents: 30, log1p: true, maxCells: 100000 };

/**
 * Load a real table as a feature matrix.
 *
 * The pipeline is the conventional one — select genes, `log1p`, highly-variable, PCA —
 * and it is here rather than inside `umap()` because the page needs the intermediate: the
 * principal components ARE the programme toggles, so they have to exist before the graph
 * is built.
 */
export async function storeDataset(url: string, opts: StoreLoadOptions): Promise<UmapDataset> {
  const say = opts.onProgress ?? (() => {});
  const maxGenes = opts.maxGenes ?? DEFAULTS.maxGenes;
  const nHvg = opts.nHvg ?? DEFAULTS.nHvg;
  const nComponents = opts.nComponents ?? DEFAULTS.nComponents;
  const matrix = opts.matrix ?? "X";

  const sdata = await openSpatialData(url);
  const cat = await listVars(sdata, opts.table);
  if (cat.error) throw new Error(`umapSource: ${opts.table}: ${cat.error}`);
  if (cat.nCells < 10) throw new Error(`umapSource: ${opts.table} has only ${cat.nCells} cells`);

  // Which genes to read. When the panel fits under the cap, all of them; otherwise a
  // deterministic RANDOM sample rather than the first N — var order in a store is
  // alphabetical or arbitrary, so "the first 400" is a biased slice dressed up as a
  // default, while a fixed-seed sample is at least unbiased and reproducible.
  const wanted = cat.nVars <= maxGenes ? Array.from({ length: cat.nVars }, (_, i) => i) : sampleIndices(cat.nVars, maxGenes, 0x51ed);
  const cost = selectionCost(cat, wanted.length);
  say(`reading ${wanted.length} of ${cat.nVars} genes (${(cost * 100).toFixed(0)}% of the stored matrix)…`);

  const cols = await readVarColumns(sdata, {
    table: opts.table,
    vars: wanted,
    matrix,
    names: wanted.map((v) => cat.names[v] ?? `var${v}`),
  });
  const nCells = cols.nCells;
  say(`${nCells.toLocaleString()} cells x ${wanted.length} genes read; selecting variable genes…`);

  // `readVarColumns` is column-major, which is the right layout for the variance pass and
  // the wrong one for everything after it; the transpose happens once, below.
  const values = cols.values;
  const log1p = opts.log1p ?? DEFAULTS.log1p;
  if (log1p) for (let t = 0; t < values.length; t++) values[t] = Math.log1p(Math.max(0, values[t]!));

  const variance = new Float64Array(wanted.length);
  for (let g = 0; g < wanted.length; g++) {
    const base = g * nCells;
    let mean = 0;
    for (let i = 0; i < nCells; i++) mean += values[base + i]!;
    mean /= nCells;
    let acc = 0;
    for (let i = 0; i < nCells; i++) acc += (values[base + i]! - mean) ** 2;
    variance[g] = acc / Math.max(1, nCells - 1);
  }
  const keep = Array.from({ length: wanted.length }, (_, g) => g)
    .sort((a, b) => variance[b]! - variance[a]!)
    .slice(0, Math.min(nHvg, wanted.length))
    // Back into var order, so the gene list stays readable rather than variance-sorted.
    .sort((a, b) => a - b);
  // A gene with zero variance carries nothing and makes the covariance singular.
  const hvg = keep.filter((g) => variance[g]! > 0);
  if (hvg.length < 2) throw new Error(`umapSource: ${opts.table}/${matrix} has fewer than 2 genes with any variance`);

  // Subsample AFTER the variance pass, so highly-variable genes are chosen from the whole
  // table rather than from the sample — the gene selection is cheap to do properly and is
  // the part a subsample would bias most.
  const maxCells = Math.max(10, opts.maxCells ?? DEFAULTS.maxCells);
  const rows = nCells <= maxCells ? undefined : sampleIndices(nCells, maxCells, 0xc0ffee);
  const outCells = rows?.length ?? nCells;
  if (rows) say(`subsampling ${outCells.toLocaleString()} of ${nCells.toLocaleString()} cells…`);

  const dense = new Float32Array(outCells * hvg.length);
  for (let c = 0; c < hvg.length; c++) {
    const base = hvg[c]! * nCells;
    for (let i = 0; i < outCells; i++) dense[i * hvg.length + c] = values[base + (rows ? rows[i]! : i)]!;
  }

  say(`PCA over ${hvg.length} variable genes…`);
  const comps = Math.min(nComponents, hvg.length, outCells);
  const reduced = await reducePca(dense, outCells, hvg.length, comps, say);

  const full = await readObsLabels(sdata, opts.table, nCells, opts.labelColumn);
  const label = rows ? Uint8Array.from(rows, (r) => full.label[r]!) : full.label;

  return {
    kind: "store",
    title: `${opts.table} · ${matrix}`,
    n: outCells,
    dim: reduced.nComponents,
    values: reduced.scores,
    blocks: componentBlocks(reduced.nComponents, reduced.explainedVarianceRatio),
    label,
    labelNames: full.labelNames,
    note:
      `${outCells.toLocaleString()}${rows ? ` of ${nCells.toLocaleString()}` : ""} cells · ${hvg.length} of ${cat.nVars} genes · ` +
      `${reduced.nComponents} PCs${log1p ? " · log1p" : ""}${full.column ? ` · coloured by ${full.column}` : ""}`,
  };
}

/**
 * PCA on the device, falling back to the host if there is no usable GPU.
 *
 * The fallback is not politeness — it is the difference between a page that loads slowly
 * and a page that does not load. But it is worth *saying* when it happens, because the two
 * are very far apart: the host path is 6 seconds of frozen tab at 60k x 300 and 24 at 200k
 * (`src/spatial/pca.ts` has the table), against well under a second on the device. A user
 * who thinks the page has hung deserves to know it is a fallback rather than the norm.
 */
async function reducePca(dense: Float32Array, n: number, dim: number, nComponents: number, say: (msg: string) => void): Promise<PcaResult> {
  try {
    return await pcaGpu(dense, n, dim, { nComponents, standardise: true });
  } catch (err) {
    say(
      `PCA on the GPU failed (${err instanceof Error ? err.message : String(err)}); falling back to the host — this will block for a few seconds…`,
    );
    return pca(dense, n, dim, { nComponents, standardise: true });
  }
}

/** Deterministic sample of `count` distinct indices below `n`, in ascending order. */
function sampleIndices(n: number, count: number, seed: number): number[] {
  const picked = new Set<number>();
  let a = seed >>> 0;
  while (picked.size < count) {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    picked.add(((t ^ (t >>> 14)) >>> 0) % n);
  }
  return [...picked].sort((x, y) => x - y);
}

/**
 * Principal components as toggleable blocks.
 *
 * The leading components get a control each — those are the ones carrying a named split,
 * and switching PC2 off to watch a separation vanish is the real-data equivalent of
 * switching a gene programme off. The tail goes into one bundle: individually they are
 * noise, collectively they are the difference between a crisp embedding and a smear, and
 * a control per component would be twenty buttons nobody would press.
 */
function componentBlocks(nComponents: number, explained: Float64Array): FeatureBlock[] {
  const individual = Math.min(10, nComponents);
  const blocks: FeatureBlock[] = [];
  for (let c = 0; c < individual; c++) {
    blocks.push({ name: `PC${c + 1} (${(explained[c]! * 100).toFixed(1)}%)`, columns: [c] });
  }
  if (nComponents > individual) {
    let rest = 0;
    for (let c = individual; c < nComponents; c++) rest += explained[c]!;
    blocks.push({
      name: `PC${individual + 1}–${nComponents} (${(rest * 100).toFixed(1)}%)`,
      columns: Array.from({ length: nComponents - individual }, (_, i) => individual + i),
    });
  }
  return blocks;
}

// --- obs labels -----------------------------------------------------------------------

/** Read a whole 1-D zarr array as numbers, chunk by chunk. */
async function readNumbers1D(arr: ZarrArrayLike): Promise<Float64Array> {
  const n = arr.shape[0] ?? 0;
  const cn = arr.chunks[0] ?? (n || 1);
  const out = new Float64Array(n);
  for (let ci = 0; ci < Math.ceil(n / cn); ci++) {
    const chunk = await arr.getChunk([ci]);
    const base = ci * cn;
    const extent = Math.min(cn, n - base);
    const stride = chunk.stride[0] ?? 1;
    for (let i = 0; i < extent; i++) out[base + i] = Number(chunk.data[i * stride]!);
  }
  return out;
}

/** Categorical obs columns are a group of `codes` + `categories`, which is the case worth
 *  having because it comes with names. Anything else is read as bare integers. */
export async function readObsLabels(
  sdata: Awaited<ReturnType<typeof openSpatialData>>,
  table: string,
  nCells: number,
  column?: string,
): Promise<{ label: Uint8Array; labelNames: string[]; column?: string }> {
  const none = { label: new Uint8Array(nCells), labelNames: ["all cells"] };
  if (!column) return none;
  try {
    const tree = (sdata as unknown as { rootStore: { tree: { tables?: Record<string, unknown> } } }).rootStore.tree;
    const obs = (tree.tables?.[table] as Record<string, unknown> | undefined)?.obs;
    const node = obs && typeof obs === "object" ? (obs as Record<string, unknown>)[column] : undefined;
    if (!node) return none;

    let codes: Float64Array;
    let names: string[];
    if (isLeaf(node)) {
      codes = await readNumbers1D(await node.get());
      const distinct = [...new Set(Array.from(codes))].sort((a, b) => a - b);
      names = distinct.map((v) => String(v));
      const remap = new Map(distinct.map((v, i) => [v, i]));
      codes = Float64Array.from(codes, (v) => remap.get(v) ?? 0);
    } else {
      codes = await readNumbers1D(await leafAt(node, ["codes"]).get());
      names = await readStrings1D(await leafAt(node, ["categories"]).get());
    }
    if (codes.length !== nCells) return none;

    // The colour palette is generated, so there is no fixed ceiling — but a column with
    // hundreds of levels is a cell id, not a label, and colouring by it says nothing.
    if (names.length > 64) return none;
    const label = new Uint8Array(nCells);
    // AnnData writes -1 for a missing categorical; it becomes a trailing "(unassigned)"
    // level rather than silently colouring as category 0.
    let missing = false;
    for (let i = 0; i < nCells; i++) {
      const c = codes[i]!;
      if (c < 0 || c >= names.length) {
        missing = true;
        label[i] = names.length;
      } else label[i] = c;
    }
    return { label, labelNames: missing ? [...names, "(unassigned)"] : names, column };
  } catch {
    return none;
  }
}

/** Categorical obs columns worth offering as a colour channel. */
export function labelColumnsOf(info: TableInfo | undefined): ColumnInfo[] {
  return (info?.columns ?? []).filter((c) => c.kind === "categorical" && (c.nCategories ?? 0) > 1 && (c.nCategories ?? 0) <= 64);
}
