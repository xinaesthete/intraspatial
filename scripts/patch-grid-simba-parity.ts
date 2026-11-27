#!/usr/bin/env tsx
// Parity check: the patch grid (R4) feeding zonal statistics (R3), end to end, against SIMBA's
// per-patch tables on the registered TNBC rasters. Builds the support masks and cell centroids
// from the OME-TIFFs (row bands, no full-image buffer), lays the grid with `patchGrid`, hands the
// kept patches to `zonalStatsTiled` as `patchPolygons`, and compares:
//   regions   bounds, ids, edge flags, support values   <leap>_patch_regions.csv
//   copper    laicpmsi_cu_p50, n_tissue_pixels, tissue_fraction   <leap>_copper_patch_features.csv
//   collagen  imc_collagen_median, imc_acquired_fraction   features/<run>/<leap>_imc_patch_features.csv
//
// SIMBA's rules (TNBC-MMB src/feature_extraction, src/continuous_anchor_multimodal_rel):
//   copper    footprint = tissue = any of Cu/Zn/Fe/Mg/Mn != 0; keep tissue_fraction >= 0.5
//             (and a finite Cu p50); Cu p50 = np.median(Cu[patch & tissue])
//   collagen  footprint = imc_roi_label_map > 0; keep acquired_fraction >= 0.5 and >= 50 cells,
//             a cell = a (ROI, local id) pair of imc_cell_segmentation with a decoded type
//             (max imc_cell_types id in 1..21), placed at its rounded, clamped centroid;
//             imc_collagen_median = np.median(collagen[patch & acquired])
//
//   pnpm patchgrid:parity [--leap Leap034] [--anchor copper|collagen] [--size 300]
//                         [--batch 3000] [--registered ~/data/tnbc_registered_res_1.0] [--simba ~/data/tnbc_simba_runs]
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import { fromFile, type GeoTIFFImage } from "geotiff";
import { TileCache } from "../src/datasource/tileCache";
import type { ChunkId, Loader, Multiscale, Tile } from "../src/datasource/types";
import { getDevice, releaseDevice } from "../src/gpu/device";
import { type TiledRaster, zonalStatsTiled } from "../src/gpu/spatial/zonalTiled";
import {
  type MaskRuns,
  type Patch,
  maskRunsFromRows,
  patchGrid,
  patchPolygons,
  pointPixels,
  positiveFinite,
  type SupportRule,
} from "../src/spatial/patchGrid";

function fail(message: string): never {
  console.error(`patchgrid:parity: ${message}`);
  process.exit(2);
}
/** `--name value`, or `dflt` when the flag is absent; a flag without a value is an error. */
const arg = (name: string, dflt: string): string => {
  const i = process.argv.indexOf(`--${name}`);
  if (i < 0) return dflt;
  const v = process.argv[i + 1];
  if (v === undefined || v.startsWith("--")) fail(`--${name} needs a value`);
  return v;
};
/** A whole number >= 1 (checked before any raster is read: building Leap035's masks takes a minute). */
const positiveInt = (name: string, dflt: string): number => {
  const v = Number(arg(name, dflt));
  if (!Number.isInteger(v) || v < 1) fail(`--${name} must be a whole number >= 1`);
  return v;
};
const leap = arg("leap", "Leap034");
const anchor = arg("anchor", "copper");
if (anchor !== "copper" && anchor !== "collagen") fail(`--anchor must be copper or collagen (got ${anchor})`);
const size = positiveInt("size", "300");
if (size % 2 !== 0) fail(`--size must be even: SIMBA's stride is half the size (got ${size})`);
const stride = size / 2;
// Patches per zonalStatsTiled call. The exact pass holds every value in each row's target bins
// (zonal-stats.md, "Exact-pass memory"): 14k overlapping 300 µm patches on Leap035 collect 73M
// values, past the default 128 MiB binding. Rows are independent and ids run top to bottom, so
// a batch is a band of the slide and touches mostly its own tiles.
const batch = positiveInt("batch", "3000");
const registered = arg("registered", join(homedir(), "data/tnbc_registered_res_1.0"));
const simba = arg("simba", join(homedir(), "data/tnbc_simba_runs"));
const METALS = ["Cu", "Zn", "Fe", "Mg", "Mn"];
const BAND = 512;

const image = async (file: string): Promise<GeoTIFFImage> => (await fromFile(join(registered, leap, file))).getImage(0);

/** Rows [y0, y1) of band 0, full width. */
async function band(img: GeoTIFFImage, y0: number, y1: number): Promise<ArrayLike<number>> {
  const r = await img.readRasters({ window: [0, y0, img.getWidth(), y1], interleave: false, samples: [0] });
  return r[0] as ArrayLike<number>;
}

/** Runs of "any of these images is non-zero" (numpy's `a != 0 | b != 0`, NaN included), read in
 *  row bands into one byte per pixel (522 MB for Leap035). */
async function nonZeroRuns(imgs: readonly GeoTIFFImage[]): Promise<MaskRuns> {
  const w = imgs[0]!.getWidth();
  const h = imgs[0]!.getHeight();
  const mask = new Uint8Array(w * h);
  for (let y0 = 0; y0 < h; y0 += BAND) {
    const y1 = Math.min(h, y0 + BAND);
    for (const img of imgs) {
      const v = await band(img, y0, y1);
      for (let i = 0; i < (y1 - y0) * w; i++) if (v[i] !== 0) mask[y0 * w + i] = 1;
    }
  }
  return maskRunsFromRows(w, h, (y) => mask.subarray(y * w, (y + 1) * w), positiveFinite);
}

/** Labelled cells' centroids (x = column, y = row), SIMBA's `build_global_segmentation` +
 *  `extract_cell_properties` rule. */
async function cellCentroids(): Promise<{ xs: Float64Array; ys: Float64Array }> {
  const [roi, seg, types] = await Promise.all(
    ["imc_roi_label_map.ome.tiff", "imc_cell_segmentation.ome.tiff", "imc_cell_types.ome.tiff"].map(image),
  );
  const w = roi!.getWidth();
  const h = roi!.getHeight();
  const cells = new Map<number, { n: number; sx: number; sy: number; t: number }>();
  for (let y0 = 0; y0 < h; y0 += BAND) {
    const y1 = Math.min(h, y0 + BAND);
    const [r, s, t] = await Promise.all([roi!, seg!, types!].map((img) => band(img, y0, y1)));
    for (let i = 0; i < (y1 - y0) * w; i++) {
      if (r![i] === 0 || s![i] === 0) continue;
      const key = r![i]! * 65536 + s![i]!;
      let c = cells.get(key);
      if (!c) cells.set(key, (c = { n: 0, sx: 0, sy: 0, t: 0 }));
      c.n++;
      c.sx += i % w;
      c.sy += y0 + Math.floor(i / w);
      if (t![i]! > c.t) c.t = t![i]!;
    }
  }
  const kept = [...cells.values()].filter((c) => c.t >= 1 && c.t <= 21);
  return { xs: Float64Array.from(kept, (c) => c.sx / c.n), ys: Float64Array.from(kept, (c) => c.sy / c.n) };
}

/** One OME-TIFF as a one-level pyramid of its own tiles (as `zonal-simba-parity.ts`). */
async function tiffRaster(file: string): Promise<TiledRaster> {
  const img = await image(file);
  const [w, h, cw, ch] = [img.getWidth(), img.getHeight(), img.getTileWidth(), img.getTileHeight()];
  const ms: Multiscale = {
    voxelDims0: [w, h, 1],
    chunkShape: [cw, ch, 1],
    levelCount: 1,
    placements: [
      {
        system: "global",
        worldFromArray: {
          origin: [0, 0, 0],
          axes: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
          ],
        },
      },
    ],
    element: { kind: "scalar" },
    dtype: "f32",
  };
  const loader: Loader<Tile> = {
    async getChunk(id: ChunkId): Promise<Tile> {
      const x0 = id.x * cw;
      const y0 = id.y * ch;
      const x1 = Math.min(w, x0 + cw);
      const y1 = Math.min(h, y0 + ch);
      const r = await img.readRasters({ window: [x0, y0, x1, y1], interleave: false, samples: [0] });
      return {
        id,
        dims: [x1 - x0, y1 - y0, 1],
        element: { kind: "scalar" },
        dtype: "f32",
        data: Float32Array.from(r[0] as ArrayLike<number>),
      };
    },
  };
  return { ms, loader, cache: new TileCache<Tile>({ maxBytes: 2 ** 28 }) };
}

async function csv(path: string): Promise<Record<string, string>[]> {
  const [head, ...lines] = (await readFile(path, "utf8")).trim().split(/\r?\n/);
  const cols = head!.split(",");
  return lines.map((l) => Object.fromEntries(l.split(",").map((v, i) => [cols[i]!, v])));
}

const key = (x0: number, y0: number, x1: number, y1: number) => `${x0},${y0},${x1},${y1}`;
const rowKey = (r: Record<string, string>) => key(+r.start_x!, +r.start_y!, +r.end_x!, +r.end_y!);

/** Compare one per-patch value; `ours` is indexed like `patches`. */
function report(
  what: string,
  patches: readonly Patch[],
  ours: ArrayLike<number>,
  theirs: Map<string, Record<string, string>>,
  col: string,
  tol = 0,
) {
  let n = 0;
  let worst = 0;
  const off: string[] = [];
  patches.forEach((p, i) => {
    const r = theirs.get(key(p.x0, p.y0, p.x1, p.y1));
    if (!r || r[col] === "" || r[col] === undefined) return;
    n++;
    const a = ours[i]!;
    const b = Number(r[col]);
    // NaN on one side only (or ±Inf against a finite value) is a mismatch, not a NaN that compares
    // false against the tolerance; NaN on both sides agrees.
    const bothNaN = Number.isNaN(a) && Number.isNaN(b);
    const rel =
      a === b || bothNaN ? 0 : Number.isFinite(a) && Number.isFinite(b) ? Math.abs(a - b) / Math.max(1e-30, Math.abs(b)) : Infinity;
    worst = Math.max(worst, rel);
    if (rel > tol) off.push(`${r.patch_id}: ours ${a}, SIMBA ${b}`);
  });
  console.log(`${what}: ${n} patches compared, max rel diff ${worst.toExponential(2)}, ${off.length} over ${tol}`);
  for (const o of off.slice(0, 5)) console.log(`   ${o}`);
}

async function main(): Promise<void> {
  const run =
    anchor === "copper"
      ? `copper_level/regions/mintissuefrac0.5_patchum${size}_strideum${stride}`
      : `${size === 300 ? "collagen_level" : `collagen_level_${size}`}/regions/minacqfrac0.5_mincells50_patchum${size}_strideum${stride}`;
  // Fail on a missing SIMBA table before the rasters are read, not after.
  const regionsCsv = join(simba, run, `${leap}_patch_regions.csv`);
  if (!existsSync(regionsCsv)) fail(`no SIMBA patch table for ${leap} ${anchor} ${size} µm: ${regionsCsv}`);
  const t0 = performance.now();
  let footprint: MaskRuns;
  const rules: SupportRule[] = [];
  if (anchor === "copper") {
    footprint = await nonZeroRuns(await Promise.all(METALS.map((m) => image(`laicpmsi_processed_channel_${m}.ome.tiff`))));
    rules.push({ kind: "fraction", name: "laicpmsi_tissue_fraction", mask: footprint, min: 0.5 });
  } else {
    footprint = await nonZeroRuns([await image("imc_roi_label_map.ome.tiff")]);
    const { xs, ys } = await cellCentroids();
    rules.push({ kind: "fraction", name: "imc_acquired_fraction", mask: footprint, min: 0.5 });
    rules.push({ kind: "count", name: "n_imc_cells", points: pointPixels(xs, ys, footprint.width, footprint.height), min: 50 });
  }
  const grid = patchGrid({ footprint, size, stride, units: "um", pixelSize: 1, rules });
  const tGrid = performance.now();

  const regions = new Map((await csv(regionsCsv)).map((r) => [rowKey(r), r]));
  const mine = new Set(grid.patches.map((p) => key(p.x0, p.y0, p.x1, p.y1)));
  const onlySimba = [...regions.keys()].filter((k) => !mine.has(k)).length;
  const onlyOurs = [...mine].filter((k) => !regions.has(k)).length;
  const idOff = grid.patches.filter((p) => {
    const r = regions.get(key(p.x0, p.y0, p.x1, p.y1));
    return r && (Number(r.patch_id!.slice(6)) !== p.id || (r.edge_patch === "True") !== p.edge);
  }).length;
  console.log(
    `${leap} ${anchor} ${size}/${stride} µm: ${grid.patches.length} kept of ${grid.candidates.length} candidates (SIMBA ${regions.size}); ` +
      `only SIMBA ${onlySimba}, only ours ${onlyOurs}, id/edge differences ${idOff}; masks + grid ${((tGrid - t0) / 1000).toFixed(1)} s`,
  );
  grid.rules.forEach((name, k) =>
    report(
      `grid ${name}`,
      grid.patches,
      grid.patches.map((p) => p.support[k]!),
      regions,
      name,
    ),
  );

  // R3 on the kept patches: polygon k = patch k.
  const device = await getDevice();
  const polygons = patchPolygons(grid.patches);
  const channels =
    anchor === "copper"
      ? [
          {
            source: await tiffRaster("laicpmsi_processed_channel_Cu.ome.tiff"),
            support: await Promise.all(METALS.map((m) => tiffRaster(`laicpmsi_processed_channel_${m}.ome.tiff`))),
            name: "cu",
          },
        ]
      : [
          {
            source: await tiffRaster("imc_processed_channel_Collage-Type_I.ome.tiff"),
            support: await tiffRaster("imc_roi_label_map.ome.tiff"),
            name: "collagen",
          },
        ];
  const t1 = performance.now();
  const columns = new Map<string, number[]>();
  let tiles = 0;
  for (let k = 0; k < polygons.length; k += batch) {
    const res = await zonalStatsTiled(
      {
        level: 0,
        regions: { kind: "polygons", polygons: polygons.slice(k, k + batch) },
        channels,
        bins: 1024,
        quantiles: [0.25, 0.5, 0.75],
        exact: true,
      },
      device,
    );
    tiles += res.tiles;
    for (const [name, v] of res.table.columns) {
      const all = columns.get(name) ?? [];
      for (const x of v) all.push(x);
      columns.set(name, all);
    }
  }
  const calls = Math.ceil(polygons.length / batch);
  console.log(
    `zonalStatsTiled: ${polygons.length} patch polygons in ${calls} call(s), ${tiles} tile visits, ${((performance.now() - t1) / 1000).toFixed(1)} s`,
  );
  const col = (name: string) => columns.get(name)!;
  if (anchor === "copper") {
    const features = new Map((await csv(join(simba, run, `${leap}_copper_patch_features.csv`))).map((r) => [rowKey(r), r]));
    report("zonal cu_median vs laicpmsi_cu_p50", grid.patches, col("cu_median"), features, "laicpmsi_cu_p50");
    report("zonal cu_supported vs laicpmsi_n_tissue_pixels", grid.patches, col("cu_supported"), features, "laicpmsi_n_tissue_pixels");
    report(
      "zonal cu_supportFraction vs laicpmsi_tissue_fraction",
      grid.patches,
      col("cu_supportFraction"),
      features,
      "laicpmsi_tissue_fraction",
    );
  } else {
    const features = new Map(
      (await csv(join(simba, run.replace("/regions/", "/features/"), `${leap}_imc_patch_features.csv`))).map((r) => [rowKey(r), r]),
    );
    // np.median on float32 averages the two middle values in f32; we interpolate in f64 (R3's doc).
    report("zonal collagen_median vs imc_collagen_median", grid.patches, col("collagen_median"), features, "imc_collagen_median", 1e-6);
    report(
      "zonal collagen_supportFraction vs imc_acquired_fraction",
      grid.patches,
      col("collagen_supportFraction"),
      regions,
      "imc_acquired_fraction",
    );
  }
}

// Release the device on failure too (a no-op when none was opened).
await main().finally(releaseDevice);
