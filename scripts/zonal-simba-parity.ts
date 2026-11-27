#!/usr/bin/env tsx
// Parity check: zonal statistics (R3) against SIMBA's per-ST-ROI features, on the registered
// TNBC rasters. Reads the OME-TIFFs tile by tile (their own 512² tiles, as a one-level pyramid),
// runs `zonalStatsTiled` on the GPU with the exact second pass, and compares with SIMBA's
// `st_roi_features/<leap>_aoi_features.csv`. The data stays where it is; nothing is copied.
//
// SIMBA's rules (TNBC-MMB src/feature_extraction; mask_dilation_pct = 0 in these runs):
//   region         st_day_a_roi_label_map == roi_id             (st.py RoiMaskSource)
//   imc_collagen_median = np.median(collagen[region & imc_roi_label_map > 0])   (imc.py)
//   laicpmsi_cu_p50     = np.median(Cu[region & tissue]), tissue = any of Cu/Zn/Fe/Mg/Mn != 0
//   laicpmsi_n_tissue_pixels = |region & tissue|, tissue_fraction = that / |region|  (laicpmsi.py)
// which here are: labels = the ST ROI map; collagen with support = the IMC ROI map; Cu with
// support = the five metals (any non-zero); valid = "finite"; exact quantiles.
//
//   pnpm zonal:parity [--leap Leap034] [--registered ~/data/tnbc_registered_res_1.0]
//                     [--simba ~/data/tnbc_simba_runs] [--bins 1024]
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import { fromFile } from "geotiff";
import type { ChunkId, Loader, Multiscale, Tile } from "../src/datasource/types";
import { TileCache } from "../src/datasource/tileCache";
import { getDevice, releaseDevice } from "../src/gpu/device";
import { type TiledRaster, zonalStatsTiled } from "../src/gpu/spatial/zonalTiled";

const arg = (name: string, dflt: string) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1]! : dflt;
};
const leap = arg("leap", "Leap034");
const registered = arg("registered", join(homedir(), "data/tnbc_registered_res_1.0"));
const simba = arg("simba", join(homedir(), "data/tnbc_simba_runs"));
const bins = Number(arg("bins", "1024"));

/** One OME-TIFF as a one-level pyramid whose chunks are the file's own tiles. */
async function tiffRaster(file: string, cache: TileCache<Tile>): Promise<TiledRaster> {
  const tiff = await fromFile(join(registered, leap, file));
  const image = await tiff.getImage(0);
  const w = image.getWidth();
  const h = image.getHeight();
  const cw = image.getTileWidth();
  const ch = image.getTileHeight();
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
      const rasters = await image.readRasters({ window: [x0, y0, x1, y1], interleave: false, samples: [0] });
      const band = rasters[0] as ArrayLike<number>;
      return { id, dims: [x1 - x0, y1 - y0, 1], element: { kind: "scalar" }, dtype: "f32", data: Float32Array.from(band) };
    },
  };
  return { ms, loader, cache };
}

/** Minimal CSV (quoted fields allowed) → rows of name → string. */
function parseCsv(text: string): Record<string, string>[] {
  const lines = text.split(/\r?\n/).filter((l) => l.length);
  const split = (line: string) => {
    const out: string[] = [];
    let cur = "";
    let q = false;
    for (const c of line) {
      if (c === '"') q = !q;
      else if (c === "," && !q) {
        out.push(cur);
        cur = "";
      } else cur += c;
    }
    out.push(cur);
    return out;
  };
  const head = split(lines[0]!);
  return lines.slice(1).map((l) => Object.fromEntries(split(l).map((v, i) => [head[i]!, v])));
}

async function simbaRows(dir: string): Promise<Map<number, Record<string, string>>> {
  const rows = parseCsv(await readFile(join(simba, dir, "st_roi_features", `${leap}_aoi_features.csv`), "utf8"));
  // One value per ROI, repeated on each segment row.
  return new Map(rows.filter((r) => r.day_id === "a").map((r) => [Number(r.roi_id), r]));
}

async function main(): Promise<void> {
  const device = await getDevice();
  // One cache per raster (keys are chunk ids), each big enough for the whole level (62 Mpx × 4 B),
  // so the label scan, range, bin and exact passes decode each tile once.
  const r = (f: string) => tiffRaster(f, new TileCache<Tile>({ maxBytes: 2 ** 28 }));
  const metals = ["Cu", "Zn", "Fe", "Mg", "Mn"];
  const [labels, imcAcquired, collagen, ...metalRasters] = await Promise.all([
    r("st_day_a_roi_label_map.ome.tiff"),
    r("imc_roi_label_map.ome.tiff"),
    r("imc_processed_channel_Collage-Type_I.ome.tiff"),
    ...metals.map((m) => r(`laicpmsi_processed_channel_${m}.ome.tiff`)),
  ]);
  const t0 = performance.now();
  const res = await zonalStatsTiled(
    {
      level: 0,
      regions: { kind: "labels", source: labels },
      channels: [
        { source: collagen, support: imcAcquired, name: "collagen" },
        { source: metalRasters[0]!, support: metalRasters, name: "cu" },
      ],
      bins,
      quantiles: [0.25, 0.5, 0.75],
      exact: true,
    },
    device,
  );
  const ms = performance.now() - t0;
  const [w, h] = labels.ms.voxelDims0;
  console.log(
    `${leap}: ${w}×${h} px, ${res.tiles} tiles with a label, ${res.histogram.regions} label ids, ${bins} bins, exact pass collected ${res.plan!.total} values; ${(ms / 1000).toFixed(1)} s`,
  );

  const col = (name: string) => res.table.columns.get(name)!;
  const imc = await simbaRows("collagen_level");
  const la = await simbaRows("copper_level");
  const ids = [...new Set([...imc.keys(), ...la.keys()])].sort((a, b) => a - b);

  type Cmp = { id: number; ours: number; theirs: number };
  const report = (what: string, ours: (g: number) => number, theirs: (id: number) => number | undefined) => {
    const both: Cmp[] = [];
    let oursOnly = 0;
    let theirsOnly = 0;
    let neither = 0;
    for (const id of ids) {
      const g = id - 1;
      const a = g < res.histogram.regions ? ours(g) : Number.NaN;
      const b = theirs(id);
      const hasA = Number.isFinite(a);
      const hasB = b !== undefined && Number.isFinite(b);
      if (hasA && hasB) both.push({ id, ours: a, theirs: b });
      else if (hasA) oursOnly++;
      else if (hasB) theirsOnly++;
      else neither++;
    }
    const rel = both.map((c) => Math.abs(c.ours - c.theirs) / Math.max(1e-30, Math.abs(c.theirs)));
    const off = both.filter((_, i) => rel[i]! > 1e-6);
    console.log(
      `${what}: ${both.length} ROIs with both, max rel diff ${Math.max(0, ...rel).toExponential(2)}, ${off.length} over 1e-6; ours only ${oursOnly}, SIMBA only ${theirsOnly}, neither ${neither}`,
    );
    for (const c of off.slice(0, 10)) console.log(`   roi ${c.id}: ours ${c.ours}, SIMBA ${c.theirs}`);
    return { both, off };
  };
  const num = (row: Record<string, string> | undefined, key: string) => (row && row[key] !== "" ? Number(row[key]) : undefined);

  report(
    "imc_collagen_median (exact)",
    (g) => col("collagen_median")[g]!,
    (id) => num(imc.get(id), "imc_collagen_median"),
  );
  report(
    "laicpmsi_cu_p50 (exact)",
    (g) => col("cu_median")[g]!,
    (id) => num(la.get(id), "laicpmsi_cu_p50"),
  );
  report(
    "laicpmsi_cu_iqr (exact)",
    (g) => col("cu_iqr")[g]!,
    (id) => num(la.get(id), "laicpmsi_cu_iqr"),
  );
  report(
    "laicpmsi_n_tissue_pixels",
    (g) => col("cu_supported")[g]!,
    (id) => num(la.get(id), "laicpmsi_n_tissue_pixels"),
  );
  report(
    "laicpmsi_effective_region_mask_pixels",
    (g) => col("cu_pixels")[g]!,
    (id) => num(la.get(id), "laicpmsi_effective_region_mask_pixels"),
  );
  report(
    "laicpmsi_tissue_fraction",
    (g) => col("cu_supportFraction")[g]!,
    (id) => num(la.get(id), "laicpmsi_tissue_fraction"),
  );

  // Where only one side (or neither) has a value, say why.
  const missing = ids.filter((id) => !Number.isFinite(col("collagen_median")[id - 1]!));
  console.log(
    `no collagen value (neither side): ROIs ${missing.join(", ")}; their IMC-acquired pixels: ${missing.map((id) => col("collagen_supported")[id - 1]).join(", ")}`,
  );
  const unreported = [...Array(res.histogram.regions).keys()].filter((g) => col("cu_pixels")[g]! > 0 && !ids.includes(g + 1));
  console.log(`labels with pixels that SIMBA does not report: ${unreported.length ? unreported.map((g) => g + 1).join(", ") : "none"}`);

  // The binned estimate against the exact one, in bin widths.
  const { histogram } = res;
  const binned = (await import("../src/spatial/zonal")).zonalTable(histogram, { names: ["collagen", "cu"], quantiles: [0.25, 0.5, 0.75] });
  for (const name of ["collagen", "cu"]) {
    let worst = 0;
    for (let g = 0; g < histogram.regions; g++) {
      const e = col(`${name}_median`)[g]!;
      const b = binned.columns.get(`${name}_median`)![g]!;
      if (Number.isFinite(e) && Number.isFinite(b)) worst = Math.max(worst, Math.abs(e - b) / col(`${name}_binWidth`)[g]!);
    }
    console.log(
      `${name}: binned median within ${worst.toFixed(3)} bin widths of exact (bin width ${col(`${name}_binWidth`)[0]!.toPrecision(3)})`,
    );
  }
  await releaseDevice();
}

await main();
