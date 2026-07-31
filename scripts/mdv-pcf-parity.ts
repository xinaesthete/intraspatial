#!/usr/bin/env tsx
// Parity sweep: our cross-PCF against a published one, at the scale the publication ran it.
//
// The covid MDV project (docs/decisions — see the design note this belongs to) carries a
// `spatial_stats` table of 76,832 rows = 32 ROIs × 49² cell-type pairs, of which 70,742 hold finite
// `gr10` / `gr20` values computed by the SpOOx/MuSpAn pipeline behind the Nature Comms paper. That
// is an external oracle for `crossPCFMatrixBinned` — not a handful of hand-checked pairs, but every
// pair in every ROI, produced by someone else's code from the same centroids.
//
// WHAT IS BEING COMPARED. `gr20` is g_AB over the annulus [20,30) µm and `gr10` over [10,20), with
// 10 µm bins — established by scanning every bin for the best fit rather than assumed, and it is
// bin 2 and bin 1 by a wide margin. ρ_B uses the ROI's recorded `roi_area`, which for 30 of the 32
// ROIs is exactly the region image's W×H.
//
// THE TYPE AXIS IS CHECKED, NOT TRUSTED. The cell table's `annotations` and the stats table's
// `Cell Type 1` are different categoricals with different orders, and they disagree on one label's
// capitalisation (`Blood vessels` / `Blood Vessels`); the cell table also has an `NA` category the
// stats table lacks. So before any g is compared, the per-type cell counts we derive are checked
// against the stored `cell 1 number` column. A mismatch there means the axes are misaligned and
// every downstream number is meaningless — it is a hard failure, not a warning.
//
// WHAT THE ANSWER LOOKS LIKE. Two estimators are reported: `plain` (full annulus, what
// `crossPCF` has always done) and `corrected` (each anchor's annulus clipped to the ROI, eq 8).
// The uncorrected one is systematically LOW by an amount that scales with perimeter·r/area, so its
// disagreement is a property of ROI geometry rather than of the implementation; the corrected one is
// the actual parity claim.
//
//   pnpm mdv:parity <store.zarr> [options]
//
//     --cells <ds>     cell table datasource      (default: cells)
//     --stats <ds>     stats table datasource     (default: spatial_stats)
//     --samples <ds>   sample table datasource    (default: samples)
//     --min-n <n>      min(n_A, n_B) for the headline subset (default: 50)
//     --limit <n>      only the first n ROIs
//     --csv <path>     write every compared row
//     --quiet          summary only, no per-ROI table

import { writeFileSync } from "node:fs";
import { MdvStore, readRegionCells } from "../src/datasource/mdvStore";
import { crossPCFMatrixBinned } from "../src/spatial/pcf";

/** Stored column → bin index, given 10 µm bins. `gr20` is the annulus [20,30) = bin 2. */
const BIN_OF = { gr10: 1, gr20: 2 } as const;
const R_MAX = 30;
const N_BINS = 3;

interface Args {
  store: string;
  cells: string;
  stats: string;
  samples: string;
  minN: number;
  limit?: number;
  csv?: string;
  quiet: boolean;
}

function parseArgs(argv: string[]): Args {
  const a: Args = { store: "", cells: "cells", stats: "spatial_stats", samples: "samples", minN: 50, quiet: false };
  const pos: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i]!;
    if (t === "--cells") a.cells = argv[++i]!;
    else if (t === "--stats") a.stats = argv[++i]!;
    else if (t === "--samples") a.samples = argv[++i]!;
    else if (t === "--min-n") a.minN = Number(argv[++i]);
    else if (t === "--limit") a.limit = Number(argv[++i]);
    else if (t === "--csv") a.csv = argv[++i]!;
    else if (t === "--quiet") a.quiet = true;
    else if (t.startsWith("--")) throw new Error(`unknown option ${t}`);
    else pos.push(t);
  }
  if (pos.length !== 1) throw new Error("usage: pnpm mdv:parity <store.zarr> [--min-n n] [--limit n] [--csv path] [--quiet]");
  a.store = pos[0]!;
  return a;
}

function median(v: number[]): number {
  if (v.length === 0) return Number.NaN;
  const s = [...v].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

function quantile(v: number[], q: number): number {
  if (v.length === 0) return Number.NaN;
  const s = [...v].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))]!;
}

/** Summary of a set of predicted/stored ratios, on the log scale so over- and under-estimates are
 *  symmetric and a ratio of 2 and of 1/2 are the same distance from agreement. */
function summarise(ratios: number[]) {
  const logs = ratios.map(Math.log);
  const absLogs = logs.map(Math.abs);
  const within = (t: number) => absLogs.filter((l) => l < Math.log(1 + t)).length / absLogs.length;
  return {
    n: ratios.length,
    medianRatio: Math.exp(median(logs)),
    medianAbsPct: 100 * (Math.exp(median(absLogs)) - 1),
    p95AbsPct: 100 * (Math.exp(quantile(absLogs, 0.95)) - 1),
    within1: within(0.01),
    within5: within(0.05),
  };
}

const pct = (x: number) => `${(100 * x).toFixed(1)}%`;

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const store = await MdvStore.open(args.store);

  // ---- stats table (the oracle) ----------------------------------------------------------------
  const sRegion = await store.readCategorical(args.stats, "sample_id");
  const sTypeA = await store.readCategorical(args.stats, "Cell Type 1");
  const sTypeB = await store.readCategorical(args.stats, "Cell Type 2");
  const nA = await store.readF64(args.stats, "cell 1 number");
  const nB = await store.readF64(args.stats, "cell 2 number");
  const stored: Record<keyof typeof BIN_OF, Float64Array> = {
    gr10: await store.readF64(args.stats, "gr10"),
    gr20: await store.readF64(args.stats, "gr20"),
  };
  const typeLabels = sTypeA.labels;
  if (sTypeB.labels.length !== typeLabels.length) throw new Error("Cell Type 1 and Cell Type 2 have different category sets");

  // ---- sample table (roi_area) -----------------------------------------------------------------
  const mRegion = await store.readCategorical(args.samples, "sample_id");
  const mArea = await store.readF64(args.samples, "roi_area");
  const areaOf = new Map<string, number>();
  for (let i = 0; i < mArea.length; i++) areaOf.set(mRegion.labels[mRegion.codes[i]!]!, mArea[i]!);

  // ---- cell table, split per ROI onto the stats table's type axis -------------------------------
  const regions = await readRegionCells(store, args.cells, {
    typeField: "annotations",
    regionField: "sample_id",
    typeLabels,
    caseInsensitiveTypes: true,
  });

  // Stats rows grouped by ROI, so each ROI's matrix is computed once.
  const rowsByRegion = new Map<string, number[]>();
  for (let i = 0; i < sRegion.codes.length; i++) {
    const name = sRegion.labels[sRegion.codes[i]!]!;
    const list = rowsByRegion.get(name);
    if (list) list.push(i);
    else rowsByRegion.set(name, [i]);
  }

  const names = [...rowsByRegion.keys()].sort();
  const todo = args.limit ? names.slice(0, args.limit) : names;
  const K = typeLabels.length;

  const csv: string[] = [];
  if (args.csv) csv.push("region,typeA,typeB,nA,nB,pairs,column,stored,plain,corrected");

  const all: Record<string, number[]> = { plainAll: [], corrAll: [], plainBig: [], corrBig: [], bboxBig: [] };
  const byColumn: Record<string, number[]> = { gr10: [], gr20: [] };
  const perRegion: { name: string; n: number; cells: number; area: number; plain: number; corr: number; bbox: number }[] = [];

  for (const name of todo) {
    const cells = regions.get(name);
    if (!cells) {
      console.warn(`! ${name}: no cells — skipped`);
      continue;
    }
    const rows = rowsByRegion.get(name)!;
    const roiArea = areaOf.get(name);
    if (roiArea === undefined) throw new Error(`no roi_area for ${name}`);

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < cells.xs.length; i++) {
      const x = cells.xs[i]!;
      const y = cells.ys[i]!;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const bbox = [minX, minY, maxX, maxY] as const;
    const p = { bbox, rMax: R_MAX, nBins: N_BINS, nTypes: K, roiArea } as const;
    const plain = crossPCFMatrixBinned(cells, p);
    const corr = crossPCFMatrixBinned(cells, { ...p, edgeCorrected: true });
    // ρ_B enters only as a final scale (g = weighted / (N_A · N_B / area)), and the edge correction
    // depends on the rectangle rather than on the area, so the bounding-box convention is an exact
    // rescale of the same pass rather than a second one. Worth reporting because the two conventions
    // disagree on exactly the ROIs where `roi_area` is not the region the cells occupy.
    const bboxArea = (maxX - minX) * (maxY - minY);
    const bboxScale = bboxArea / roiArea;

    // Type-axis check: our per-type counts must equal the stored ones. See the header — a silent
    // misalignment here would produce plausible-looking garbage everywhere downstream.
    for (const row of rows) {
      const a = sTypeA.codes[row]!;
      const b = sTypeB.codes[row]!;
      if (plain.counts[a] !== nA[row] || plain.counts[b] !== nB[row]) {
        throw new Error(
          `type axis misaligned in ${name}: '${typeLabels[a]}' has ${plain.counts[a]} cells here but the stats table says ${nA[row]} ` +
            `(and '${typeLabels[b]}': ${plain.counts[b]} vs ${nB[row]})`,
        );
      }
    }

    const regionPlain: number[] = [];
    const regionCorr: number[] = [];
    const regionBbox: number[] = [];
    for (const row of rows) {
      const a = sTypeA.codes[row]!;
      const b = sTypeB.codes[row]!;
      const small = Math.min(nA[row]!, nB[row]!);
      for (const col of Object.keys(BIN_OF) as (keyof typeof BIN_OF)[]) {
        const k = BIN_OF[col];
        const obs = stored[col][row]!;
        if (!Number.isFinite(obs) || obs <= 0) continue;
        const at = a * K * N_BINS + b * N_BINS + k;
        const gp = plain.g[at]!;
        const gc = corr.g[at]!;
        if (args.csv) {
          csv.push(`${name},${typeLabels[a]},${typeLabels[b]},${nA[row]},${nB[row]},${plain.pairs[at]},${col},${obs},${gp},${gc}`);
        }
        if (!(gp > 0) || !(gc > 0)) continue;
        all.plainAll!.push(gp / obs);
        all.corrAll!.push(gc / obs);
        if (small >= args.minN) {
          all.plainBig!.push(gp / obs);
          all.corrBig!.push(gc / obs);
          all.bboxBig!.push((gc * bboxScale) / obs);
          byColumn[col]!.push(gc / obs);
          regionPlain.push(gp / obs);
          regionCorr.push(gc / obs);
          regionBbox.push((gc * bboxScale) / obs);
        }
      }
    }
    const sp = summarise(regionPlain);
    const sc = summarise(regionCorr);
    const sb = summarise(regionBbox);
    perRegion.push({ name, n: sc.n, cells: cells.xs.length, area: roiArea, plain: sp.medianAbsPct, corr: sc.medianAbsPct, bbox: sb.medianAbsPct });
    if (!args.quiet) {
      console.log(
        `${name.padEnd(26)} cells=${String(cells.xs.length).padStart(6)} pairs=${String(sc.n).padStart(5)}  ` +
          `plain ${sp.medianAbsPct.toFixed(2).padStart(6)}%  corrected ${sc.medianAbsPct.toFixed(2).padStart(6)}%  ` +
          `corrected/bbox ${sb.medianAbsPct.toFixed(2).padStart(6)}%  (ratio ${sc.medianRatio.toFixed(4)}, roi_area/bbox ${(1 / bboxScale).toFixed(3)})`,
      );
    }
  }

  const line = (label: string, r: number[]) => {
    const s = summarise(r);
    console.log(
      `  ${label.padEnd(30)} n=${String(s.n).padStart(6)}  median |Δ| ${s.medianAbsPct.toFixed(2).padStart(6)}%  ` +
        `p95 ${s.p95AbsPct.toFixed(1).padStart(6)}%  median ratio ${s.medianRatio.toFixed(4)}  within 1% ${pct(s.within1)}  within 5% ${pct(s.within5)}`,
    );
  };

  console.log(`\n=== parity vs stored gr10/gr20 (${todo.length} ROIs) ===`);
  line("plain, all pairs", all.plainAll!);
  line("corrected, all pairs", all.corrAll!);
  line(`plain, min(n) >= ${args.minN}`, all.plainBig!);
  line(`corrected, min(n) >= ${args.minN}`, all.corrBig!);
  line(`corrected/bbox area, min(n) >= ${args.minN}`, all.bboxBig!);
  console.log("");
  line("corrected, gr10 only", byColumn.gr10!);
  line("corrected, gr20 only", byColumn.gr20!);

  const worst = [...perRegion].sort((a, b) => b.corr - a.corr).slice(0, 3);
  console.log(`\nworst ROIs (corrected): ${worst.map((w) => `${w.name} ${w.corr.toFixed(2)}%`).join(", ")}`);

  if (args.csv) {
    writeFileSync(args.csv, csv.join("\n"));
    console.log(`wrote ${csv.length - 1} rows to ${args.csv}`);
  }
}

main().catch((e) => {
  console.error(e instanceof Error ? e.message : e);
  process.exit(1);
});
