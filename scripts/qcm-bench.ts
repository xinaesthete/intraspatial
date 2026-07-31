#!/usr/bin/env tsx
// CPU vs GPU for the QCM's permutation null, on a real ROI — and, more importantly, whether they
// agree. A speedup that changed the answer would be worth nothing, so this reports the agreement
// first and the timing second.
//
// The two paths sample the null differently by construction (Fisher-Yates against a Feistel
// permutation), so the comparison is between two Monte Carlo estimates of the same distribution.
// What must match is the DECISION: the same pairs surviving Benjamini–Hochberg.
//
//   PATH="$HOME/.volta/tools/image/node/22.23.1/bin:$PATH" pnpm qcm:bench <store.zarr> [roi] [sims]

import { MdvStore, readRegionCells } from "../src/datasource/mdvStore";
import { quadratCorrelationGpu } from "../src/gpu/spatial/quadratCorrelationGpu";
import { quadratCorrelation } from "../src/spatial/quadratCorrelation";

const storePath = process.argv[2];
if (!storePath) throw new Error("usage: pnpm qcm:bench <store.zarr> [roi] [sims]");
const roiWanted = process.argv[3] ?? "COVID_SAMPLE_16_ROI_3";
const SIMS = Number(process.argv[4] ?? 999);

const store = await MdvStore.open(storePath);
const sTypeA = await store.readCategorical("spatial_stats", "Cell Type 1");
const K = sTypeA.labels.length;
const regions = await readRegionCells(store, "cells", {
  typeField: "annotations",
  regionField: "sample_id",
  typeLabels: sTypeA.labels,
  caseInsensitiveTypes: true,
});
const cells = regions.get(roiWanted);
if (!cells) throw new Error(`no ROI ${roiWanted} — have ${[...regions.keys()].slice(0, 3).join(", ")} …`);

let maxX = 0;
let maxY = 0;
for (let i = 0; i < cells.xs.length; i++) {
  maxX = Math.max(maxX, cells.xs[i]!);
  maxY = Math.max(maxY, cells.ys[i]!);
}
const params = { bbox: [0, 0, maxX, maxY] as [number, number, number, number], quadratSize: 100, nTypes: K, simulations: SIMS, seed: 0x5eed };

// One warm-up pass builds the pipelines and pools; timing that would measure compilation.
await quadratCorrelationGpu(cells, { ...params, simulations: 64 });

const tGpu0 = performance.now();
const gpu = await quadratCorrelationGpu(cells, params);
const tGpu = performance.now() - tGpu0;

const tCpu0 = performance.now();
const cpu = quadratCorrelation(cells, params);
const tCpu = performance.now() - tCpu0;

const present = new Set(cells.typeId);
let pairs = 0;
let agree = 0;
let bothSig = 0;
let onlyGpu = 0;
let onlyCpu = 0;
let worstSes = 0;
const alpha = 0.05;
for (const a of present) {
  for (const b of present) {
    if (a >= b) continue;
    const i = a * K + b;
    if (!Number.isFinite(cpu.pc[i]!)) continue;
    pairs++;
    const g = gpu.pcQ[i]! < alpha;
    const c = cpu.pcQ[i]! < alpha;
    if (g === c) agree++;
    if (g && c) bothSig++;
    else if (g) onlyGpu++;
    else if (c) onlyCpu++;
    if (Number.isFinite(gpu.pcSes[i]!) && Number.isFinite(cpu.pcSes[i]!)) worstSes = Math.max(worstSes, Math.abs(gpu.pcSes[i]! - cpu.pcSes[i]!));
  }
}

// The observed statistic is CPU-computed in both paths, so this must be exact, not merely close.
let obsMax = 0;
for (let i = 0; i < K * K; i++) {
  if (Number.isFinite(cpu.pc[i]!) && Number.isFinite(gpu.pc[i]!)) obsMax = Math.max(obsMax, Math.abs(cpu.pc[i]! - gpu.pc[i]!));
}

console.log(`${roiWanted} — ${cells.xs.length} cells, ${present.size}/${K} types present, ${cpu.quadrats} quadrats, ${SIMS} shuffles`);
console.log(`\n=== agreement (${pairs} testable pairs) ===`);
console.log(`  observed MH_PC, GPU vs CPU      : max |Δ| = ${obsMax.toExponential(2)}   (must be 0 — same f64 code both sides)`);
console.log(`  partial SES, GPU vs CPU         : max |Δ| = ${worstSes.toFixed(3)}   (two Monte Carlo estimates of one null)`);
console.log(`  BH verdict at q<${alpha} agrees      : ${agree}/${pairs} (${((100 * agree) / pairs).toFixed(1)}%)`);
console.log(`    significant on both           : ${bothSig}`);
console.log(`    GPU only / CPU only           : ${onlyGpu} / ${onlyCpu}`);
console.log(`\n=== timing ===`);
console.log(`  CPU : ${tCpu.toFixed(0)} ms`);
console.log(`  GPU : ${tGpu.toFixed(0)} ms   —  ${(tCpu / tGpu).toFixed(1)}× faster`);
