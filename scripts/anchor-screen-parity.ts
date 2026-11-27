#!/usr/bin/env tsx
// Parity of `anchorScreen` against SIMBA's continuous-anchor scorer (R5; `docs/anchor-screen.md`).
//
// Inputs are local-only and not in the repo: per-block matrices exported with SIMBA's own builders
// (`build_patch_block`, `build_st_block`) by `~/data/tnbc_simba_runs/scripts/export_anchor_blocks.py`,
// and the scores SIMBA's `compute_feature_block_scores` wrote (`simba_anchor_check.py`).
//
//   pnpm anchor:parity <run_root> [--leap Leap034] [--gpu [--feistel]]
//
// <run_root>/check/blocks/<leap>__<source>.{json,f64} and <run_root>/check/<leap>_scores.csv.
// ρ and n_finite must match per feature; p is a Monte Carlo estimate from a different RNG, so it is
// compared as z = Δp / sqrt(2 p(1−p) / (N+1)).

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { mulberry32 } from "../src/evo/rng";
import {
  ANCHOR_SCREEN_PERMUTATIONS,
  anchorEmpiricalP,
  anchorNullExceedances,
  anchorSampleSeed,
  prepareAnchorBlock,
} from "../src/spatial/anchorScreen";
import { randomPermutation } from "../src/spatial/permute";

const SHUFFLE = process.argv.includes("--feistel") ? "feistel" : "fisher-yates";

interface Meta {
  leap_id: string;
  feature_source: string;
  n: number;
  m: number;
  features: [string, string, string][];
}

function readCsv(path: string): Record<string, string>[] {
  const lines = readFileSync(path, "utf8").trim().split("\n");
  const split = (l: string) => {
    const out: string[] = [];
    let cur = "";
    let q = false;
    for (const ch of l) {
      if (ch === '"') q = !q;
      else if (ch === "," && !q) {
        out.push(cur);
        cur = "";
      } else cur += ch;
    }
    out.push(cur);
    return out;
  };
  const head = split(lines[0]!);
  return lines.slice(1).map((l) => Object.fromEntries(split(l).map((v, i) => [head[i]!, v])));
}

async function main() {
  const args = process.argv.slice(2);
  const root = args.find((a) => !a.startsWith("--"));
  if (!root) throw new Error("usage: anchor-screen-parity <run_root> [--leap Leap034] [--gpu [--feistel]]");
  const leapArg = args.indexOf("--leap");
  const onlyLeap = leapArg >= 0 ? args[leapArg + 1] : undefined;
  const useGpu = args.includes("--gpu");
  const S = ANCHOR_SCREEN_PERMUTATIONS;
  const blocksDir = join(root, "check", "blocks");
  const stems = readdirSync(blocksDir)
    .filter((f) => f.endsWith(".json"))
    .map((f) => f.slice(0, -5))
    .filter((s) => !onlyLeap || s.startsWith(`${onlyLeap}__`))
    .sort();
  const gpu = useGpu ? await import("../src/gpu/spatial/anchorScreenGpu") : undefined;

  for (const stem of stems) {
    const meta = JSON.parse(readFileSync(join(blocksDir, `${stem}.json`), "utf8")) as Meta;
    const bytes = readFileSync(join(blocksDir, `${stem}.f64`));
    // readFileSync may return a view into a pooled buffer at an unaligned offset: copy it out.
    const raw = new Float64Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    const anchor = raw.subarray(0, meta.n);
    const cols = Array.from({ length: meta.m }, (_, j) => raw.subarray(meta.n * (1 + j), meta.n * (2 + j)));
    const simba = new Map(
      readCsv(join(root, "check", `${meta.leap_id}_scores.csv`)).map((r) => [`${r.modality}|${r.feature_group}|${r.feature}`, r]),
    );

    const t0 = performance.now();
    const block = prepareAnchorBlock(anchor, cols);
    const t1 = performance.now();
    const cpuNull = () => {
      const rnd = mulberry32(anchorSampleSeed(0, meta.leap_id));
      const buf = new Uint32Array(block.nAnchor);
      return anchorNullExceedances(block, S, () => randomPermutation(block.nAnchor, rnd, buf));
    };
    const exceed = gpu
      ? await gpu.anchorNullExceedancesGpu(block, S, { seed: anchorSampleSeed(0, meta.leap_id), shuffle: SHUFFLE })
      : cpuNull();
    const t2 = performance.now();
    // With the default shuffle the GPU draws the CPU's permutations: count where the counts differ.
    let vsCpu = 0;
    if (gpu) {
      const cpu = cpuNull();
      for (let j = 0; j < cpu.length; j++) if (cpu[j] !== exceed[j]) vsCpu++;
    }
    const p = anchorEmpiricalP(block.rho, exceed, S);

    let maxRho = 0;
    let nanMismatch = 0;
    let finiteMismatch = 0;
    let missing = 0;
    let scored = 0;
    let z3 = 0;
    let maxZ = 0;
    let sumZ2 = 0;
    meta.features.forEach(([mod, grp, feat], j) => {
      const r = simba.get(`${mod}|${grp}|${feat}`);
      if (!r) {
        missing++;
        return;
      }
      const sr = r.observed_rho === "" ? Number.NaN : Number(r.observed_rho);
      if (Number(r.n_finite_regions) !== block.nFinite[j]) finiteMismatch++;
      if (Number.isNaN(sr) !== Number.isNaN(block.rho[j]!)) nanMismatch++;
      if (Number.isNaN(sr)) return;
      maxRho = Math.max(maxRho, Math.abs(sr - block.rho[j]!));
      const sp = Number(r.empirical_p);
      const pm = (sp + p[j]!) / 2;
      const sd = Math.sqrt((2 * pm * (1 - pm)) / (S + 1));
      const z = sd > 0 ? Math.abs(p[j]! - sp) / sd : 0;
      scored++;
      sumZ2 += z * z;
      if (z > 3) z3++;
      maxZ = Math.max(maxZ, z);
    });
    console.log(
      `${stem}: n=${meta.n} m=${meta.m} scored=${scored} | max|Δρ|=${maxRho.toExponential(2)} ` +
        `nFinite≠${finiteMismatch} NaN≠${nanMismatch} missing=${missing} | p: mean z²=${(sumZ2 / Math.max(scored, 1)).toFixed(2)} ` +
        `|z|>3: ${z3} max|z|=${maxZ.toFixed(2)} | prep ${(t1 - t0).toFixed(0)} ms, null ${(t2 - t1).toFixed(0)} ms${gpu ? ` (gpu, ${SHUFFLE}; counts ≠ CPU: ${vsCpu})` : ""}`,
    );
    for (const want of [
      "COL1A1",
      "SOX18",
      "COX6C",
      "laicpmsi_zn_p50",
      "laicpmsi_mg_p50",
      "laicpmsi_cu_p50",
      "imc_lineage_tumor_fraction",
    ]) {
      const j = meta.features.findIndex(([, , f]) => f === want);
      if (j < 0) continue;
      const r = simba.get(meta.features[j]!.join("|"));
      if (!r) {
        console.log(`    ${want}: no SIMBA row`);
        continue;
      }
      const sr = r.observed_rho === "" ? Number.NaN : Number(r.observed_rho);
      console.log(
        `    ${want}: ρ ${block.rho[j]!.toFixed(4)} (SIMBA ${sr.toFixed(4)}), p ${p[j]!.toFixed(3)} (SIMBA ${Number(r.empirical_p).toFixed(3)}), n ${block.nFinite[j]}`,
      );
    }
  }
}

void main().then(
  () => process.exit(0),
  (e: unknown) => {
    console.error(e);
    process.exit(1);
  },
);
