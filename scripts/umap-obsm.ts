#!/usr/bin/env tsx
// Offline UMAP → `obsm`. The project's first piece of real work done OUTSIDE the
// browser: open an AnnData / SpatialData zarr store, compute an embedding, write it
// back as `obsm/<key>` for scanpy, MDV or the in-browser viewer to pick up.
//
// It shares every line of its maths with the interactive path (`src/spatial/umap.ts`);
// the only things this file adds are argument parsing, progress reporting, and the
// write. That is the point — an `obsm` entry is only worth writing if it agrees with
// what the viewer shows.
//
// The k-NN runs on the GPU in this process, like any other consumer of the library —
// no subprocess, no special casing. That is worth stating because it briefly was not
// true: an earlier version quarantined the device in a child process to survive a crash
// during the zarr writes. The crash was our own Instance-lifetime bug in
// `src/gpu/device.ts`, now fixed, so the quarantine is gone. `--cpu` forces the host
// path, which is exact and useful for cross-checking.
//
// `releaseDevice()` is called once the last kernel has read back: holding Dawn's
// Instance is what keeps GPU work safe, and dropping it is what lets the process exit.
//
// The k-NN is chosen by size: exact (GPU) below ~5k cells, NN-descent above, because the
// exact search is O(N^2 * D) and no device fixes that. `--exact` / `--approx` override.
// See `docs/umap-on-anndata.md` §3.
//
//   pnpm umap:obsm <store> [options]
//
//     --table <path>     AnnData group inside the store (default: auto-detect)
//     --matrix <name>    "X" or a layers/<name>            (default: X)
//     --key <name>       obsm key to write                 (default: X_umap)
//     --genes <n,m,...>  var indices to use                (default: all)
//     --n-neighbors <k>  (default 15)   --min-dist <d>     (default 0.1)
//     --n-epochs <n>     (default 200)  --components <n>   PCA dims (default 50)
//     --dim <n>          embedding dimension               (default 2)
//     --seed <n>         (default 42)
//     --cpu              force the host k-NN instead of the GPU
//     --exact            force the exact k-NN even above the crossover
//     --approx           force approximate (NN-descent) even below it
//     --obsp             also write the neighbour graph to obsp + uns/neighbors
//     --force            overwrite an existing obsm/obsp key
//     --dry-run          compute and report, write nothing

import {
  fuzzyGraphToCsr,
  knnToCsr,
  openStore,
  readExpressionMatrix,
  readNObs,
  writeNeighborsUns,
  writeObsm,
  writeObsp,
} from "../src/datasource/annDataIo";
import { knnDescentCpu, knnStrategyFor } from "../src/spatial/knnDescent";
import { umapGraphFor } from "../src/spatial/umap";
import type { KnnResult } from "../src/spatial/umapGraph";
import { fitAB, optimizeLayout } from "../src/spatial/umapLayout";

interface Args {
  store: string;
  table?: string;
  matrix: string;
  key: string;
  genes?: number[];
  nNeighbors: number;
  minDist: number;
  nEpochs: number;
  components: number;
  dim: number;
  seed: number;
  cpu: boolean;
  exact: boolean;
  approx: boolean;
  obsp: boolean;
  force: boolean;
  dryRun: boolean;
}

function parseArgs(argv: string[]): Args {
  const positional: string[] = [];
  const flags = new Map<string, string | true>();
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (!a.startsWith("--")) {
      positional.push(a);
      continue;
    }
    const name = a.slice(2);
    const next = argv[i + 1];
    if (next !== undefined && !next.startsWith("--")) {
      flags.set(name, next);
      i++;
    } else {
      flags.set(name, true);
    }
  }
  const num = (name: string, fallback: number) => {
    const v = flags.get(name);
    if (v === undefined || v === true) return fallback;
    const parsed = Number(v);
    if (!Number.isFinite(parsed)) throw new Error(`--${name} expects a number, got "${v}"`);
    return parsed;
  };
  const str = (name: string, fallback: string) => {
    const v = flags.get(name);
    return v === undefined || v === true ? fallback : v;
  };

  const store = positional[0];
  if (!store) throw new Error("usage: pnpm umap:obsm <store> [options] (see the header of scripts/umap-obsm.ts)");

  const genesRaw = flags.get("genes");
  const genes =
    typeof genesRaw === "string"
      ? genesRaw
          .split(",")
          .map((s) => Number(s.trim()))
          .filter((v) => Number.isInteger(v) && v >= 0)
      : undefined;

  return {
    store,
    table: typeof flags.get("table") === "string" ? (flags.get("table") as string) : undefined,
    matrix: str("matrix", "X"),
    key: str("key", "X_umap"),
    genes,
    nNeighbors: num("n-neighbors", 15),
    minDist: num("min-dist", 0.1),
    nEpochs: num("n-epochs", 200),
    components: num("components", 50),
    dim: num("dim", 2),
    seed: num("seed", 42),
    cpu: flags.get("cpu") === true,
    exact: flags.get("exact") === true,
    approx: flags.get("approx") === true,
    obsp: flags.get("obsp") === true,
    force: flags.get("force") === true,
    dryRun: flags.get("dry-run") === true,
  };
}

/** SpatialData nests AnnData under `tables/<name>`; a bare AnnData store has it at the
 *  root. Probe rather than make the user know which they have. */
async function detectTablePath(loc: Awaited<ReturnType<typeof openStore>>, explicit?: string): Promise<string> {
  if (explicit !== undefined) return explicit;
  if ((await readNObs(loc, "")) !== undefined) return "";
  const store = (loc as unknown as { store: { get(k: string): Promise<Uint8Array | undefined> } }).store;
  for (const marker of ["/tables/.zgroup", "/tables/zarr.json"]) {
    try {
      if (!(await store.get(marker))) continue;
    } catch {
      continue;
    }
    for (const name of ["table", "cells", "anndata"]) {
      if ((await readNObs(loc, `tables/${name}`)) !== undefined) return `tables/${name}`;
    }
  }
  throw new Error("could not find an AnnData group; pass --table <path> explicitly");
}

/**
 * Choose the k-NN.
 *
 * Two independent axes: **where** it runs (GPU or host) and **whether it is exact**. The
 * exact search is O(N^2 * D) and stops being viable somewhere in the tens of thousands of
 * cells whichever device runs it, so past the crossover we switch to NN-descent — which
 * is host-side today, hence `--exact` for anyone who would rather wait for the exact
 * answer.
 */
function resolveKnn(
  args: Args,
  nCells: number,
): {
  fn?: (d: ArrayLike<number>, n: number, dim: number, k: number) => Promise<KnnResult>;
  label: string;
  gpu: boolean;
} {
  // The crossover depends on which exact search we would otherwise run: the tiled GPU
  // kernel stays ahead of the host descent past 50k, whereas a host brute force is
  // overtaken around 5k. Measured — see the table in `knnDescent.ts`.
  const crossover = args.cpu ? 5_000 : 50_000;
  const wantApprox = args.approx || (!args.exact && knnStrategyFor(nCells, crossover) === "descent");
  if (wantApprox) {
    const why = args.approx ? "--approx" : `${nCells} cells is past the ${crossover} crossover`;
    return {
      fn: async (d, n, dim, k) => knnDescentCpu(d, n, dim, { k, seed: args.seed }),
      label: `NN-descent, approximate (${why})`,
      gpu: false,
    };
  }
  if (args.cpu) return { label: "exact, host (--cpu)", gpu: false };
  return {
    fn: async (d, n, dim, k) => {
      const { knnGpu } = await import("../src/gpu/spatial/knn");
      return knnGpu(d, { n, dim, k });
    },
    label: "exact, GPU",
    gpu: true,
  };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const started = Date.now();

  const loc = await openStore(args.store);
  const tablePath = await detectTablePath(loc, args.table);
  console.log(`store    ${args.store}`);
  console.log(`table    ${tablePath || "<root>"}`);

  const matrix = await readExpressionMatrix(loc, { tablePath, matrix: args.matrix, vars: args.genes });
  console.log(`matrix   ${args.matrix} — ${matrix.nCells} cells x ${matrix.nVars} genes`);

  const { fn: knn, label, gpu } = resolveKnn(args, matrix.nCells);
  usedGpu = gpu;
  console.log(`k-NN     ${label}`);
  if (matrix.nCells > 20000 && label.startsWith("exact")) {
    console.warn(`  warning: ${matrix.nCells} cells through an exact O(N^2) k-NN will be slow; drop --exact to approximate`);
  }

  // All GPU work happens in this call. The device is NOT released here: the k-NN's
  // pooled buffers and pipelines outlive the call, and dropping the Instance while they
  // are still alive faults on their finalisers. Release happens once, at the very end.
  const graph = await umapGraphFor(matrix.values, matrix.nCells, matrix.nVars, {
    nNeighbors: args.nNeighbors,
    nComponents: args.components,
    knn,
  });
  console.log(`graph    ${graph.graph.nEdges} directed edges${graph.reducedDim ? `, after PCA to ${graph.reducedDim} dims` : ", no PCA"}`);

  const ab = fitAB(args.minDist, 1);
  const embedding = optimizeLayout(graph.graph, {
    dim: args.dim,
    nEpochs: args.nEpochs,
    seed: args.seed,
    ab,
  });
  console.log(`layout   ${args.nEpochs} epochs, ${args.dim}-D, a=${ab.a.toFixed(4)} b=${ab.b.toFixed(4)}`);

  if (args.dryRun) {
    console.log(`dry-run  would write obsm/${args.key} (${matrix.nCells} x ${args.dim}); nothing written`);
    return;
  }

  const target = await writeObsm(loc, embedding, matrix.nCells, args.dim, { tablePath, key: args.key, force: args.force });
  // `target` is already store-relative and includes the table path.

  if (args.obsp) {
    // `obsm` hands over our picture; `obsp` hands over our GRAPH, which is the more
    // useful half — scanpy's leiden/louvain cluster on `connectivities`, so a
    // collaborator can re-cluster on this manifold instead of building their own and
    // wondering why the labels disagree. `uns/neighbors` is what points scanpy at them;
    // without it the obsp write is only half the handover.
    const conn = await writeObsp(loc, fuzzyGraphToCsr(graph.graph), { tablePath, key: "connectivities", force: args.force });
    const dist = await writeObsp(loc, knnToCsr(graph.knn), { tablePath, key: "distances", force: args.force });
    await writeNeighborsUns(loc, {
      tablePath,
      connectivitiesKey: "connectivities",
      distancesKey: "distances",
      nNeighbors: args.nNeighbors,
    });
    console.log(`obsp     ${conn} + ${dist} + uns/neighbors`);
  }
  console.log(`wrote    ${target} in ${((Date.now() - started) / 1000).toFixed(1)}s`);
}

/** Release Dawn's Instance so the process can exit. Must be the LAST thing that happens
 *  — see the note in `src/gpu/device.ts`. */
async function shutdown(gpu: boolean): Promise<void> {
  if (!gpu) return;
  const { releaseDevice } = await import("../src/gpu/device");
  await releaseDevice();
}

let usedGpu = false;
main()
  .then(() => shutdown(usedGpu))
  .catch(async (err) => {
    await shutdown(usedGpu);
    console.error(`umap-obsm: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  });
