#!/usr/bin/env tsx
// How far does this actually scale? — a measured answer, stage by stage.
//
// "It scales to N cells" is only worth saying if the shape of the data is held fixed
// while N moves, and if every stage is timed separately. Both matter here:
//
//   • **Fixed shape.** The sweep runs one generator from
//     `src/spatial/syntheticManifolds.ts` across sizes, and those generators are
//     normalised so the difficulty does not drift with n. Sweeping n over a dataset that
//     gets easier as it grows measures the generator, not the code.
//   • **Stage by stage.** k-NN is O(n²·D) and the layout is O(edges) per epoch. They hit
//     their walls at completely different sizes, so a single wall-clock number would hide
//     which one you are actually waiting for — and therefore which one is worth fixing.
//
// It also reports **trustworthiness**, because a throughput table with no quality column
// invites the reading that the fast path is as good as the slow one. The approximate k-NN
// is scored by recall against the exact result wherever the exact result is affordable.
//
//   pnpm umap:bench [options]
//
//     --shape <key>      generator (default branching); see MANIFOLDS
//     --sizes <a,b,c>    cell counts        (default 1000,4000,16000,50000)
//     --genes <n>        genes per latent axis                (default 6)
//     --n-neighbors <k>  (default 15)       --epochs <n>      (default 200)
//     --no-gpu           host only (skips the GPU k-NN and the GPU layout)
//     --no-descent       skip the approximate k-NN
//     --no-exact-above <n>  do not run the exact k-NN past this size (default 50000)

import { getDevice, releaseDevice } from "../src/gpu/device";
import { knnGpu } from "../src/gpu/spatial/knn";
import { GpuUmapLayout } from "../src/gpu/spatial/umapLayoutGpu";
import { knnDescentCpu, knnRecall } from "../src/spatial/knnDescent";
import { expressManifold, MANIFOLDS, makeManifold } from "../src/spatial/syntheticManifolds";
import { fuzzySimplicialSet, type KnnResult, knnBruteForceCpu } from "../src/spatial/umapGraph";
import { fitAB, initLayout, optimizeLayoutStep, trustworthiness } from "../src/spatial/umapLayout";

interface Args {
  shape: string;
  sizes: number[];
  genes: number;
  nNeighbors: number;
  epochs: number;
  gpu: boolean;
  descent: boolean;
  exactLimit: number;
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    shape: "branching",
    sizes: [1000, 4000, 16000, 50000],
    genes: 6,
    nNeighbors: 15,
    epochs: 200,
    gpu: true,
    descent: true,
    exactLimit: 50000,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i] ?? "";
    if (a === "--shape") args.shape = next();
    else if (a === "--sizes") args.sizes = next().split(",").map(Number).filter(Number.isFinite);
    else if (a === "--genes") args.genes = Number(next());
    else if (a === "--n-neighbors") args.nNeighbors = Number(next());
    else if (a === "--epochs") args.epochs = Number(next());
    else if (a === "--no-gpu") args.gpu = false;
    else if (a === "--no-descent") args.descent = false;
    else if (a === "--no-exact-above") args.exactLimit = Number(next());
    else if (a.startsWith("--")) throw new Error(`unknown option ${a}`);
  }
  if (!MANIFOLDS.some((m) => m.key === args.shape)) {
    throw new Error(`unknown shape '${args.shape}'; one of: ${MANIFOLDS.map((m) => m.key).join(", ")}`);
  }
  return args;
}

async function time<T>(fn: () => Promise<T> | T): Promise<[T, number]> {
  const t0 = performance.now();
  const value = await fn();
  return [value, performance.now() - t0];
}

/** Peak resident set, in MB. Reported because the memory wall arrives before the time
 *  wall for the dense feature matrix, and a table of seconds would not show it. */
function rssMb(): number {
  return process.memoryUsage().rss / (1024 * 1024);
}

interface Row {
  n: number;
  edges: number;
  knnExactGpu?: number;
  knnExactHost?: number;
  knnDescent?: number;
  descentRecall?: number;
  graph: number;
  layoutHost?: number;
  layoutGpu?: number;
  trust?: number;
  rss: number;
}

/** Trustworthiness is O(n² log n); above this it is sampled down rather than skipped, so
 *  the quality column does not silently vanish at exactly the sizes it matters most. */
const TRUST_SAMPLE = 3000;

function trustSampled(features: Float32Array, embedding: Float32Array, n: number, dim: number): number {
  if (n <= TRUST_SAMPLE) return trustworthiness(features, embedding, n, dim, 2, 10);
  // A contiguous slice, not a random one: the generators emit rows in no meaningful order
  // (branch membership is drawn per row), so a prefix is already an unbiased sample and
  // avoids building two permuted copies of a large matrix.
  const m = TRUST_SAMPLE;
  return trustworthiness(features.subarray(0, m * dim), embedding.subarray(0, m * 2), m, dim, 2, 10);
}

async function runSize(args: Args, n: number): Promise<Row> {
  const manifold = makeManifold(args.shape, n, 11);
  const expressed = expressManifold(manifold, { genesPerAxis: args.genes, seed: 11 });
  const { values, dim } = expressed;
  const k = args.nNeighbors - 1;
  const row: Row = { n, edges: 0, graph: 0, rss: 0 };

  // --- k-NN, every implementation that is affordable at this size ---------------------
  let exact: KnnResult | undefined;
  if (args.gpu && n <= args.exactLimit) {
    const [res, ms] = await time(() => knnGpu(values, { n, dim, k }));
    exact = res;
    row.knnExactGpu = ms;
  }
  if (!args.gpu && n <= args.exactLimit) {
    const [res, ms] = await time(() => knnBruteForceCpu(values, n, dim, k));
    exact = res;
    row.knnExactHost = ms;
  }
  let approx: KnnResult | undefined;
  if (args.descent) {
    const [res, ms] = await time(() => knnDescentCpu(values, n, dim, { k, seed: 42 }));
    approx = res;
    row.knnDescent = ms;
    if (exact) row.descentRecall = knnRecall(res, exact);
  }

  const knn = exact ?? approx;
  if (!knn) throw new Error(`no k-NN ran at n=${n} — --no-descent with n past --no-exact-above leaves nothing`);

  // --- graph --------------------------------------------------------------------------
  const [graph, graphMs] = await time(() => fuzzySimplicialSet(knn, { nNeighbors: args.nNeighbors }));
  row.graph = graphMs;
  row.edges = graph.nEdges;

  // --- layout, per epoch ---------------------------------------------------------------
  const ab = fitAB(0.1, 1);
  // The host layout is O(edges) per epoch in JavaScript; past a few hundred thousand edges
  // a full run would dominate the sweep, so a fixed small number of epochs is timed and the
  // per-epoch cost reported. That is the number that matters anyway — it is what the page
  // pays every frame.
  const HOST_EPOCHS = Math.max(3, Math.min(args.epochs, Math.ceil(2e6 / Math.max(graph.nEdges, 1))));
  const hostState = initLayout(graph, { dim: 2, nEpochs: args.epochs, seed: 7 });
  const [, hostMs] = await time(() => {
    for (let e = 0; e < HOST_EPOCHS; e++) optimizeLayoutStep(hostState, graph, { nEpochs: args.epochs, seed: 7, ab });
  });
  row.layoutHost = hostMs / HOST_EPOCHS;

  if (args.gpu) {
    const gl = await GpuUmapLayout.create(graph, { dim: 2, nEpochs: args.epochs, seed: 7 });
    // One warm-up submit first: the first step pays pipeline creation and buffer upload,
    // which is a one-off and would otherwise be charged to the per-epoch figure.
    gl.step(1);
    await gl.read();
    const [, gpuMs] = await time(async () => {
      gl.step(args.epochs);
      await gl.read();
    });
    row.layoutGpu = gpuMs / args.epochs;

    // Quality is measured on the GPU layout, since that is the one the page runs.
    const embedding = await gl.read();
    row.trust = trustSampled(values, embedding, n, dim);
    gl.destroy();
  } else {
    for (let e = HOST_EPOCHS; e < args.epochs; e++) optimizeLayoutStep(hostState, graph, { nEpochs: args.epochs, seed: 7, ab });
    row.trust = trustSampled(values, hostState.embedding, n, dim);
  }

  row.rss = rssMb();
  return row;
}

function fmt(v: number | undefined, digits = 0): string {
  return v === undefined ? "—" : v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function report(args: Args, rows: Row[]): void {
  const header = [
    "cells",
    "edges",
    "knn gpu",
    "knn host",
    "descent",
    "recall",
    "graph",
    "epoch host",
    "epoch gpu",
    "speedup",
    "trust",
    "rss",
  ];
  const body = rows.map((r) => [
    fmt(r.n),
    fmt(r.edges),
    r.knnExactGpu === undefined ? "—" : `${fmt(r.knnExactGpu)} ms`,
    r.knnExactHost === undefined ? "—" : `${fmt(r.knnExactHost)} ms`,
    r.knnDescent === undefined ? "—" : `${fmt(r.knnDescent)} ms`,
    r.descentRecall === undefined ? "—" : r.descentRecall.toFixed(3),
    `${fmt(r.graph)} ms`,
    r.layoutHost === undefined ? "—" : `${fmt(r.layoutHost, 2)} ms`,
    r.layoutGpu === undefined ? "—" : `${fmt(r.layoutGpu, 2)} ms`,
    r.layoutHost !== undefined && r.layoutGpu ? `${(r.layoutHost / r.layoutGpu).toFixed(0)}x` : "—",
    r.trust === undefined ? "—" : r.trust.toFixed(3),
    `${fmt(r.rss)} MB`,
  ]);
  const widths = header.map((h, c) => Math.max(h.length, ...body.map((b) => b[c]!.length)));
  const line = (cells: string[]) => cells.map((v, c) => v.padStart(widths[c]!)).join("  ");
  console.log(`\nshape ${args.shape} · ${args.nNeighbors} neighbours · ${args.epochs} epochs · ${args.genes} genes per latent axis\n`);
  console.log(line(header));
  console.log(widths.map((w) => "-".repeat(w)).join("  "));
  for (const b of body) console.log(line(b));
  console.log(
    "\nknn/descent/graph are one-off; epoch columns are per epoch. `trust` is trustworthiness at k=10" +
      `${rows.some((r) => r.n > TRUST_SAMPLE) ? `, sampled to ${TRUST_SAMPLE.toLocaleString()} rows past that size` : ""}.`,
  );
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.gpu) {
    const device = await getDevice();
    const info = (device as { adapterInfo?: { device?: string; description?: string } }).adapterInfo;
    console.log(`device   ${info?.device || info?.description || "WebGPU"}`);
  }
  const rows: Row[] = [];
  for (const n of args.sizes) {
    process.stdout.write(`n=${n.toLocaleString()} … `);
    const row = await runSize(args, n);
    rows.push(row);
    process.stdout.write("done\n");
  }
  report(args, rows);
}

/** Release Dawn's Instance so the process can exit — LAST, per `src/gpu/device.ts`. */
main()
  .then(() => releaseDevice())
  .catch(async (err) => {
    await releaseDevice();
    console.error(`umap-bench: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  });
