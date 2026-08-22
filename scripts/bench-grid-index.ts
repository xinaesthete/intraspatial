// How long does the GPU uniform-grid index build take, and how does it compare with the host
// counting sort it replaces? Same shape as `bench-scan.ts` (a plain process, not a vitest
// benchmark — see that file for why), and the same two costs separated:
//   • the BUILD (histogram + scan + scatter, one submit), timed to `onSubmittedWorkDone`, which
//     is what a resident consumer such as `crossPcf.ts` actually pays;
//   • the READBACK of `start` + `items`, the cost `buildGridIndexGpu` adds on top for callers
//     that want the host `BucketGrid`, and the reason resident consumers should not.
//
//   pnpm bench:grid-index

import { getDevice, releaseDevice } from "../src/gpu/device";
import { ensureBuf } from "../src/gpu/scan/prefixSum";
import { buildGridIndexGpu, encodeGridIndex, getGridIndexCtx } from "../src/gpu/spatial/gridIndex";
import { buildBucketGrid, latticeFor } from "../src/spatial/bucketGrid";

const SIZES = [100_000, 1_000_000, 4_000_000, 10_000_000];
/** Points per cell on average — the density `cell = query radius` gives on a typical slide. */
const PER_CELL = 8;

function median(xs: number[]): number {
  const sorted = [...xs].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? Number.NaN;
}

async function timeIt(reps: number, warm: number, fn: () => Promise<unknown>) {
  for (let i = 0; i < warm; i++) await fn();
  const ts: number[] = [];
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    await fn();
    ts.push(performance.now() - t0);
  }
  return median(ts);
}

async function main() {
  const ctx = await getGridIndexCtx();
  const device = await getDevice();
  console.log("Medians of 3 after 1 warm-up. Points uniform in a square; cell sized for ~8 points per cell.");
  console.log("n           cells       cpu build   gpu build   gpu build+readback");
  for (const n of SIZES) {
    const side = 1000;
    const cell = side / Math.sqrt(n / PER_CELL);
    const pts = new Float32Array(2 * n);
    const xs = new Float32Array(n);
    const ys = new Float32Array(n);
    let s = 1;
    for (let i = 0; i < n; i++) {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      xs[i] = ((s >>> 8) / 8388608) * side;
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      ys[i] = ((s >>> 8) / 8388608) * side;
      pts[2 * i] = xs[i]!;
      pts[2 * i + 1] = ys[i]!;
    }
    const bounds: [number, number, number, number] = [0, 0, side, side];
    const lattice = latticeFor(xs, ys, cell, bounds);
    const cells = lattice.cols * lattice.rows;

    const cpuMs = await timeIt(3, 1, async () => buildBucketGrid(xs, ys, cell, bounds));

    const ptsBuf = ensureBuf(device, "bench:pts", n * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(ptsBuf, 0, pts);
    const buildMs = await timeIt(3, 1, async () => {
      const enc = device.createCommandEncoder();
      encodeGridIndex(ctx, ptsBuf, n, lattice, enc);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    });
    const fullMs = await timeIt(3, 1, () => buildGridIndexGpu(pts, { cell, bounds }));

    const f = (ms: number) => `${ms.toFixed(1)} ms`.padEnd(12);
    console.log(`${String(n).padEnd(12)}${String(cells).padEnd(12)}${f(cpuMs)}${f(buildMs)}${f(fullMs)}`);
  }
  await releaseDevice();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
