// How long does the scan / stream-compaction primitive take at MDV scale?
//
// Deliberately NOT a vitest benchmark, for the reason `bench-idwt-readback.ts` documents:
// inside the vitest fork `mapAsync` completion is only observed on a coarse tick, so every
// size reports the same flat number and the ordering against size inverts. Run as a plain
// process the timings scale properly.
//
// Two numbers matter and are separated here:
//   • the DISPATCH, which is what the ~2 s GPU watchdog kills silently. The scan is
//     bandwidth-bound, so this is where "35M rows is safe" has to be shown rather than
//     assumed.
//   • the READBACK, which is the cost this repo has repeatedly measured as the expensive
//     step, and the reason the on-device compact list is worth more than the host copy.
//
//   pnpm bench:scan
import { releaseDevice } from "../src/gpu/device";
import { exclusiveScanGpu } from "../src/gpu/scan/prefixSum";
import { streamCompactGpu } from "../src/gpu/scan/streamCompact";

const SIZES = [1_000_000, 5_000_000, 10_000_000, 20_000_000, 33_000_000];

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
  console.log("n           scan(u32)   compact ~6%  compact 50%     selected");
  for (const n of SIZES) {
    // Deterministic mask, so two runs are comparable. The shift is load-bearing: an LCG's
    // low bits have a short period, so `s % 20` is nowhere near 1-in-20.
    const mask = new Uint8Array(n);
    let s = 1;
    for (let i = 0; i < n; i++) {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      mask[i] = (s >>> 8) % 20 === 0 ? 0 : 1;
    }
    const dense = new Uint8Array(n);
    for (let i = 0; i < n; i++) dense[i] = i % 2;

    const values = new Uint32Array(n).fill(1);
    let scanMs: number;
    try {
      scanMs = await timeIt(3, 1, () => exclusiveScanGpu(values));
    } catch (err) {
      // 33M u32 is 132 MB, past the 128 MiB default maxStorageBufferBindingSize.
      console.log(`${String(n).padEnd(12)}${(err as Error).message}`);
      continue;
    }
    const sparse = await timeIt(3, 1, () => streamCompactGpu(mask, { pass: "eq", value: 0 }));
    const half = await timeIt(3, 1, () => streamCompactGpu(dense, { pass: "gt", value: 0 }));
    const { count } = await streamCompactGpu(mask, { pass: "eq", value: 0 });

    console.log(
      `${String(n).padEnd(12)}${scanMs.toFixed(1).padStart(7)} ms${sparse.toFixed(1).padStart(9)} ms${half.toFixed(1).padStart(11)} ms${String(count).padStart(14)}`,
    );
  }
  console.log("\nMedians of 3 after 1 warm-up. scan(u32) includes reading back all n offsets;");
  console.log("compact includes both readbacks (the 4-byte count, then the index list).");
  await releaseDevice();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
