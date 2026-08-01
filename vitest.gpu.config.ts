import typegpu from "unplugin-typegpu/vite";
import { defineConfig } from "vitest/config";

// GPU tests (`*.gpu.test.ts`) — run via the Dawn (`webgpu`) native addon.
//
// **2026-07-29: the Dawn instability this config was built around was a bug in our own
// `src/gpu/device.ts`, and it is fixed.** `create([])` returns Dawn's Instance, which
// owns the native event loop and the mutexes every later device call takes — and we
// were letting it fall out of scope as soon as `getDevice()` resolved. V8 then collected
// it whenever it felt like it and the N-API finaliser destroyed the Instance out from
// under a live device, producing `mutex lock failed: Invalid argument` or a plain
// segfault. Because it was GC-timing dependent it looked like ambient flakiness, which
// is how it survived so long. `device.ts` now holds the Instance and adapter for the
// process lifetime; see the comment there.
//
// What that changed, measured over 5 runs each:
//   • `fileParallelism: false` → `true`   — 9.5s to 3.2s, identical results.
//   • `dangerouslyIgnoreUnhandledErrors`  — removed. Zero "Worker exited unexpectedly"
//                                           errors now occur, so the suite gets the same
//                                           strictness as the CPU one, and a genuine
//                                           worker crash will fail the run again
//                                           instead of being silently tolerated.
//   • 93 tests pass deterministically. The only failing files are the ones that import
//     `rust/htj2k-core/pkg` and need `pnpm build:wasm` first — not a GPU issue.
//
// One historical note is kept, because it remains the reason for a choice:
//   • **Forks, not worker threads.** Keeps the native GPU handle out of a shared thread.
//
// **The Node-22 pin is gone (2026-08-01).** It existed because newer V8/Node shutdown
// interacted worse with Dawn's atexit — which was indeed this same lifetime bug from
// another angle. Full `pnpm test` and `pnpm bench:readback` pass on 24.18.0 and 26.5.0,
// 9/9 clean exits, so `volta.node` is now 24.18.0. The three Dawn limits recorded in
// ADR-0003 went the same way; probes in `test/dawn-limits-sweep.gpu.test.ts`.
//
// If flakiness reappears, suspect a native object we are failing to keep alive before
// concluding that Dawn is unreliable.
export default defineConfig({
  // Transpiles `"use gpu"` TS kernels to WGSL at build time (TGSL). Compute
  // primitives that fit TGSL are authored in TypeScript; ones needing atomics /
  // workgroup shared memory / barriers stay as WGSL templates (resolveWithContext).
  plugins: [typegpu()],
  test: {
    pool: "forks",
    fileParallelism: true,
    testTimeout: 30_000,
    include: ["test/**/*.gpu.test.ts", "src/**/*.gpu.test.ts"],
  },
});
