import typegpu from "unplugin-typegpu/vite";
import { defineConfig } from "vitest/config";

// GPU tests (`*.gpu.test.ts`) — run via the Dawn (`webgpu`) native addon.
// Each file runs in its OWN fork process, serially (fileParallelism: false):
// Dawn's process-exit teardown segfaults once a single process has created a
// device and done enough work across multiple files, so we isolate each file
// to a fresh process that creates one device and exits cleanly. Forks (not
// worker threads) keep the native GPU handle out of a shared thread.
//
// Two findings from the 2026-07 toolchain bump, so this isn't re-litigated:
//   • Per-file isolation is OPTIMAL, not incidental. The tempting inverse — one
//     shared fork (`maxWorkers:1, isolate:false`) — is catastrophic: a mid-run
//     Dawn crash takes the remaining files down with it (a run collapsed to 2
//     tests). No single file crashes when run alone; the crash is a teardown
//     race that only emerges under the churn of many device-creating forks.
//   • Do NOT bump Node past 22 while on Dawn 0.4.0 (latest). Newer V8/Node
//     shutdown interacts worse with Dawn's atexit: measured ~3 teardown crashes
//     on Node 22 vs 6–8 on 24 and 8 on 26 — and the extra crashes silently DROP
//     passing GPU results (counted tests 50 → ~40). Revisit when a newer Dawn
//     tears down cleanly. (Deno's wgpu backend has the same device-destroy crash
//     class — not an escape.)
export default defineConfig({
  // Transpiles `"use gpu"` TS kernels to WGSL at build time (TGSL). Compute
  // primitives that fit TGSL are authored in TypeScript; ones needing atomics /
  // workgroup shared memory / barriers stay as WGSL templates (resolveWithContext).
  plugins: [typegpu()],
  test: {
    pool: "forks",
    fileParallelism: false,
    testTimeout: 30_000,
    include: ["test/**/*.gpu.test.ts", "src/**/*.gpu.test.ts"],
    // Dawn's native atexit teardown still segfaults a few fork processes AFTER their
    // tests have passed and reported (the isolation above shrinks it to a couple of
    // files, not all). Vitest 2 tolerated that dirty worker exit; Vitest 4 escalates
    // it to a run-failing "Worker exited unexpectedly" unhandled error. This flag
    // restores the v2 behaviour — scoped to this GPU config ONLY (the CPU suite keeps
    // full strictness) — so a benign post-report segfault no longer fails a green run.
    dangerouslyIgnoreUnhandledErrors: true,
  },
});
