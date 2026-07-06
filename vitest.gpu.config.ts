import typegpu from "unplugin-typegpu/vite";
import { defineConfig } from "vitest/config";

// GPU tests (`*.gpu.test.ts`) — run via the Dawn (`webgpu`) native addon.
// Each file runs in its OWN fork process, serially (fileParallelism: false):
// Dawn's process-exit teardown segfaults once a single process has created a
// device and done enough work across multiple files, so we isolate each file
// to a fresh process that creates one device and exits cleanly. Forks (not
// worker threads) keep the native GPU handle out of a shared thread.
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
  },
});
