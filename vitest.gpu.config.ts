import { defineConfig } from "vitest/config";

// GPU tests (`*.gpu.test.ts`) — run in their own process via the Dawn
// (`webgpu`) native addon. Single fork, no module isolation, so one cached
// GPUDevice is shared across files (creating multiple Dawn devices, or mixing
// with heavy wasm in the same process, destabilises Dawn's teardown).
export default defineConfig({
  test: {
    pool: "forks",
    poolOptions: { forks: { singleFork: true } },
    isolate: false,
    testTimeout: 30_000,
    include: ["test/**/*.gpu.test.ts", "src/**/*.gpu.test.ts"],
  },
});
