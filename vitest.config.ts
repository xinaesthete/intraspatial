import { defineConfig } from "vitest/config";

// CPU / wasm tests. GPU tests (`*.gpu.test.ts`) run separately via
// `vitest.gpu.config.ts` because the Dawn (`webgpu`) native addon is unstable
// when a process has also done heavy wasm work — its process-exit teardown can
// segfault. Keeping the two in separate processes avoids that. `pnpm test` runs
// both (see package.json).
export default defineConfig({
  test: {
    pool: "forks",
    poolOptions: { forks: { singleFork: true } },
    testTimeout: 30_000,
    include: ["test/**/*.test.ts", "src/**/*.test.ts"],
    exclude: ["**/node_modules/**", "**/dist/**", "**/*.gpu.test.ts"],
  },
});
