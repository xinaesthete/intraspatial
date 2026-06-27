import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // The Dawn (`webgpu`) native addon is happiest in a single process; avoid
    // worker-thread interactions with the native GPU handle.
    pool: "forks",
    poolOptions: { forks: { singleFork: true } },
    testTimeout: 30_000,
    include: ["test/**/*.test.ts", "src/**/*.test.ts"],
  },
});
