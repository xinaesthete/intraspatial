import { defineConfig } from "vite-plus";

// Workspace-root Vite+ config: the one place `vp fmt`, `vp lint`, `vp check` and `vp test` read.
// Format and lint settings here apply to every package (Vite+ does not honour nested lint/fmt blocks).
// The library build (`vite.lib.config.ts`), the GPU suite (`vitest.gpu.config.ts`) and the playground
// app (`playground/vite.config.ts`) keep their own configs because each needs plugins or process
// settings the others must not get.
export default defineConfig({
  // CPU / wasm tests. GPU tests (`*.gpu.test.ts`) run separately via `vitest.gpu.config.ts` because the
  // Dawn (`webgpu`) native addon is unstable when a process has also done heavy wasm work — its
  // process-exit teardown can segfault. Keeping the two in separate processes avoids that. `pnpm test`
  // runs both (see package.json).
  test: {
    // Vitest 4 removed `poolOptions`; the old `forks.singleFork: true` is now the
    // top-level `maxWorkers: 1` + `isolate: false` (one shared fork process).
    pool: "forks",
    maxWorkers: 1,
    isolate: false,
    testTimeout: 30_000,
    include: ["test/**/*.test.ts", "src/**/*.test.ts", "packages/*/src/**/*.test.ts"],
    exclude: ["**/node_modules/**", "**/dist/**", "**/*.gpu.test.ts"],
  },

  // Oxfmt, set to reproduce the house style Biome enforced (2-space, 140 columns, double quotes).
  fmt: {
    printWidth: 140,
    sortPackageJson: false,
    ignorePatterns: [
      // Prose and hand-laid data: Biome never formatted these, and reflowing them is churn, not style.
      "**/*.md",
      "**/*.mdx",
      "**/*.html",
      "**/*.css",
      "**/*.yaml",
      "**/*.yml",
      "**/*.svg",
      ".fallow-baseline/**",
      "**/rust/**",
      "**/target/**",
      "**/.astro/**",
      "**/public/dwt-primer/**",
      "viz/**",
      "pnpm-lock.yaml",
    ],
  },

  lint: {
    // Oxlint's `correctness` category (its default) across the TS, React and import plugins.
    plugins: ["eslint", "typescript", "unicorn", "oxc", "import", "react"],
    categories: { correctness: "error" },
    ignorePatterns: ["**/rust/**", "**/target/**", "**/.astro/**", "**/public/**", "viz/**", "scratch/**"],
    jsPlugins: [
      { name: "vite-plus", specifier: "vite-plus/oxlint-plugin" },
      { name: "intraspatial", specifier: "./scripts/lint/gpu-resource-label.mjs" },
    ],
    rules: {
      // Test and config imports go through `vite-plus/test` / `vite-plus`, so there is one Vitest and one Vite.
      "vite-plus/prefer-vite-plus-imports": "error",

      // Deliberate numeric idioms, not mistakes: `new Array(n)` to presize, `0 * w + i` to keep an
      // index expression's shape visible, and `[...map.keys()]` to snapshot keys before a loop
      // that deletes from the map (the executor's lease sweep relies on exactly that).
      "unicorn/no-new-array": "off",
      "oxc/erasing-op": "off",
      "unicorn/no-useless-spread": "off",
      // Cannot see Vite query imports (`?worker&url`); TypeScript checks default imports anyway.
      "import/default": "off",

      // The playground and docs-site React is prototype UI. Keep these visible, not blocking.
      "react-hooks/exhaustive-deps": "warn",
      "react/refs": "warn",
      "react/immutability": "warn",
      "react/set-state-in-effect": "warn",
      "react/preserve-manual-memoization": "warn",
    },
    overrides: [
      {
        // Every GPU object the library and its packages create is named — see the rule's header.
        files: ["src/**/*.ts", "packages/*/src/**/*.ts", "packages/*/src/**/*.tsx"],
        excludeFiles: ["**/*.test.ts", "**/*.test.tsx"],
        rules: { "intraspatial/gpu-resource-label": "error" },
      },
    ],
  },
});
