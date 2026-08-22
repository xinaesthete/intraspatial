// Library build for the `intraspatial` package.
//
// Why a Vite build and not plain `tsc`: the `"use gpu"` kernels (26 files in `src/`) are
// TypeGPU/TGSL functions that `unplugin-typegpu` rewrites at build time — attaching the
// parsed-AST metadata `tgpu.resolve` needs. If we emitted untransformed JS, every consumer
// would have to run the plugin over our `node_modules` entry (and Vite's dep pre-bundling
// would still bypass it in dev). So the package ships PRE-TRANSFORMED kernels; consumers need
// only `typegpu` (a peer) and nothing in their bundler config. Types come from
// `tsconfig.build.json` (declaration-only emit into the same `dist/` tree).
//
// `preserveModules` keeps `dist/` a 1:1 mirror of `src/` (minus tests) so the `exports`
// map's `./*` wildcard and per-directory indexes both resolve, and tree-shaking stays
// with the consumer.
import { readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import typegpu from "unplugin-typegpu/vite";
import { defineConfig } from "vite";

const SRC = "src";
const isTest = (f: string) => /\.test\.ts$/.test(f);

function entries(dir: string, out: Record<string, string> = {}): Record<string, string> {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) entries(p, out);
    else if (name.endsWith(".ts") && !isTest(name)) out[relative(SRC, p).replace(/\.ts$/, "")] = p;
  }
  return out;
}

export default defineConfig({
  plugins: [typegpu()],
  build: {
    lib: { entry: entries(SRC), formats: ["es"] },
    outDir: "dist",
    emptyOutDir: true,
    minify: false,
    sourcemap: true,
    target: "esnext",
    rollupOptions: {
      // Everything that is not a relative/absolute import is a dependency the consumer installs.
      external: (id) => !id.startsWith(".") && !id.startsWith("/") && !id.includes("\0"),
      output: { preserveModules: true, preserveModulesRoot: SRC, entryFileNames: "[name].js" },
    },
  },
});
