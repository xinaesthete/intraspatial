import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import typegpu from "unplugin-typegpu/vite";
import { defineConfig } from "vite";

// The composer imports the toolbox's own runtime + ops from ../src. Those modules
// pull in src/gpu/device.ts, which statically imports the Node-only `webgpu` (Dawn)
// package — in the browser we use `navigator.gpu` instead, so we alias `webgpu` to a
// stub whose `create`/`globals` are never actually called (device.ts prefers
// navigator.gpu when present). unplugin-typegpu transpiles the `"use gpu"` kernels,
// exactly as the GPU vitest config does, so the SAME op definitions run here.
export default defineConfig({
  plugins: [react(), typegpu()],
  // Keep the WHOLE zarrextra package out of Vite dep pre-bundling (SpatialDataLoader, ADR-0010).
  // Two reasons, both from pre-bundling: (1) `zarrextra/workers` resolves its worker via
  // `new URL('./codec-worker.js', import.meta.url)`, which pre-bundling rewrites to a
  // `.vite/deps/codec-worker.js` that is never emitted → the worker 404s; (2) more subtly,
  // pre-bundling `zarrextra` but not `zarrextra/workers` splits `chunkDecode` into two module
  // instances, so `enableWorkerChunkDecode()` flips the worker backend on one instance while
  // `getTile` reads the other (still inline) → decode silently falls back to the main thread and
  // fails to resolve the codec. Excluding the whole package gives ONE instance served from
  // node_modules, so the worker actually intercepts decode. Under Vite 8 `optimizeDeps` runs
  // through Rolldown (was esbuild), but `exclude` is still the sanctioned fix for the
  // `new Worker(new URL(..., import.meta.url))` pattern — excluding preserves the relative URL.
  optimizeDeps: { exclude: ["zarrextra", "zarrextra/workers"] },
  resolve: {
    alias: {
      webgpu: fileURLToPath(new URL("./src/webgpu-stub.ts", import.meta.url)),
    },
    // The composer's own imports and the ../src imports must share ONE TypeGPU
    // instance, or its internal registries (and `instanceof` checks) break.
    dedupe: ["typegpu"],
  },
  server: {
    port: Number(process.env.PORT) || 5173,
    fs: { allow: [".."] }, // allow importing ../src/...
  },
  build: {
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL("./index.html", import.meta.url)),
        datasource: fileURLToPath(new URL("./datasource.html", import.meta.url)),
        hspf: fileURLToPath(new URL("./hspf.html", import.meta.url)),
      },
    },
  },
});
