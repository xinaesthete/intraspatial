import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import typegpu from "unplugin-typegpu/vite";
import { defineConfig, type Plugin } from "vite";

const require_ = createRequire(import.meta.url);

/** Emit the OpenJPH WASM binaries next to the bundled Emscripten glue.
 *
 *  `openjph-wasm` loads its glue via `new URL("./wasm/libopenjph.mjs", import.meta.url)`, so a build
 *  bundles the glue into `assets/libopenjph-<hash>.mjs` — but the sibling `.wasm` it then asks for
 *  (`new URL("libopenjph.wasm", import.meta.url)`, resolved against the GLUE's own url) is never
 *  copied, so the volume viewer's decoder 404s in a production build (dev is fine: the glue is
 *  served straight from node_modules beside its wasm). Emitting the binaries UNHASHED into
 *  `assets/` is what makes that runtime-relative lookup resolve. */
function emitOpenJphWasm(): Plugin {
  const files = ["libopenjph.wasm", "libopenjph_simd.wasm"];
  return {
    name: "emit-openjph-wasm",
    apply: "build",
    generateBundle() {
      for (const name of files) {
        try {
          const path = require_.resolve(`openjph-wasm/wasm/${name}`);
          this.emitFile({ type: "asset", fileName: `assets/${name}`, source: readFileSync(path) });
        } catch {
          this.warn(`emit-openjph-wasm: could not resolve openjph-wasm/wasm/${name} — the volume viewer will not decode.`);
        }
      }
    },
  };
}

// The composer imports the toolbox's own runtime + ops from ../src. Those modules
// pull in src/gpu/device.ts, which statically imports the Node-only `webgpu` (Dawn)
// package — in the browser we use `navigator.gpu` instead, so we alias `webgpu` to a
// stub whose `create`/`globals` are never actually called (device.ts prefers
// navigator.gpu when present). unplugin-typegpu transpiles the `"use gpu"` kernels,
// exactly as the GPU vitest config does, so the SAME op definitions run here.
export default defineConfig({
  plugins: [react(), typegpu(), emitOpenJphWasm()],
  // Relative asset URLs, so the built prototypes work at ANY mount point. The docs site copies this
  // dist to `<pages base>/playground/`, where the default absolute `/assets/...` would 404.
  base: "./",
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
  // `openjph-wasm` is excluded for the same reason as zarrextra: it loads its Emscripten glue via
  // `new URL("./wasm/libopenjph.mjs", import.meta.url)`, which pre-bundling would rewrite to a
  // `.vite/deps` path that never emits the sibling `.wasm`.
  optimizeDeps: { exclude: ["zarrextra", "zarrextra/workers", "openjph-wasm"] },
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
        spatialdata: fileURLToPath(new URL("./spatialdata.html", import.meta.url)),
        hspf: fileURLToPath(new URL("./hspf.html", import.meta.url)),
        geometry: fileURLToPath(new URL("./geometry.html", import.meta.url)),
        raymarch: fileURLToPath(new URL("./raymarch.html", import.meta.url)),
        spatialscene: fileURLToPath(new URL("./spatialscene.html", import.meta.url)),
        spatialvolume: fileURLToPath(new URL("./spatialvolume.html", import.meta.url)),
        rasterstat: fileURLToPath(new URL("./rasterstat.html", import.meta.url)),
        cellstats: fileURLToPath(new URL("./cellstats.html", import.meta.url)),
        cellmodes: fileURLToPath(new URL("./cellmodes.html", import.meta.url)),
        r3fspike: fileURLToPath(new URL("./r3fspike.html", import.meta.url)),
      },
    },
  },
});
