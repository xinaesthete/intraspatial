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
  // `@spatialdata/core` is deliberately NOT in this list — it used to be, for a third variant of the
  // same problem, and MDV still excludes it (`~/code/www/MDV/vite.config.mts`). Up to core 0.8.0 the
  // cause was concrete: core deferred its vendored parquet-wasm behind
  // `import(/* @vite-ignore */ "../vendor/parquet-wasm/parquet_wasm.js")`, a path relative to core's
  // OWN dist. Pre-bundled, that module is served from `.vite/deps/`, where `../vendor/...` points at
  // nothing — Vite failed the import analysis and the whole `@spatialdata_core.js` chunk 500'd, so
  // every page touching sd.js died on `Failed to fetch dynamically imported module` while the parquet
  // path was named only in the SERVER log. core 0.10.0 fixed that upstream: the vendored wasm is a
  // real `./parquet-wasm` subpath export, imported as a bare specifier, which the optimizer resolves.
  // Removal verified on 0.10.0 against a real store (2026-09-23), with the dep cache cleared first:
  // core pre-bundles to `.vite/deps/@spatialdata_core.js` AND the vendored wasm comes along as
  // `.vite/deps/parquet_wasm-*.js` — the very module that used to 500. The parquet path then really
  // runs: `getParquetRowCount()` → 12,165,021 rows, `loadPolygonShapes()` → 162,254 polygons,
  // `getPointsTilingMetadata()` → real bounds. Image / volume / cell-table / scene pages all render.
  // Pre-bundling core is also what we WANT — it is a large dep with apache-arrow behind it.
  // This does not weaken the zarrextra rules above: core reaches zarrextra through the bare
  // specifier, which still resolves to the one excluded, node_modules-served instance.
  // If sd.js pages ever die on `Failed to fetch dynamically imported module` again, put
  // `@spatialdata/core` back in the exclude list and check whether upstream reintroduced a
  // dist-relative dynamic import.
  // NB MDV also excludes `zod`, because MDV is on Zod 3 while core wants Zod 4. That does not apply
  // here: `pnpm why zod` reports one version (4.x) across the workspace.
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
        umap: fileURLToPath(new URL("./umap.html", import.meta.url)),
      },
    },
  },
});
