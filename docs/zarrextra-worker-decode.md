# Decoding HTJ2K off-main-thread with zarrextra's codec worker

What it actually took to get `zarrextra`'s worker-pool HTJ2K decode working for a real store here
(ADR-0010), and what would make it easier to set up upstream. Companion to
[`decisions/0010-spatialdata-js-as-loader-source.md`](./decisions/0010-spatialdata-js-as-loader-source.md).

**Note on MDV.** MDV's `ensureChunkWorker` calls only `enableWorkerChunkDecode()`. That gets the
worker *infrastructure* running, but on its own it does **not** decode real HTJ2K — the pipeline
still errors `Unknown codec: experimental.openjph_htj2k`. It's likely MDV had not yet exercised the
path where an element's chunks actually use this codec, so it would hit the same errors we did. The
recipe below is the full set of steps that were actually required.

## What we needed (the working recipe)

Current versions: `@spatialdata/core@0.8.0`, `zarrextra@0.4.0`, and zarrextra's optional worker deps
`@fideus-labs/fizarrita@1.4.1`, `@fideus-labs/worker-pool@1.0.0`. HTJ2K now decodes through
`openjph-wasm@0.1.0`, inlined into `codec-worker.js` (base64 data URI) along with its wasm;
`@cornerstonejs/codec-openjpeg@1.3.0` remains for legacy JPEG 2000 only.

> **Upgraded 2026-08-22 (IntraSpatial), zarrextra 0.2.3 → 0.4.0 and core 0.2.5 → 0.8.0.** The two move
> together: core pins `zarrextra` exactly (`@spatialdata/core@0.8.0` pins `0.4.0`), so the playground
> pins `zarrextra` to exactly `0.4.0` as well — any other pair leaves two zarrextra instances in the
> graph, and worker decode silently reads through whichever one the bundler chose. Lockfile effects:
> `apache-arrow` 17 → 21, and `parquet-wasm` drops out as a separate package because core vendors it.
> This supersedes the version-specific parts of ADR-0010, which records the 0.2.x state it decided in.

1. **Install** — `pnpm add @spatialdata/core zarrextra`. The worker deps above are
   `optionalDependencies` of zarrextra and pnpm pulls them automatically. (They link *under
   zarrextra's own* `node_modules`, so a `require.resolve(...)` from the app root reports "missing"
   even though the worker resolves them fine — a red herring.) Check `pnpm why zarrextra` reports
   **one** version: adding 0.4.0 while core still pins an older one puts two in the graph, and the
   read goes through whichever instance the bundler happened to give the reader.

2. **Two calls, once, before any read** — registration and worker enablement are *separate*
   concerns and **both** are required:
   ```ts
   const { registerExperimentalHtj2kCodec } = await import('zarrextra');
   const { enableWorkerChunkDecode } = await import('zarrextra/workers');
   registerExperimentalHtj2kCodec(); // says WHAT the codec is (id → decoder).
   enableWorkerChunkDecode();        // says WHERE decode runs (a worker pool bundling the codec+wasm).
   ```
   `@spatialdata/core@0.8.0` does **not** do either for you — its `enablePointsWorker` is an
   unrelated, points-only worker — so MDV's comment that its `ensureChunkWorker` call "is currently
   redundant" is wrong: without it, nothing routes chunk decode off-thread.

3. **Vite (v8): exclude the *whole* zarrextra package, plus `openjph-wasm` and `@spatialdata/core`,
   from dep pre-bundling.**
   ```ts
   optimizeDeps: { exclude: ['zarrextra', 'zarrextra/workers', 'openjph-wasm', '@spatialdata/core'] }
   ```

4. **Read through `getTile`** — `readZarr(url)` → `sdata.images[name]` →
   `loadOmeZarrMultiscalesFromStore(img.getStore())` → `source.getTile({ x, y, selection })`.
   Decode now happens in the worker. **That seam is the only one the worker sees** — see the last
   failure mode below.

## Why each step was needed (the failure modes, all cryptic)

- **Skip `registerExperimentalHtj2kCodec()`** → `Unknown codec: experimental.openjph_htj2k`.
  `enableWorkerChunkDecode()` does not register image-codec ids.
- **Pre-bundle `zarrextra/workers`** → the worker script is loaded via
  `new URL('./codec-worker.js', import.meta.url)`; Vite rewrites that to a `.vite/deps/codec-worker.js`
  it never emits → the worker **404s and decode hangs forever with no error**.
- **Pre-bundle `zarrextra` but exclude only `zarrextra/workers`** → `chunkDecode` exists as **two
  module instances** (one pre-bundled, one from node_modules). `enableWorkerChunkDecode()` flips the
  worker backend on the node_modules instance; `getTile` reads the backend from the pre-bundled
  instance, still `inline` → **silent fall back to main-thread decode**, which then fails to resolve
  the codec package as a bare specifier. The symptom (a module-resolution error) points nowhere near
  the real cause (a duplicated module). Excluding the whole package gives one shared instance.
- **Pre-bundle `@spatialdata/core`** → core defers its vendored parquet-wasm behind
  `import(/* @vite-ignore */ '../vendor/parquet-wasm/parquet_wasm.js')`, a path relative to core's
  own `dist`. Served from `.vite/deps/`, `../vendor/…` points at nothing; Vite fails import analysis
  and the whole `@spatialdata_core.js` chunk 500s, so every page touching sd.js dies on
  `Failed to fetch dynamically imported module` — with the parquet path named only in the *server*
  log, not the browser. MDV excludes core for this same reason.
- **Decode on the main thread at all** (no worker, or a read the worker can't see) →
  `CodecPipelineError: Failed to decode chunk via codec "experimental.openjph_htj2k"`, whose `cause`
  is `TypeError: Failed to resolve module specifier 'openjph-wasm'`. zarrextra's built-in decoder
  reaches its codec through `Function('specifier', 'return import(specifier)')` — deliberately opaque
  to bundlers, and a browser cannot resolve a bare specifier at runtime. **The fix is to pass the
  module yourself**: `registerExperimentalHtj2kCodec({ decoder: createOpenJphDecoder(decode) })` with
  `decode` from your own `import('openjph-wasm')`. Only the worker path works decoder-free, because
  `codec-worker.js` inlines openjph.
- **Read through anything other than `getTile`** → the worker is bypassed, silently. The backend set
  by `enableWorkerChunkDecode()` is consulted by exactly one function, zarrextra's internal
  `getZarrChunk` (a `zarr.get` selection read, which `getTile` calls); it is not exported. A caller
  holding a zarrita `Array` and calling `Array.getChunk(coords)` — as `spatialDataVolume.ts` does, to
  read a native 3-D brick in one fetch instead of 32 `getTile`s — never reaches it, so decode runs
  inline no matter how many workers were spawned.

(We are on Vite 8, and still exclude. MDV is on Vite 7 with `worker: { format: 'iife' }`; we have
not needed that — the worker loads from `node_modules` with its `import.meta.url` intact, and both
the dev server and a production build emit it.)

## Recommended `zarrextra` changes (to file upstream)

Ordered by how much confusion each would remove. **Still open as of 0.4.0** — the list was written
against 0.2.3 and re-checked against the shipped 0.4.0 source on 2026-08-10; none of items 1–6 has
landed, and 0.4.0 adds two more (7, 8). What 0.4.0 *did* fix is the multi-component defect: HTJ2K
goes through `openjph-wasm`, so a z-deep chunk no longer decodes as component 0 replicated.

1. **Make the worker backend a cross-instance singleton.** Store the decode-backend state on
   `globalThis` under a symbol rather than in module scope. A bundler that duplicates the module
   (the #1 time-sink above) would then still share the backend, so `enableWorkerChunkDecode()` is
   seen by every `getTile`. This alone removes the worst, most-mysterious failure.

2. **Fail loudly instead of hanging / mis-pointing.**
   - If the worker script fails to load, reject the decode with a clear error (and/or a timeout)
     rather than hanging forever.
   - If the worker backend is enabled but a decode runs inline, emit a one-time warning
     (`worker decode enabled but ran inline — likely a duplicate zarrextra instance from bundler
     pre-bundling; see <docs>`).
   - If an image codec id is unregistered, the "Unknown codec" error should name the fix
     (`call registerExperimentalHtj2kCodec()` / `registerJpeg2kCodec()`).

3. **One call for the common case.** Export a convenience that does both registration and worker
   enablement, e.g. `enableHtj2kWorkerDecode()` → `registerExperimentalHtj2kCodec()` +
   `enableWorkerChunkDecode()`. The current split (register in `zarrextra`, enable in
   `zarrextra/workers`) is easy to half-implement — as MDV's `ensureChunkWorker` did.

4. **Document the bundler story.** A short "Using the worker with Vite/webpack/etc." section: the
   `new URL(import.meta.url)` worker pattern, the `optimizeDeps.exclude` (or `worker.format`)
   requirement, and how it varies by Vite 5 vs 7. Optionally ship a tiny Vite plugin or exported
   recommended config.

5. **Clarify the codec deps.** They are `optionalDependencies`, so a working setup depends on the
   package manager pulling them and linking them under zarrextra. Either promote them to documented
   peer deps with an install snippet, or add a startup check that reports precisely which codec
   package is missing. Today their absence surfaces only as a downstream bare-specifier resolution
   error.

6. **A README "decode a real HTJ2K store in a worker" happy-path** — the full recipe in §"What we
   needed", so the next integrator doesn't rediscover the register+enable+bundler triple.

7. **Don't reach for the codec through `Function('specifier', 'return import(specifier)')`.** That
   construction exists to hide the import from bundlers, but the cost is that a browser then has to
   resolve a bare specifier at runtime, which it cannot do — so the *default* main-thread decoder is
   unusable in every bundled web app, and only fails at first decode. A plain
   `import('openjph-wasm')` is what bundlers already know how to rewrite; if the goal is to keep the
   dependency optional, catch the import failure and say so.

8. **Expose a chunk-level read that the worker backend serves.** `enableWorkerChunkDecode()` only
   affects `getZarrChunk`, which is internal and reached only via `PixelSource.getTile`. Anyone
   reading volumetric data wants the store's native 3-D chunk in one call (`Array.getChunk`), not a
   2-D tile per z slice, and today that means giving up worker decode entirely. Either export
   `getZarrChunk`, or route the worker backend at the zarrita codec layer so any read benefits.
