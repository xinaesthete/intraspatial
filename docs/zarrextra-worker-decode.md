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

Versions match MDV: `@spatialdata/core@0.2.5`, `zarrextra@0.2.3`, and zarrextra's optional worker
deps `@cornerstonejs/codec-openjph@2.4.7`, `@fideus-labs/fizarrita@1.4.1`,
`@fideus-labs/worker-pool@1.0.0`.

1. **Install** — `pnpm add @spatialdata/core zarrextra`. The worker deps above are
   `optionalDependencies` of zarrextra and pnpm pulls them automatically. (They link *under
   zarrextra's own* `node_modules`, so a `require.resolve('@cornerstonejs/codec-openjph')` from the
   app root reports "missing" even though the worker resolves them fine — a red herring.)

2. **Two calls, once, before any read** — registration and worker enablement are *separate*
   concerns and **both** are required:
   ```ts
   const { registerExperimentalHtj2kCodec } = await import('zarrextra');
   const { enableWorkerChunkDecode } = await import('zarrextra/workers');
   registerExperimentalHtj2kCodec(); // says WHAT the codec is (id → decoder). No custom decoder
                                     // needed once @cornerstonejs/codec-openjph is installed.
   enableWorkerChunkDecode();        // says WHERE decode runs (a worker pool bundling the codec+wasm).
   ```

3. **Vite (v5): exclude the *whole* zarrextra package from dep pre-bundling.**
   ```ts
   optimizeDeps: { exclude: ['zarrextra', 'zarrextra/workers'] }
   ```

4. **Read normally** — `readZarr(url)` → `sdata.images[name]` →
   `loadOmeZarrMultiscalesFromStore(img.getStore())` → `source.getTile({ x, y, selection })`.
   Decode now happens in the worker.

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
  `@cornerstonejs/codec-openjph` as a bare specifier. The symptom (a cornerstone resolution error)
  points nowhere near the real cause (a duplicated module). Excluding the whole package gives one
  shared instance.

(We are on Vite 5. MDV is on Vite 7 with `worker: { format: 'iife' }` and does *not* exclude — the
bundler trade-off differs by version, which is itself worth zarrextra documenting.)

## Recommended `zarrextra` changes (to file upstream)

Ordered by how much confusion each would remove:

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
