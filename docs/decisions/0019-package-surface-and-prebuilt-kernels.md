# ADR-0019 — Package surface: one `intraspatial` package, subpath exports, pre-transformed kernels

Status: **accepted, landed** (2026-08-22). Promoted from the design note
[`packaging-and-consumers.md`](../packaging-and-consumers.md) the day the package was cut.
Implementation: `package.json` (`exports`, `files`, peer deps), `vite.lib.config.ts`,
`tsconfig.build.json`, `src/index.ts` + per-catalogue `index.ts` barrels, `src/gpu/device.ts`.

## Context

The engine and its catalogues were only ever consumed from inside this repo (playground, docs-site
via `../src`). After the 2026-08-22 split into IntraSpatial (MIT) and the `tgpu-htj2k` codec, the
next goal was a package that psychogeo/TerraCognita, MDV + spatialdata.js, and the art repos can
install. Three specific blockers were known: `webgpu` (Node-only Dawn) was a hard dependency; there
was no `exports` map or build; and — the least understood — the `"use gpu"` TGSL kernels are
rewritten by `unplugin-typegpu` in the *consumer's* bundler config, so it was unclear whether a
package could ship them at all.

## Decisions

1. **One package, `intraspatial`, with subpath exports** — not `@intraspatial/core` plus siblings.
   Package boundaries are for forced dependencies; `src/` has none that differ between catalogues
   (everything needs `typegpu`; nothing needs three/deck). Subpaths give the same ergonomics:
   `intraspatial` (engine = `gpu/graph` + `getDevice`), `/graph`, `/graph/*`, `/device`,
   `/datasource`, `/geometry`, `/evo`, `/spatial` (CPU goldens), `/color`, `/gpu/spatial`,
   `/gpu/sim`, `/gpu/interop`, and a **`./*` wildcard** so any module deep-imports as
   `intraspatial/<src path>` — which is the form the docs-site already documents
   (`intraspatial/gpu/spatial/anni`). The scope `@intraspatial/*` stays reserved for the viewer
   packages (`viewer-three`, `viewer-deck`) that *do* carry a forced dependency.

2. **Ship pre-transformed kernels.** `unplugin-typegpu`'s default filter (`/\.m?[jt]sx?$/`, no
   `node_modules` exclusion, early-pruned on the `"use gpu"` literal) means a consumer's plugin
   *could* transform our JS — but Vite dep pre-bundling bypasses plugins in dev, so that path is
   fragile. Instead the package is built by **Vite library mode with the same plugin**
   (`vite.lib.config.ts`): every non-test `src/**/*.ts` is an entry, `preserveModules` keeps
   `dist/` a 1:1 mirror of `src/`, and all bare imports are external. Consumers need only the
   `typegpu` peer; no bundler configuration. Types come from a separate declaration-only
   `tsc -p tsconfig.build.json` into the same tree, so the `./*` wildcard resolves `.d.ts` too.
   **Verified** by a scratch consumer (tarball + `typegpu` + `webgpu`, no plugin) running
   `nearestNeighborDistancesGpu` under plain Node and typechecking subpath imports under
   `moduleResolution: Bundler`.

3. **`webgpu` (Dawn) is an optional peer, loaded lazily.** `src/gpu/device.ts` no longer imports
   it statically: when `navigator.gpu` is absent it `await import(specifier)`s Dawn with the
   specifier held in a variable (`/* @vite-ignore */`), so browser bundles never resolve it and a
   missing install produces a clear error. `typegpu` is a **peer** (one instance per app — its
   registries and `instanceof` checks break otherwise) and a devDependency here. The playground's
   `webgpu` alias stub is now redundant but harmless.

4. **Barrels are generated, not curated.** `src/spatial`, `src/gpu/spatial`, `src/color`,
   `src/gpu/sim` got `export *` indexes. Two collisions were resolved by explicit re-export
   (`FuzzyAdjacency` → fixed-kernel module wins; `mulberry32` → `kernelAnalysis`). The duplicate
   `mulberry32` in `umapLayout.ts` is a dedup follow-up, not addressed here.

## Consequences

- `pnpm build:lib` (also `prepack`) produces `dist/`; `files` ships only `dist`, README, LICENSE.
  Version bumped to **0.1.0**. **Not published** — `npm publish` needs Peter's credentials.
- `fallow` (run 2026-08-22, maintainability 89) surfaced follow-ups that the export map now makes
  public surface: `src/gpu/sim/jet.ts` is 100 % dead; `ops/danceForces.ts` exports 13 unused
  symbols; `src/geometry/swept.ts` ↔ `geometry/index.ts` is an import cycle. Prune before 0.2.
- Anything exported from `src/` is now API. The `./*` wildcard is deliberate openness at 0.x;
  tighten it when the surface stabilises.
- The viewer layer in `playground/` remains the next structural move (design note, sequencing §2).
