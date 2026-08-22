# IntraSpatial

An **in-GPU operation-graph runtime** for spatial data, with domain catalogues of ops and
sources built on it — spatial statistics, geometry, simulation, evolutionary form, colour,
and tiled/multiscale datasources — all on [TypeGPU](https://typegpu.com) / WebGPU.

"Intra-spatial": one substrate that lives *inside* many disciplines (spatial biology, GIS
and terrain, generative art) rather than a bridge between two. The engine's job is to make
crossings between those domains cheap and expressive; the catalogues are where each domain's
primitives live.

> **Runtime: Node + pnpm + vitest, not Bun.** The Dawn `webgpu` native binding segfaults
> under Bun on the compute path, so the toolchain matches the sibling `psychogeo` project.
> Use `pnpm`, not `bun`. See [`docs/decisions/0002-runtime-node-not-bun.md`](docs/decisions/0002-runtime-node-not-bun.md).

> **Lineage.** This repo was split out of `tgpu-htj2k` on 2026-08-22 with history preserved.
> The HTJ2K codec (Rust/wasm core + GPU DWT kernels) stays in
> [`tgpu-htj2k`](https://github.com/xinaesthete/tgpu-htj2k) under BSD-2-Clause; this repo is
> pure MIT and has no dependency on it. Older design notes under `docs/` still mention the
> codec where it was historically relevant.

## FAIR by design

This is research tooling, and it is built to serve the
[FAIR principles](https://www.go-fair.org/fair-principles/) — **F**indable,
**A**ccessible, **I**nteroperable, **R**eusable — rather than treat them as an
afterthought:

- **Findable** — every op and source is a self-describing entry (name, description,
  category, rendered maths) in a discoverable catalogue; the executor is content-addressed
  so intermediate results have stable identities.
- **Accessible** — runs in any WebGPU browser over standard `navigator.gpu`, no
  proprietary runtime or install; imagery comes in through open formats (OME-Zarr /
  SpatialData, GeoTIFF, HTJ2K via the external `openjph-wasm`), and composed graphs
  serialise to plain JSON.
- **Interoperable** — primitives take and return plain typed arrays with an explicit
  shape/element/basis schema, so kernels port to deck.gl / MDV / SpatialData.js instead of
  locking analysis in.
- **Reusable** — a declarative operation graph *is* its own provenance; runs are
  reproducible (seeded RNG, CPU goldens with bit-exact/bounded-error GPU tests), and named
  subgraphs make reuse first-class. Code is MIT-licensed.

See [FAIR by design](docs-site/src/content/docs/concepts/fair.md) for the full treatment.

## Using it as a package

```sh
pnpm add intraspatial typegpu        # browser: that's all
pnpm add webgpu                      # headless Node only (Dawn; optional peer)
```

```ts
import { nearestNeighborDistancesGpu } from "intraspatial/gpu/spatial";
import { Graph, pullData } from "intraspatial/graph";
import { getDevice } from "intraspatial/device";
```

Subpaths: `intraspatial` (engine), `/graph`, `/device`, `/datasource`, `/geometry`, `/evo`,
`/spatial` (CPU goldens), `/color`, `/gpu/spatial`, `/gpu/sim`, `/gpu/interop`, and any module by
its `src/` path (`intraspatial/gpu/spatial/anni`). The `"use gpu"` kernels ship **pre-transformed**,
so no bundler plugin is needed — see
[ADR-0019](docs/decisions/0019-package-surface-and-prebuilt-kernels.md). Not yet on npm; build
locally with `pnpm build:lib` and `pnpm pack`.

## Layout

| Path | What |
| :--- | :--- |
| [`src/gpu/graph/`](src/gpu/graph/) | The **engine**: lazy-pull operation-graph runtime (executor, memo/pool, resident buffers, placement, browser + Node backends). |
| [`src/gpu/spatial/`](src/gpu/spatial/) | Spatial-statistics catalogue — neighbour distances, KDE splat, Getis-Ord, ANNI, cKNN, fuzzy adjacency, separable convolution, … each with a CPU golden. |
| [`src/gpu/sim/`](src/gpu/sim/) | Simulation catalogue — reaction–diffusion, force/body/spline dynamics, the HsPf figure sim. |
| [`src/gpu/fields/`](src/gpu/fields/), [`src/gpu/interop/`](src/gpu/interop/) | Field types and the three.js / canvas interop seams (no `three` import inside `src/`). |
| [`src/geometry/`](src/geometry/), [`src/evo/`](src/evo/) | Swept/superellipsoid geometry and the Mutator-lineage evolutionary catalogue (ParamSpec, pedigree, trait space). |
| [`src/datasource/`](src/datasource/) | Tiled / multiscale sources: view-driven tile-to-field, caches, synthetic and SpatialData-backed loaders. |
| [`src/spatial/`](src/spatial/), [`src/color/`](src/color/), [`src/coords.ts`](src/coords.ts) | CPU-side spatial utilities, colour spaces, coordinate systems (ADR-0015). |
| [`playground/`](playground/) | The **operation-graph composer** and viewer prototypes — React Flow + three.js apps that run the engine in the browser. See [`playground/README.md`](playground/README.md). |
| [`docs-site/`](docs-site/) | The **Astro/Starlight docs site** with interactive React-island demos. See [`docs-site/README.md`](docs-site/README.md). |
| [`docs/`](docs/) | Design notes, the [gap analysis](docs/gap-analysis.md), the [packaging plan](docs/packaging-and-consumers.md), and [Architecture Decision Records](docs/decisions/). |
| [`viz/`](viz/) | A standalone, dependency-free DWT primer (open `viz/index.html`). See [`viz/README.md`](viz/README.md). |
| [`test/`](test/) | GPU integration/bench tests (run under the separate `vitest.gpu` config). |

## Prerequisites

- **Node 22+** — pinned via [Volta](https://volta.sh). The pin is conservatism: Node 24
  and 26 also pass the full suite (see `docs/umap-on-anndata.md` §5).
- **pnpm 11** (`packageManager` is pinned). This is a pnpm **workspace** — one
  `pnpm install` at the root installs the root package, `playground`, and `docs-site`.
- A **WebGPU** device: the GPU test suite runs headless through Dawn (`webgpu` npm
  package); the apps need a WebGPU-capable browser.

## Install

```sh
pnpm install
```

The external `openjph-wasm` sibling (`../codecs/openjph-wasm`, a test-only reference
fixture) is an `optionalDependency`; it is skipped cleanly in worktrees and CI where the
sibling checkout isn't present. The playground depends on the *published* `openjph-wasm`
for its volume viewer.

## Common tasks

All run from the repo root.

| Command | Action |
| :--- | :--- |
| `pnpm dev` | Run the **playground** composer (alias for `dev:playground`) → http://localhost:5173 |
| `pnpm dev:playground` | Run the operation-graph composer (needs a WebGPU-capable browser) |
| `pnpm dev:docs` | Run the docs site → http://localhost:4321 |
| `pnpm build:playground` | Production build of the composer |
| `pnpm build:docs` | Production build of the docs site |
| `pnpm build:lib` | Build the publishable package to `dist/` (Vite lib build + declaration emit) |
| `pnpm test` | Full test suite — CPU then GPU vitest projects |
| `pnpm test:cpu` | CPU vitest suite only |
| `pnpm test:gpu` | GPU vitest suite only (Dawn `webgpu`; one fork per file) |
| `pnpm test:watch` | Watch-mode CPU tests |
| `pnpm typecheck` | `tsc --noEmit` over core and playground |
| `pnpm lint` | Biome check |

The two front-end apps also have their own `pnpm dev` / `pnpm build` if you prefer to run
them from inside `playground/` or `docs-site/`.

## Docs & decisions

Start with the toolbox overviews in [`docs/`](docs/) (e.g.
[`gpu-primitives-toolbox.md`](docs/gpu-primitives-toolbox.md),
[`gpu-resource-sync.md`](docs/gpu-resource-sync.md)), the
[gap analysis](docs/gap-analysis.md) for what is planned, and the ADRs in
[`docs/decisions/`](docs/decisions/) — notably the Node-not-Bun runtime decision (0002),
the `"use gpu"`/TGSL kernel approach (0003), and the field-type / wavelet-domain model
(0004, 0006) that the operation graph is built on. ADRs are written only for work in
flight; speculative directions live as design notes in `docs/*.md`.

## License

[MIT](LICENSE).
