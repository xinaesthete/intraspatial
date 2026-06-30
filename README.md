# tgpu-htj2k

WebGPU **HTJ2K** (JPEG 2000 Part 15 / High-Throughput JPEG 2000) codec work, plus a
growing **in-GPU operation-graph runtime** and spatial-analysis toolbox built on
[TypeGPU](https://typegpu.com). A Rust/wasm CPU core handles the bitstream; a
TypeScript/TypeGPU layer runs the DWT, reaction–diffusion, and spatial primitives on the
GPU.

> **Runtime: Node + pnpm + vitest, not Bun.** The Dawn `webgpu` native binding segfaults
> under Bun on the compute path, so the toolchain matches the sibling `psychogeo` project.
> Use `pnpm`, not `bun`. See [`docs/decisions/0002-runtime-node-not-bun.md`](docs/decisions/0002-runtime-node-not-bun.md).

## Layout

| Path | What |
| :--- | :--- |
| [`src/`](src/) | The TypeScript GPU layer. `src/gpu/` holds the DWT kernels (5/3, 9/7), the [operation-graph runtime](src/gpu/graph/) (`graph/`), the reaction–diffusion sim, and spatial index. `src/wasm/` wraps the Rust core. |
| [`rust/htj2k-core/`](rust/htj2k-core/) | The Rust/wasm codec core (bitstream, block coder), built with `wasm-pack`. |
| [`playground/`](playground/) | The **operation-graph composer** — a React Flow app to wire ops on a canvas and run them in the browser via WebGPU. See [`playground/README.md`](playground/README.md). |
| [`docs-site/`](docs-site/) | The **Astro/Starlight docs site**, including interactive React-island demos (DWT drawing, filtration). See [`docs-site/README.md`](docs-site/README.md). |
| [`docs/`](docs/) | Design notes and [Architecture Decision Records](docs/decisions/) (`docs/decisions/`). |
| [`viz/`](viz/) | A standalone, dependency-free DWT primer (open `viz/index.html`). See [`viz/README.md`](viz/README.md). |
| [`test/`](test/) | GPU integration/bench tests (run under the separate `vitest.gpu` config). |

## Prerequisites

- **Node 22** — pinned via [Volta](https://volta.sh) (`"volta": { "node": "22.23.1" }`).
  Volta switches automatically inside the repo; otherwise use Node 22 yourself.
- **pnpm 11** (`packageManager` is pinned). This is a pnpm **workspace** — one
  `pnpm install` at the root installs the root package, `playground`, and `docs-site`.
- **Rust + `wasm-pack`** — only needed to rebuild the wasm codec core.

## Install

```sh
pnpm install
```

The external `openjph-wasm` sibling (a test-only reference fixture) is an
`optionalDependency`; it is skipped cleanly in worktrees and CI where the sibling
checkout isn't present.

## Common tasks

All run from the repo root.

| Command | Action |
| :--- | :--- |
| `pnpm dev` | Run the **playground** composer (alias for `dev:playground`) → http://localhost:5173 |
| `pnpm dev:playground` | Run the operation-graph composer (needs a WebGPU-capable browser) |
| `pnpm dev:docs` | Run the docs site → http://localhost:4321 |
| `pnpm build:playground` | Production build of the composer |
| `pnpm build:docs` | Production build of the docs site |
| `pnpm test` | Full test suite — CPU then GPU vitest projects |
| `pnpm test:cpu` | CPU vitest suite only |
| `pnpm test:gpu` | GPU vitest suite only (Dawn `webgpu`; one fork per file) |
| `pnpm test:watch` | Watch-mode CPU tests |
| `pnpm test:rust` | `cargo test` for the Rust core |
| `pnpm typecheck` | `tsc --noEmit` over the TS layer |
| `pnpm build:wasm` | Build the Rust core to wasm (`build:wasm:dev` for a debug build) |
| `pnpm bench:gpu` | IDWT GPU microbenchmark |

The two front-end apps also have their own `pnpm dev` / `pnpm build` if you prefer to run
them from inside `playground/` or `docs-site/`.

## Docs & decisions

Start with the toolbox overviews in [`docs/`](docs/) (e.g.
[`gpu-primitives-toolbox.md`](docs/gpu-primitives-toolbox.md),
[`gpu-resource-sync.md`](docs/gpu-resource-sync.md)) and the ADRs in
[`docs/decisions/`](docs/decisions/) — notably the Node-not-Bun runtime decision, the
`"use gpu"`/TGSL kernel approach (0003), and the field-type / wavelet-domain model
(0004, 0006) that the operation graph is built on.
