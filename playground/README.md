# Operation-graph composer

A visual composer for the in-GPU operation-graph runtime (`src/gpu/graph/`). Wire
source generators and ops on a React Flow canvas, select a node, and pull its output —
the canvas serialises to the runtime's `Graph` IR and the executor runs the minimal
required subgraph in the browser via WebGPU (`navigator.gpu`).

It reuses the **same op definitions** that the Node vitest suite tests: the only
difference is the backend (`src/backend.browser.ts` vs `src/gpu/graph/backend.node.ts`)
and a Vite alias stubbing the Node-only `webgpu` package (the browser uses
`navigator.gpu`). `unplugin-typegpu` transpiles the `"use gpu"` kernels, exactly as the
GPU vitest config does.

## Run

From the repo root (`pnpm install` there installs the whole workspace):

```sh
pnpm dev:playground   # or just `pnpm dev` — http://localhost:5173
```

Or from inside this directory: `pnpm dev`. Needs a WebGPU-capable browser.

Node 22 is required (see the repo's Volta pin). The dev server allows importing from
`../src`, so edits to the toolbox's ops are picked up live.

Try the **Examples** in the sidebar — reaction–diffusion (a single complex-field
feedback loop), a reusable “Hotspots” subgraph instantiated twice, and a wavelet denoise
(forward DWT → threshold detail → inverse DWT).

## What's here

- `App.tsx` — the composer (palette, canvas, inspector, run/preview).
- `OpNode.tsx` — the custom node (typed input/output handles per op port).
- `buildGraph.ts` — React Flow state → `Graph` IR → `pull`.
- `sources.ts` — source generators (ring/blob points, Gray–Scott seed (scalar &
  complex), noise and wave grids).
- `extraOps.ts` — registers the opt-in op packs (element/complex, wavelet, calculus)
  into the app's module instance.
- `examples.ts` — the prebuilt example graphs loadable from the sidebar.
- `specs.ts` / `opMeta.ts` — unified op/source metadata + categories/help driving the
  grouped palette and inspector.
- `Preview.tsx` — grid/matrix heatmaps (element-aware, with a re/im/lane selector),
  persistence diagrams, scalars.
- `FieldTooltip.tsx` / `PortHover.tsx` — the port/edge hover inspector (resolved type +
  a mini data view).
- `backend.browser.ts` — the `GpuBackend` for `navigator.gpu`.

See `docs-site/.../concepts/operation-graphs.md` and `docs/gpu-resource-sync.md` for the
runtime model.
