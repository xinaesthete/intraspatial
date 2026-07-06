# AGENTS.md — index for AI agents

A small map. Follow the links for detail; don't duplicate it here.

- **Priorities & conventions:** [`CLAUDE.md`](CLAUDE.md) (incl. runtime: Node + pnpm + vitest)
- **Datasource/rendering vocabulary:** [`CONTEXT.md`](CONTEXT.md)
- **Design decisions:** [`docs/decisions/`](docs/decisions/)

## Two standing priorities

1. **Share one GPUDevice — no readback.** Run compute on the host renderer's device so buffers/
   textures pass without a CPU roundtrip. Seam: [`src/gpu/interop/adoptDevice.ts`](src/gpu/interop/adoptDevice.ts); direction: [ADR-0009](docs/decisions/0009-rendering-as-ops.md).

2. **Prefer our op library over throwaway host shaders.** Reach for (or add, with a CPU golden) a
   TypeGPU op in [`src/gpu/graph/ops/`](src/gpu/graph/ops/) rather than one-off three.js TSL / per-host
   WGSL. Pattern: `ops/threshold.ts`; rationale: [ADR-0003](docs/decisions/0003-use-gpu-tgsl-kernels.md).
