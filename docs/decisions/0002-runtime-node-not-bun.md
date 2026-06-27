# ADR-0002 — Runtime: Node toolchain, not Bun

Status: **accepted** (2026-06-27)

## Decision

This project uses **Node + pnpm + vitest** with the `webgpu` (Dawn) binding for headless
WebGPU, matching the sibling `psychogeo` project — **not** Bun, despite the repo's original
`bun init` scaffold and `CLAUDE.md` Bun-first default.

## Context & provenance

The plan started Bun-first (per `CLAUDE.md`) with a fallback if WebGPU couldn't be hosted. We
probed it directly (`scratch/webgpu-probe.ts`):

- Bun 1.3.0 has **no native WebGPU** (`navigator.gpu` is undefined).
- The `webgpu` npm package (Dawn N-API addon, v0.4.0) loads under Bun and returns a `GPUDevice`,
  but Bun **segfaults** on the compute-dispatch / `mapAsync` readback path
  (`panic(main thread): Segmentation fault`). This is a Bun↔Dawn N-API bug, not our code.
- The **identical** code runs correctly under **Node 18** (raw compute round-trips:
  `[0..7] << 1 == [0,2,..,14]`).

The user pre-authorised this exact fallback: "if not [ok under Bun] then we should not fall back
to Playwright but rather change to be more similar to psychogeo; Bun is not an absolute
requirement here." psychogeo already runs `typegpu` + the `webgpu` binding + `vitest` on Node.

## Consequences

- Package manager: pnpm. Test runner: vitest (Node). Dev server (later, for `viz/`): vite.
- Native GPU addon + workers: run vitest with a single fork pool to avoid threading issues with
  the Dawn addon.
- `CLAUDE.md` carries a note pointing here so future sessions don't reintroduce Bun for runtime
  code. (Bun may still be fine for unrelated scripts, but the GPU/test path is Node.)
- Revisit if Bun fixes the Dawn N-API crash, or if we later target only the browser.
