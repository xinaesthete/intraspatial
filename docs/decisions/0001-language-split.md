# ADR-0001 — Language split: Rust/wasm CPU core + TypeScript/TypeGPU GPU layer

Status: **accepted** (2026-06-27)

## Decision

- **Rust → wasm** (`wasm-bindgen`, `wasm-pack`) for the CPU core: codestream parser and the
  bit-serial **HT block decoder**. Publishable as a standalone crate (`htj2k-core`).
- **TypeScript + TypeGPU** for the GPU layer: inverse DWT, dequantization, DC level-shift — and
  the surface for a future user-extension-hook mechanism.

## Context & provenance

- Kakadu ICIP2019 and OpenJPH establish that the **inverse DWT is the GPU win** (separable /
  embarrassingly parallel) while the **HT block decoder is bit-serial** — a poor GPU fit and the
  genuinely-hard, reusable IP. Rust is the right home for it (and a crate other languages can
  bind to).
- The user wants (a) a future **extension-hook** mechanism — most ergonomic in TS/TypeGPU, and
  living at the coefficient→pixel/GPU stage — and (b) potential **cross-language reuse** (a Rust
  crate). The CPU-Rust / GPU-TypeGPU split satisfies both at once; all-Rust+wgpu would weaken
  hook ergonomics, all-TS would forgo the crate and put bit-twiddling in the weakest language.
- Matches the user's existing `psychogeo` pattern (Rust/wasm for CPU crunching via
  `shp_processor_wasm`; TypeGPU for GPU).

## Scope clamp (v1)

Single component, reversible **5/3**, decode only, no MCT, single tile / default progression —
driven by actual zarrextra usage. Multi-component (volumetric, z-as-components) is a deliberate
later differentiator, validated against `openjph-wasm` (our from-source OpenJPH build, which —
unlike the stale cornerstone WASM — round-trips independent components correctly).
