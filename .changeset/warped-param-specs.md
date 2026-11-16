---
"intraspatial": minor
---

Parameters can say what they're measured in, and parameters whose natural scale isn't linear now mutate and map evenly. Frequencies and times can be declared in their own units (e.g. `freq` in Hz) instead of as MIDI-valued workarounds. Specs without the new fields behave exactly as before.

- `ParamSpec` gains three optional fields:
  - `warp?: "lin" | "exp" | "amp" | "db" | number`: how a normalised position in [0, 1] maps onto [min, max], using SuperCollider's ControlSpec warp names. A number is a curve.
  - `delta?: number`: how far one mutation typically moves the param, in its own units. Number and int params only: a `delta` on an enum is ignored with a warning.
  - `units?: string`: what the value is measured in. It can be a fixed unit ("Hz", "s", "dB"), which is just a label. It can also be a relative unit, which means something only against a context and converts through a resolver. Absent means undeclared.
- New `UnitResolver` interface in `intraspatial/graph`, with `resolveUnit`, `convertUnit` and `fixedUnit`. A resolver claims some unit names and resolves each against its context into a label plus a map into a base unit. Two units convert into each other exactly when they share a base, so other domains (sample rates, tempi, musical scales) can supply their own resolvers.
- `spatialUnits` resolves "cells" and "world" against a field's placement and the store's µm-per-unit scale (ADR-0018 decision 5, now amended). A world length's base is µm when the scale is known. A cell length reaches the same base through the placement's cell size, so "2 cells = 0.8 µm" can be computed. The spatial params of `convolveSeparable`, `morphology`, `getisOrd`, the structure-tensor op, `splatDensity`, `maskedDensity`, `fuzzyAdjacency` and `gridIndex` now declare their units. No op converts units yet.
- `warpMap`, `warpUnmap`, `warpValid`, `warpIsLinear` and the `Warp` / `WarpSpec` types are exported from `intraspatial/graph`. They are a port of sclang's Warp classes and are tested against values sclang 3.13 computes.
- In `intraspatial/evo`, a trait's coord goes through its param's warp. `{ name: "freq", type: "number", default: 500, min: 50, max: 3000, warp: "exp" }` reads 387 Hz, the geometric mean, at coord 0.5. Mutation, marriage and steering therefore move evenly in pitch.
  - An exp warp over a range touching or crossing zero falls back to linear with a console warning.
  - `Trait` gains optional `warp` and `deltaNorm`.
- `mutate` scales each trait's Gaussian step by its `deltaNorm`, the param's `delta` in coord units (σ = `rate × deltaNorm`).
  - For a linear param, `deltaNorm` is `delta / (max - min)`.
  - For a warped param, it is the coord distance from the default to `default + delta`. That keeps an exp param's step the same ratio across its range.
  - A param with no `delta` mutates with σ = `rate`, as before.
