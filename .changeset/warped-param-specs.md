---
"intraspatial": minor
---

Parameters whose natural scale isn't linear now mutate and map evenly. Frequencies and times can be declared in their own units (e.g. `freq` in Hz) instead of as MIDI-valued workarounds. Specs without the new fields behave exactly as before.

- `ParamSpec` gains two optional fields:
  - `warp?: "lin" | "exp" | "amp" | "db" | number`: how a normalised position in [0, 1] maps onto [min, max], using SuperCollider's ControlSpec warp names. A number is a curve.
  - `delta?: number`: how far one mutation typically moves the param, in its own units.
- `warpMap`, `warpUnmap`, `warpValid`, `warpIsLinear` and the `Warp` / `WarpSpec` types are exported from `intraspatial/graph`. They are a port of sclang's Warp classes and are tested against values sclang 3.13 computes.
- In `intraspatial/evo`, a trait's coord goes through its param's warp. `{ name: "freq", type: "number", default: 500, min: 50, max: 3000, warp: "exp" }` reads 387 Hz, the geometric mean, at coord 0.5. Mutation, marriage and steering therefore move evenly in pitch.
  - An exp warp over a range touching or crossing zero falls back to linear with a console warning.
  - `Trait` gains optional `warp` and `deltaNorm`.
- `mutate` scales each trait's Gaussian step by its `deltaNorm`, the param's `delta` in coord units (σ = `rate × deltaNorm`).
  - For a linear param, `deltaNorm` is `delta / (max - min)`.
  - For a warped param, it is the coord distance from the default to `default + delta`. That keeps an exp param's step the same ratio across its range.
  - A param with no `delta` mutates with σ = `rate`, as before.
