---
"intraspatial": minor
---

Op parameters can now say what they are measured in, and how their range should be traversed. Specs that don't use the new fields behave exactly as before.

- `ParamSpec` gains three optional fields:
  - `warp?: "lin" | "exp" | "amp" | "db" | number` sets how a position in [0, 1] (a slider, a controller, a bred trait) maps onto [min, max]. With `"exp"`, each step multiplies the value by the same factor. That suits scale-like params such as bandwidths, radii, times and frequencies, where 0.1–40 should not spend most of its travel above 20. A number gives a curve. The names follow SuperCollider's, so specs written there carry over.
  - `delta?: number` is how far one mutation typically moves the param, in its own units.
  - `units?: string` is what the value is measured in. A plain name such as `"Hz"` or `"s"` is a label. A relative unit such as `"cells"` or `"world"` means something only in context and can be converted by a resolver (see below). Absent means undeclared.
- New in `intraspatial/graph`:
  - `warpMap` and `warpUnmap` convert between positions and values for any warp.
  - `UnitResolver`, `resolveUnit` and `convertUnit` provide a pluggable way to resolve relative units. Two units convert into each other when they resolve to the same base unit, so other domains can supply their own resolvers.
  - `spatialUnits` resolves `"cells"` and `"world"` against a field's placement and the dataset's µm-per-unit scale, so a UI can say "2 cells = 0.8 µm" (ADR-0018 decision 5, now amended).
  - The spatial params of `convolveSeparable`, `morphology`, `getisOrd`, the structure-tensor op, `splatDensity`, `maskedDensity` and `fuzzyAdjacency` now declare their units. Ops don't convert units yet.
- In `intraspatial/evo`, breeding follows each param's warp and delta:
  - A param with `warp: "exp"` over 50–3000 reads 387 (the geometric mean) at the midpoint of its trait, and mutation moves it by even ratios.
  - `mutate` scales each param's step by its `delta`. A param without one mutates as before.
  - An exp warp over a range that touches or crosses zero, or a curve too steep to evaluate, falls back to linear with a console warning. A `delta` on an enum is ignored with a warning, since its options have no distance between them.
