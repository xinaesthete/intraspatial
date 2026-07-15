# Context Map

This repo now hosts more than one bounded context, each with its own ubiquitous language.

## Contexts

- [View-driven data loading & rendering](./CONTEXT.md) — the datasource / demand-pull /
  multiscale-rendering design (Resource → Multiscale → Select → Resolve → Tileset → render).
- [Procedural geometry-ops](./src/geometry/CONTEXT.md) — the composable geometry-ops catalogue
  (horn-grammar-first swept geometry, extensible to CSG/implicit), recorded in ADR-0010.
- [HsPf spatial population-genetics example](./src/gpu/sim/hspf/CONTEXT.md) — a stand-alone
  selection–migration GPU sim over a raster map of Africa (genotype-frequency field + LD,
  sampled mosquito-bite neighbourhood, GeoTIFF scaffold), recorded in ADR-0011.

## Relationships

- **Geometry-ops → op-graph.** A geometry expression lowers to Level-1 nodes in `src/gpu/graph`
  and is pulled through the same executor/memo/backend seam as every other op (ADR-0003, ADR-0010).
- **Geometry-ops ↔ expression-IR (ADR-0007).** Geometry reuses the typed expression-IR substrate
  for its op parameters (`ParamSpec`-carrying literals, `{s,θ}` free-variable expressions). It is
  the first concrete consumer of that IR, and lowers to the same **CPU-golden + TGSL** backends.
- **Geometry-ops ↔ evo/Mutator.** Op-parameter `ParamSpec`s are the breeding surface; a built
  geometry expression is a first-class value the Mutator can breed (`src/evo`).
- **Geometry-ops → rendering (ADR-0009).** A geometry's resident GPU buffer feeds the
  rendering-as-ops path; three.js/TSL is presentation only.
- **HsPf → library primitives, not the op-graph.** HsPf reuses `getDevice`, the field/element
  types (ADR-0004), seeded RNG and the `ParamSpec`/Mutator surface (`src/evo`), and tgpu-managed
  buffers + `.read()` — but is a *stand-alone page*, not a composer op. Its kernel is raw WGSL (a
  recorded deviation from ADR-0003). It is the library-reuse counterpart to the framework-native
  `reactionDiffusion` sim (ADR-0011).
- **HsPf ↔ evo/Mutator.** HsPf exposes a `ParamSpec` seam (dotted-path names + tags) and is
  `Params`-driven; the general Mutator UI and MIDI consume that seam *by availability* rather than
  being built into the artefact. HsPf is the first consumer of the *filter-then-apply* param model.
- **HsPf → rendering (ADR-0009, nuanced).** Rendering is a direct GPU-resident pass (palette +
  `fwidth` iso-line contours + nodata coastline), not yet a render op; it refactors
  `playground/src/Preview.tsx` onto shared field-viz infrastructure.
- **HsPf ⇢ datasource (ADR-0008).** GeoTIFF is decoded in-page via `geotiff.js` with **no** reusable
  Loader; a general geospatial Loader belongs to the datasource context, not this artefact.
