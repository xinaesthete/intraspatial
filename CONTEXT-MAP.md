# Context Map

This repo now hosts more than one bounded context, each with its own ubiquitous language.

## Contexts

- [View-driven data loading & rendering](./CONTEXT.md) — the datasource / demand-pull /
  multiscale-rendering design (Resource → Multiscale → Select → Resolve → Tileset → render).
- [Procedural geometry-ops](./src/geometry/CONTEXT.md) — the composable geometry-ops catalogue
  (horn-grammar-first swept geometry, extensible to CSG/implicit), recorded in ADR-0010.

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
