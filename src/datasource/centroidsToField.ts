// B2a — the points half of the SpatialData → op-graph bridge (docs/stream-b-bridge-plan.md).
//
// Turns ONE cell type's parallel centroid arrays (`xs`/`ys`) into a graph points `FieldValue`
// (`{kind:"points", n}`, packed `[x0,y0,x1,y1,...]` exactly as `graph.points` does), so the
// existing points ops (`splatDensity`, …) run on real cell centroids. Pure and **dependency-free**
// (no zarr, no sd.js, no GPU): it only packs the numbers it is handed and stamps the coordinate +
// provenance facets it is *handed*.
//
// Ownership boundary (ADR-0015/0018): this converter **consumes resolved values only**. It never
// opens a store, reads a table, or composes a transform — the playground/sd.js side reads
// `obsm['spatial']`, groups by `cell_type_id`, and resolves the `placement` (per-`(table,region)`
// cloud), then passes those in `opts`. Absent facet ⇒ today's behaviour (array space, no
// provenance), so the converter is additive.
//
// Per ADR-0018 the coordinates are carried, NOT host-transformed: a placed cloud's world position
// is applied on the GPU (`splatDensity` splats in the points' own system and stamps the output
// grid's placement), so the raw centroids flow through unchanged.

import type { FieldProvenance, FieldValue, ResolvedPlacement } from "../gpu/graph/handle";

/** Facets stamped onto the produced points cloud (all optional — absent ⇒ unchanged / today's
 *  behaviour). Every value here is **already resolved** by the caller (ADR-0015); the converter
 *  never derives them. */
export interface CentroidsToFieldOpts {
  /** Where the cloud's centroids sit in world space (ADR-0018). Absent ⇒ array space (unitless). */
  placement?: ResolvedPlacement;
  /** Where these centroids came from — the `(region, instanceKey, cellTypeId)` this cloud was split
   *  on (ADR-0018 keys a cloud per `(table, region, cell_type)`). Absent ⇒ none recorded. */
  provenance?: FieldProvenance;
}

/** `xs`/`ys` → points `FieldValue`. Packs the parallel arrays into one interleaved `[x0,y0,…]`
 *  `Float32Array` (the layout `graph.points` and `splatDensity` expect) and stamps the optional
 *  placement + provenance. Throws if the two arrays differ in length. An empty cloud (`n === 0`)
 *  is valid and yields a zero-length points value. */
export function centroidsToField(xs: ArrayLike<number>, ys: ArrayLike<number>, opts: CentroidsToFieldOpts = {}): FieldValue {
  const n = xs.length;
  if (ys.length !== n) {
    throw new Error(`centroidsToField: xs and ys length mismatch (${n} vs ${ys.length})`);
  }
  const data = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    data[i * 2] = xs[i]!;
    data[i * 2 + 1] = ys[i]!;
  }
  return {
    shape: { kind: "points", n },
    dtype: "f32",
    data,
    placement: opts.placement,
    provenance: opts.provenance,
  };
}
