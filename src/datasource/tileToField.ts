// B1a — the raster half of the SpatialData → op-graph bridge (docs/stream-b-bridge-plan.md).
//
// Turns a decoded datasource `Tile` (a plane, `dims[2] === 1`) into a graph `FieldValue` of
// `{kind:"grid"}`, so the existing grid ops (`getisOrd`, `convolveSeparable`, …) run on real
// data. Pure and **dependency-free** (no sd.js, no graph builder): it only reshapes the tile's
// already-decoded `Float32Array` and stamps the coordinate facets it is *handed*.
//
// Ownership boundary (ADR-0015): this converter **consumes resolved values only**. It never reads
// a store, parses NGFF, or composes a transform — the sd.js/playground side resolves the
// `placement` / channel-`axes` / `role` and passes them in `opts`. Absent facet ⇒ today's
// behaviour (array space, no axes, intensity), so the converter is additive.

import type { Graph } from "../gpu/graph/graph";
import type { FieldRole, FieldValue, GpuField, ResolvedPlacement, TensorAxis } from "../gpu/graph/handle";
import { elementLanes } from "../gpu/graph/handle";
import { hostSamples, type Tile } from "./types";

/** Facets stamped onto the produced grid (all optional — absent ⇒ unchanged / today's behaviour).
 *  Every value here is **already resolved** by the caller (ADR-0015); the converter never derives them. */
export interface TileToFieldOpts {
  /** Where the grid's samples sit in world space (ADR-0018). Absent ⇒ array space (unitless, cell-indexed). */
  placement?: ResolvedPlacement;
  /** Open tensor axes — channel/time (ADR-0004/0015). Absent ⇒ none. */
  axes?: readonly TensorAxis[];
  /** Field polarity — intensity vs label (ADR-0015). Absent ⇒ intensity. */
  role?: FieldRole;
  /** Extract a single interleaved lane from a multi-lane tile (element `vec`/`complex`/`quaternion`),
   *  yielding a **scalar** grid — e.g. picking one morphology channel out of an interleaved RAW
   *  tile so a scalar-grid op like `getisOrd` can run. Still pure (de-interleave, no transform).
   *  Absent ⇒ carry the tile's element and data unchanged. */
  channel?: number;
}

/** `Tile → FieldValue(grid)`. Requires a plane (`dims[2] === 1`); a true volume is out of scope for
 *  the grid-shaped ops. `data`/`element`/`dtype` are carried straight from the tile (or, with
 *  `opts.channel`, one de-interleaved lane as a scalar grid). */
export function tileToField(tile: Tile, opts: TileToFieldOpts = {}): FieldValue {
  const [width, height, depth] = tile.dims;
  if (depth !== 1) {
    throw new Error(`tileToField: expected a plane (dims[2] === 1) for a grid, got depth ${depth}`);
  }
  const cells = width * height;
  const lanes = elementLanes(tile.element);
  const expected = cells * lanes;
  // A device-resident tile could in principle pass its `buffer` straight through as the
  // FieldValue's resident payload (the layouts are identical by construction) — but only once a
  // resident `source` op exists to receive it, so for now this stays the host path and says so.
  const data = hostSamples(tile);
  if (data.length !== expected) {
    throw new Error(`tileToField: data length ${data.length} != width*height*lanes (${width}*${height}*${lanes} = ${expected})`);
  }

  const shape = { kind: "grid", width, height } as const;
  const base = { placement: opts.placement, axes: opts.axes, role: opts.role };

  if (opts.channel !== undefined) {
    if (opts.channel < 0 || opts.channel >= lanes) {
      throw new Error(`tileToField: channel ${opts.channel} out of range [0, ${lanes})`);
    }
    const out = new Float32Array(cells);
    for (let i = 0; i < cells; i++) out[i] = data[i * lanes + opts.channel]!;
    // A single de-interleaved lane is a scalar grid; the multi-lane `element` no longer applies.
    return { shape, dtype: tile.dtype, element: { kind: "scalar" }, data: out, ...base };
  }

  return { shape, dtype: tile.dtype, element: tile.element, data, ...base };
}

/** Convenience: convert a `Tile` (B1a) and add it as a graph `source` in one call, returning the
 *  lazy handle. Lives on the datasource side (datasource → graph is the allowed layer direction;
 *  the graph core must not depend on datasource), so `Graph.source` stays generic. */
export function sourceFromTile(graph: Graph, tile: Tile, opts?: TileToFieldOpts): GpuField {
  return graph.source(tileToField(tile, opts), "tile");
}
