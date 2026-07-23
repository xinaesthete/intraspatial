// B1b (sd.js-facing half) — map a loaded SpatialData image onto an op-graph source
// (docs/stream-b-bridge-plan.md). Thin adapter: it pulls the already-resolved placement + channel
// metadata off a `SpatialDataImage` (which `spatialDataLoader` composed on the sd.js side) and hands
// them to the pure `src/datasource` packers. No transform composition happens here — that reasoning
// already lives in `spatialDataLoader.readGlobalFromArray` / the omero seeding (ADR-0015 boundary).

import type { Tile } from "../../../src/datasource";
import { channelAxisFrom, placementFromMatrix, sourceFromTile, type TileToFieldOpts } from "../../../src/datasource";
import type { Graph } from "../../../src/gpu/graph/graph";
import type { GpuField, ResolvedPlacement, TensorAxis } from "../../../src/gpu/graph/handle";
import type { SpatialDataImage } from "./spatialDataLoader";

/** The op-graph coordinate facets for a loaded SpatialData image:
 *  - `placement`: the REAL sd.js level-0 array→`global` matrix, when the store carries one (absent ⇒
 *    left absent = array space; never fabricated to identity, per ADR-0018 / the plan).
 *  - `axes`: a single `channel` `TensorAxis` carrying the omero-seeded per-channel entries. */
export function imageFacets(img: SpatialDataImage): { placement?: ResolvedPlacement; axes?: readonly TensorAxis[] } {
  const placement = placementFromMatrix(img.globalFromArray);
  const axis = channelAxisFrom(img.channels);
  return { placement, axes: axis ? [axis] : undefined };
}

/** Convert one decoded `Tile` of `img` into a graph `source`, stamping the image's placement and
 *  channel axis (B1a `tileToField` ∘ B1b facets). Pass `opts.channel` to de-interleave one
 *  morphology lane into a **scalar** grid — what grid-shaped ops (`getisOrd`, `convolveSeparable`)
 *  consume; in that mode the multi-channel axis no longer describes the grid, so it is dropped while
 *  the placement (unaffected by lane selection) is kept. */
export function imageTileSource(g: Graph, img: SpatialDataImage, tile: Tile, opts?: { channel?: number }): GpuField {
  const { placement, axes } = imageFacets(img);
  const facets: TileToFieldOpts = opts?.channel === undefined ? { placement, axes } : { placement, channel: opts.channel };
  return sourceFromTile(g, tile, facets);
}
