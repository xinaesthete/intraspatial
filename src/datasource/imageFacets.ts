// B1b (pure half) — pack *already-resolved* image metadata into op-graph facets
// (docs/stream-b-bridge-plan.md). Dependency-free: no sd.js, no THREE, no store access.
//
// Ownership boundary (ADR-0015): these functions **consume resolved values only** and never
// compose a transform or parse NGFF. The sd.js side (playground/spatialDataLoader) has already
// collapsed the element∘dataset transform into one `Affine3` and resolved omero channel metadata;
// here we merely wrap those into the graph's `ResolvedPlacement` / `TensorAxis` shapes. The thin
// sd.js-facing adapter that pulls these off a `SpatialDataImage` lives in
// `playground/src/datasource/imageToGraph.ts`, which delegates here.

import type { Affine3 } from "../coords";
import type { ChannelEntry, ResolvedPlacement, TensorAxis } from "../gpu/graph/handle";

/** A channel's already-resolved render metadata — the fields the sd.js loader seeds from
 *  `omero.channels` (structurally the playground's `ChannelSettings`). `contrastLimits` are in the
 *  field's normalised value space (the loader normalises decoded samples into `[0,1]`). */
export interface ResolvedChannel {
  readonly label: string;
  /** Display colour, rgb each in `[0,1]`. */
  readonly color: readonly [number, number, number];
  /** Contrast window `[start, end]` in the field's normalised value space. */
  readonly contrastLimits: readonly [number, number];
  readonly visible?: boolean;
}

/** rgb floats in `[0,1]` → an uppercase `"RRGGBB"` hex string (ADR-0015 `ChannelEntry.color`). */
export function rgbToHex(color: readonly [number, number, number]): string {
  const h = (v: number) =>
    Math.max(0, Math.min(255, Math.round(v * 255)))
      .toString(16)
      .toUpperCase()
      .padStart(2, "0");
  return `${h(color[0])}${h(color[1])}${h(color[2])}`;
}

/** A resolved array→world `Affine3` (sd.js already composed it) → a `ResolvedPlacement`. Pass the
 *  value only when the store actually carries a transform; **do not fabricate identity** — absent
 *  ⇒ leave placement absent (array space), the ADR-0018 distinction. `system` defaults to `"global"`. */
export function placementFromMatrix(worldFromArray: Affine3 | undefined, system = "global"): ResolvedPlacement | undefined {
  if (!worldFromArray) return undefined;
  return { system, worldFromArray };
}

/** Resolved channel metadata → an open `channel` `TensorAxis` with per-index `entries`
 *  (ADR-0015 fork B: channel semantics live on the axis). Empty input ⇒ `undefined` (no axis). */
export function channelAxisFrom(channels: readonly ResolvedChannel[], name = "c"): TensorAxis | undefined {
  if (channels.length === 0) return undefined;
  const entries: ChannelEntry[] = channels.map((c) => ({
    label: c.label,
    color: rgbToHex(c.color),
    window: { min: 0, max: 1, start: c.contrastLimits[0], end: c.contrastLimits[1] },
    active: c.visible ?? true,
  }));
  return { name, type: "channel", length: channels.length, entries };
}
