// Transcript channels: per-gene (or per-gene-set) point clouds for the Gram form, built from
// resolved points tiles (docs/cell-stats.md §4, §12).
//
// A channel is a WEIGHT VECTOR over feature codes: one gene is the one-hot case, a gene set is just
// more entries. Weights are non-negative (the Gram normalisation divides by mass), so a signed
// contrast is not a channel. A transcript contributes `weights[code]` to the channel listing its code.
//
// Channels must be DISJOINT — no code in two channels. `gram.ts` and `permute.ts` decide whether
// channels share points by array identity, so two overlapping sets built as separate arrays would
// get a zero off-diagonal `selfTerm` and be permuted as if disjoint: a plausible, wrong answer.
// Overlap is therefore refused here rather than computed.

import type { PointsTile, Rect } from "../datasource/points";
import type { ChannelCloud } from "./gram";

export interface WeightedChannel {
  readonly label: string;
  /** Feature code → weight. Entries ≤ 0 are ignored (a weight must be non-negative for `gram.ts`). */
  readonly weights: ReadonlyMap<number, number>;
}

export interface TranscriptChannelOptions {
  /** Analysis window. Points farther than `apron` outside it cannot reach it and are dropped. */
  readonly window: Rect;
  /** Splat support radius — the reach of a point outside the window. */
  readonly apron: number;
  /** Keep only points whose passthrough `column` is ≥ `value`, e.g. `{ column: "qv", value: 20 }`. */
  readonly minimum?: { readonly column: string; readonly value: number };
}

export interface TranscriptChannels {
  readonly clouds: ChannelCloud[];
  /** `window` as the `[minX, minY, maxX, maxY]` tuple `GramParams.bbox` takes. */
  readonly bbox: readonly [number, number, number, number];
  readonly stats: {
    /** Points in the tiles. */
    readonly considered: number;
    /** Dropped for lying beyond `window ⊕ apron`. */
    readonly outOfReach: number;
    /** Dropped by `minimum`. */
    readonly belowMinimum: number;
    /** In reach and passing `minimum`, but in no channel. */
    readonly unselected: number;
    readonly perChannel: readonly number[];
  };
}

/** Build one `ChannelCloud` per channel from resolved tiles. */
export function transcriptChannels(
  tiles: Iterable<PointsTile>,
  channels: readonly WeightedChannel[],
  opts: TranscriptChannelOptions,
): TranscriptChannels {
  const channelOf = new Map<number, number>();
  const weightOf = new Map<number, number>();
  channels.forEach((ch, c) => {
    for (const [code, w] of ch.weights) {
      if (!(w > 0)) continue;
      const prior = channelOf.get(code);
      if (prior !== undefined) {
        throw new Error(
          `transcriptChannels: feature code ${code} is in both "${channels[prior]?.label}" and "${ch.label}"; channels must be disjoint`,
        );
      }
      channelOf.set(code, c);
      weightOf.set(code, w);
    }
  });

  const { window: w, apron, minimum } = opts;
  const loX = w.minX - apron;
  const hiX = w.maxX + apron;
  const loY = w.minY - apron;
  const hiY = w.maxY + apron;
  const tileList = [...tiles];
  let considered = 0;
  let outOfReach = 0;
  let belowMinimum = 0;
  let unselected = 0;

  // Two passes: count, then fill preallocated arrays. `member[t][i]` is the channel or -1.
  const counts = new Array<number>(channels.length).fill(0);
  const membership = tileList.map((tile) => {
    const floor = minimum ? tile.columns[minimum.column] : undefined;
    if (minimum && !floor) throw new Error(`transcriptChannels: tile has no "${minimum.column}" column to filter on`);
    const member = new Int32Array(tile.count);
    considered += tile.count;
    for (let i = 0; i < tile.count; i++) {
      member[i] = -1;
      const x = tile.xs[i] ?? Number.NaN;
      const y = tile.ys[i] ?? Number.NaN;
      if (!(x >= loX && x <= hiX && y >= loY && y <= hiY)) {
        outOfReach++;
        continue;
      }
      if (floor && minimum && !((floor[i] ?? Number.NaN) >= minimum.value)) {
        belowMinimum++;
        continue;
      }
      const c = channelOf.get(tile.codes[i] ?? -1);
      if (c === undefined) {
        unselected++;
        continue;
      }
      member[i] = c;
      counts[c] = (counts[c] ?? 0) + 1;
    }
    return member;
  });

  const xs = counts.map((n) => new Float32Array(n));
  const ys = counts.map((n) => new Float32Array(n));
  const ws = counts.map((n) => new Float32Array(n));
  const unit = channels.map(() => true);
  const cursor = new Array<number>(channels.length).fill(0);
  tileList.forEach((tile, t) => {
    const member = membership[t];
    if (!member) return;
    for (let i = 0; i < tile.count; i++) {
      const c = member[i] ?? -1;
      if (c < 0) continue;
      const k = cursor[c] ?? 0;
      cursor[c] = k + 1;
      const weight = weightOf.get(tile.codes[i] ?? -1) ?? 0;
      (xs[c] as Float32Array)[k] = tile.xs[i] ?? 0;
      (ys[c] as Float32Array)[k] = tile.ys[i] ?? 0;
      (ws[c] as Float32Array)[k] = weight;
      if (weight !== 1) unit[c] = false;
    }
  });

  const clouds: ChannelCloud[] = channels.map((ch, c) => ({
    label: ch.label,
    xs: xs[c] ?? new Float32Array(0),
    ys: ys[c] ?? new Float32Array(0),
    ...(unit[c] ? {} : { weights: ws[c] }),
  }));
  return {
    clouds,
    bbox: [w.minX, w.minY, w.maxX, w.maxY],
    stats: { considered, outOfReach, belowMinimum, unselected, perChannel: counts },
  };
}
