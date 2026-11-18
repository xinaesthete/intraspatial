// Niche map: a per-bin label layer — which mix of the current channels each bin of tissue holds —
// fitted once on a whole-element sample and assigned tile by tile as the view loads
// (docs/emeans-clustering.md, "The operating model").
//
// 1. Fit. The sample is the page's own tiles over the element's full extent, which a points-zarr
//    element serves from a coarse level: a uniform random sample (on Atera, 1/256 of the molecules).
//    Each bin's channel counts weight a molecule by 1/its sample fraction; its descriptor is the
//    Hellinger mix and its weight the molecules it stands for (`binComposition`). Weighted e-means
//    runs in a worker.
// 2. Assign. Tiles come from ONE fixed grid level, one finer than the sample's, whatever the zoom —
//    so a bin's molecules are always the same read, and its label never changes as the view moves.
//    Each tile is binned and labelled against the frozen table as it arrives (`assignCentroids`),
//    and drawn as its own little label image.

import { BitmapLayer } from "@deck.gl/layers";
import type { Matrix4 } from "@math.gl/core";
import { useEffect, useMemo, useReducer, useState } from "react";
import { type PointsGrid, pointsGridAtLevel, type Rect, selectPointsTiles } from "../../../src/datasource/points";
import { chunkKey } from "../../../src/datasource/tileCache";
import type { ChunkId } from "../../../src/datasource/types";
import { binComposition, type ChannelBins, channelBins, pooledBins } from "../../../src/spatial/channelBins";
import { assignCentroids, type CentroidTable } from "../../../src/spatial/emeans";
import type { WeightedChannel } from "../../../src/spatial/transcriptChannels";
import type { PointsSource } from "../datasource/pointsTileLoader";
import { useAsync } from "../hooks/useAsync";
import { usePointsTiles } from "../hooks/useSpatialData";
import type { NicheFitRequest, NicheFitResponse, NicheFitSummary } from "./nicheWorker";

/** Categorical niche colours (Okabe–Ito, then Tableau's), dark-background safe. Past 16 they repeat. */
const NICHE_COLOURS: readonly (readonly [number, number, number])[] = [
  [230, 159, 0],
  [86, 180, 233],
  [0, 158, 115],
  [240, 228, 66],
  [0, 114, 178],
  [213, 94, 0],
  [204, 121, 167],
  [255, 255, 255],
  [255, 157, 167],
  [176, 122, 161],
  [156, 117, 95],
  [118, 183, 178],
  [237, 201, 72],
  [89, 161, 79],
  [186, 176, 172],
  [225, 87, 89],
];
export const nicheColour = (j: number) => NICHE_COLOURS[j % NICHE_COLOURS.length]!;

/** Tile reads at once while labelling a view. */
const CONCURRENCY = 6;
const RESTARTS = 4;

export interface NicheOptions {
  readonly enabled: boolean;
  readonly k: number;
  /** Target bin side; the bin is the nearest quadtree level over the extent, no larger than a tile. */
  readonly binSide: number;
  /** Fewest channel molecules a bin must have READ to be fitted or labelled. */
  readonly minSampled: number;
}

/** The grids a niche map works on: the sample's tiles, the fixed level views are labelled at, and the bins. */
interface NichePlan {
  readonly sample: PointsGrid;
  readonly tiles: PointsGrid;
  readonly bins: PointsGrid;
}

function nichePlan(source: PointsSource, binSide: number): NichePlan {
  const { bounds } = source.grid;
  const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY);
  const sampleLevel = (source.gridFor?.(bounds, 0) ?? source.grid).level;
  // One level finer than the sample, so a tile reads deeper than the fit did, but a whole-slide view
  // still costs a few hundred tiles. An element with one grid has no choice.
  const tileLevel = source.gridFor ? Math.min(source.grid.level, sampleLevel + 1) : source.grid.level;
  const binLevel = Math.max(tileLevel, Math.round(Math.log2(span / binSide)));
  return {
    sample: pointsGridAtLevel(bounds, source.totalRows, sampleLevel),
    tiles: pointsGridAtLevel(bounds, source.totalRows, tileLevel),
    bins: pointsGridAtLevel(bounds, source.totalRows, binLevel),
  };
}

/** The tiles of `grid` around `id` (up to 8), whose bins a pooled neighbourhood can reach. */
function neighbourIds(id: ChunkId, { cols, rows }: PointsGrid): ChunkId[] {
  const out: ChunkId[] = [];
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++) {
      const x = id.x + dx;
      const y = id.y + dy;
      if ((dx || dy) && x >= 0 && y >= 0 && x < cols && y < rows) out.push({ ...id, x, y });
    }
  return out;
}

export interface NicheFit extends CentroidTable {
  readonly summary: NicheFitSummary;
  /** What the table was fitted for: labelling anything else would mismatch its dimensions. */
  readonly channels: readonly WeightedChannel[];
  readonly plan: NichePlan;
  readonly minSampled: number;
  /** Bins and molecules read in the sample, and the bins kept for the fit. */
  readonly sample: {
    readonly bins: number;
    readonly kept: number;
    readonly molecules: number;
    /** Smallest share of the element (or of a gene, when genes were read separately) a molecule was read from. */
    readonly fraction: number;
    readonly perGene: boolean;
  };
}

function runWorker(req: NicheFitRequest, signal: AbortSignal): Promise<NicheFitSummary> {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL("./nicheWorker.ts", import.meta.url), { type: "module" });
    const done = () => w.terminate();
    signal.addEventListener("abort", () => {
      done();
      reject(new Error("superseded"));
    });
    w.onmessage = (e: MessageEvent<NicheFitResponse>) => {
      done();
      if (e.data.ok) resolve(e.data.fit);
      else reject(new Error(`niche fit: ${e.data.error}`));
    };
    w.onerror = (e) => {
      done();
      reject(new Error(e.message ? `niche worker: ${e.message}` : "the niche worker could not load its script — reload the page"));
    };
    w.postMessage(req, [req.data.buffer, req.weights.buffer]);
  });
}

/** Fit the niche table to the whole element. Reruns only when channels, k, bins or the element change. */
function useNicheFit(source: PointsSource | undefined, channels: readonly WeightedChannel[], opts: NicheOptions) {
  const on = opts.enabled && channels.length >= 2;
  const plan = useMemo(() => (source ? nichePlan(source, opts.binSide) : undefined), [source, opts.binSide]);
  const sample = usePointsTiles(source, on ? source?.grid.bounds : undefined, 0);
  const { k, minSampled } = opts;
  const fit = useAsync<NicheFit>(
    (signal) => {
      const tiles = sample.value?.tiles;
      if (!on || !tiles || !plan) return undefined;
      const rows: number[] = [];
      const weights: number[] = [];
      let bins = 0;
      let molecules = 0;
      // Every sample tile is here, so each is pooled with the ones around it at once.
      const binned = new Map(tiles.map((t) => [chunkKey(t.id), channelBins(t, plan.bins, channels)]));
      for (const t of tiles) {
        const own = binned.get(chunkKey(t.id))!;
        const around = neighbourIds(t.id, plan.sample).flatMap((id) => binned.get(chunkKey(id)) ?? []);
        const desc = binComposition(pooledBins(own, around, plan.bins), minSampled);
        bins += desc.n;
        for (let i = 0; i < desc.n; i++) {
          const w = desc.weights[i]!;
          if (w === 0) continue;
          weights.push(w);
          for (let c = 0; c < desc.d; c++) rows.push(desc.data[i * desc.d + c]!);
        }
        molecules += t.count;
      }
      const d = channels.length;
      const n = weights.length;
      if (n < k)
        return Promise.reject(
          new Error(`only ${n} bins of the sample hold ${minSampled}+ channel molecules; use bigger bins or fewer niches`),
        );
      const perGene = tiles.some((t) => t.featureFractions);
      const fraction = Math.min(1, ...tiles.flatMap((t) => [t.sampleFraction ?? 1, ...(t.featureFractions?.values() ?? [])]));
      const req = { data: Float64Array.from(rows), weights: Float64Array.from(weights), n, d, k, restarts: RESTARTS };
      return runWorker(req, signal).then((summary) => ({
        k,
        d,
        centroids: summary.centroids,
        summary,
        channels,
        plan,
        minSampled,
        sample: { bins, kept: n, molecules, fraction, perGene },
      }));
    },
    [on, sample.value, plan, channels, k, minSampled],
    source,
  );
  // A table for other channels or bins is no stand-in: its labels would mean something else.
  const value = fit.value && fit.value.channels === channels && fit.value.plan === plan ? fit.value : undefined;
  return { ...fit, value, plan: on ? plan : undefined, sampleProgress: sample.progress, sampleError: sample.error };
}

/** One tile, binned and labelled. */
interface NicheTile {
  readonly key: string;
  readonly bins: ChannelBins;
  readonly labels: Int32Array;
  readonly image: ImageData;
}

/**
 * The fixed-level tiles covering `window`, each binned once as it arrives (kept per channels and
 * plan, so a pan back costs nothing) and labelled against `fit` once the tiles around it are binned
 * too — a bin's pooled neighbourhood reaches into them, so its label is final when it first shows.
 */
function useNicheTiles(source: PointsSource | undefined, fit: NicheFit | undefined, window: Rect | undefined) {
  const plan = fit?.plan;
  const channels = fit?.channels;
  // Binned tiles depend on the source, the bin grid and the channels, not on the fit: k can change
  // without a single tile being read again.
  const binned = useMemo(() => new Map<string, ChannelBins>(), [source, plan?.bins, channels]);
  const [version, bump] = useReducer((n: number) => n + 1, 0);
  const [progress, setProgress] = useState<{ done: number; total: number }>();
  const [error, setError] = useState<Error>();
  const ids = useMemo<ChunkId[]>(
    () => (plan && window ? selectPointsTiles(plan.tiles, window, 0).chunks.map((c) => c.id) : []),
    [plan, window],
  );
  // What to read: the tiles drawn, nearest the middle first, then the ring around them.
  const reads = useMemo<ChunkId[]>(() => {
    if (!plan) return [];
    const seen = new Set(ids.map(chunkKey));
    const ring = ids.flatMap((id) => neighbourIds(id, plan.tiles)).filter((id) => !seen.has(chunkKey(id)) && seen.add(chunkKey(id)));
    return [...ids, ...ring];
  }, [plan, ids]);

  useEffect(() => {
    if (!source || !plan || !channels) return;
    const todo = reads.filter((id) => !binned.has(chunkKey(id)));
    setError(undefined);
    if (!todo.length) {
      setProgress(undefined);
      return;
    }
    let live = true;
    let done = 0;
    setProgress({ done, total: todo.length });
    const pull = async () => {
      for (let id = todo.shift(); id && live; id = todo.shift()) {
        const tile = await source.loader.getChunk(id);
        // Kept even if the view moved on: it is the same tile whichever view asked for it.
        binned.set(chunkKey(id), channelBins(tile, plan.bins, channels));
        if (!live) return;
        setProgress({ done: ++done, total: done + todo.length });
        bump();
      }
    };
    Promise.all(Array.from({ length: CONCURRENCY }, pull)).then(
      () => live && setProgress(undefined),
      (err: unknown) => {
        if (!live) return;
        setProgress(undefined);
        setError(err instanceof Error ? err : new Error(String(err)));
      },
    );
    return () => {
      live = false;
    };
  }, [source, plan, channels, reads, binned]);

  // Labels and images per fit: a refit relabels the binned tiles without reading any.
  const labelled = useMemo(() => new Map<string, NicheTile>(), [fit]);
  const tiles = useMemo(() => {
    const out: NicheTile[] = [];
    if (!fit) return out;
    for (const id of ids) {
      const key = chunkKey(id);
      let t = labelled.get(key);
      const own = binned.get(key);
      if (!t && own) {
        const ring = neighbourIds(id, fit.plan.tiles);
        const around = ring.flatMap((n) => binned.get(chunkKey(n)) ?? []);
        if (around.length === ring.length) labelled.set(key, (t = labelTile(key, pooledBins(own, around, fit.plan.bins), fit)));
      }
      if (t) out.push(t);
    }
    return out;
    // `version` stands for `binned`'s contents, which change in place as tiles arrive.
  }, [fit, ids, binned, labelled, version]);
  return { tiles, progress, error, pending: ids.length - tiles.length };
}

function labelTile(key: string, bins: ChannelBins, fit: NicheFit): NicheTile {
  const desc = binComposition(bins, fit.minSampled);
  const { labels } = assignCentroids(fit, desc.data, desc.n, { weights: desc.weights });
  const { x0, y0, width, height } = bins;
  const cols = fit.plan.bins.cols;
  const rgba = new Uint8ClampedArray(width * height * 4);
  bins.bins.forEach((b, i) => {
    const l = labels[i]!;
    if (l < 0) return;
    const p = ((Math.floor(b / cols) - y0) * width + (b % cols) - x0) * 4;
    const [r, g, bl] = nicheColour(l);
    rgba.set([r, g, bl, 255], p);
  });
  return { key, bins, labels, image: new ImageData(rgba, width, height) };
}

export function useNicheMap(
  source: PointsSource | undefined,
  channels: readonly WeightedChannel[],
  window: Rect | undefined,
  opts: NicheOptions,
) {
  const fit = useNicheFit(source, channels, opts);
  const tiles = useNicheTiles(opts.enabled ? source : undefined, fit.value, window);
  return { fit, tiles };
}

export type NicheMap = ReturnType<typeof useNicheMap>;

/** One bitmap per labelled tile, in the element's frame. Bins are drawn whole, so the image spans
 *  the tile's block of the bin grid even where the extent clips it. */
export function nicheLayers(fit: NicheFit, tiles: readonly NicheTile[], modelMatrix: Matrix4, opacity: number): BitmapLayer[] {
  const { bounds, tileSize: s } = fit.plan.bins;
  return tiles.map(
    (t) =>
      new BitmapLayer({
        id: `niche-${t.key}`,
        image: t.image,
        // [left, bottom, right, top]: the image's first row is bin row y0, at the lower y.
        bounds: [
          bounds.minX + t.bins.x0 * s,
          bounds.minY + (t.bins.y0 + t.bins.height) * s,
          bounds.minX + (t.bins.x0 + t.bins.width) * s,
          bounds.minY + t.bins.y0 * s,
        ],
        textureParameters: { minFilter: "nearest", magFilter: "nearest", mipmapFilter: "nearest", lodMaxClamp: 0 },
        opacity,
        modelMatrix,
        pickable: false,
      }),
  );
}
