// Gene modules for a whole points element, as channels the transcript Gram can take.
//
// Fit once on a sample of the whole element, never on the view (docs/emeans-clustering.md): modules
// fitted to what is in view would change as you pan, so the same tissue would get different channels
// in different places. The sample comes through the page's own tiles over the element's full extent:
// a points-zarr element serves those from a coarse level, which is a uniform random sample (on Atera,
// 4.6M of 1.19B molecules). The embedding and clustering run in a worker.

import { useMemo } from "react";
import type { PointsTile, Rect } from "../../../src/datasource/points";
import type { WeightedChannel } from "../../../src/spatial/transcriptChannels";
import type { PointsFeature, PointsSource } from "../datasource/pointsTileLoader";
import { useAsync } from "../hooks/useAsync";
import { usePointsTiles } from "../hooks/useSpatialData";
import { isControl } from "./channelModel";
import type { GeneModulesRequest, GeneModulesResponse, GeneModulesSummary } from "./geneModulesWorker";

export interface GeneModuleChannels {
  readonly channels: WeightedChannel[];
  readonly summary: GeneModulesSummary;
  /** Share of the element the sample was read from. */
  readonly sampleFraction: number;
}

/** "M3 · PCK1 LRP2 GATM": a module's label is its index and its most structured genes. */
export function moduleLabel(i: number, codes: readonly number[], names: ReadonlyMap<number, string>): string {
  return `M${i + 1} · ${codes
    .slice(0, 3)
    .map((c) => names.get(c) ?? `#${c}`)
    .join(" ")}`;
}

function concatTiles(tiles: readonly PointsTile[]) {
  const n = tiles.reduce((s, t) => s + t.count, 0);
  const xs = new Float32Array(n);
  const ys = new Float32Array(n);
  const codes = new Int32Array(n);
  let o = 0;
  for (const t of tiles) {
    xs.set(t.xs.subarray(0, t.count), o);
    ys.set(t.ys.subarray(0, t.count), o);
    codes.set(t.codes.subarray(0, t.count), o);
    o += t.count;
  }
  return { xs, ys, codes };
}

function runWorker(req: GeneModulesRequest, signal: AbortSignal): Promise<GeneModulesSummary> {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL("./geneModulesWorker.ts", import.meta.url), { type: "module" });
    const done = () => w.terminate();
    signal.addEventListener("abort", () => {
      done();
      reject(new Error("superseded"));
    });
    w.onmessage = (e: MessageEvent<GeneModulesResponse>) => {
      done();
      if (e.data.ok) resolve(e.data.summary);
      else reject(new Error(`gene modules: ${e.data.error}`));
    };
    // Failures inside the computation come back as messages, so an error event is the worker itself.
    w.onerror = (e) => {
      done();
      reject(
        new Error(e.message ? `gene modules worker: ${e.message}` : "the gene modules worker could not load its script — reload the page"),
      );
    };
    w.postMessage(req, [req.xs.buffer, req.ys.buffer, req.codes.buffer, req.exclude.buffer]);
  });
}

/** Modules for `source`, while `enabled`. `k` is the number of modules asked for. */
export function useGeneModules(source: PointsSource | undefined, enabled: boolean, k: number) {
  const extent: Rect | undefined = enabled ? source?.grid.bounds : undefined;
  const sample = usePointsTiles(source, extent, 0);
  const features = source?.features;
  const names = useMemo(() => new Map((features ?? []).map((f: PointsFeature) => [f.code, f.name])), [features]);
  const state = useAsync<GeneModuleChannels>(
    (signal) => {
      const tiles = sample.value?.tiles;
      if (!enabled || !tiles || !features || !extent) return undefined;
      const { xs, ys, codes } = concatTiles(tiles);
      const nFeatures = features.reduce((m, f) => Math.max(m, f.code + 1), 0);
      const exclude = Int32Array.from(features.filter((f) => isControl(f.name)).map((f) => f.code));
      const area = (extent.maxX - extent.minX) * (extent.maxY - extent.minY);
      const sampleFraction = tiles.length ? Math.min(...tiles.map((t) => t.sampleFraction ?? 1)) : 1;
      return runWorker({ xs, ys, codes, nFeatures, exclude, area, k }, signal).then((summary) => ({
        summary,
        sampleFraction,
        channels: summary.modules.map((codes, i) => ({ label: moduleLabel(i, codes, names), weights: new Map(codes.map((c) => [c, 1])) })),
      }));
    },
    [enabled, sample.value, features, k, names],
    source,
  );
  return { ...state, sampleProgress: sample.progress, sampleError: sample.error };
}
