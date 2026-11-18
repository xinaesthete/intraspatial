// The niche fit off the main thread: weighted e-means on the whole-element sample's bins, best of a
// few k-means++ starts. Cheap next to the gene modules (tens of thousands of bins, a dozen channels),
// but it runs once per change of channels, k or bin size, and a pan must never wait on it.

import { assignCentroids, emeansFit } from "../../../src/spatial/emeans";

export interface NicheFitRequest {
  /** n × d Hellinger rows (`binComposition`), only bins of weight > 0. */
  readonly data: Float64Array;
  /** Molecules each bin stands for. */
  readonly weights: Float64Array;
  readonly n: number;
  readonly d: number;
  readonly k: number;
  readonly restarts: number;
}

export interface NicheFitSummary {
  /** k × d, ordered by mass, largest first. */
  readonly centroids: Float64Array;
  /** Molecules per niche in the sample, under the final labels, same order. */
  readonly mass: Float64Array;
  /** Bins per niche, same order. */
  readonly bins: Int32Array;
  /** Weighted mean squared Hellinger distance to the niche (the objective, per molecule). */
  readonly mse: number;
  /** Mini-batch size and steps per epoch the scaling laws chose. */
  readonly batch: number;
  readonly stepsPerEpoch: number;
  readonly ms: number;
}

export type NicheFitResponse = { readonly ok: true; readonly fit: NicheFitSummary } | { readonly ok: false; readonly error: string };

self.onmessage = (e: MessageEvent<NicheFitRequest>) => {
  const { data, weights, n, d, k, restarts } = e.data;
  const t0 = performance.now();
  try {
    // Niches can be well separated (tissue against background, cortex against medulla), and e-means
    // cannot move a centroid across empty space: k-means++ seeds, best of several starts.
    let best: { fit: ReturnType<typeof emeansFit>; a: ReturnType<typeof assignCentroids> } | undefined;
    for (let r = 0; r < restarts; r++) {
      const fit = emeansFit(data, n, d, k, { weights, seeding: "kmeans++", seed: r });
      const a = assignCentroids(fit, data, n, { weights });
      if (!best || a.mse < best.a.mse) best = { fit, a };
    }
    if (!best) throw new Error("no restarts");
    const order = Array.from({ length: k }, (_, j) => j).sort((p, q) => best.a.mass[q]! - best.a.mass[p]!);
    const centroids = new Float64Array(k * d);
    order.forEach((j, o) => centroids.set(best.fit.centroids.subarray(j * d, j * d + d), o * d));
    const fit: NicheFitSummary = {
      centroids,
      mass: Float64Array.from(order, (j) => best.a.mass[j]!),
      bins: Int32Array.from(order, (j) => best.a.counts[j]!),
      mse: best.a.mse,
      batch: best.fit.schedule.batch,
      stepsPerEpoch: best.fit.schedule.stepsPerEpoch,
      ms: performance.now() - t0,
    };
    postMessage({ ok: true, fit } satisfies NicheFitResponse, { transfer: [centroids.buffer] });
  } catch (err) {
    postMessage({ ok: false, error: err instanceof Error ? err.message : String(err) } satisfies NicheFitResponse);
  }
};
