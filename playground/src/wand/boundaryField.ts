// The boundary field of a Gram result, pulled through the wand's field subgraph (`wandField` in
// src/gpu/graph/wand.ts): the Gram's raster stack → channels → u = A·z → the Di Zenzo tensor of u.
// On the device; the tensor and u come back to the host, for the overlay and the patch.

import { browserBackend } from "../../../src/gpu/graph/backend.browser";
import { pull } from "../../../src/gpu/graph/executor";
import { Graph } from "../../../src/gpu/graph/graph";
import type { GramRasters } from "../../../src/gpu/graph/ops/channelOps";
import { registerWandOps, wandField } from "../../../src/gpu/graph/wand";
import type { GramMatrixGpuResult } from "../../../src/gpu/spatial/gramMatrix";
import type { BoundaryField } from "./boundaryStrength";

registerWandOps();

const host = (data: ArrayLike<number> | undefined): Float32Array => (data instanceof Float32Array ? data : Float32Array.from(data ?? []));

/**
 * Boundary strength for `res` (its rasters snapshot `rasters`) under metric `A`, in whitened distance
 * per kernel radius. `mean`/`sd` are the standardisation the map is painted with (a locked basis's, or
 * res's own). `q` is how many rows of `A` are main modes; the rest take their gradient from the
 * channels blurred at σ = `smoothR` kernel radii (0 = no blur).
 */
export async function boundaryField(
  res: GramMatrixGpuResult,
  rasters: GPUBuffer,
  mean: ArrayLike<number>,
  sd: ArrayLike<number>,
  A: ArrayLike<number>,
  radius: number,
  q: number,
  smoothR: number,
): Promise<BoundaryField> {
  const K = res.labels.length;
  const { width, height } = res;
  const n = width * height;
  // Raster pixels per kernel radius: the per-pixel strength times this is per radius.
  const pixel = (res.bbox[2] - res.bbox[0]) / width;
  const perR = pixel > 0 ? radius / pixel : 1;
  const payload: GramRasters = {
    kind: "gramRasters",
    buffer: rasters,
    rowWords: res.resident.rowWords,
    half: res.resident.precision === "f16",
  };
  const g = new Graph();
  const src = g.source(
    { shape: { kind: "grid", width, height }, dtype: "f32", axes: [{ name: "c", type: "channel", length: K }], payload },
    "gram rasters",
  );
  const x = g.op("gramChannels", { rasters: src })[0]!;
  const { u, tensor } = wandField(g, x, {
    A,
    q,
    mean: Array.from(mean),
    sd: Array.from(sd),
    perR,
    residualSigma: Math.max(0, smoothR) * perR,
  });
  const ctx = { backend: browserBackend };
  // Two pulls rather than one: the executor returns one sink, and recomputing u (a mix of K planes)
  // costs less than holding its lease across pulls.
  const field = host((await pull(g, tensor, { ctx })).data);
  const uHost = host((await pull(g, u, { ctx })).data);
  const s = new Float32Array(n);
  for (let i = 0; i < n; i++) s[i] = field[i * 4]!;
  s.sort();
  const at = (p: number) => s[Math.min(n - 1, Math.floor(p * n))] ?? 0;
  return { field, u: uHost, K, perR, quantiles: { p50: at(0.5), p90: at(0.9), p99: at(0.99), max: s[n - 1] ?? 0 } };
}
