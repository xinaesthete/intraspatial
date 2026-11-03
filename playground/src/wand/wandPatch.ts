// PROTOTYPE: the wand's patch at one seed (docs/wand-metric-and-regions.md): the cell of the boundary
// field around the seed under the chosen rule, optionally focused by a broader scale's cell, then
// opened so branches narrower than the opening drop. The field is computed elsewhere.
//
// The plain rule (Laplacian or tolerance, then the opening) runs as a graph of general ops
// (`wandPatchOnGraph`, src/gpu/graph/wand.ts). The experiments (ridges, two scales, graded
// necks) still run here as one function, `wandPatch`.

registerWandOps();

import { browserBackend } from "../../../src/gpu/graph/backend.browser";
import { pull } from "../../../src/gpu/graph/executor";
import { Graph } from "../../../src/gpu/graph/graph";
import { registerWandOps, wandPatch as patchSubgraph } from "../../../src/gpu/graph/wand";
import { nearestPixel } from "../../../src/spatial/regions";
import type { Raster } from "../selection/worldMask";
import { type BoundaryField, focusCell, laplacianCell, openCell, ridgeCell, ridges } from "./boundaryStrength";
import { gradedCell } from "./gradedNecks";

/** How the patch is bounded: the Laplacian of the wand distance, a plain tolerance on it, or the ridge lines. */
export type PatchRule = "laplacian" | "threshold" | "ridges";

export interface PatchSettings {
  readonly rule: PatchRule;
  /** σ, in kernel radii, of the smoothing on d before its Laplacian. */
  readonly dSmooth: number;
  /** Also stop where d exceeds this; 0 = off. */
  readonly tol: number;
  /** Opening radius, in kernel radii; 0 = off. */
  readonly open: number;
  /** Grade the opening by neck width rather than keep or drop parts whole. */
  readonly graded: boolean;
  /** How far, in r, the edge may sit outside the broad cell. */
  readonly slack: number;
  /** The ridge rule's strength threshold. */
  readonly theta: number;
}

/** The broader scale's field, on its own raster, when the patch is judged at two scales. */
export interface BroadField {
  readonly field: BoundaryField;
  readonly res: Raster;
}

export interface Patch {
  /** 0/1 per map pixel. */
  readonly mask: Float32Array;
  /** With graded necks: membership in [0, 1], and where it is above zero. */
  readonly weight?: Float32Array;
  readonly support?: Float32Array;
  /** The seed it was computed for, in the transcripts' frame. */
  readonly seed: readonly [number, number];
  readonly count: number;
  readonly total: number;
  readonly dEdge?: number;
  readonly gated?: number;
  /** The opening removed the seed's own pixel: nothing to pick. */
  readonly moved: boolean;
  readonly scales?: { readonly broad: number; readonly narrow: number };
}

/** The seed as a fractional map pixel (pixel centres at integers). */
function pixelOf(res: Raster, seed: readonly [number, number]): { col: number; row: number } {
  const { width, height, bbox } = res;
  const [minX, minY, maxX, maxY] = bbox;
  return { col: ((seed[0] - minX) / (maxX - minX)) * width - 0.5, row: ((maxY - seed[1]) / (maxY - minY)) * height - 0.5 };
}

/** The patch at `seed` on `res`'s map, or undefined when the seed is off it. */
export function wandPatch(
  field: BoundaryField,
  res: Raster,
  seed: readonly [number, number],
  s: PatchSettings,
  broad?: BroadField,
): Patch | undefined {
  const { width, height } = res;
  const { col, row } = pixelOf(res, seed);
  if (col < -0.5 || row < -0.5 || col > width - 0.5 || row > height - 0.5) return undefined;
  let inside: Uint8Array;
  let dEdge: number | undefined;
  let gated: number | undefined;
  let moved = false;
  let weight: Float32Array | undefined;
  let support: Float32Array | undefined;
  let scales: { broad: number; narrow: number } | undefined;
  if (s.rule !== "ridges") {
    const lap = s.rule === "laplacian";
    ({ mask: inside, dEdge, gated } = laplacianCell(field, width, height, { col, row }, s.dSmooth, s.tol, lap));
    if (broad) {
      // The same seed on the broad raster, then the broad cell at the same σ (in its own r).
      const { width: bw, height: bh } = broad.res;
      const outer = laplacianCell(broad.field, bw, bh, pixelOf(broad.res, seed), s.dSmooth, s.tol, lap).mask;
      const share = (m: Uint8Array) => m.reduce((a, v) => a + v, 0) / m.length;
      scales = { broad: share(outer), narrow: share(inside) };
      inside = focusCell(inside, width, height, outer, bw, bh, { col, row }, s.slack * field.perR);
    }
    const rho = s.open * field.perR;
    if (s.graded && rho >= 0.5) {
      const start = Math.max(0, Math.min(height - 1, Math.round(row))) * width + Math.max(0, Math.min(width - 1, Math.round(col)));
      ({ weight, support, mask: inside } = gradedCell(inside, width, height, { col, row }, rho));
      moved = !inside[start];
    } else ({ mask: inside, moved } = openCell(inside, width, height, { col, row }, rho));
  } else {
    inside = ridgeCell(ridges(field.field, width, height, s.theta), width, height, { col, row });
  }
  let count = 0;
  for (const v of inside) count += v;
  return { mask: Float32Array.from(inside), weight, support, seed, count, total: width * height, dEdge, gated, moved, scales };
}

/** Whether the settings need the experiments' path rather than the graph. */
export const needsExperiments = (s: PatchSettings, broad: BroadField | undefined): boolean => s.rule === "ridges" || s.graded || !!broad;

/** The plain rule's patch at `seed`, pulled through the wand's subgraph, or undefined off the map. */
export async function wandPatchOnGraph(
  field: BoundaryField,
  res: Raster,
  seed: readonly [number, number],
  s: PatchSettings,
): Promise<Patch | undefined> {
  const { width, height } = res;
  const { col, row } = pixelOf(res, seed);
  if (col < -0.5 || row < -0.5 || col > width - 0.5 || row > height - 0.5) return undefined;
  const g = new Graph();
  const u = g.source(
    { shape: { kind: "grid", width, height }, dtype: "f32", data: field.u, axes: [{ name: "modes", type: "channel", length: field.K }] },
    "u",
  );
  const rho = s.open * field.perR;
  const { patch } = patchSubgraph(g, u, { col, row, sigma: s.dSmooth * field.perR, tol: s.tol, laplacian: s.rule === "laplacian", rho });
  // On the host: the patch is per pointer move on a map of some 10⁵ pixels, and every step after the
  // first is a fill or a transform with no GPU kernel yet.
  const out = await pull(g, patch, { mode: "cpu", ctx: { backend: browserBackend } });
  const mask = out.data instanceof Float32Array ? out.data : Float32Array.from(out.data ?? []);
  let count = 0;
  for (const v of mask) count += v;
  return { mask, seed, count, total: width * height, moved: rho >= 0.5 && !mask[nearestPixel(width, height, col, row)] };
}
