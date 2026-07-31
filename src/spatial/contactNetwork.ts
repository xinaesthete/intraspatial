// The cell-type contact network — who touches whom, and how much of each type's contact budget goes
// where.
//
// ## What the published columns mean, recovered exactly
//
// The covid project stores five network columns per (ROI, A, B). Their relationships to each other
// were recovered from the stored values and hold to floating-point on all 70,742 rows, so the
// quantities below are the published ones, not an interpretation of them:
//
//     %contacts   = 100 · contacts / n_A                    (checked: 100% of rows within 1e-4)
//     mean degree = Network / n_A                           (checked: 100%)
//     Network(%)  = 100 · Network / Σ_B Network(A,B)        (checked: 100%)
//
// and the two primitives they are built from, identified by their algebra: `Network(A,B)` is an
// integer and SYMMETRIC — the count of A–B edges — while `contacts(A,B)` is an integer, asymmetric,
// and never exceeds n_A, which fixes it as *the number of A cells having at least one B neighbour*.
// A cell with six B neighbours adds six to `Network` and one to `contacts`; that is the whole
// difference between them, and it is why `mean degree` and `%contacts` are not proportional.
//
// ## What is NOT reproduced, and why it is stated rather than fudged
//
// The graph itself. Across the 32 covid ROIs the stored network's mean degree runs 4.13 to 7.34
// (median 5.75), bracketing the ~6.0 that any planar triangulation of the same points must give — so
// it is Delaunay-SHAPED without being reproducible from the centroids: on COVID_SAMPLE_16_ROI_3 a
// Delaunay of those points gives 97,679 edges against 108,364 stored, and no distance-capped
// Delaunay or fixed-radius graph tried came within 36% of the stored per-pair counts. The likely
// source is adjacency taken from SEGMENTATION MASKS — two cells are in contact when their labelled
// regions touch — which is not a function of the centroids at all, and the project ships its masks
// only as RGB PNGs. So the counts below are computed from an explicit, stated graph rather than from
// theirs, and they are comparable in kind, not in value.
//
// The construction here is the fixed-radius contact graph: an edge when two centroids lie within
// `radius`. It is the honest centroid-only definition — "contact" becomes a distance you choose and
// can see — and it makes the radius a parameter of the answer instead of a hidden property of a
// segmentation.

import type { LabelledCells } from "./pcf";

export interface ContactNetworkParams {
  /** Two cells are in contact when their centroids are within this distance (world units). */
  readonly radius: number;
  /** Length of the type axis; ids must lie in `[0, nTypes)`. Defaults to `max(typeId) + 1`. */
  readonly nTypes?: number;
}

export interface ContactNetworkResult {
  readonly nTypes: number;
  /** Cell count per type. */
  readonly counts: number[];
  /** `edges[a*K + b]` — undirected A–B edge count, SYMMETRIC. The diagonal counts each within-type
   *  edge once, matching the published `Network`. */
  readonly edges: Float64Array;
  /** `contacts[a*K + b]` — A cells with at least one B neighbour. Asymmetric, ≤ `counts[a]`. */
  readonly contacts: Float64Array;
  /** `100 · contacts / n_A`. */
  readonly pctContacts: Float64Array;
  /** `edges / n_A` — mean number of B neighbours per A cell. */
  readonly meanDegree: Float64Array;
  /** `100 · edges(A,B) / Σ_B edges(A,B)` — the share of A's contacts that go to B. NaN for a type
   *  with no edges at all, which is a different statement from 0%. */
  readonly networkPct: Float64Array;
  /** Total undirected edges in the graph, and the mean degree over all cells — the two numbers that
   *  say what KIND of graph this is, and the ones to compare against a published network. */
  readonly totalEdges: number;
  readonly graphMeanDegree: number;
}

/**
 * Contact network statistics for every ordered type pair, in one bucket-grid pass.
 *
 * `radius` is the contact threshold; there is no default, because there is no natural one — it is
 * the definition of "touching" and belongs to the caller.
 */
export function contactNetwork(cells: LabelledCells, p: ContactNetworkParams): ContactNetworkResult {
  const n = cells.xs.length;
  let K = p.nTypes ?? 0;
  if (!p.nTypes) for (let i = 0; i < n; i++) K = Math.max(K, cells.typeId[i]! + 1);
  const counts = new Array<number>(K).fill(0);
  for (let i = 0; i < n; i++) {
    const t = cells.typeId[i]!;
    if (t < 0 || t >= K) throw new Error(`contactNetwork: type id ${t} outside [0, ${K})`);
    counts[t]!++;
  }

  const edges = new Float64Array(K * K);
  const contacts = new Float64Array(K * K);
  if (n === 0 || !(p.radius > 0)) {
    return {
      nTypes: K,
      counts,
      edges,
      contacts,
      pctContacts: new Float64Array(K * K),
      meanDegree: new Float64Array(K * K),
      networkPct: new Float64Array(K * K).fill(Number.NaN),
      totalEdges: 0,
      graphMeanDegree: 0,
    };
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = cells.xs[i]!;
    const y = cells.ys[i]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const cell = p.radius;
  const cols = Math.max(1, Math.ceil((maxX - minX) / cell) + 1);
  const rows = Math.max(1, Math.ceil((maxY - minY) / cell) + 1);
  const buckets: number[][] = Array.from({ length: cols * rows }, () => []);
  const colOf = (x: number) => Math.min(cols - 1, Math.max(0, Math.floor((x - minX) / cell)));
  const rowOf = (y: number) => Math.min(rows - 1, Math.max(0, Math.floor((y - minY) / cell)));
  for (let i = 0; i < n; i++) buckets[rowOf(cells.ys[i]!) * cols + colOf(cells.xs[i]!)]!.push(i);

  const r2 = p.radius * p.radius;
  // Which neighbour types this cell has already been counted against, reset via `touched` so the
  // clear is O(types seen) rather than O(K) per cell.
  const seen = new Uint8Array(K);
  const touched: number[] = [];
  let totalEdges = 0;

  for (let i = 0; i < n; i++) {
    const ax = cells.xs[i]!;
    const ay = cells.ys[i]!;
    const a = cells.typeId[i]!;
    const c0 = colOf(ax);
    const r0 = rowOf(ay);
    for (let dRow = -1; dRow <= 1; dRow++) {
      const rr = r0 + dRow;
      if (rr < 0 || rr >= rows) continue;
      for (let dCol = -1; dCol <= 1; dCol++) {
        const cc = c0 + dCol;
        if (cc < 0 || cc >= cols) continue;
        for (const j of buckets[rr * cols + cc]!) {
          if (j === i) continue;
          const dx = cells.xs[j]! - ax;
          const dy = cells.ys[j]! - ay;
          if (dx * dx + dy * dy >= r2) continue;
          const b = cells.typeId[j]!;
          // Each unordered pair is visited twice (once from each end). Count the edge on one visit
          // only, so `edges` is a true undirected count with each within-type edge counted once.
          if (j > i) {
            edges[a * K + b]! += 1;
            if (a !== b) edges[b * K + a]! += 1;
            totalEdges++;
          }
          if (!seen[b]) {
            seen[b] = 1;
            touched.push(b);
          }
        }
      }
    }
    for (const b of touched) {
      contacts[a * K + b]! += 1;
      seen[b] = 0;
    }
    touched.length = 0;
  }

  const pctContacts = new Float64Array(K * K);
  const meanDegree = new Float64Array(K * K);
  const networkPct = new Float64Array(K * K).fill(Number.NaN);
  for (let a = 0; a < K; a++) {
    const nA = counts[a]!;
    let rowSum = 0;
    for (let b = 0; b < K; b++) rowSum += edges[a * K + b]!;
    for (let b = 0; b < K; b++) {
      const at = a * K + b;
      pctContacts[at] = nA > 0 ? (100 * contacts[at]!) / nA : 0;
      meanDegree[at] = nA > 0 ? edges[at]! / nA : 0;
      if (rowSum > 0) networkPct[at] = (100 * edges[at]!) / rowSum;
    }
  }
  return {
    nTypes: K,
    counts,
    edges,
    contacts,
    pctContacts,
    meanDegree,
    networkPct,
    totalEdges,
    graphMeanDegree: n > 0 ? (2 * totalEdges) / n : 0,
  };
}
