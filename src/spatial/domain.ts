// The region a set of cells was observed in — the `W` that the cross-PCF's ρ_B, edge correction and
// membership all refer to. See docs/stats-on-arbitrary-domains.md.
//
// Three kinds. A **rect** is the analytic case `edgeCorrection.ts` already handles exactly. A
// **polygon** is a drawn shape — an MDV gate, a SpatialData `shapes` annotation, a TMA core outline —
// and is clipped exactly too. A **mask** is a raster of in/out pixels, which is how a shape with no
// outline gets in: a tissue mask, a density threshold, the Gram wand.
//
// ## The polygon is clipped exactly
//
// `|disk(c, r) ∩ P|` is the sum, over P's edges (p, q), of the signed area of `disk ∩ triangle(c, p, q)`.
// Each term is closed form: split the edge where it crosses the circle, and every piece is either
// wholly inside (a triangle) or wholly outside (a circular sector). Rings are oriented on entry so
// that the sum is the even–odd region — holes and disjoint parts need no flag. Annulus k is the
// difference of the disks at its two edges. Only edges within rMax of an anchor need evaluating (see
// "Only edges near the anchor cost anything"), and anchors farther in take the full-annulus fast path.
//
// ## The mask is clipped exactly — its only error is its staircase
//
// The clipped annulus area is `A_k(x) = Σ_{mask pixels p} |annulus_k(x) ∩ p|`, and every term is a
// `diskRectArea` call, so the integration is exact *for the union of mask pixels*. What remains is how
// well that union represents the true boundary. Measured (`pnpm bench:domain-mask`) at the covid
// binning, that costs ≤ 0.1% on g at 1–2 µm pixels and ≤ 0.34% at 4 µm. Counting pixel centres
// instead is indistinguishable in g but 2–6× noisier per anchor on oblique edges, and per-anchor
// values are what highlights and weights are built from.
//
// Most pixels are not cut by any bin edge and drop into a single bin as their full area, so only
// the pixels a bin-edge circle passes through pay for `diskRectArea`.
//
// ## Raster convention
//
// Same as `ScalarField` and the GPU splats: row-major `width × height` over `bbox`, row 0 at the TOP
// (worldY = maxY). Non-zero = inside. Pixels need not be square.

import { annulusAreasInto, diskRectArea } from "./edgeCorrection";

type Box = readonly [number, number, number, number];

export interface RectDomain {
  readonly kind: "rect";
  /** `[minX, minY, maxX, maxY]`, closed. */
  readonly bbox: Box;
}

export interface MaskDomain {
  readonly kind: "mask";
  /** World extent the raster spans, `[minX, minY, maxX, maxY]`. Outside it is outside the domain. */
  readonly bbox: Box;
  readonly width: number;
  readonly height: number;
  /** Row-major, row 0 = top (maxY). Non-zero = inside. */
  readonly data: Uint8Array;
  /** Type ids whose cells the mask was BUILT from — a density threshold, a wand similarity — when
   *  it came from the data rather than being drawn. A cross-PCF pair with both members in here is
   *  measured through a window its own cells chose, and `crossPCFMatrixBinned` flags it as
   *  `confounded`. Absent for a drawn or external mask. */
  readonly derivedFrom?: readonly number[];
}

export interface PolygonDomain {
  readonly kind: "polygon";
  /** The rings' bounding box. */
  readonly bbox: Box;
  /** Flat `[x0, y0, x1, y1, …]` per ring, open (no repeated closing vertex), oriented so outer
   *  boundaries run counter-clockwise and holes clockwise: the even–odd region of the input. */
  readonly rings: readonly Float64Array[];
}

export type Domain = RectDomain | MaskDomain | PolygonDomain;

type Ring = readonly (readonly [number, number])[];

/** How a set of rings, possibly crossing, encloses a region. `evenodd`: a point is inside when a ray
 *  from it crosses the rings an odd number of times (SVG/canvas `evenodd`, and what a ray-casting
 *  point-in-polygon filter computes). `nonzero`: inside when the rings wind round it a non-zero
 *  number of times: the windings of all rings round it, counter-clockwise +1 and clockwise −1,
 *  must not sum to zero. Overlaps of rings running the same way merge; one ring each way over the
 *  same area cancels to zero and is outside, which is how a hole must be drawn under this rule. */
export type FillRule = "evenodd" | "nonzero";

/**
 * A polygon domain from GeoJSON-style rings: one ring `[[x, y], …]`, or several (`[[[x, y], …], …]`,
 * a Polygon's coordinates or a MultiPolygon's rings flattened). The region is even–odd: a ring
 * inside another is a hole, a ring inside a hole is an island. A closing vertex equal to the first
 * is dropped. Rings must not cross each other or themselves (this throws, naming the ring and edge
 * indices); touching at a vertex is allowed. Orientation does not matter.
 *
 * Pass `fillRule` to accept crossing or overlapping rings (e.g. a self-intersecting lasso): they
 * are first normalised by `normaliseRings` under that rule. Choose the rule the caller's own
 * selection uses, so the domain covers exactly what it selects. Without it nothing is repaired.
 */
export function polygonDomain(shape: Ring | readonly Ring[], opts: { readonly fillRule?: FillRule }): PolygonDomain;
// The one-argument form last, so `shapes.map(polygonDomain)` infers from it rather than reading
// the index as options.
export function polygonDomain(shape: Ring | readonly Ring[]): PolygonDomain;
export function polygonDomain(shape: Ring | readonly Ring[], opts: { readonly fillRule?: FillRule } = {}): PolygonDomain {
  const parsed = parseRings(shape, "polygonDomain");
  const flat = opts.fillRule === undefined ? parsed : fillRings(parsed, opts.fillRule, "polygonDomain");
  const bbox = ringsBbox(flat);
  // Normalised rings are simple by construction; checking again costs little and turns any
  // numerical slip into an error rather than a silently biased domain.
  assertNoCrossings(flat, bbox);
  // A ring nested in an odd number of others is a hole: orient it clockwise, the rest counter-
  // clockwise, so the signed sums below count the even–odd region once. Normalised rings already
  // carry that orientation (filled side on the left).
  const oriented =
    opts.fillRule !== undefined
      ? flat
      : flat.map((f, r) => {
          let depth = 0;
          for (let o = 0; o < flat.length; o++) {
            if (o === r) continue;
            const [x, y] = probeOff(f, flat[o]!);
            if (inRing(flat[o]!, x, y)) depth++;
          }
          const wantCcw = depth % 2 === 0;
          return signedRingArea(f) > 0 === wantCcw ? f : reverseRing(f);
        });
  if (oriented.some((f) => signedRingArea(f) === 0)) throw new Error("polygonDomain: a ring has zero area");
  return { kind: "polygon", bbox, rings: oriented };
}

/**
 * Crossing or overlapping rings rewritten as simple rings that cover the same region under
 * `fillRule`, for `polygonDomain` (which `polygonDomain(shape, { fillRule })` does for you). A
 * figure-eight becomes two rings touching at its crossing; under `evenodd` a loop drawn twice round
 * the same area is outside; under `nonzero` a part is outside only where the windings sum to zero,
 * so same-way overlaps merge into one ring.
 *
 * Output rings meet only at shared vertices. Each runs with the region on its left (outer
 * boundaries counter-clockwise, holes clockwise) and open. The input is not modified: normalise when building a domain, and keep the
 * shape the user edits as drawn. Throws, as `polygonDomain` does, on a ring with fewer than 3
 * vertices, a non-finite vertex or zero area, and when the region under the rule is empty.
 */
export function normaliseRings(shape: Ring | readonly Ring[], fillRule: FillRule): [number, number][][] {
  return fillRings(parseRings(shape, "normaliseRings"), fillRule, "normaliseRings").map((f) =>
    Array.from({ length: f.length / 2 }, (_, i): [number, number] => [f[2 * i]!, f[2 * i + 1]!]),
  );
}

/** Rings as flat open `Float64Array`s, checked: at least 3 distinct vertices, all finite. */
function parseRings(shape: Ring | readonly Ring[], who: string): Float64Array[] {
  const first = shape[0];
  const rings: readonly Ring[] = first !== undefined && typeof first[0] === "number" ? [shape as Ring] : (shape as readonly Ring[]);
  const flat = rings.map((ring, r) => {
    let n = ring.length;
    if (n > 1 && ring[0]![0] === ring[n - 1]![0] && ring[0]![1] === ring[n - 1]![1]) n--;
    if (n < 3) throw new Error(`${who}: ring ${r} has ${n} distinct vertices, fewer than 3`);
    const f = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      const [x, y] = ring[i]!;
      if (!(Number.isFinite(x) && Number.isFinite(y))) throw new Error(`${who}: ring ${r} vertex ${i} is not finite`);
      f[2 * i] = x;
      f[2 * i + 1] = y;
    }
    return f;
  });
  if (flat.length === 0) throw new Error(`${who}: no rings`);
  return flat;
}

function ringsBbox(rings: readonly Float64Array[]): Box {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const f of rings) {
    for (let i = 0; i < f.length; i += 2) {
      minX = Math.min(minX, f[i]!);
      maxX = Math.max(maxX, f[i]!);
      minY = Math.min(minY, f[i + 1]!);
      maxY = Math.max(maxY, f[i + 1]!);
    }
  }
  return [minX, minY, maxX, maxY];
}

/** `fillRegion`, with `polygonDomain`'s zero-area error per ring and an error for an empty region. */
function fillRings(rings: readonly Float64Array[], rule: FillRule, who: string): Float64Array[] {
  // A ring with non-zero signed area winds round something. One with zero signed area may still
  // enclose area (a figure-eight), so ask the arrangement.
  for (const f of rings) {
    if (signedRingArea(f) === 0 && fillRegion([f], "nonzero").length === 0) throw new Error(`${who}: a ring has zero area`);
  }
  const out = fillRegion(rings, rule);
  if (out.length === 0) throw new Error(`${who}: the rings enclose no area under the ${rule} fill rule`);
  return out;
}

/** Proper crossing or positive-length collinear overlap of segments (a→b) and (c→d). Touching at
 *  an endpoint, or an endpoint resting on the other segment, is not a crossing. */
function segmentsCross(ax: number, ay: number, bx: number, by: number, cx: number, cy: number, dx: number, dy: number): boolean {
  if (!boxesMeet(ax, ay, bx, by, cx, cy, dx, dy)) return false;
  const o1 = Math.sign((bx - ax) * (cy - ay) - (by - ay) * (cx - ax));
  const o2 = Math.sign((bx - ax) * (dy - ay) - (by - ay) * (dx - ax));
  const o3 = Math.sign((dx - cx) * (ay - cy) - (dy - cy) * (ax - cx));
  const o4 = Math.sign((dx - cx) * (by - cy) - (dy - cy) * (bx - cx));
  if (o1 * o2 < 0 && o3 * o4 < 0) return true;
  if (o1 === 0 && o2 === 0) {
    // Collinear: overlap of positive length along the dominant axis.
    const useX = Math.abs(bx - ax) >= Math.abs(by - ay);
    const [a0, a1] = useX ? [ax, bx] : [ay, by];
    const [c0, c1] = useX ? [cx, dx] : [cy, dy];
    return Math.min(Math.max(a0, a1), Math.max(c0, c1)) > Math.max(Math.min(a0, a1), Math.min(c0, c1));
  }
  return false;
}

/** Whether the bounding boxes of segments (a→b) and (c→d) meet. Segments whose boxes are apart
 *  cannot touch, and checking first stops rounding in the orientation tests from reporting a
 *  crossing between two nearly collinear pieces of one line that lie end to end. */
function boxesMeet(ax: number, ay: number, bx: number, by: number, cx: number, cy: number, dx: number, dy: number): boolean {
  return (
    Math.max(Math.min(ax, bx), Math.min(cx, dx)) <= Math.min(Math.max(ax, bx), Math.max(cx, dx)) &&
    Math.max(Math.min(ay, by), Math.min(cy, dy)) <= Math.min(Math.max(ay, by), Math.max(cy, dy))
  );
}

/**
 * Calls `visit(k, m)` once for each pair of edges whose bounding boxes share a cell of a coarse
 * grid, which includes every pair that touches. Near-linear for ordinary polygons. `e` is laid out
 * as `edgesOf` lays it out.
 */
function forEachCandidatePair(e: Float64Array, bbox: Box, visit: (k: number, m: number) => void): void {
  const nE = e.length / 4;
  const cell = Math.max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / Math.max(1, Math.ceil(Math.sqrt(nE)));
  const ix = buildEdgeIndex(e, bbox, cell);
  for (let c = 0; c < ix.nx * ix.ny; c++) {
    const ci = c % ix.nx;
    const cj = (c - ci) / ix.nx;
    for (let p = ix.start[c]!; p < ix.start[c + 1]!; p++) {
      const k = ix.ids[p]!;
      const [ki, , kj] = edgeCells(e, k, ix.minX, ix.minY, ix.cell);
      for (let q = p + 1; q < ix.start[c + 1]!; q++) {
        const m = ix.ids[q]!;
        const [mi, , mj] = edgeCells(e, m, ix.minX, ix.minY, ix.cell);
        // Long edges share many cells; visit the pair only in the first cell both reach.
        if (Math.max(ki, mi) === ci && Math.max(kj, mj) === cj) visit(k, m);
      }
    }
  }
}

/**
 * Throws if any two edges of the rings cross (each other or themselves).
 * Edge `i` of a ring runs from vertex `i - 1` to vertex `i`, matching `edgesOf`.
 */
function assertNoCrossings(rings: readonly Float64Array[], bbox: Box): void {
  const e = edgesOf(rings);
  const ringStart: number[] = [];
  let acc = 0;
  for (const f of rings) {
    ringStart.push(acc);
    acc += f.length / 2;
  }
  const where = (k: number) => {
    let r = ringStart.length - 1;
    while (ringStart[r]! > k) r--;
    return { ring: r, edge: k - ringStart[r]!, len: rings[r]!.length / 2 };
  };
  forEachCandidatePair(e, bbox, (k, m) => {
    const wk = where(k);
    const wm = where(m);
    if (wk.ring === wm.ring) {
      const gap = Math.abs(wk.edge - wm.edge);
      if (gap === 1 || gap === wk.len - 1) return; // adjacent edges share a vertex
    }
    if (segmentsCross(e[4 * k]!, e[4 * k + 1]!, e[4 * k + 2]!, e[4 * k + 3]!, e[4 * m]!, e[4 * m + 1]!, e[4 * m + 2]!, e[4 * m + 3]!)) {
      throw new Error(
        `polygonDomain: edges cross (ring ${wk.ring} edge ${wk.edge} and ring ${wm.ring} edge ${wm.edge}; edge i runs from vertex i-1 to i). ` +
          "Rings must not cross themselves or each other; repair the polygon upstream, or pass a fillRule.",
      );
    }
  });
}

// ## Normalising under a fill rule
//
// The rings are cut into a planar arrangement: every edge is split wherever another edge crosses
// it or a vertex rests on it, so the pieces meet only at shared vertices. Each piece carries its
// net multiplicity — how many input edges run along it one way minus the other — and crossing a
// piece from its right to its left raises the winding number by that much. Pieces with net 0
// separate nothing and are dropped.
//
// Faces are traced by always taking the first piece clockwise from the one arrived on, which keeps
// each face on the left. Within a connected component, windings spread from the outer face across
// pieces; the outer face takes the winding the other components have at the component's leftmost
// vertex. The fill rule turns windings into in/out, and the pieces with in on the left and out on
// the right, traced by the same turning rule, are the output rings.
//
// Input vertices are kept exactly; only computed crossing points, which are rounded, snap to a
// vertex within 1e-10 of the crossing edges' length. That is far below any drawn feature and keeps
// three edges through one point meeting at one vertex.

/** The simple, region-on-the-left rings bounding `rings`' region under `rule`; empty when it is. */
function fillRegion(rings: readonly Float64Array[], rule: FillRule): Float64Array[] {
  const bbox = ringsBbox(rings);
  const extent = Math.max(bbox[2] - bbox[0], bbox[3] - bbox[1]);
  // Coordinates' rounding unit: a snap can never be finer than a few of these.
  const ulp = Math.max(Math.abs(bbox[0]), Math.abs(bbox[1]), Math.abs(bbox[2]), Math.abs(bbox[3])) * Number.EPSILON;
  const snapCell = Math.max(4e-10 * extent, 16 * ulp) || 1;
  const vx: number[] = [];
  const vy: number[] = [];
  const exact = new Map<string, number>();
  const snapCells = new Map<string, number[]>();
  /** Vertex id for `(x, y)`: an exact match, or with `tol > 0` (a computed crossing) any vertex
   *  within `tol`, so three edges through one point meet at one vertex. `tol` ≤ `snapCell`. */
  const vertex = (x: number, y: number, tol = 0): number => {
    const i = Math.floor(x / snapCell);
    const j = Math.floor(y / snapCell);
    const hit = exact.get(`${x},${y}`);
    if (hit !== undefined) return hit;
    if (tol > 0) {
      for (let dj = -1; dj <= 1; dj++) {
        for (let di = -1; di <= 1; di++) {
          for (const id of snapCells.get(`${i + di},${j + dj}`) ?? []) {
            if (Math.abs(vx[id]! - x) <= tol && Math.abs(vy[id]! - y) <= tol) return id;
          }
        }
      }
    }
    const id = vx.length;
    exact.set(`${x},${y}`, id);
    vx.push(x);
    vy.push(y);
    const key = `${i},${j}`;
    const bucket = snapCells.get(key);
    if (bucket) bucket.push(id);
    else snapCells.set(key, [id]);
    return id;
  };

  // Input edges between snapped vertices, zero-length ones dropped.
  const segA: number[] = [];
  const segB: number[] = [];
  for (const f of rings) {
    const n = f.length / 2;
    const ids = Array.from({ length: n }, (_, i) => vertex(f[2 * i]!, f[2 * i + 1]!));
    for (let i = 0; i < n; i++) {
      const a = ids[(i + n - 1) % n]!;
      const b = ids[i]!;
      if (a !== b) {
        segA.push(a);
        segB.push(b);
      }
    }
  }
  const nS = segA.length;
  const e = new Float64Array(4 * nS);
  for (let k = 0; k < nS; k++) {
    e[4 * k] = vx[segA[k]!]!;
    e[4 * k + 1] = vy[segA[k]!]!;
    e[4 * k + 2] = vx[segB[k]!]!;
    e[4 * k + 3] = vy[segB[k]!]!;
  }

  // Split points per edge, as (t along the edge, vertex id).
  const splits: { t: number; v: number }[][] = Array.from({ length: nS }, () => []);
  const restsOn = (k: number, v: number) => {
    if (v === segA[k] || v === segB[k]) return;
    const ax = e[4 * k]!;
    const ay = e[4 * k + 1]!;
    const dx = e[4 * k + 2]! - ax;
    const dy = e[4 * k + 3]! - ay;
    const t = ((vx[v]! - ax) * dx + (vy[v]! - ay) * dy) / (dx * dx + dy * dy);
    if (t > 0 && t < 1) splits[k]!.push({ t, v });
  };
  forEachCandidatePair(e, bbox, (k, m) => {
    const ax = e[4 * k]!;
    const ay = e[4 * k + 1]!;
    const bx = e[4 * k + 2]!;
    const by = e[4 * k + 3]!;
    const cx = e[4 * m]!;
    const cy = e[4 * m + 1]!;
    const dx = e[4 * m + 2]!;
    const dy = e[4 * m + 3]!;
    if (!boxesMeet(ax, ay, bx, by, cx, cy, dx, dy)) return;
    const o1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    const o2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax);
    const o3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx);
    const o4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx);
    if (Math.sign(o1) * Math.sign(o2) < 0 && Math.sign(o3) * Math.sign(o4) < 0) {
      // o3 + t (o4 − o3) is the side of c→d that a + t (b − a) is on; it is zero at the crossing.
      const t = o3 / (o3 - o4);
      // Snap within a hair of the longer edge's length (never below the coordinates' rounding),
      // so a feature far from this one cannot widen the tolerance.
      const scale = Math.max(Math.abs(bx - ax), Math.abs(by - ay), Math.abs(dx - cx), Math.abs(dy - cy));
      const tol = Math.min(snapCell, Math.max(1e-10 * scale, 16 * ulp));
      const v = vertex(ax + t * (bx - ax), ay + t * (by - ay), tol);
      splits[k]!.push({ t, v });
      splits[m]!.push({ t: o1 / (o1 - o2), v });
      return;
    }
    // A vertex resting on the other edge (this includes collinear overlaps).
    if (o1 === 0) restsOn(k, segA[m]!);
    if (o2 === 0) restsOn(k, segB[m]!);
    if (o3 === 0) restsOn(m, segA[k]!);
    if (o4 === 0) restsOn(m, segB[k]!);
  });

  // Pieces, merged where input edges overlap: net multiplicity along lo → hi.
  const nV = vx.length;
  const net = new Map<number, number>();
  for (let k = 0; k < nS; k++) {
    const chain = [segA[k]!, ...splits[k]!.sort((p, q) => p.t - q.t).map((s) => s.v), segB[k]!];
    for (let i = 1; i < chain.length; i++) {
      const u = chain[i - 1]!;
      const v = chain[i]!;
      if (u === v) continue;
      const key = u < v ? u * nV + v : v * nV + u;
      net.set(key, (net.get(key) ?? 0) + (u < v ? 1 : -1));
    }
  }

  // Half-edges: 2i runs lo → hi with multiplicity c, 2i + 1 back with −c. Twin of h is h ^ 1.
  const hFrom: number[] = [];
  const hTo: number[] = [];
  const hNet: number[] = [];
  for (const [key, c] of net) {
    if (c === 0) continue;
    const lo = Math.floor(key / nV);
    const hi = key - lo * nV;
    hFrom.push(lo, hi);
    hTo.push(hi, lo);
    hNet.push(c, -c);
  }
  const nH = hFrom.length;
  if (nH === 0) return [];

  // Outgoing half-edges per vertex, counter-clockwise by angle.
  const outStart = new Uint32Array(nV + 1);
  for (let h = 0; h < nH; h++) outStart[hFrom[h]! + 1]!++;
  for (let v = 0; v < nV; v++) outStart[v + 1]! += outStart[v]!;
  const fill = outStart.slice(0, nV);
  const out = new Uint32Array(nH);
  for (let h = 0; h < nH; h++) out[fill[hFrom[h]!]!++] = h;
  const angle = (h: number) => Math.atan2(vy[hTo[h]!]! - vy[hFrom[h]!]!, vx[hTo[h]!]! - vx[hFrom[h]!]!);
  const pos = new Uint32Array(nH);
  for (let v = 0; v < nV; v++) {
    const list = Array.from(out.subarray(outStart[v]!, outStart[v + 1]!)).sort((p, q) => angle(p) - angle(q));
    list.forEach((h, i) => {
      out[outStart[v]! + i] = h;
      pos[h] = i;
    });
  }
  /** The first half-edge clockwise from `h`'s twin, among those `accept` allows. */
  const turn = (h: number, accept: (g: number) => boolean): number => {
    const v = hTo[h]!;
    const deg = outStart[v + 1]! - outStart[v]!;
    const p = pos[h ^ 1]!;
    for (let q = 1; q <= deg; q++) {
      const g = out[outStart[v]! + ((p - q + deg) % deg)]!;
      if (accept(g)) return g;
    }
    throw new Error("normaliseRings: inconsistent arrangement (no way on from a vertex)");
  };

  // Faces, each half-edge's being the face on its left.
  const face = new Int32Array(nH).fill(-1);
  const faceFirst: number[] = [];
  const faceArea: number[] = [];
  const any = () => true;
  for (let h0 = 0; h0 < nH; h0++) {
    if (face[h0] !== -1) continue;
    const f = faceFirst.length;
    let area = 0;
    let h = h0;
    for (let steps = 0; face[h] === -1; steps++) {
      if (steps > nH) throw new Error("normaliseRings: inconsistent arrangement (face does not close)");
      face[h] = f;
      area += (vx[hFrom[h]!]! - bbox[0]) * (vy[hTo[h]!]! - bbox[1]) - (vx[hTo[h]!]! - bbox[0]) * (vy[hFrom[h]!]! - bbox[1]);
      h = turn(h, any);
    }
    faceFirst.push(h0);
    faceArea.push(area / 2);
  }

  // Connected components, and each one's outer face: the one with the most negative area.
  const parent = Array.from({ length: nV }, (_, i) => i);
  const find = (v: number): number => {
    while (parent[v] !== v) v = parent[v] = parent[parent[v]!]!;
    return v;
  };
  for (let h = 0; h < nH; h += 2) parent[find(hFrom[h]!)] = find(hTo[h]!);
  const outerOf = new Map<number, number>();
  for (let f = 0; f < faceFirst.length; f++) {
    const c = find(hFrom[faceFirst[f]!]!);
    const o = outerOf.get(c);
    if (o === undefined || faceArea[f]! < faceArea[o]!) outerOf.set(c, f);
  }

  // Windings: the outer face of each component from the others' winding at its leftmost vertex,
  // then across pieces (left = right + multiplicity).
  const wind = new Float64Array(faceFirst.length).fill(Number.NaN);
  const queue: number[] = [];
  const leftmost = new Map<number, number>();
  for (let h = 0; h < nH; h++) {
    const v = hFrom[h]!;
    const c = find(v);
    const p = leftmost.get(c);
    if (p === undefined || vx[v]! < vx[p]! || (vx[v]! === vx[p]! && vy[v]! < vy[p]!)) leftmost.set(c, v);
  }
  const bands = outerOf.size > 1 ? pieceBands(nH, hFrom, hTo, vy, bbox) : undefined;
  for (const [c, f] of outerOf) {
    const p = leftmost.get(c)!;
    wind[f] = bands ? windingAt(vx[p]!, vy[p]!, bands, (h) => find(hFrom[h]!) !== c, hFrom, hTo, hNet, vx, vy) : 0;
    queue.push(f);
  }
  while (queue.length > 0) {
    const f = queue.pop()!;
    let h = faceFirst[f]!;
    do {
      const g = face[h ^ 1]!;
      if (Number.isNaN(wind[g]!)) {
        wind[g] = wind[f]! - hNet[h]!;
        queue.push(g);
      }
      h = turn(h, any);
    } while (h !== faceFirst[f]);
  }

  const filled = (w: number) => (rule === "evenodd" ? Math.abs(w) % 2 === 1 : w !== 0);
  const isBoundary = (h: number) => filled(wind[face[h]!]!) && !filled(wind[face[h ^ 1]!]!);

  // Trace the boundary pieces into rings.
  const used = new Uint8Array(nH);
  const ringIds: number[][] = [];
  for (let h0 = 0; h0 < nH; h0++) {
    if (used[h0] || !isBoundary(h0)) continue;
    const ids: number[] = [];
    let h = h0;
    do {
      if (used[h]) throw new Error("normaliseRings: inconsistent arrangement (boundary does not close)");
      used[h] = 1;
      ids.push(hFrom[h]!);
      h = turn(h, isBoundary);
    } while (h !== h0);
    // A walk that passes a vertex twice (a hole touching its shell) is cut there into loops, each
    // keeping its direction, so every output ring is simple.
    const stack: number[] = [];
    const at = new Map<number, number>();
    for (const v of ids) {
      const p = at.get(v);
      if (p !== undefined) {
        const loop = stack.splice(p);
        for (const u of loop) at.delete(u);
        ringIds.push(loop);
      }
      at.set(v, stack.length);
      stack.push(v);
    }
    ringIds.push(stack);
  }

  return ringIds.map((ids) => {
    const ring = new Float64Array(2 * ids.length);
    for (let i = 0; i < ids.length; i++) {
      const v = ids[i]!;
      ring[2 * i] = vx[v]!;
      ring[2 * i + 1] = vy[v]!;
    }
    return ring;
  });
}

/** The even half-edges (one per piece) bucketed by the horizontal bands their y-spans touch. */
interface PieceBands {
  readonly minY: number;
  readonly band: number;
  readonly lists: readonly number[][];
}

function pieceBands(nH: number, hFrom: readonly number[], hTo: readonly number[], vy: readonly number[], bbox: Box): PieceBands {
  const nb = Math.max(1, Math.min(4096, Math.ceil(Math.sqrt(nH / 2))));
  const band = Math.max((bbox[3] - bbox[1]) / nb, Number.MIN_VALUE);
  const lists: number[][] = Array.from({ length: nb }, () => []);
  const at = (y: number) => Math.min(nb - 1, Math.max(0, Math.floor((y - bbox[1]) / band)));
  for (let h = 0; h < nH; h += 2) {
    const ya = vy[hFrom[h]!]!;
    const yb = vy[hTo[h]!]!;
    for (let b = at(Math.min(ya, yb)); b <= at(Math.max(ya, yb)); b++) lists[b]!.push(h);
  }
  return { minY: bbox[1], band, lists };
}

/** Winding number at `(x, y)` of the pieces that `include` allows, with multiplicity, by Sunday's
 *  crossing rule. `(x, y)` must not lie on any of them. Only pieces whose y-span reaches `y` can
 *  count, so only `y`'s band is read. */
function windingAt(
  x: number,
  y: number,
  bands: PieceBands,
  include: (h: number) => boolean,
  hFrom: readonly number[],
  hTo: readonly number[],
  hNet: readonly number[],
  vx: readonly number[],
  vy: readonly number[],
): number {
  let w = 0;
  const b = Math.min(bands.lists.length - 1, Math.max(0, Math.floor((y - bands.minY) / bands.band)));
  for (const h of bands.lists[b]!) {
    if (!include(h)) continue;
    const ax = vx[hFrom[h]!]!;
    const ay = vy[hFrom[h]!]!;
    const bx = vx[hTo[h]!]!;
    const by = vy[hTo[h]!]!;
    const side = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
    if (ay <= y) {
      if (by > y && side > 0) w += hNet[h]!;
    } else if (by <= y && side < 0) w -= hNet[h]!;
  }
  return w;
}

/** Shoelace area, about the ring's first vertex so that a ring far from the origin keeps its
 *  low-order bits (a unit square at 2²⁷ would otherwise read as zero). */
function signedRingArea(f: Float64Array): number {
  let s = 0;
  const n = f.length;
  const x0 = f[0]!;
  const y0 = f[1]!;
  for (let i = 0; i < n; i += 2) {
    const j = (i + 2) % n;
    s += (f[i]! - x0) * (f[j + 1]! - y0) - (f[j]! - x0) * (f[i + 1]! - y0);
  }
  return s / 2;
}

function reverseRing(f: Float64Array): Float64Array {
  const n = f.length / 2;
  const out = new Float64Array(f.length);
  for (let i = 0; i < n; i++) {
    out[2 * i] = f[2 * (n - 1 - i)]!;
    out[2 * i + 1] = f[2 * (n - 1 - i) + 1]!;
  }
  return out;
}

/** Whether `(x, y)` lies on one of the ring's edges, exactly. */
function onRing(f: Float64Array, x: number, y: number): boolean {
  const n = f.length;
  for (let i = 0, j = n - 2; i < n; j = i, i += 2) {
    const ax = f[j]!;
    const ay = f[j + 1]!;
    const bx = f[i]!;
    const by = f[i + 1]!;
    if ((bx - ax) * (y - ay) - (by - ay) * (x - ax) !== 0) continue;
    if (x >= Math.min(ax, bx) && x <= Math.max(ax, bx) && y >= Math.min(ay, by) && y <= Math.max(ay, by)) return true;
  }
  return false;
}

/** A point of ring `f` not on ring `o`, to test which side of `o` the ring lies: rings that do
 *  not cross lie wholly on one side, but a touching point answers either way. Tries the
 *  vertices; failing that, the middle of the first stretch of an edge between the points where
 *  `o`'s vertices rest on it. Rings that pass the crossing check share no stretch of edge, so
 *  that point is off `o`. */
function probeOff(f: Float64Array, o: Float64Array): readonly [number, number] {
  const n = f.length;
  for (let i = 0; i < n; i += 2) if (!onRing(o, f[i]!, f[i + 1]!)) return [f[i]!, f[i + 1]!];
  for (let i = 0, j = n - 2; i < n; j = i, i += 2) {
    const ax = f[j]!;
    const ay = f[j + 1]!;
    const dx = f[i]! - ax;
    const dy = f[i + 1]! - ay;
    const ts = [0, 1];
    for (let k = 0; k < o.length; k += 2) {
      const px = o[k]! - ax;
      const py = o[k + 1]! - ay;
      if (dx * py - dy * px !== 0) continue;
      const t = (px * dx + py * dy) / (dx * dx + dy * dy);
      if (t > 0 && t < 1) ts.push(t);
    }
    ts.sort((p, q) => p - q);
    for (let k = 1; k < ts.length; k++) {
      if (ts[k]! === ts[k - 1]!) continue;
      const t = (ts[k]! + ts[k - 1]!) / 2;
      const x = ax + t * dx;
      const y = ay + t * dy;
      if (!onRing(o, x, y)) return [x, y];
    }
  }
  return [f[0]!, f[1]!];
}

/** Even–odd crossing test against one ring. */
function inRing(f: Float64Array, x: number, y: number): boolean {
  let inside = false;
  const n = f.length;
  for (let i = 0, j = n - 2; i < n; j = i, i += 2) {
    const xi = f[i]!;
    const yi = f[i + 1]!;
    const xj = f[j]!;
    const yj = f[j + 1]!;
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Signed area of `disk(0, r) ∩ triangle(0, a, b)`, positive when a → b turns counter-clockwise. */
function diskTriangleArea(ax: number, ay: number, bx: number, by: number, r: number): number {
  const dx = bx - ax;
  const dy = by - ay;
  const r2 = r * r;
  // |a + t·d|² = r² → t² (d·d) + 2t (a·d) + (a·a − r²) = 0.
  const A = dx * dx + dy * dy;
  if (A === 0) return 0;
  const B = ax * dx + ay * dy;
  const C = ax * ax + ay * ay - r2;
  const disc = B * B - A * C;
  let t0 = 1;
  let t1 = 1;
  if (disc > 0) {
    const sq = Math.sqrt(disc);
    t0 = Math.min(1, Math.max(0, (-B - sq) / A));
    t1 = Math.min(1, Math.max(0, (-B + sq) / A));
  }
  // Pieces [0, t0] and [t1, 1] lie outside the circle (sectors); [t0, t1] inside (a triangle).
  const px = ax + t0 * dx;
  const py = ay + t0 * dy;
  const qx = ax + t1 * dx;
  const qy = ay + t1 * dy;
  const sector = (ux: number, uy: number, vx: number, vy: number) => (r2 * Math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)) / 2;
  if (disc <= 0 || t0 >= t1) return sector(ax, ay, bx, by);
  return sector(ax, ay, px, py) + (px * qy - py * qx) / 2 + sector(qx, qy, bx, by);
}

// ## Only edges near the anchor cost anything
//
// An edge wholly outside the circle contributes its sector, r²·θ/2, and the θs of every edge sum to
// 2π × the winding number, which the orientation above makes 1 inside the region and 0 outside. So
//   |disk(c, r) ∩ P| = π r² · [c ∈ P] + Σ_{edges within r of c} (term − r²·θ/2),
// and a grid of edges finds the few within rMax. Containment uses horizontal bands of edges.

interface EdgeIndex {
  /** Per edge: a then b, `[ax, ay, bx, by]`. */
  readonly e: Float64Array;
  readonly minX: number;
  readonly minY: number;
  readonly cell: number;
  readonly nx: number;
  readonly ny: number;
  /** CSR buckets: edges of grid cell `c` are `ids[start[c] .. start[c + 1])`. */
  readonly start: Uint32Array;
  readonly ids: Uint32Array;
  /** Per edge, the last query that visited it, to visit each once. */
  readonly seen: Uint32Array;
  query: number;
}

const MAX_CELLS_PER_AXIS = 512;

function edgesOf(rings: readonly Float64Array[]): Float64Array {
  let n = 0;
  for (const f of rings) n += f.length / 2;
  const e = new Float64Array(4 * n);
  let k = 0;
  for (const f of rings) {
    const m = f.length;
    for (let i = 0, j = m - 2; i < m; j = i, i += 2) {
      e[k++] = f[j]!;
      e[k++] = f[j + 1]!;
      e[k++] = f[i]!;
      e[k++] = f[i + 1]!;
    }
  }
  return e;
}

/** Grid cells `[i0, i1, j0, j1]` that edge `k`'s bounding box spans. */
function edgeCells(e: Float64Array, k: number, minX: number, minY: number, cell: number): readonly [number, number, number, number] {
  const i0 = Math.floor((Math.min(e[4 * k]!, e[4 * k + 2]!) - minX) / cell);
  const i1 = Math.floor((Math.max(e[4 * k]!, e[4 * k + 2]!) - minX) / cell);
  const j0 = Math.floor((Math.min(e[4 * k + 1]!, e[4 * k + 3]!) - minY) / cell);
  const j1 = Math.floor((Math.max(e[4 * k + 1]!, e[4 * k + 3]!) - minY) / cell);
  return [i0, i1, j0, j1];
}

/** Edges bucketed by the grid cells their bounding boxes touch. Rows and columns overlap so a
 *  query of radius ≤ cell visits at most 3×3 cells. */
function buildEdgeIndex(e: Float64Array, bbox: Box, cellWanted: number): EdgeIndex {
  const [minX, minY, maxX, maxY] = bbox;
  const cell = Math.max(cellWanted, (maxX - minX) / MAX_CELLS_PER_AXIS, (maxY - minY) / MAX_CELLS_PER_AXIS, 1e-9);
  const nx = Math.floor((maxX - minX) / cell) + 1;
  const ny = Math.floor((maxY - minY) / cell) + 1;
  const nE = e.length / 4;
  const range = (k: number) => edgeCells(e, k, minX, minY, cell);
  const count = new Uint32Array(nx * ny + 1);
  for (let k = 0; k < nE; k++) {
    const [i0, i1, j0, j1] = range(k);
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) count[j * nx + i + 1]!++;
  }
  for (let c = 0; c < nx * ny; c++) count[c + 1]! += count[c]!;
  const start = count.slice();
  const fill = count.slice(0, nx * ny);
  const ids = new Uint32Array(start[nx * ny]!);
  for (let k = 0; k < nE; k++) {
    const [i0, i1, j0, j1] = range(k);
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) ids[fill[j * nx + i]!++] = k;
  }
  return { e, minX, minY, cell, nx, ny, start, ids, seen: new Uint32Array(nE), query: 0 };
}

/** Calls `visit(k)` once for every edge whose bucket meets the square of half-side `r` round `(x, y)`. */
function nearEdges(ix: EdgeIndex, x: number, y: number, r: number, visit: (k: number) => void): void {
  const i0 = Math.max(0, Math.floor((x - r - ix.minX) / ix.cell));
  const i1 = Math.min(ix.nx - 1, Math.floor((x + r - ix.minX) / ix.cell));
  const j0 = Math.max(0, Math.floor((y - r - ix.minY) / ix.cell));
  const j1 = Math.min(ix.ny - 1, Math.floor((y + r - ix.minY) / ix.cell));
  if (i0 > i1 || j0 > j1) return;
  const q = ++ix.query;
  for (let j = j0; j <= j1; j++) {
    for (let i = i0; i <= i1; i++) {
      const c = j * ix.nx + i;
      for (let t = ix.start[c]!; t < ix.start[c + 1]!; t++) {
        const k = ix.ids[t]!;
        if (ix.seen[k] === q) continue;
        ix.seen[k] = q;
        visit(k);
      }
    }
  }
}

/** Horizontal bands of edges for the crossing test: a point's crossings come only from its band. */
interface BandIndex {
  readonly e: Float64Array;
  readonly minY: number;
  readonly band: number;
  readonly start: Uint32Array;
  readonly ids: Uint32Array;
}

function buildBandIndex(d: PolygonDomain): BandIndex {
  const e = edgesOf(d.rings);
  const nE = e.length / 4;
  const [, minY, , maxY] = d.bbox;
  const nb = Math.max(1, Math.min(4096, Math.ceil(Math.sqrt(nE))));
  const band = Math.max((maxY - minY) / nb, 1e-12);
  const span = (k: number) => {
    const lo = Math.max(0, Math.floor((Math.min(e[4 * k + 1]!, e[4 * k + 3]!) - minY) / band));
    const hi = Math.min(nb - 1, Math.floor((Math.max(e[4 * k + 1]!, e[4 * k + 3]!) - minY) / band));
    return [lo, hi] as const;
  };
  const count = new Uint32Array(nb + 1);
  for (let k = 0; k < nE; k++) {
    const [lo, hi] = span(k);
    for (let b = lo; b <= hi; b++) count[b + 1]!++;
  }
  for (let b = 0; b < nb; b++) count[b + 1]! += count[b]!;
  const fill = count.slice(0, nb);
  const ids = new Uint32Array(count[nb]!);
  for (let k = 0; k < nE; k++) {
    const [lo, hi] = span(k);
    for (let b = lo; b <= hi; b++) ids[fill[b]!++] = k;
  }
  return { e, minY, band, start: count, ids };
}

const bandIndices = new WeakMap<PolygonDomain, BandIndex>();

function polygonContains(d: PolygonDomain, x: number, y: number): boolean {
  let ix = bandIndices.get(d);
  if (!ix) {
    ix = buildBandIndex(d);
    bandIndices.set(d, ix);
  }
  const nb = ix.start.length - 1;
  const b = Math.min(nb - 1, Math.floor((y - ix.minY) / ix.band));
  if (b < 0) return false;
  const e = ix.e;
  let inside = false;
  for (let t = ix.start[b]!; t < ix.start[b + 1]!; t++) {
    const k = 4 * ix.ids[t]!;
    const xj = e[k]!;
    const yj = e[k + 1]!;
    const xi = e[k + 2]!;
    const yi = e[k + 3]!;
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Edge grids per domain, keyed by cell size: a query radius sets the cell. */
const edgeIndices = new WeakMap<PolygonDomain, Map<number, EdgeIndex>>();

function edgeIndex(d: PolygonDomain, cell: number): EdgeIndex {
  let byCell = edgeIndices.get(d);
  if (!byCell) {
    byCell = new Map();
    edgeIndices.set(d, byCell);
  }
  let ix = byCell.get(cell);
  if (!ix) {
    ix = buildEdgeIndex(edgesOf(d.rings), d.bbox, cell);
    byCell.set(cell, ix);
  }
  return ix;
}

function segmentDistance2(e: Float64Array, k: number, x: number, y: number): number {
  const ax = e[4 * k]!;
  const ay = e[4 * k + 1]!;
  const dx = e[4 * k + 2]! - ax;
  const dy = e[4 * k + 3]! - ay;
  const L = dx * dx + dy * dy;
  const t = L > 0 ? Math.min(1, Math.max(0, ((x - ax) * dx + (y - ay) * dy) / L)) : 0;
  const ex = ax + t * dx - x;
  const ey = ay + t * dy - y;
  return ex * ex + ey * ey;
}

/** Annulus areas `[k·dr, (k+1)·dr)` of a polygon, exact. */
function polygonAnnulusAreasInto(out: Float64Array, d: PolygonDomain, cx: number, cy: number, dr: number, nBins: number): void {
  const rMax = nBins * dr;
  const ix = edgeIndex(d, rMax);
  const e = ix.e;
  const near: number[] = [];
  nearEdges(ix, cx, cy, rMax, (k) => {
    if (segmentDistance2(e, k, cx, cy) < rMax * rMax) near.push(k);
  });
  const inside = polygonContains(d, cx, cy) ? 1 : 0;
  let prev = 0;
  for (let b = 0; b < nBins; b++) {
    const r = (b + 1) * dr;
    let s = Math.PI * r * r * inside;
    for (const k of near) {
      const ax = e[4 * k]! - cx;
      const ay = e[4 * k + 1]! - cy;
      const bx = e[4 * k + 2]! - cx;
      const by = e[4 * k + 3]! - cy;
      s += diskTriangleArea(ax, ay, bx, by, r) - (r * r * Math.atan2(ax * by - ay * bx, ax * bx + ay * by)) / 2;
    }
    const a = Math.max(0, s);
    out[b] = Math.max(0, a - prev);
    prev = a;
  }
}

/** Whether the disk of `rMax` round `(x, y)` lies wholly inside the polygon. */
function polygonInterior(d: PolygonDomain, rMax: number): (x: number, y: number) => boolean {
  const ix = edgeIndex(d, rMax);
  const r2 = rMax * rMax;
  return (x, y) => {
    if (!polygonContains(d, x, y)) return false;
    let clear = true;
    nearEdges(ix, x, y, rMax, (k) => {
      if (clear && segmentDistance2(ix.e, k, x, y) < r2) clear = false;
    });
    return clear;
  };
}

export function rectDomain(bbox: Box): RectDomain {
  if (!(bbox[2] > bbox[0] && bbox[3] > bbox[1])) throw new Error(`rectDomain: bbox [${bbox.join(", ")}] has no positive extent`);
  return { kind: "rect", bbox };
}

/** A mask domain, checked: a raster whose length disagrees with its shape would be read as a
 *  different, silently sheared region. */
export function maskDomain(bbox: Box, width: number, height: number, data: Uint8Array, derivedFrom?: readonly number[]): MaskDomain {
  if (!(bbox[2] > bbox[0] && bbox[3] > bbox[1])) throw new Error(`maskDomain: bbox [${bbox.join(", ")}] has no positive extent`);
  if (!(Number.isInteger(width) && Number.isInteger(height) && width > 0 && height > 0)) {
    throw new Error(`maskDomain: ${width}×${height} is not a raster shape`);
  }
  if (data.length !== width * height) throw new Error(`maskDomain: ${data.length} pixels for a ${width}×${height} raster`);
  return derivedFrom ? { kind: "mask", bbox, width, height, data, derivedFrom } : { kind: "mask", bbox, width, height, data };
}

/**
 * A mask from any raster in the `ScalarField` convention — a density field, a mode projection, the
 * Gram wand's `similarityField` — inside wherever `keep(value)` holds.
 *
 * The raster's `bbox` becomes the domain's, so build the field over the OBSERVATION window: a field
 * padded past it (as `gaussianKdeField` pads by default) can mark unobserved ground as inside, which
 * is the loose-boundary bias of docs/stats-on-arbitrary-domains.md source 4.
 *
 * A mask built from the same cells the statistic is then measured on is a data-dependent window.
 * Measured (docs/stats-on-arbitrary-domains.md, "Masks from the data"): on independently clustered
 * types, a mask cut from both A's and B's densities reads g_AB 14–16% low, where the truth is 1. A
 * mask that never saw B's cells leaves g_AB unbiased. Pass `derivedFrom` — the type ids whose cells
 * built the field — so the PCF can flag the pairs this applies to.
 */
export function maskWhere(
  field: { readonly data: ArrayLike<number>; readonly width: number; readonly height: number; readonly bbox: Box },
  keep: (value: number) => boolean,
  opts: { readonly derivedFrom?: readonly number[] } = {},
): MaskDomain {
  const n = field.width * field.height;
  if (field.data.length !== n) throw new Error(`maskWhere: ${field.data.length} values for a ${field.width}×${field.height} raster`);
  const data = new Uint8Array(n);
  for (let i = 0; i < n; i++) data[i] = keep(field.data[i]!) ? 1 : 0;
  return maskDomain(field.bbox, field.width, field.height, data, opts.derivedFrom);
}

/** |W|. For a mask, the area of its non-zero pixels; for a polygon, of its even–odd region. */
export function domainArea(d: Domain): number {
  if (d.kind === "polygon") return d.rings.reduce((s, f) => s + signedRingArea(f), 0);
  const [minX, minY, maxX, maxY] = d.bbox;
  if (d.kind === "rect") return (maxX - minX) * (maxY - minY);
  let n = 0;
  for (let i = 0; i < d.data.length; i++) if (d.data[i] !== 0) n++;
  return (n * (maxX - minX) * (maxY - minY)) / (d.width * d.height);
}

/** Whether `(x, y)` lies in the domain. A mask answers by the pixel containing the point, with its
 *  max edges closed so a cell exactly on `maxX`/`minY` is not lost to rounding. */
export function domainContains(d: Domain, x: number, y: number): boolean {
  const [minX, minY, maxX, maxY] = d.bbox;
  if (!(x >= minX && x <= maxX && y >= minY && y <= maxY)) return false;
  if (d.kind === "rect") return true;
  if (d.kind === "polygon") return polygonContains(d, x, y);
  const i = Math.min(d.width - 1, Math.floor(((x - minX) / (maxX - minX)) * d.width));
  const j = Math.min(d.height - 1, Math.floor(((maxY - y) / (maxY - minY)) * d.height));
  return d.data[j * d.width + i] !== 0;
}

/**
 * Clipped areas of every annulus `[k·dr, (k+1)·dr)` around `(cx, cy)`, written into `out` (length
 * `nBins`). Exact for a rect and a polygon, and exact for the pixel union of a mask.
 */
export function domainAnnulusAreasInto(out: Float64Array, d: Domain, cx: number, cy: number, dr: number, nBins: number): void {
  const [minX, minY, maxX, maxY] = d.bbox;
  if (d.kind === "rect") {
    annulusAreasInto(out, cx, cy, dr, nBins, minX, minY, maxX, maxY);
    return;
  }
  if (d.kind === "polygon") {
    polygonAnnulusAreasInto(out, d, cx, cy, dr, nBins);
    return;
  }
  out.fill(0, 0, nBins);
  const rMax = nBins * dr;
  const pw = (maxX - minX) / d.width;
  const ph = (maxY - minY) / d.height;
  const full = pw * ph;
  const i0 = Math.max(0, Math.floor((cx - rMax - minX) / pw));
  const i1 = Math.min(d.width - 1, Math.floor((cx + rMax - minX) / pw));
  const j0 = Math.max(0, Math.floor((maxY - (cy + rMax)) / ph));
  const j1 = Math.min(d.height - 1, Math.floor((maxY - (cy - rMax)) / ph));
  for (let j = j0; j <= j1; j++) {
    const y1 = maxY - j * ph;
    const y0 = y1 - ph;
    const dyMin = Math.max(y0 - cy, 0, cy - y1);
    const dyMax = Math.max(Math.abs(cy - y0), Math.abs(cy - y1));
    for (let i = i0; i <= i1; i++) {
      if (d.data[j * d.width + i] === 0) continue;
      const x0 = minX + i * pw;
      const x1 = x0 + pw;
      const dxMin = Math.max(x0 - cx, 0, cx - x1);
      const dMin = Math.sqrt(dxMin * dxMin + dyMin * dyMin);
      if (dMin >= rMax) continue;
      const dxMax = Math.max(Math.abs(cx - x0), Math.abs(cx - x1));
      const dMax = Math.sqrt(dxMax * dxMax + dyMax * dyMax);
      // Walk the bin edges that pass through this pixel. `prev` is the area of the pixel inside the
      // disk of the last edge crossed; the part beyond the final edge, if the pixel ends inside
      // rMax, falls wholly in the bin that edge opens. An uncut pixel takes the first branch at once.
      let prev = 0;
      for (let e = Math.floor(dMin / dr) + 1; ; e++) {
        const r = e * dr;
        if (r >= dMax) {
          if (e <= nBins) out[e - 1]! += full - prev;
          break;
        }
        const a = diskRectArea(cx, cy, r, x0, y0, x1, y1);
        out[e - 1]! += a - prev;
        prev = a;
        if (e === nBins) break;
      }
    }
  }
}

/**
 * Per-anchor reciprocal annulus areas `1/A_k(x)` — the weight the cross-PCF puts on a pair in bin
 * `k`. An anchor whose disk of `rMax` the boundary cannot reach takes the precomputed full-annulus
 * weights; that is an exact answer, not an approximation, since nothing is clipped. Bins with no
 * area left get weight 0.
 *
 * The returned array is reused between boundary anchors: read it before the next call.
 */
export function inverseAnnulusAreas(d: Domain, rMax: number, nBins: number): (x: number, y: number) => Float64Array {
  const dr = rMax / nBins;
  const invFull = new Float64Array(nBins);
  for (let k = 0; k < nBins; k++) {
    const r0 = k * dr;
    const r1 = r0 + dr;
    invFull[k] = 1 / (Math.PI * (r1 * r1 - r0 * r0));
  }
  const scratch = new Float64Array(nBins);
  const inv = new Float64Array(nBins);
  const interior = interiorTest(d, rMax);
  return (x, y) => {
    if (interior(x, y)) return invFull;
    domainAnnulusAreasInto(scratch, d, x, y, dr, nBins);
    for (let k = 0; k < nBins; k++) inv[k] = scratch[k]! > 0 ? 1 / scratch[k]! : 0;
    return inv;
  };
}

/** Whether the disk of `rMax` around a point is wholly inside the domain. Exact for a rect and a
 *  polygon (inside, and no edge nearer than rMax). Conservative for a mask:
 *  it checks the disk's bounding square through a summed-area table of OUTSIDE pixels, so an anchor
 *  near a concave corner may be sent down the exact path needlessly, never the reverse. */
function interiorTest(d: Domain, rMax: number): (x: number, y: number) => boolean {
  const [minX, minY, maxX, maxY] = d.bbox;
  if (d.kind === "rect") return (x, y) => x - minX >= rMax && maxX - x >= rMax && y - minY >= rMax && maxY - y >= rMax;
  if (d.kind === "polygon") return polygonInterior(d, rMax);
  const w = d.width;
  const h = d.height;
  const pw = (maxX - minX) / w;
  const ph = (maxY - minY) / h;
  // sat[(j+1)(w+1) + (i+1)] = outside pixels in rows ≤ j, columns ≤ i.
  const sat = new Int32Array((w + 1) * (h + 1));
  for (let j = 0; j < h; j++) {
    let row = 0;
    for (let i = 0; i < w; i++) {
      if (d.data[j * w + i] === 0) row++;
      sat[(j + 1) * (w + 1) + i + 1] = sat[j * (w + 1) + i + 1]! + row;
    }
  }
  return (x, y) => {
    const i0 = Math.floor((x - rMax - minX) / pw);
    const i1 = Math.floor((x + rMax - minX) / pw);
    const j0 = Math.floor((maxY - (y + rMax)) / ph);
    const j1 = Math.floor((maxY - (y - rMax)) / ph);
    // A square reaching past the raster reaches outside the domain.
    if (i0 < 0 || j0 < 0 || i1 >= w || j1 >= h) return false;
    const W = w + 1;
    return sat[(j1 + 1) * W + i1 + 1]! - sat[j0 * W + i1 + 1]! - sat[(j1 + 1) * W + i0]! + sat[j0 * W + i0]! === 0;
  };
}
