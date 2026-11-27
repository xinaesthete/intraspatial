// How big the wand's connected region is at EVERY tolerance, from one flood rather than one fill per
// tolerance — the "area vs tolerance" curve a user reads to pick a tolerance below the point where
// the region leaks into a neighbouring basin.
//
// ## The bottleneck cost
//
// `connectedRegion(d, w, h, seed, t)` keeps the pixels reachable from the seed by edge-sharing steps
// through pixels with d ≤ t. So pixel p is in the region at t exactly when some path from the seed to
// p has max(d) ≤ t. The smallest such max over all paths — p's BOTTLENECK cost — is therefore a
// single number per pixel, and region(t) = { p : cost(p) ≤ t }. One minimax flood (Dijkstra with
// max in place of +) gives every cost; area(t) is then a count, read off the sorted costs.
//
// The flood uses a binary heap keyed on the exact cost, which is always one of the input values (a
// max never rounds), so area(t) equals the fill's pixel count for every t, not just on a grid. A
// bucket queue over a histogram's bins would quantise the costs to bins: still exact AT bin edges
// (a pixel's cost is ≤ an edge iff its bucket is below it, given edges compared in the same f32
// arithmetic), but not between them, and the curve's leaks would be smeared to the bin width. The
// heap costs O(n log n) over the reached pixels and is fast enough for the 2048² rasters the wand
// sees.
//
// ## The seed
//
// `connectedRegion` starts from the nearest pixel to the seed if it is in range, else from the
// closest in-range corner of the cell the seed sits in (`wandSeedPixel`). The start pixel therefore
// depends on t, but only through which of (at most) four corner pixels are in range — so the t axis
// splits into at most five intervals with a fixed start each, and one flood per start that is ever
// used gives the exact composite curve. For t ≥ d(nearest) the start is always the nearest pixel,
// so the other floods are only run for the tolerances below that, and only when they are reached.
//
// Validity: a NaN pixel is never in range (NaN ≤ t is false), so it is never reached; ±Inf compare
// as numbers do, as they do in the fill.

/** A seed in raster coordinates, pixel centres at whole numbers (as the wand's marker). */
export interface RasterSeed {
  readonly col: number;
  readonly row: number;
}

/**
 * The pixel a fill at tolerance `tol` starts from: the nearest to `seed`, else the closest in-range
 * corner of the cell `seed` sits in, else -1. Shared with `connectedRegion`, so the curve and the
 * fill cannot disagree about where the region starts.
 */
export function wandSeedPixel(d: ArrayLike<number>, width: number, height: number, seed: RasterSeed, tol: number): number {
  const nearest = nearestPixel(width, height, seed);
  if (d[nearest]! <= tol) return nearest;
  let best = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const { i, dist } of seedCorners(width, height, seed))
    if (d[i]! <= tol && dist < bestDist) {
      best = i;
      bestDist = dist;
    }
  return best;
}

const clampTo = (v: number, n: number) => Math.max(0, Math.min(n - 1, v));

function nearestPixel(width: number, height: number, seed: RasterSeed): number {
  return clampTo(Math.round(seed.row), height) * width + clampTo(Math.round(seed.col), width);
}

/** The four corners of the seed's cell, clamped to the raster, in the order `wandSeedPixel` scans. */
function seedCorners(width: number, height: number, seed: RasterSeed): { i: number; dist: number }[] {
  const out: { i: number; dist: number }[] = [];
  for (const r of [Math.floor(seed.row), Math.ceil(seed.row)])
    for (const c of [Math.floor(seed.col), Math.ceil(seed.col)]) {
      const cc = clampTo(c, width);
      const rr = clampTo(r, height);
      out.push({ i: rr * width + cc, dist: Math.hypot(cc - seed.col, rr - seed.row) });
    }
  return out;
}

/**
 * Bottleneck cost of every pixel from `start`: the least, over 4-connected paths from `start`, of
 * the path's largest d. NaN where unreachable (through NaN pixels only, or `start` itself NaN).
 */
export function bottleneckCosts(d: ArrayLike<number>, width: number, height: number, start: number): Float64Array {
  const n = width * height;
  const cost = new Float64Array(n).fill(Number.NaN);
  if (start < 0 || start >= n || Number.isNaN(d[start]!)) return cost;
  // Indexed binary min-heap on cost, with decrease-key: each pixel is in the heap at most once, so
  // the heap never exceeds the raster. pos[i] = -1 never queued, -2 settled.
  const heap = new Int32Array(n);
  const pos = new Int32Array(n).fill(-1);
  let size = 0;
  const swap = (a: number, b: number) => {
    const ia = heap[a]!;
    const ib = heap[b]!;
    heap[a] = ib;
    heap[b] = ia;
    pos[ib] = a;
    pos[ia] = b;
  };
  const up = (k: number) => {
    while (k > 0) {
      const p = (k - 1) >> 1;
      if (cost[heap[p]!]! <= cost[heap[k]!]!) return;
      swap(k, p);
      k = p;
    }
  };
  const down = (k: number) => {
    for (;;) {
      const l = 2 * k + 1;
      if (l >= size) return;
      const r = l + 1;
      const m = r < size && cost[heap[r]!]! < cost[heap[l]!]! ? r : l;
      if (cost[heap[k]!]! <= cost[heap[m]!]!) return;
      swap(k, m);
      k = m;
    }
  };
  const relax = (j: number, through: number) => {
    const dj = d[j]!;
    if (Number.isNaN(dj) || pos[j] === -2) return;
    const c = dj > through ? dj : through;
    if (pos[j] === -1) {
      cost[j] = c;
      heap[size] = j;
      pos[j] = size;
      up(size++);
    } else if (c < cost[j]!) {
      cost[j] = c;
      up(pos[j]!);
    }
  };
  cost[start] = d[start]!;
  heap[size] = start;
  pos[start] = size++;
  while (size > 0) {
    const i = heap[0]!;
    pos[i] = -2;
    size--;
    if (size > 0) {
      heap[0] = heap[size]!;
      pos[heap[0]!] = 0;
      down(0);
    }
    const ci = cost[i]!;
    const x = i % width;
    if (x > 0) relax(i - 1, ci);
    if (x < width - 1) relax(i + 1, ci);
    if (i >= width) relax(i - width, ci);
    if (i < n - width) relax(i + width, ci);
  }
  return cost;
}

/** The region's pixel count as a step function of the tolerance. */
export interface FloodAreaCurve {
  /** Tolerances at which the area changes, ascending and distinct. */
  readonly tol: Float64Array;
  /** `area[k]` is the region's pixel count for every t in `[tol[k], tol[k+1])`; 0 below `tol[0]`. */
  readonly area: Float64Array;
  /** width × height. */
  readonly pixels: number;
  /** Bottleneck cost of every pixel from the pixel nearest the seed (NaN where unreachable, and
   *  absent when that pixel is NaN): the tolerance at which each pixel joins the region, for every
   *  tolerance at or above the nearest pixel's own d, where the fill always starts there. */
  readonly nearestCosts?: Float64Array;
}

/** Number of entries of ascending `sorted` that are ≤ t. */
function countAtMost(sorted: ArrayLike<number>, t: number): number {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid]! <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * The area-vs-tolerance curve of `connectedRegion(d, width, height, seed, t)`: for every t,
 * `floodAreaAt(curve, t)` is exactly that fill's pixel count. See the note at the top.
 */
export function floodAreaByTolerance(d: ArrayLike<number>, width: number, height: number, seed: RasterSeed): FloodAreaCurve {
  if (d.length !== width * height) throw new Error(`floodAreaByTolerance: d has ${d.length} values, not ${width}×${height}`);
  const nearest = nearestPixel(width, height, seed);
  const starts = [nearest, ...seedCorners(width, height, seed).map((c) => c.i)];
  // Where the start can change: each candidate start's own d. Between two consecutive change points
  // the set of in-range candidates, hence the start, is fixed.
  const changes = [...new Set(starts.map((i) => d[i]!).filter((v) => !Number.isNaN(v)))].sort((a, b) => a - b);
  const sortedCosts = new Map<number, Float64Array>();
  let nearestCosts: Float64Array | undefined;
  const costsFrom = (start: number) => {
    let s = sortedCosts.get(start);
    if (!s) {
      const c = bottleneckCosts(d, width, height, start);
      if (start === nearest) nearestCosts = c;
      s = c.filter((v) => !Number.isNaN(v)).sort();
      sortedCosts.set(start, s);
    }
    return s;
  };
  const tol: number[] = [];
  const area: number[] = [];
  const push = (t: number, a: number) => {
    if (area.length > 0 && area[area.length - 1] === a) return;
    tol.push(t);
    area.push(a);
  };
  for (let j = 0; j < changes.length; j++) {
    const a = changes[j]!;
    const b = changes[j + 1] ?? Number.POSITIVE_INFINITY;
    const start = wandSeedPixel(d, width, height, seed, a);
    const costs = costsFrom(start);
    push(a, countAtMost(costs, a));
    // Every distinct cost strictly inside (a, b) is a step. Equal costs are adjacent in `costs`, so
    // the count after a run of equal values is the index just past it.
    for (let k = countAtMost(costs, a); k < costs.length && costs[k]! < b;) {
      const v = costs[k]!;
      while (k < costs.length && costs[k] === v) k++;
      push(v, k);
    }
  }
  return {
    tol: Float64Array.from(tol),
    area: Float64Array.from(area),
    pixels: width * height,
    ...(nearestCosts ? { nearestCosts } : {}),
  };
}

/** The region's pixel count at tolerance `t`, read off the curve. */
export function floodAreaAt(curve: FloodAreaCurve, t: number): number {
  const k = countAtMost(curve.tol, t);
  return k === 0 ? 0 : curve.area[k - 1]!;
}

export interface FloodLeakOptions {
  /** A step is a leak when the area grows by at least this fraction of the area just below it. Default 0.5. */
  readonly growth?: number;
  /** …from a region already at least this fraction of the raster's pixels. Default 0.005. */
  readonly minFraction?: number;
}

/**
 * Indices into `curve.tol` where the region LEAKS: a single tolerance at which a region of at least
 * `minFraction` of the raster grows by at least `growth` of itself at once.
 *
 * Why single steps: when the tolerance reaches a saddle of d, every pixel of the basin beyond it
 * that is lower than the saddle has the saddle as its bottleneck cost, so the whole basin joins at
 * one exact tolerance. Smooth growth spreads its pixels over many distinct costs instead. The size
 * floor keeps the seed's first few pixels out (1 → 2 is a 100% jump), as rings of tied values
 * around it would otherwise be.
 */
export function floodLeaks(curve: FloodAreaCurve, opts: FloodLeakOptions = {}): number[] {
  const growth = opts.growth ?? 0.5;
  const minBefore = (opts.minFraction ?? 0.005) * curve.pixels;
  const out: number[] = [];
  for (let k = 1; k < curve.area.length; k++) {
    const before = curve.area[k - 1]!;
    const jump = curve.area[k]! - before;
    if (before >= minBefore && jump >= growth * before) out.push(k);
  }
  return out;
}
