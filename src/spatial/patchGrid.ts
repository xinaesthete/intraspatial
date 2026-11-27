// Patch grid: square patches tiled over a support footprint, kept by minimum-support rules.
// R4 of the region/raster statistics plan (docs/patch-grid.md). Its kept patches go to zonal
// statistics (R3, `zonalStatsTiled`) as polygons (`patchPolygons`), or as label rasters.
//
// The layout is SIMBA's (`src/feature_extraction/patch_grid.py`, TNBC-MMB) bit for bit:
//   - the footprint is split into 4-connected components and EACH COMPONENT gets its own grid,
//     anchored at its bounding box, not one global lattice;
//   - along an axis, a component no wider than a patch gets one patch ending at its far edge;
//     a wider one gets starts every `stride` from its near edge plus, when the stride does not
//     land on it, one trailing start ending exactly at its far edge (flagged `edge`);
//   - starts are clamped into the image; a patch is never partial unless the image itself is
//     smaller than a patch (then `end = min(start + size, extent)`);
//   - patches two components share are kept once; ids are the index in the (y0, x0) order of
//     every candidate, so `id` = SIMBA's `patch_%05d` number, before any support rule.
//
// Support is counted, not sampled: a mask's support in a patch is the number of its pixels in
// the rectangle (the fraction divides by the rectangle's area), a point set's is the number of
// points whose rounded (half-to-even, as numpy) and clamped pixel lies in it.
//
// Masks are held as row runs (`MaskRuns`), so a 1 µm/px canvas of half a gigapixel costs the
// runs, not the pixels: components come from merging runs of adjacent rows, rectangle counts
// from a prefix sum over run lengths. The GPU twin (`src/gpu/spatial/patchSupport.ts`) counts
// a resident dense mask with a flat prefix scan instead; both give identical integers.
import { type PolygonDomain, polygonDomain } from "./domain";

/** A pixel test: is this mask value support? */
export type SupportTest = (v: number) => boolean;

/** Finite and > 0: the default reading of a 0/1 mask or a non-negative raster. */
export const positiveFinite: SupportTest = (v) => v > 0 && v !== Infinity;
/** Any non-zero value, NaN included: numpy's `img != 0` (SIMBA's LA tissue mask). */
export const nonZero: SupportTest = (v) => v !== 0;

/** A binary mask as runs of support pixels, row by row: row `y`'s runs are
 *  `[x0[k], x1[k])` for `k` in `rowStart[y] .. rowStart[y + 1]`, left to right. */
export interface MaskRuns {
  readonly width: number;
  readonly height: number;
  readonly rowStart: Uint32Array;
  readonly x0: Uint32Array;
  readonly x1: Uint32Array;
}

class U32Builder {
  buf = new Uint32Array(1024);
  n = 0;
  push(v: number): void {
    if (this.n === this.buf.length) {
      const next = new Uint32Array(this.buf.length * 2);
      next.set(this.buf);
      this.buf = next;
    }
    this.buf[this.n++] = v;
  }
  done(): Uint32Array {
    return this.buf.slice(0, this.n);
  }
}

/** Runs of a mask read one row at a time (`rowAt(y)` returns that row's `width` values), so a
 *  canvas too big to hold as pixels can be streamed. */
export function maskRunsFromRows(
  width: number,
  height: number,
  rowAt: (y: number) => ArrayLike<number>,
  test: SupportTest = positiveFinite,
): MaskRuns {
  const rowStart = new Uint32Array(height + 1);
  const x0 = new U32Builder();
  const x1 = new U32Builder();
  for (let y = 0; y < height; y++) {
    const row = rowAt(y);
    if (row.length < width) throw new Error(`maskRuns: row ${y} has ${row.length} values, need ${width}`);
    let start = -1;
    for (let x = 0; x < width; x++) {
      const on = test(row[x]!);
      if (on && start < 0) start = x;
      else if (!on && start >= 0) {
        x0.push(start);
        x1.push(x);
        start = -1;
      }
    }
    if (start >= 0) {
      x0.push(start);
      x1.push(width);
    }
    rowStart[y + 1] = x0.n;
  }
  return { width, height, rowStart, x0: x0.done(), x1: x1.done() };
}

/** Runs of a row-major `width × height` mask. */
export function maskRuns(data: ArrayLike<number>, width: number, height: number, test: SupportTest = positiveFinite): MaskRuns {
  if (data.length !== width * height) throw new Error(`maskRuns: ${data.length} values for ${width}x${height}`);
  const row = new Float64Array(width);
  return maskRunsFromRows(
    width,
    height,
    (y) => {
      for (let x = 0; x < width; x++) row[x] = data[y * width + x]!;
      return row;
    },
    test,
  );
}

/** The union of masks on one canvas (a pixel is support when it is in any of them). */
export function unionRuns(masks: readonly MaskRuns[]): MaskRuns {
  if (masks.length === 0) throw new Error("unionRuns: no masks");
  const { width, height } = masks[0]!;
  for (const m of masks) if (m.width !== width || m.height !== height) throw new Error("unionRuns: masks differ in size");
  const row = new Uint8Array(width);
  return maskRunsFromRows(
    width,
    height,
    (y) => {
      row.fill(0);
      for (const m of masks) for (let k = m.rowStart[y]!; k < m.rowStart[y + 1]!; k++) row.fill(1, m.x0[k]!, m.x1[k]!);
      return row;
    },
    positiveFinite,
  );
}

/** Support pixels in the mask. */
export function runPixelCount(runs: MaskRuns): number {
  let n = 0;
  for (let k = 0; k < runs.x0.length; k++) n += runs.x1[k]! - runs.x0[k]!;
  return n;
}

/** Bounding box of a pixel set, exclusive ends (skimage's `bbox`). */
export interface PixelBox {
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
}

/** Bounding boxes of the mask's 4-connected components (runs of adjacent rows that overlap in
 *  x join), in order of each component's first run. */
export function componentBoxes(runs: MaskRuns): PixelBox[] {
  const n = runs.x0.length;
  const parent = new Uint32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (i: number): number => {
    let r = i;
    while (parent[r] !== r) r = parent[r]!;
    while (parent[i] !== r) {
      const next = parent[i]!;
      parent[i] = r;
      i = next;
    }
    return r;
  };
  const { rowStart, x0, x1 } = runs;
  for (let y = 0; y + 1 < runs.height; y++) {
    // Two-pointer sweep over the runs of rows y and y + 1: overlapping x ranges share an edge.
    let a = rowStart[y]!;
    let b = rowStart[y + 1]!;
    const aEnd = rowStart[y + 1]!;
    const bEnd = rowStart[y + 2]!;
    while (a < aEnd && b < bEnd) {
      if (x0[a]! < x1[b]! && x0[b]! < x1[a]!) {
        const ra = find(a);
        const rb = find(b);
        if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
      }
      if (x1[a]! < x1[b]!) a++;
      else b++;
    }
  }
  const boxOf = new Map<number, { x0: number; y0: number; x1: number; y1: number }>();
  for (let y = 0; y < runs.height; y++) {
    for (let k = rowStart[y]!; k < rowStart[y + 1]!; k++) {
      const r = find(k);
      const b = boxOf.get(r);
      if (!b) boxOf.set(r, { x0: x0[k]!, y0: y, x1: x1[k]!, y1: y + 1 });
      else {
        if (x0[k]! < b.x0) b.x0 = x0[k]!;
        if (x1[k]! > b.x1) b.x1 = x1[k]!;
        b.y1 = y + 1;
      }
    }
  }
  return [...boxOf.values()];
}

/** SIMBA's `_axis_starts`: patch starts along one axis for a component spanning `[lo, hi)`,
 *  each with whether it was added only to cover the trailing edge. */
export function axisStarts(lo: number, hi: number, extent: number, size: number, stride: number): [number, boolean][] {
  if (extent <= 0) return [];
  const maxStart = Math.max(extent - size, 0);
  if (hi - lo <= size) return [[Math.min(Math.max(hi - size, 0), maxStart), size > extent]];
  const first = Math.min(Math.max(lo, 0), maxStart);
  const last = Math.min(Math.max(hi - size, 0), maxStart);
  if (last < first) return [[last, false]];
  const starts: [number, boolean][] = [];
  for (let s = first; s <= last; s += stride) starts.push([s, false]);
  if (starts.length > 0 && starts[starts.length - 1]![0] !== last) starts.push([last, true]);
  return starts;
}

/** One grid patch: a half-open pixel rectangle `[x0, x1) × [y0, y1)`. */
export interface PatchRect extends PixelBox {
  /** Index among every candidate in (y0, x0) order: SIMBA's `patch_%05d`. */
  readonly id: number;
  /** Placed only to cover a component's trailing edge (or the patch is larger than the image). */
  readonly edge: boolean;
}

/** Every candidate patch over the footprint's components, deduplicated, in (y0, x0) order. */
export function patchCandidates(boxes: readonly PixelBox[], width: number, height: number, size: number, stride: number): PatchRect[] {
  if (!(Number.isInteger(size) && size > 0)) throw new Error(`patchGrid: size must be a positive integer of pixels (got ${size})`);
  if (!(Number.isInteger(stride) && stride > 0)) throw new Error(`patchGrid: stride must be a positive integer of pixels (got ${stride})`);
  // end = min(start + size, extent) is fixed by the start, so the start pair is the key.
  const edgeAt = new Map<number, boolean>();
  for (const b of boxes) {
    const ys = axisStarts(b.y0, b.y1, height, size, stride);
    const xs = axisStarts(b.x0, b.x1, width, size, stride);
    for (const [sy, ey] of ys) {
      for (const [sx, ex] of xs) {
        const key = sy * width + sx;
        edgeAt.set(key, (edgeAt.get(key) ?? false) || ey || ex);
      }
    }
  }
  const keys = [...edgeAt.keys()].sort((a, b) => a - b);
  return keys.map((key, id) => {
    const y0 = Math.floor(key / width);
    const x0 = key - y0 * width;
    return { id, x0, y0, x1: Math.min(x0 + size, width), y1: Math.min(y0 + size, height), edge: edgeAt.get(key)! };
  });
}

/** Prefix sum of run lengths: `cum[k]` support pixels in runs before run `k` (any row). */
function runPrefix(runs: MaskRuns): Float64Array {
  const cum = new Float64Array(runs.x0.length + 1);
  for (let k = 0; k < runs.x0.length; k++) cum[k + 1] = cum[k]! + runs.x1[k]! - runs.x0[k]!;
  return cum;
}

/** First index in `a[lo, hi)` whose value is >= `v` (all of them sorted ascending). */
function lowerBound(a: ArrayLike<number>, lo: number, hi: number, v: number): number {
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid]! < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Support pixels of `runs` inside each rectangle. */
export function maskCountsInRects(runs: MaskRuns, rects: readonly PixelBox[]): Float64Array {
  const cum = runPrefix(runs);
  const { rowStart, x0, x1 } = runs;
  // Support pixels of row y left of column X: whole runs that start before X, less the part of
  // the last one at or past X.
  const left = (a: number, b: number, X: number): number => {
    const j = lowerBound(x0, a, b, X); // runs a..j-1 start before X
    if (j === a) return 0;
    return cum[j]! - cum[a]! - Math.max(0, x1[j - 1]! - X);
  };
  const out = new Float64Array(rects.length);
  rects.forEach((r, i) => {
    if (r.x1 > runs.width || r.y1 > runs.height || r.x0 < 0 || r.y0 < 0) throw new Error("maskCountsInRects: rectangle leaves the canvas");
    let s = 0;
    for (let y = r.y0; y < r.y1; y++) {
      const a = rowStart[y]!;
      const b = rowStart[y + 1]!;
      if (a !== b) s += left(a, b, r.x1) - left(a, b, r.x0);
    }
    out[i] = s;
  });
  return out;
}

/** numpy's `round`: half to even. */
export function roundHalfEven(v: number): number {
  const f = Math.floor(v);
  const d = v - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

/** Points binned to pixels by row: row `y`'s pixel columns, ascending, are
 *  `cols[rowStart[y] .. rowStart[y + 1])`. */
export interface PointPixels {
  readonly width: number;
  readonly height: number;
  readonly rowStart: Uint32Array;
  readonly cols: Uint32Array;
}

/** Bin points (pixel coordinates `x` = column, `y` = row, pixel centres at integers) to the
 *  pixel each rounds to, half to even and clamped to the canvas: SIMBA's `centroid_pixels`. */
export function pointPixels(xs: ArrayLike<number>, ys: ArrayLike<number>, width: number, height: number): PointPixels {
  const n = xs.length;
  if (ys.length !== n) throw new Error("pointPixels: xs and ys differ in length");
  const row = new Uint32Array(n);
  const col = new Uint32Array(n);
  const rowStart = new Uint32Array(height + 1);
  for (let i = 0; i < n; i++) {
    row[i] = Math.min(Math.max(roundHalfEven(ys[i]!), 0), height - 1);
    col[i] = Math.min(Math.max(roundHalfEven(xs[i]!), 0), width - 1);
    rowStart[row[i]! + 1]!++;
  }
  for (let y = 0; y < height; y++) rowStart[y + 1]! += rowStart[y]!;
  const fill = rowStart.slice(0, height);
  const cols = new Uint32Array(n);
  for (let i = 0; i < n; i++) cols[fill[row[i]!]!++] = col[i]!;
  for (let y = 0; y < height; y++) cols.subarray(rowStart[y]!, rowStart[y + 1]!).sort();
  return { width, height, rowStart, cols };
}

/** Points inside each rectangle. */
export function pointCountsInRects(points: PointPixels, rects: readonly PixelBox[]): Float64Array {
  const { rowStart, cols } = points;
  return Float64Array.from(rects, (r) => {
    let s = 0;
    for (let y = r.y0; y < r.y1; y++) {
      const a = rowStart[y]!;
      const b = rowStart[y + 1]!;
      if (a !== b) s += lowerBound(cols, a, b, r.x1) - lowerBound(cols, a, b, r.x0);
    }
    return s;
  });
}

/** A minimum-support rule. A patch is kept when every rule holds: the value is finite and at
 *  least `min` (SIMBA's `filter_supported`, and its `< min` failure flags). */
export type SupportRule =
  | { readonly kind: "fraction"; readonly name: string; readonly mask: MaskRuns; readonly min: number }
  | { readonly kind: "count"; readonly name: string; readonly points: PointPixels; readonly min: number };

export interface PatchGridOptions {
  /** The mask the grid is laid over: its components anchor the per-component grids. */
  readonly footprint: MaskRuns;
  /** Patch side and stride, in `units`. */
  readonly size: number;
  readonly stride: number;
  /** "px" (default) or "um"; µm need `pixelSize`. */
  readonly units?: "px" | "um";
  /** µm per pixel, for `units: "um"`. */
  readonly pixelSize?: number;
  readonly rules?: readonly SupportRule[];
}

/** A candidate patch with its support values (one per rule, in rule order) and verdict. */
export interface Patch extends PatchRect {
  readonly support: readonly number[];
  readonly kept: boolean;
}

export interface PatchGrid {
  readonly width: number;
  readonly height: number;
  readonly sizePx: number;
  readonly stridePx: number;
  /** Rule names, the order of every patch's `support`. */
  readonly rules: readonly string[];
  /** Every candidate, kept or not (SIMBA's `keep_invalid` table). */
  readonly candidates: readonly Patch[];
  /** The kept candidates, in id order. */
  readonly patches: readonly Patch[];
}

/** A length in `units` as whole pixels. Non-integer pixel counts round (SIMBA's µm are px at
 *  1 µm/px, so its sizes are always whole). */
export function toPixels(length: number, units: "px" | "um" = "px", pixelSize?: number): number {
  let px = length;
  if (units === "um") {
    if (!(pixelSize !== undefined && pixelSize > 0)) throw new Error("patchGrid: units 'um' need a positive pixelSize");
    px = length / pixelSize;
  }
  const r = Math.round(px);
  if (r < 1) throw new Error(`patchGrid: ${length} ${units} is less than one pixel`);
  return r;
}

/** Lay the grid over `footprint`, measure every rule on every candidate, keep the supported. */
export function patchGrid(opts: PatchGridOptions): PatchGrid {
  const { footprint } = opts;
  const { width, height } = footprint;
  const sizePx = toPixels(opts.size, opts.units, opts.pixelSize);
  const stridePx = toPixels(opts.stride, opts.units, opts.pixelSize);
  const rules = opts.rules ?? [];
  for (const r of rules) {
    const src = r.kind === "fraction" ? r.mask : r.points;
    if (src.width !== width || src.height !== height)
      throw new Error(`patchGrid: rule '${r.name}' is ${src.width}x${src.height}, footprint ${width}x${height}`);
  }
  const rects = patchCandidates(componentBoxes(footprint), width, height, sizePx, stridePx);
  const values = rules.map((r) => {
    if (r.kind === "count") return pointCountsInRects(r.points, rects);
    const counts = maskCountsInRects(r.mask, rects);
    return counts.map((c, i) => c / ((rects[i]!.x1 - rects[i]!.x0) * (rects[i]!.y1 - rects[i]!.y0)));
  });
  const candidates = rects.map((rect, i): Patch => {
    const support = values.map((v) => v[i]!);
    const kept = support.every((v, k) => Number.isFinite(v) && v >= rules[k]!.min);
    return { ...rect, support, kept };
  });
  return {
    width,
    height,
    sizePx,
    stridePx,
    rules: rules.map((r) => r.name),
    candidates,
    patches: candidates.filter((p) => p.kept),
  };
}

/**
 * Patches as R3's region polygons: `zonalStatsTiled({ regions: { kind: "polygons", polygons } })`
 * then has row k = patch k. R3 counts a pixel in a polygon when its centre (i + ½, j + ½) is
 * inside, so each rectangle's corners at its integer bounds cover exactly its pixels
 * `[x0, x1) × [y0, y1)`, the pixels its support was counted over. R3 splits overlapping polygons
 * into layers itself (`polygonLayers`, the same greedy split as `patchLayers`).
 */
export function patchPolygons(rects: readonly PixelBox[]): PolygonDomain[] {
  return rects.map((r) =>
    polygonDomain([
      [r.x0, r.y0],
      [r.x1, r.y0],
      [r.x1, r.y1],
      [r.x0, r.y1],
    ]),
  );
}

/** Rectangles as `n × 4` u32 `[x0, y0, x1, y1]` rows: the GPU support and zonal input. */
export function rectsArray(rects: readonly PixelBox[]): Uint32Array {
  const out = new Uint32Array(rects.length * 4);
  rects.forEach((r, i) => out.set([r.x0, r.y0, r.x1, r.y1], i * 4));
  return out;
}

const overlaps = (a: PixelBox, b: PixelBox): boolean => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

/**
 * Split rectangles into layers whose members do not overlap, so each layer is one label image.
 * Greedy in input order: each rectangle joins the first layer it overlaps nothing in. A lattice
 * with `size = k · stride` gives k² layers (4 at SIMBA's half overlap); a component's trailing
 * edge patches and patches of differently anchored components can add a few more. Returns the
 * indices (into `rects`) of each layer, ascending.
 */
export function patchLayers(rects: readonly PixelBox[]): number[][] {
  if (rects.length === 0) return [];
  // A bucket grid of the largest side per layer, so the overlap test looks at neighbours only.
  let cell = 1;
  for (const r of rects) cell = Math.max(cell, r.x1 - r.x0, r.y1 - r.y0);
  const layers: { members: number[]; buckets: Map<number, number[]> }[] = [];
  const bucketsOf = (r: PixelBox): number[] => {
    const keys: number[] = [];
    for (let by = Math.floor(r.y0 / cell); by <= Math.floor((r.y1 - 1) / cell); by++)
      for (let bx = Math.floor(r.x0 / cell); bx <= Math.floor((r.x1 - 1) / cell); bx++) keys.push(by * 1048576 + bx);
    return keys;
  };
  rects.forEach((r, i) => {
    const keys = bucketsOf(r);
    let layer = layers.find((l) => keys.every((k) => (l.buckets.get(k) ?? []).every((j) => !overlaps(rects[j]!, r))));
    if (!layer) {
      layer = { members: [], buckets: new Map() };
      layers.push(layer);
    }
    layer.members.push(i);
    for (const k of keys) {
      const b = layer.buckets.get(k);
      if (b) b.push(i);
      else layer.buckets.set(k, [i]);
    }
  });
  return layers.map((l) => l.members);
}

/**
 * A u32 label image of one layer, in R3's label convention (label L is region L − 1): pixel =
 * (index into `rects`) + 1 of the rectangle holding it,
 * 0 outside every one. `scale` > 1 gives a canvas `scale` times coarser (ceil), each label pixel
 * taking the rectangle that holds its centre, for zonal stats on a downsampled raster. The
 * members of `layer` must not overlap (`patchLayers`); an overlapping later one would win.
 */
export function patchLabelImage(
  rects: readonly PixelBox[],
  layer: readonly number[],
  width: number,
  height: number,
  scale = 1,
): Uint32Array {
  if (!(Number.isInteger(scale) && scale >= 1)) throw new Error(`patchLabelImage: scale must be an integer >= 1 (got ${scale})`);
  const w = Math.ceil(width / scale);
  const h = Math.ceil(height / scale);
  const out = new Uint32Array(w * h);
  // Label pixel j covers fine pixels [j·s, (j+1)·s) and takes the rectangle holding j·s + s/2:
  // the first j with j·s + s/2 >= lo, i.e. j >= (lo - s/2) / s.
  const first = (lo: number) => Math.max(0, Math.ceil((lo - scale / 2) / scale));
  for (const i of layer) {
    const r = rects[i]!;
    const jx0 = first(r.x0);
    const jx1 = Math.min(w, first(r.x1));
    for (let jy = first(r.y0); jy < Math.min(h, first(r.y1)); jy++) out.fill(i + 1, jy * w + jx0, jy * w + Math.max(jx0, jx1));
  }
  return out;
}
