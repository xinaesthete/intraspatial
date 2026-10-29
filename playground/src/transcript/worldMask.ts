// PROTOTYPE: a selection mask that outlives the window it was drawn in. Each window's map is rebuilt
// on every pan or zoom, so the mask lives on its own fixed grid in the transcripts' frame: a set of
// cells `cell` µm wide, keyed by their integer position. Patches computed on a window's map are
// carried onto it, combined Photoshop-style, and carried back onto whatever map is showing. A drag is
// one stroke: its patches are unioned, then combined with the mask as it was when the stroke began.

/** The window map a patch was computed on: row 0 at bbox maxY, as `GramMatrixGpuResult`. */
export interface Raster {
  readonly width: number;
  readonly height: number;
  readonly bbox: readonly [number, number, number, number];
}

export interface WorldMask {
  /** Grid spacing in the transcripts' frame. */
  readonly cell: number;
  readonly keys: ReadonlySet<number>;
}

export type MaskOp = "replace" | "add" | "subtract" | "intersect";

const OFFSET = 2 ** 20;
const key = (ix: number, iy: number) => (ix + OFFSET) * 2 ** 21 + (iy + OFFSET);

/** Cells whose centres fall in a span; the cell holding its middle when none do (a span narrower than a cell). */
function span(lo: number, hi: number, cell: number): [number, number] {
  const a = Math.ceil(lo / cell - 0.5);
  const b = Math.floor(hi / cell - 0.5);
  if (b >= a) return [a, b];
  const m = Math.floor((lo + hi) / 2 / cell);
  return [m, m];
}

/** The grid cells covered by the pixels of `patch` (nonzero = inside) on `res`. */
export function footprint(patch: ArrayLike<number>, res: Raster, cell: number): Set<number> {
  const { width, height, bbox } = res;
  const [minX, minY, maxX, maxY] = bbox;
  const px = (maxX - minX) / width;
  const py = (maxY - minY) / height;
  const out = new Set<number>();
  for (let row = 0; row < height; row++) {
    const [y0, y1] = span(maxY - (row + 1) * py, maxY - row * py, cell);
    for (let col = 0; col < width; col++) {
      if (!patch[row * width + col]) continue;
      const [x0, x1] = span(minX + col * px, minX + (col + 1) * px, cell);
      for (let iy = y0; iy <= y1; iy++) for (let ix = x0; ix <= x1; ix++) out.add(key(ix, iy));
    }
  }
  return out;
}

/** Add to `out` the grid cells whose centres lie within `r` of the segment from `a` to `b` (a brush
 *  dab when `a` and `b` coincide): a stroke is swept, so a fast drag leaves no gaps. */
export function sweep(out: Set<number>, a: readonly [number, number], b: readonly [number, number], r: number, cell: number): void {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  const x0 = Math.floor((Math.min(a[0], b[0]) - r) / cell);
  const x1 = Math.floor((Math.max(a[0], b[0]) + r) / cell);
  const y0 = Math.floor((Math.min(a[1], b[1]) - r) / cell);
  const y1 = Math.floor((Math.max(a[1], b[1]) + r) / cell);
  for (let iy = y0; iy <= y1; iy++) {
    const cy = (iy + 0.5) * cell;
    for (let ix = x0; ix <= x1; ix++) {
      const cx = (ix + 0.5) * cell;
      const t = len2 > 0 ? Math.max(0, Math.min(1, ((cx - a[0]) * dx + (cy - a[1]) * dy) / len2)) : 0;
      const ex = cx - (a[0] + t * dx);
      const ey = cy - (a[1] + t * dy);
      if (ex * ex + ey * ey <= r * r) out.add(key(ix, iy));
    }
  }
}

/** The grid a new mask takes: the pixel size of the window map it starts on. */
export const gridFor = (res: Raster): number => (res.bbox[2] - res.bbox[0]) / res.width;

/** `base` combined with a stroke (the union of the patches painted since the button went down). */
export function apply(base: WorldMask | undefined, stroke: ReadonlySet<number>, cell: number, op: MaskOp): WorldMask {
  if (!base || op === "replace") return { cell, keys: new Set(stroke) };
  const keys = new Set(base.keys);
  if (op === "add") for (const k of stroke) keys.add(k);
  else if (op === "subtract") for (const k of stroke) keys.delete(k);
  else for (const k of base.keys) if (!stroke.has(k)) keys.delete(k);
  return { cell: base.cell, keys };
}

/** The mask sampled at each pixel centre of `res`, 1 inside. */
export function onRaster(mask: WorldMask, res: Raster): Float32Array {
  const { width, height, bbox } = res;
  const [minX, minY, maxX, maxY] = bbox;
  const px = (maxX - minX) / width;
  const py = (maxY - minY) / height;
  const out = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    const iy = Math.floor((maxY - (row + 0.5) * py) / mask.cell);
    for (let col = 0; col < width; col++) {
      const ix = Math.floor((minX + (col + 0.5) * px) / mask.cell);
      if (mask.keys.has(key(ix, iy))) out[row * width + col] = 1;
    }
  }
  return out;
}

/** A disc sampled at each pixel centre of `res`, 1 inside: the brush's preview on a window's map. */
export function discOnRaster(c: readonly [number, number], r: number, res: Raster): Float32Array {
  const { width, height, bbox } = res;
  const [minX, minY, maxX, maxY] = bbox;
  const px = (maxX - minX) / width;
  const py = (maxY - minY) / height;
  const out = new Float32Array(width * height);
  const c0 = Math.max(0, Math.floor((c[0] - r - minX) / px));
  const c1 = Math.min(width - 1, Math.ceil((c[0] + r - minX) / px));
  const r0 = Math.max(0, Math.floor((maxY - c[1] - r) / py));
  const r1 = Math.min(height - 1, Math.ceil((maxY - c[1] + r) / py));
  for (let row = r0; row <= r1; row++) {
    const dy = maxY - (row + 0.5) * py - c[1];
    for (let col = c0; col <= c1; col++) {
      const dx = minX + (col + 0.5) * px - c[0];
      if (dx * dx + dy * dy <= r * r) out[row * width + col] = 1;
    }
  }
  return out;
}

/** Squared distance to the nearest zero of `f` along one line (Felzenszwalb & Huttenlocher's lower
 *  envelope of parabolas); `f` is 0 at targets and huge elsewhere. Writes into `d`. */
function edt1(f: Float64Array, n: number, d: Float64Array, v: Int32Array, z: Float64Array): void {
  let k = 0;
  v[0] = 0;
  z[0] = -Infinity;
  z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s = (f[q]! + q * q - (f[v[k]!]! + v[k]! * v[k]!)) / (2 * q - 2 * v[k]!);
    while (s <= z[k]!) {
      k--;
      s = (f[q]! + q * q - (f[v[k]!]! + v[k]! * v[k]!)) / (2 * q - 2 * v[k]!);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1]! < q) k++;
    d[q] = (q - v[k]!) * (q - v[k]!) + f[v[k]!]!;
  }
}

/** Euclidean distance (pixels) from each pixel to the nearest pixel where `target` holds. */
function distanceTo(target: (i: number) => boolean, width: number, height: number): Float64Array {
  const BIG = 1e12;
  const n = Math.max(width, height);
  const f = new Float64Array(n);
  const d = new Float64Array(n);
  const v = new Int32Array(n);
  const z = new Float64Array(n + 1);
  const g = new Float64Array(width * height);
  for (let i = 0; i < width * height; i++) g[i] = target(i) ? 0 : BIG;
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) f[y] = g[y * width + x]!;
    edt1(f, height, d, v, z);
    for (let y = 0; y < height; y++) g[y * width + x] = d[y]!;
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) f[x] = g[y * width + x]!;
    edt1(f, width, d, v, z);
    for (let x = 0; x < width; x++) g[y * width + x] = Math.sqrt(d[x]!);
  }
  return g;
}

/** A 0/1 raster as a smooth field whose 0.5 level is the region's outline with the pixel staircase
 *  gone: the signed distance to the edge (half a pixel out from inside pixel centres), blurred at
 *  σ = 1 px, mapped linearly so ±3 px spans 0 to 1. For drawing only: what is selected is unchanged. */
export function smoothOutline(binary: ArrayLike<number>, width: number, height: number): Float32Array {
  const n = width * height;
  const toOut = distanceTo((i) => !binary[i], width, height);
  const toIn = distanceTo((i) => !!binary[i], width, height);
  const sd = new Float32Array(n);
  for (let i = 0; i < n; i++) sd[i] = binary[i] ? Math.min(toOut[i]!, 1e6) - 0.5 : 0.5 - Math.min(toIn[i]!, 1e6);
  // σ = 1 px, separable, edge-clamped.
  const w = [0.0044, 0.054, 0.242, 0.399, 0.242, 0.054, 0.0044];
  const tmp = new Float32Array(n);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      let a = 0;
      for (let k = -3; k <= 3; k++) a += w[k + 3]! * sd[y * width + Math.min(width - 1, Math.max(0, x + k))]!;
      tmp[y * width + x] = a;
    }
  const out = new Float32Array(n);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      let a = 0;
      for (let k = -3; k <= 3; k++) a += w[k + 3]! * tmp[Math.min(height - 1, Math.max(0, y + k)) * width + x]!;
      out[y * width + x] = Math.min(1, Math.max(0, 0.5 + a / 6));
    }
  return out;
}

/** The mask's area in the frame's units squared. */
export const maskArea = (mask: WorldMask): number => mask.keys.size * mask.cell * mask.cell;
