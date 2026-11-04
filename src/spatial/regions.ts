// Region primitives on a raster: the exact Euclidean distance transform, the connected piece holding a
// seed, and the distance from every pixel to the mixture at a seed. Each is a step of the wand's patch
// (docs/wand-metric-and-regions.md), and each is general: an opening by a disc is two distance
// transforms and two thresholds, a flood fill is a selection.

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

/** Euclidean distance (pixels, between pixel centres) from each pixel to the nearest pixel where
 *  `target` holds; about 1e6 everywhere when it holds nowhere. */
export function distanceTo(target: (i: number) => boolean, width: number, height: number): Float64Array {
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

/**
 * The distance transform of a raster: each pixel's distance to the nearest pixel below 0.5 (0 on
 * those pixels). With `border`, the ring of pixels just outside the raster counts as below too, so
 * a region is measured from the raster's edge as well as from its own.
 */
export function distanceTransform(src: ArrayLike<number>, width: number, height: number, border = false): Float32Array {
  const out = new Float32Array(width * height);
  if (!border) {
    const d = distanceTo((i) => src[i]! < 0.5, width, height);
    for (let i = 0; i < out.length; i++) out[i] = d[i]!;
    return out;
  }
  const pw = width + 2;
  const padded = distanceTo(
    (i) => {
      const x = (i % pw) - 1;
      const y = Math.floor(i / pw) - 1;
      return x < 0 || y < 0 || x >= width || y >= height || src[y * width + x]! < 0.5;
    },
    pw,
    height + 2,
  );
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) out[y * width + x] = padded[(y + 1) * pw + x + 1]!;
  return out;
}

/** The pixel nearest a fractional (col, row), clamped to the raster. */
export function nearestPixel(width: number, height: number, col: number, row: number): number {
  return Math.max(0, Math.min(height - 1, Math.round(row))) * width + Math.max(0, Math.min(width - 1, Math.round(col)));
}

/**
 * The 4-connected piece of the pixels at or above 0.5 that holds the pixel nearest (col, row), as 0/1.
 * Empty when that pixel is not in the set, unless `seedIn`, which puts it in regardless (a flood that
 * always starts at the seed).
 */
export function floodFrom(src: ArrayLike<number>, width: number, height: number, col: number, row: number, seedIn = false): Float32Array {
  const n = width * height;
  const out = new Float32Array(n);
  const start = nearestPixel(width, height, col, row);
  if (!seedIn && !(src[start]! >= 0.5)) return out;
  const queue = new Int32Array(n);
  let head = 0;
  let tail = 0;
  out[start] = 1;
  queue[tail++] = start;
  while (head < tail) {
    const i = queue[head++]!;
    const x = i % width;
    for (const j of [x > 0 ? i - 1 : -1, x < width - 1 ? i + 1 : -1, i - width, i < n - width ? i + width : -1]) {
      if (j >= 0 && !out[j] && src[j]! >= 0.5) {
        out[j] = 1;
        queue[tail++] = j;
      }
    }
  }
  return out;
}

/**
 * The Euclidean distance from every pixel's mixture to the mixture at (col, row), over `K` planes
 * (plane-major: plane k is `u[k·n … (k+1)·n)`). The reference is interpolated bilinearly at the
 * fractional position, so it moves smoothly with the seed rather than stepping at pixel crossings.
 * In a whitened metric (u = A·z) this is the wand's distance d.
 */
export function mixtureDistance(u: ArrayLike<number>, K: number, width: number, height: number, col: number, row: number): Float32Array {
  const n = width * height;
  const fx = Math.max(0, Math.min(width - 1, col));
  const fy = Math.max(0, Math.min(height - 1, row));
  const x0 = Math.max(0, Math.min(width - 2, Math.floor(fx)));
  const y0 = Math.max(0, Math.min(height - 2, Math.floor(fy)));
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const tx = fx - x0;
  const ty = fy - y0;
  const corners = [
    [y0 * width + x0, (1 - tx) * (1 - ty)],
    [y0 * width + x1, tx * (1 - ty)],
    [y1 * width + x0, (1 - tx) * ty],
    [y1 * width + x1, tx * ty],
  ] as const;
  const out = new Float32Array(n);
  for (let k = 0; k < K; k++) {
    let ref = 0;
    for (const [i, w] of corners) ref += w * u[k * n + i]!;
    for (let i = 0; i < n; i++) {
      const dv = u[k * n + i]! - ref;
      out[i]! += dv * dv;
    }
  }
  for (let i = 0; i < n; i++) out[i] = Math.sqrt(out[i]!);
  return out;
}
