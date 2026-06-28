// Points -> 2D scalar field, on the CPU. Two fields, both consumed by the same
// cubical sublevel-set persistence reducer (`sublevelsetPersistence.ts`):
//
//   • `gaussianKdeField`  — a smooth (C-infinity) kernel-density estimate. Its
//     *super*level sets {KDE >= t} are the "fuzzy" union-of-Gaussians: smooth
//     boundaries, no corners. This is the CPU twin of the GPU `splatDensity`
//     primitive (same additive-Gaussian maths), so it also serves as that
//     primitive's golden, and lets the docs demo build a KDE field with no GPU.
//
//   • `distanceField` — the distance-to-nearest-point function d(x) = min_i |x-p_i|.
//     Its *sub*level sets {d <= r} are exactly the union of radius-r balls — the
//     hard-ball (Cech-style) filtration. The ball union has non-smooth boundaries
//     (corners where two balls meet), which is what spawns the spurious, short-lived
//     features the fuzzy field avoids.
//
// Putting both behind one field type means the difference between "hard" and
// "fuzzy" topology is just *which field you sublevel-set*, reduced by one shared
// primitive — see the docs "Fuzzy filtrations" demo.

export interface ScalarField {
  /** Row-major width*height samples. Row 0 is the TOP of the bbox (worldY = maxY),
   *  matching the GPU `splatDensity` convention. */
  data: Float32Array;
  width: number;
  height: number;
  /** World extent [minX, minY, maxX, maxY] the grid spans (cell centres inset). */
  bbox: [number, number, number, number];
}

export interface FieldOptions {
  /** Output grid resolution. */
  width: number;
  height: number;
  /** World bounds [minX, minY, maxX, maxY]. Default: the points' bounds, padded. */
  bbox?: [number, number, number, number];
}

/** World coordinate of the centre of grid cell (col, row). Row 0 = top (maxY). */
export function cellCenter(
  field: Pick<ScalarField, "width" | "height" | "bbox">,
  col: number,
  row: number,
): [number, number] {
  const [minX, minY, maxX, maxY] = field.bbox;
  const x = minX + ((col + 0.5) / field.width) * (maxX - minX);
  const y = maxY - ((row + 0.5) / field.height) * (maxY - minY);
  return [x, y];
}

function resolveBbox(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  n: number,
  pad: number,
  bbox?: [number, number, number, number],
): [number, number, number, number] {
  if (bbox) return bbox;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = xs[i]!, y = ys[i]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (!isFinite(minX)) { minX = 0; minY = 0; maxX = 1; maxY = 1; }
  return [minX - pad, minY - pad, maxX + pad, maxY + pad];
}

export interface KdeOptions extends FieldOptions {
  /** Gaussian bandwidth in WORLD units. */
  sigma: number;
  /** Kernel support half-extent in units of sigma (contributions past it are
   *  dropped — see distance-decay). Default 4 (captures ~all the mass). */
  radiusSigma?: number;
  /** Per-point weights (default 1 each). */
  weights?: ArrayLike<number>;
}

/**
 * Gaussian kernel-density field: data[cell] = sum_i w_i * exp(-|c - p_i|^2 / 2sigma^2).
 *
 * The CPU twin of `gpu/spatial/splatDensity` (additive un-normalised Gaussians),
 * truncated at `radiusSigma` for speed. Its superlevel sets are the smooth
 * "fuzzy balls" of the kernel filtration.
 */
export function gaussianKdeField(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  opts: KdeOptions,
): ScalarField {
  const n = xs.length;
  if (ys.length !== n) throw new Error("gaussianKdeField: xs/ys length mismatch");
  const { width: w, height: h, sigma } = opts;
  if (w <= 0 || h <= 0) throw new Error("gaussianKdeField: width/height must be > 0");
  if (sigma <= 0) throw new Error("gaussianKdeField: sigma must be > 0");
  const radiusSigma = opts.radiusSigma ?? 4;
  const support = sigma * radiusSigma;
  const bbox = resolveBbox(xs, ys, n, support, opts.bbox);
  const [minX, minY, maxX, maxY] = bbox;
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const inv2s2 = 1 / (2 * sigma * sigma);

  const data = new Float32Array(w * h);
  // Splat each point into the cells within its square support (the O(N*k) path).
  for (let i = 0; i < n; i++) {
    const px = xs[i]!, py = ys[i]!;
    const wi = opts.weights ? opts.weights[i]! : 1;
    // World support box -> cell-index range. col grows with +X, row grows with -Y.
    const cLo = Math.max(0, Math.floor(((px - support - minX) / spanX) * w - 0.5));
    const cHi = Math.min(w - 1, Math.ceil(((px + support - minX) / spanX) * w - 0.5));
    const rLo = Math.max(0, Math.floor(((maxY - (py + support)) / spanY) * h - 0.5));
    const rHi = Math.min(h - 1, Math.ceil(((maxY - (py - support)) / spanY) * h - 0.5));
    for (let r = rLo; r <= rHi; r++) {
      const cy = maxY - ((r + 0.5) / h) * spanY;
      const dy = cy - py;
      for (let c = cLo; c <= cHi; c++) {
        const cx = minX + ((c + 0.5) / w) * spanX;
        const dx = cx - px;
        const d2 = dx * dx + dy * dy;
        if (d2 > support * support) continue;
        const idx = r * w + c;
        data[idx] = data[idx]! + wi * Math.exp(-d2 * inv2s2);
      }
    }
  }
  return { data, width: w, height: h, bbox };
}

export interface DtmOptions extends FieldOptions {
  /** Number of nearest points averaged. Larger k = more smoothing / robustness.
   *  Clamped to the point count. Default 5. */
  k?: number;
  /** Padding added around the points' bounds when `bbox` is omitted. Default 0. */
  pad?: number;
}

/**
 * Distance-to-measure (DTM) field of Chazal-Cohen-Steiner-Merigot:
 *   dtm_k(x) = sqrt( mean of the k smallest |x - p_i|^2 ).
 *
 * This is the *robust* relative of `distanceField`: an outlier has only itself
 * nearby, so the k-average keeps the field high there and the point enters the
 * sublevel filtration late (a short, near-diagonal bar) instead of spawning a
 * full-strength ball at r=0. The persistent homology of its sublevel sets is
 * provably stable to outliers (bottleneck/Wasserstein), unlike Cech/Vietoris-Rips.
 *
 * (k = 1 recovers `distanceField`.) Brute force, O(grid * N); for the small N the
 * demos use.
 */
export function dtmField(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  opts: DtmOptions,
): ScalarField {
  const n = xs.length;
  if (ys.length !== n) throw new Error("dtmField: xs/ys length mismatch");
  const { width: w, height: h } = opts;
  if (w <= 0 || h <= 0) throw new Error("dtmField: width/height must be > 0");
  const k = Math.max(1, Math.min(opts.k ?? 5, n || 1));
  const bbox = resolveBbox(xs, ys, n, opts.pad ?? 0, opts.bbox);
  const [minX, minY, maxX, maxY] = bbox;
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const d2 = new Float64Array(Math.max(n, 1)); // scratch squared distances per cell
  const data = new Float32Array(w * h);
  for (let r = 0; r < h; r++) {
    const cy = maxY - ((r + 0.5) / h) * spanY;
    for (let c = 0; c < w; c++) {
      const cx = minX + ((c + 0.5) / w) * spanX;
      for (let i = 0; i < n; i++) {
        const dx = cx - xs[i]!, dy = cy - ys[i]!;
        d2[i] = dx * dx + dy * dy;
      }
      // Mean of the k smallest squared distances (partial selection sort — k small).
      let sum = 0;
      for (let s = 0; s < k; s++) {
        let m = s;
        for (let i = s + 1; i < n; i++) if (d2[i]! < d2[m]!) m = i;
        const tmp = d2[s]!; d2[s] = d2[m]!; d2[m] = tmp;
        sum += d2[s]!;
      }
      data[r * w + c] = Math.sqrt(sum / k);
    }
  }
  return { data, width: w, height: h, bbox };
}

/**
 * Distance-to-nearest-point field: data[cell] = min_i |c - p_i| (brute force).
 *
 * Its sublevel set {d <= r} is the union of radius-r balls — the hard-ball
 * (Cech) filtration. The contrast partner of `gaussianKdeField`.
 */
export function distanceField(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  opts: FieldOptions & { pad?: number },
): ScalarField {
  const n = xs.length;
  if (ys.length !== n) throw new Error("distanceField: xs/ys length mismatch");
  const { width: w, height: h } = opts;
  if (w <= 0 || h <= 0) throw new Error("distanceField: width/height must be > 0");
  const bbox = resolveBbox(xs, ys, n, opts.pad ?? 0, opts.bbox);
  const [minX, minY, maxX, maxY] = bbox;
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const data = new Float32Array(w * h);
  for (let r = 0; r < h; r++) {
    const cy = maxY - ((r + 0.5) / h) * spanY;
    for (let c = 0; c < w; c++) {
      const cx = minX + ((c + 0.5) / w) * spanX;
      let best = Infinity;
      for (let i = 0; i < n; i++) {
        const dx = cx - xs[i]!, dy = cy - ys[i]!;
        const d2 = dx * dx + dy * dy;
        if (d2 < best) best = d2;
      }
      data[r * w + c] = Math.sqrt(best);
    }
  }
  return { data, width: w, height: h, bbox };
}
