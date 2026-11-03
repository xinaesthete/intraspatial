// PROTOTYPE (docs/wand-metric-and-regions.md, experiment 1): where the channel mixture changes
// fastest, measured in the wand's metric, and the cell rules read off it. The field itself is now
// computed by a graph (`boundaryField.ts`, `wandField` in src/gpu/graph/wand.ts); what is left here is
// the metric and the rules the graph does not cover yet (ridges, two scales).
//
// The metric is the note's PPCA form, as a K×K matrix A so the field is one `u = A·z`: the leading
// q modes at 1/√λ_k, the rest at √w/σ with σ² the mean of their λ. q = K is today's "every mode",
// q = 3 with w = 0 is today's "3 modes". Boundary strength is the Di Zenzo structure tensor of u,
// G = Σ_k ∇u_k ∇u_kᵀ, and its larger eigenvalue's root: the most the whitened distance changes per
// step, in the steepest direction. It is scaled to "per kernel radius" so a θ survives a zoom.
//
// Experiment 1b: the modes after q are spatially rough, so a gradient amplifies them well beyond
// their share of the variance. Their gradient is therefore taken from a Gaussian-blurred copy of
// the channels (blurring z then projecting = blurring those modes, the projection being linear),
// at a σ set in kernel radii. The main modes keep the raw rasters' gradient.

/** The PPCA metric as a K×K matrix (row k = mode k · weight), and the noise floor it used. */
export function ppcaMetric(
  vectors: Float64Array,
  values: Float64Array,
  K: number,
  q: number,
  w: number,
): { A: Float32Array<ArrayBuffer>; sigma2: number } {
  const qq = Math.max(1, Math.min(q, K));
  // Same floor as whiteningMatrix, so q = K reproduces today's "every mode".
  const floor = Math.max(values[0] ?? 1, 1e-12) * 1e-6;
  let tail = 0;
  for (let k = qq; k < K; k++) tail += Math.max(values[k] ?? 0, 0);
  const sigma2 = qq < K ? Math.max(tail / (K - qq), floor) : 0;
  const A = new Float32Array(K * K);
  for (let k = 0; k < K; k++) {
    const weight = k < qq ? 1 / Math.sqrt(Math.max(values[k] ?? 0, floor)) : Math.sqrt(Math.max(0, w) / sigma2);
    for (let a = 0; a < K; a++) A[k * K + a] = weight * (vectors[k * K + a] ?? 0);
  }
  return { A, sigma2 };
}

export interface BoundaryField {
  /** Per raster pixel, row-major: strength, then the tensor's xx, yy, xy. */
  readonly field: Float32Array;
  /** The wand's coordinates u = A·z per pixel, mode-major (K planes of width×height), on the host. */
  readonly u: Float32Array;
  readonly K: number;
  /** Raster pixels per kernel radius. */
  readonly perR: number;
  /** Quantiles of the strength over the window, for the readout and a θ that means something. */
  readonly quantiles: { readonly p50: number; readonly p90: number; readonly p99: number; readonly max: number };
}

/** Bilinear strength at a raster position (pixel centres at whole numbers), edge-clamped. */
function strengthAt(f: Float32Array, width: number, height: number, x: number, y: number): number {
  const cx = Math.max(0, Math.min(width - 1, x));
  const cy = Math.max(0, Math.min(height - 1, y));
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = Math.min(x0 + 1, width - 1);
  const y1 = Math.min(y0 + 1, height - 1);
  const fx = cx - x0;
  const fy = cy - y0;
  const v = (c: number, r: number) => f[(r * width + c) * 4]!;
  return (v(x0, y0) * (1 - fx) + v(x1, y0) * fx) * (1 - fy) + (v(x0, y1) * (1 - fx) + v(x1, y1) * fx) * fy;
}

/**
 * Ridge pixels: the discrete zero-crossing of the strength's derivative across the edge (Cumani's
 * rule as Canny's non-maximum suppression). A pixel is on a ridge when its strength is at least θ and
 * no less than the strength one pixel either side along the tensor's leading direction. The
 * result is 8-connected lines, which a 4-connected fill cannot cross.
 */
export function ridges(f: Float32Array, width: number, height: number, theta: number): Uint8Array {
  const out = new Uint8Array(width * height);
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      const i = (r * width + c) * 4;
      const s = f[i]!;
      // Flat ground is no ridge, even when θ is 0 (a window whose 90th percentile is 0).
      if (s < theta || s <= 0) continue;
      const ang = 0.5 * Math.atan2(2 * f[i + 3]!, f[i + 1]! - f[i + 2]!);
      const nx = Math.cos(ang);
      const ny = Math.sin(ang);
      if (s >= strengthAt(f, width, height, c + nx, r + ny) && s >= strengthAt(f, width, height, c - nx, r - ny)) out[r * width + c] = 1;
    }
  }
  return out;
}

/** The 4-connected cell holding `seed` whose pixels are not ridge pixels: 1 inside, 0 outside. */
export function ridgeCell(ridge: Uint8Array, width: number, height: number, seed: { col: number; row: number }): Uint8Array {
  const cell = new Uint8Array(width * height);
  const c = Math.max(0, Math.min(width - 1, Math.round(seed.col)));
  const r = Math.max(0, Math.min(height - 1, Math.round(seed.row)));
  const start = r * width + c;
  if (ridge[start]) return cell;
  const queue = new Int32Array(width * height);
  let head = 0;
  let tail = 0;
  cell[start] = 1;
  queue[tail++] = start;
  const visit = (i: number) => {
    if (cell[i] === 0 && ridge[i] === 0) {
      cell[i] = 1;
      queue[tail++] = i;
    }
  };
  while (head < tail) {
    const i = queue[head++]!;
    const x = i % width;
    if (x > 0) visit(i - 1);
    if (x < width - 1) visit(i + 1);
    if (i >= width) visit(i - width);
    if (i < width * (height - 1)) visit(i + width);
  }
  return cell;
}

/** A separable Gaussian on a host raster, edge-replicated; σ in pixels (≤ 0 returns a copy). */
function gaussian(src: Float32Array, width: number, height: number, sigma: number): Float32Array {
  if (sigma <= 0) return src.slice();
  const r = Math.min(48, Math.ceil(3 * sigma));
  const w = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) sum += w[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma));
  for (let i = 0; i < w.length; i++) w[i]! /= sum;
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) acc += w[i + r]! * src[y * width + Math.max(0, Math.min(width - 1, x + i))]!;
      tmp[y * width + x] = acc;
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) acc += w[i + r]! * tmp[Math.max(0, Math.min(height - 1, y + i)) * width + x]!;
      out[y * width + x] = acc;
    }
  }
  return out;
}

/**
 * The Marr–Hildreth cell for one seed: the wand distance d to the seed's mixture, smoothed at
 * σ = `smoothR` radii, and the 4-connected region around the seed where its Laplacian is ≥ 0.
 * Inside, d is a bowl (curving up); the boundary is where the rise away from the seed's mixture
 * stops accelerating: the shoulder of the step, which is a closed curve because it is the zero set
 * of one scalar field. `tol` > 0 also stops the fill where d itself exceeds it (on the raw d,
 * so the gate means what the wand's tolerance means); 0 leaves it off.
 */
export function laplacianCell(
  f: BoundaryField,
  width: number,
  height: number,
  seed: { col: number; row: number },
  smoothR: number,
  tol = 0,
  /** false: a plain flood fill on `d ≤ tol` (the Laplacian test is skipped). */
  useLaplacian = true,
): { mask: Uint8Array; d: Float32Array; dEdge: number; gated: number } {
  const n = width * height;
  const { u, K } = f;
  const c = Math.max(0, Math.min(width - 1, Math.round(seed.col)));
  const r = Math.max(0, Math.min(height - 1, Math.round(seed.row)));
  const start = r * width + c;
  // The seed's mixture, interpolated bilinearly at the seed itself: read at the nearest pixel, it
  // would step every time the pointer crossed one (a third of r on the usual raster).
  const fx = Math.max(0, Math.min(width - 1, seed.col));
  const fy = Math.max(0, Math.min(height - 1, seed.row));
  const x0 = Math.min(width - 2, Math.floor(fx));
  const y0 = Math.min(height - 2, Math.floor(fy));
  const tx = width > 1 ? fx - x0 : 0;
  const ty = height > 1 ? fy - y0 : 0;
  const corners = [
    [Math.max(0, y0) * width + Math.max(0, x0), (1 - tx) * (1 - ty)],
    [Math.max(0, y0) * width + Math.min(width - 1, x0 + 1), tx * (1 - ty)],
    [Math.min(height - 1, y0 + 1) * width + Math.max(0, x0), (1 - tx) * ty],
    [Math.min(height - 1, y0 + 1) * width + Math.min(width - 1, x0 + 1), tx * ty],
  ] as const;
  const raw = new Float32Array(n);
  for (let k = 0; k < K; k++) {
    let ref = 0;
    for (const [i, wt] of corners) ref += wt * u[k * n + i]!;
    for (let i = 0; i < n; i++) {
      const dv = u[k * n + i]! - ref;
      raw[i]! += dv * dv;
    }
  }
  for (let i = 0; i < n; i++) raw[i] = Math.sqrt(raw[i]!);
  const d = gaussian(raw, width, height, smoothR * f.perR);
  const lap = (i: number) => {
    const x = i % width;
    const y = (i - x) / width;
    const at = (xx: number, yy: number) => d[Math.max(0, Math.min(height - 1, yy)) * width + Math.max(0, Math.min(width - 1, xx))]!;
    return at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4 * d[i]!;
  };
  const mask = new Uint8Array(n);
  const queue = new Int32Array(n);
  let head = 0;
  let tail = 0;
  mask[start] = 1;
  queue[tail++] = start;
  // The seed is in regardless: the smoothed bowl's floor can be flat to rounding.
  let edgeSum = 0;
  let edgeCount = 0;
  /** Edge pixels stopped by the gate rather than by the Laplacian. */
  let gated = 0;
  const visit = (i: number) => {
    if (mask[i] !== 0) return;
    const curving = !useLaplacian || lap(i) >= 0;
    const near = tol <= 0 || raw[i]! <= tol;
    if (curving && near) {
      mask[i] = 1;
      queue[tail++] = i;
    } else {
      mask[i] = 2; // a boundary pixel: seen, outside
      if (curving) gated++;
      edgeSum += raw[i]!;
      edgeCount++;
    }
  };
  while (head < tail) {
    const i = queue[head++]!;
    const x = i % width;
    if (x > 0) visit(i - 1);
    if (x < width - 1) visit(i + 1);
    if (i >= width) visit(i - width);
    if (i < n - width) visit(i + width);
  }
  for (let i = 0; i < n; i++) if (mask[i] === 2) mask[i] = 0;
  return { mask, d: raw, dEdge: edgeCount > 0 ? edgeSum / edgeCount : 0, gated: edgeCount > 0 ? gated / edgeCount : 0 };
}

/** 4-connected component of `set` holding `start`. */
function component(set: Uint8Array, width: number, height: number, start: number): Uint8Array {
  const out = new Uint8Array(width * height);
  if (!set[start]) return out;
  const queue = new Int32Array(width * height);
  let head = 0;
  let tail = 0;
  out[start] = 1;
  queue[tail++] = start;
  while (head < tail) {
    const i = queue[head++]!;
    const x = i % width;
    for (const j of [x > 0 ? i - 1 : -1, x < width - 1 ? i + 1 : -1, i - width, i + width]) {
      if (j >= 0 && j < width * height && set[j] && !out[j]) {
        out[j] = 1;
        queue[tail++] = j;
      }
    }
  }
  return out;
}

/**
 * Morphological opening of a cell by a disk of radius `rho` pixels (erode, then dilate), then the
 * piece the seed belongs to. Opening removes every part of the cell narrower than the disk (the
 * branches and the necks between them) and leaves the rest exactly as it was. When the seed itself
 * sat on a removed branch the result is empty and `moved` says so: a piece elsewhere would be a
 * patch the pointer is not on, and it jumps about as the pointer moves.
 */
export function openCell(
  cell: Uint8Array,
  width: number,
  height: number,
  seed: { col: number; row: number },
  rho: number,
): { mask: Uint8Array; moved: boolean } {
  const n = width * height;
  const r = Math.max(0, rho);
  const start = Math.max(0, Math.min(height - 1, Math.round(seed.row))) * width + Math.max(0, Math.min(width - 1, Math.round(seed.col)));
  if (r < 0.5) return { mask: cell, moved: false };
  const R = Math.ceil(r);
  const disk: [number, number][] = [];
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) if (dx * dx + dy * dy <= r * r) disk.push([dx, dy]);
  // Outside the raster counts as outside the cell, so a cell is eroded from the window's edge too.
  const eroded = new Uint8Array(n);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!cell[y * width + x]) continue;
      let keep = 1;
      for (const [dx, dy] of disk) {
        const xx = x + dx;
        const yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= width || yy >= height || !cell[yy * width + xx]) {
          keep = 0;
          break;
        }
      }
      eroded[y * width + x] = keep;
    }
  }
  const opened = new Uint8Array(n);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!eroded[y * width + x]) continue;
      for (const [dx, dy] of disk) {
        const xx = x + dx;
        const yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < width && yy < height) opened[yy * width + xx] = 1;
      }
    }
  }
  if (opened[start]) return { mask: component(opened, width, height, start), moved: false };
  return { mask: new Uint8Array(n), moved: true };
}

/**
 * Two scales: the broad cell picks the patch, the narrow cell places its edge. The result is the
 * narrow cell inside the broad one, where the broad one is first grown by `slack` narrow-raster pixels
 * (its edge is blurred over about r₂, so the narrow edge may sit a little outside it); then the
 * click's piece. The broad cell may be on a coarser raster over the same bbox; it is sampled
 * nearest at each narrow pixel's centre.
 */
export function focusCell(
  narrow: Uint8Array,
  nw: number,
  nh: number,
  broad: Uint8Array,
  bw: number,
  bh: number,
  seed: { col: number; row: number },
  slack: number,
): Uint8Array {
  const n = nw * nh;
  const inBroad = new Uint8Array(n);
  for (let r = 0; r < nh; r++) {
    const br = Math.max(0, Math.min(bh - 1, Math.round(((r + 0.5) * bh) / nh - 0.5)));
    for (let c = 0; c < nw; c++) {
      const bc = Math.max(0, Math.min(bw - 1, Math.round(((c + 0.5) * bw) / nw - 0.5)));
      inBroad[r * nw + c] = broad[br * bw + bc]!;
    }
  }
  let grown = inBroad;
  const R = Math.max(0, slack);
  if (R >= 0.5) {
    grown = new Uint8Array(n);
    const Ri = Math.ceil(R);
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        if (!inBroad[y * nw + x]) continue;
        for (let dy = -Ri; dy <= Ri; dy++) {
          for (let dx = -Ri; dx <= Ri; dx++) {
            if (dx * dx + dy * dy > R * R) continue;
            const xx = x + dx;
            const yy = y + dy;
            if (xx >= 0 && yy >= 0 && xx < nw && yy < nh) grown[yy * nw + xx] = 1;
          }
        }
      }
    }
  }
  const both = new Uint8Array(n);
  for (let i = 0; i < n; i++) both[i] = narrow[i]! & grown[i]!;
  const start = Math.max(0, Math.min(nh - 1, Math.round(seed.row))) * nw + Math.max(0, Math.min(nw - 1, Math.round(seed.col)));
  return component(both, nw, nh, start);
}
