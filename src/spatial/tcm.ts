// Topographical Correlation Map (TCM) — MuSpAn / SpOOx, Mode 1 (faithful, global ρ_B).
//
// Reference (exact) CPU implementation: the PARITY ORACLE for TCM. It is deliberately O(N_A·N_B) for
// the neighbourhood counts — correct, not fast — so it validates the grid-accelerated / GPU path that
// follows. See docs/muspan-cell-stats-plan.md §2 and the source paper
// docs/natureCovid-TCMetc-s41467-023-42421-0.pdf (eqs 9–14).
//
//   m_ab(a) = |{ b∈B : |x_a − x_b| < radius }| / (ρ_B · A_radius)         (eq 9; A_radius = π·radius²)
//   M_ab    = markToM(m_ab, α)                                            (eqs 10–13, α=5)
//   Γ_ab(x) = Σ_a  M_ab · (1/(σ√2π)) · exp(−|x − x_a|² / 2σ²)             (eq 14, σ=50µm)
//
// ρ_B is the GLOBAL density of B over the ROI (Mode 1). Edge correction is DEFERRED here: A_radius is
// the full disk area (constant), which is exact for anchors ≥radius inside the ROI and biased near the
// boundary — the Mode-2 viewport-apron path removes that bias by construction (plan §5). Lengths
// (radius, σ) are in the clouds' own world units; feeding µm is the caller's job (placement/units).

/** Transformed mark M_ab ∈ [−1, 1] (eqs 10–13). Antisymmetric under reciprocal: `M(m) = −M(1/m)`,
 *  so clustering (m>1) and exclusion (m<1) are measured on the same scale. 0 at CSR (m=1); +1 at
 *  ≥α-fold clustering; −1 at ≥α-fold exclusion (and at m=0, no neighbours). α is the extreme
 *  threshold (paper: 5). */
export function markToM(m: number, alpha: number): number {
  if (!(alpha > 1)) throw new Error(`markToM: alpha must be > 1, got ${alpha}`);
  if (m >= alpha) return 1;
  if (m > 1) return (m - 1) / (alpha - 1);
  if (m <= 1 / alpha) return -1; // includes m = 0 (no B neighbours ⇒ maximal exclusion)
  return (1 - 1 / m) / (alpha - 1); // 1/α < m < 1  (reciprocal form — NOT a linear mirror)
}

export interface CellCloud {
  readonly xs: readonly number[];
  readonly ys: readonly number[];
}

export interface TcmParams {
  /** Output grid resolution. */
  readonly width: number;
  readonly height: number;
  /** ROI in world coords `[minX, minY, maxX, maxY]` — Mode 1's fixed region (also sets ρ_B and the grid extent). */
  readonly bbox: readonly [number, number, number, number];
  /** m_ab neighbourhood radius in world units (paper: 100 µm). */
  readonly radius: number;
  /** TCM Gaussian bandwidth in world units (paper: 50 µm). */
  readonly sigma: number;
  /** Extreme clustering/exclusion threshold α (paper: 5). */
  readonly alpha: number;
}

/** Per-A-cell mark `m_ab` (eq 9), exact disk count / CSR expectation. Exposed for validation. */
export function crossMarks(a: CellCloud, b: CellCloud, p: TcmParams): Float64Array {
  const [minX, minY, maxX, maxY] = p.bbox;
  const roiArea = Math.max((maxX - minX) * (maxY - minY), 1e-12);
  const rhoB = b.xs.length / roiArea; // global density of B (Mode 1)
  const expected = rhoB * Math.PI * p.radius * p.radius; // ρ_B · A_radius (full-disk; edge corr. deferred)
  const r2 = p.radius * p.radius;
  const nA = a.xs.length;
  const m = new Float64Array(nA);
  for (let i = 0; i < nA; i++) {
    const ax = a.xs[i]!;
    const ay = a.ys[i]!;
    let count = 0;
    for (let j = 0; j < b.xs.length; j++) {
      const dx = b.xs[j]! - ax;
      const dy = b.ys[j]! - ay;
      if (dx * dx + dy * dy < r2) count++;
    }
    m[i] = expected > 0 ? count / expected : 0;
  }
  return m;
}

/** Grid-accelerated TCM — the SAME result as `computeTcmReference` (exact counts, exact splat), but
 *  O(N_A · local B density) via a bucket grid of cell size `radius`: every B within `radius` of an
 *  anchor lies in the anchor's 3×3 bucket neighbourhood, so the exact distance test over just those
 *  candidates reproduces the O(N_A·N_B) count. This is the path the demo runs at Leap034 scale; the
 *  reference is its oracle. */
export function computeTcm(a: CellCloud, b: CellCloud, p: TcmParams): Float32Array {
  const [minX, minY] = p.bbox;
  const maxX = p.bbox[2];
  const maxY = p.bbox[3];
  const roiArea = Math.max((maxX - minX) * (maxY - minY), 1e-12);
  const expected = (b.xs.length / roiArea) * Math.PI * p.radius * p.radius; // ρ_B · A_radius
  const r = p.radius;
  const r2 = r * r;
  const nB = b.xs.length;

  // Bucket grid over B's own extent, cell size = radius.
  let bMinX = Infinity;
  let bMinY = Infinity;
  let bMaxX = -Infinity;
  let bMaxY = -Infinity;
  for (let j = 0; j < nB; j++) {
    const x = b.xs[j]!;
    const y = b.ys[j]!;
    if (x < bMinX) bMinX = x;
    if (x > bMaxX) bMaxX = x;
    if (y < bMinY) bMinY = y;
    if (y > bMaxY) bMaxY = y;
  }
  const cols = nB > 0 ? Math.max(1, Math.ceil((bMaxX - bMinX) / r) + 1) : 1;
  const rows = nB > 0 ? Math.max(1, Math.ceil((bMaxY - bMinY) / r) + 1) : 1;
  const buckets: number[][] = Array.from({ length: cols * rows }, () => []);
  const colOf = (x: number) => Math.min(cols - 1, Math.max(0, Math.floor((x - bMinX) / r)));
  const rowOf = (y: number) => Math.min(rows - 1, Math.max(0, Math.floor((y - bMinY) / r)));
  for (let j = 0; j < nB; j++) buckets[rowOf(b.ys[j]!) * cols + colOf(b.xs[j]!)]!.push(j);

  const { width: w, height: h, sigma } = p;
  const cw = (maxX - minX) / w;
  const ch = (maxY - minY) / h;
  const inv2s2 = 1 / (2 * sigma * sigma);
  const norm = 1 / (sigma * Math.sqrt(2 * Math.PI));
  const kr = Math.max(1, Math.ceil((3 * sigma) / Math.min(cw, ch)));
  const grid = new Float32Array(w * h);

  for (let i = 0; i < a.xs.length; i++) {
    const ax = a.xs[i]!;
    const ay = a.ys[i]!;
    let count = 0;
    if (nB > 0) {
      const c0 = colOf(ax);
      const r0 = rowOf(ay);
      for (let dr = -1; dr <= 1; dr++) {
        const rr = r0 + dr;
        if (rr < 0 || rr >= rows) continue;
        for (let dc = -1; dc <= 1; dc++) {
          const cc = c0 + dc;
          if (cc < 0 || cc >= cols) continue;
          for (const j of buckets[rr * cols + cc]!) {
            const dx = b.xs[j]! - ax;
            const dy = b.ys[j]! - ay;
            if (dx * dx + dy * dy < r2) count++;
          }
        }
      }
    }
    const M = markToM(expected > 0 ? count / expected : 0, p.alpha);
    if (M === 0) continue;
    const ci = (ax - minX) / cw;
    const cj = (ay - minY) / ch;
    const i0 = Math.max(0, Math.floor(ci - kr));
    const i1 = Math.min(w - 1, Math.ceil(ci + kr));
    const j0 = Math.max(0, Math.floor(cj - kr));
    const j1 = Math.min(h - 1, Math.ceil(cj + kr));
    for (let jj = j0; jj <= j1; jj++) {
      const wy = minY + (jj + 0.5) * ch;
      const dy = wy - ay;
      for (let ii = i0; ii <= i1; ii++) {
        const wx = minX + (ii + 0.5) * cw;
        const dx = wx - ax;
        grid[jj * w + ii]! += M * norm * Math.exp(-(dx * dx + dy * dy) * inv2s2);
      }
    }
  }
  return grid;
}

/** Exact TCM reference Γ_ab(x) (eq 14) as a row-major `width×height` grid. The parity oracle. */
export function computeTcmReference(a: CellCloud, b: CellCloud, p: TcmParams): Float32Array {
  const [minX, minY] = p.bbox;
  const maxX = p.bbox[2];
  const maxY = p.bbox[3];
  const { width: w, height: h, sigma } = p;
  const cw = (maxX - minX) / w;
  const ch = (maxY - minY) / h;
  const inv2s2 = 1 / (2 * sigma * sigma);
  const norm = 1 / (sigma * Math.sqrt(2 * Math.PI));
  const kr = Math.max(1, Math.ceil((3 * sigma) / Math.min(cw, ch))); // ~3σ compact support, in cells
  const m = crossMarks(a, b, p);
  const grid = new Float32Array(w * h);
  for (let idx = 0; idx < a.xs.length; idx++) {
    const M = markToM(m[idx]!, p.alpha);
    if (M === 0) continue;
    const ax = a.xs[idx]!;
    const ay = a.ys[idx]!;
    const ci = (ax - minX) / cw;
    const cj = (ay - minY) / ch;
    const i0 = Math.max(0, Math.floor(ci - kr));
    const i1 = Math.min(w - 1, Math.ceil(ci + kr));
    const j0 = Math.max(0, Math.floor(cj - kr));
    const j1 = Math.min(h - 1, Math.ceil(cj + kr));
    for (let j = j0; j <= j1; j++) {
      const wy = minY + (j + 0.5) * ch;
      const dy = wy - ay;
      for (let i = i0; i <= i1; i++) {
        const wx = minX + (i + 0.5) * cw;
        const dx = wx - ax;
        grid[j * w + i]! += M * norm * Math.exp(-(dx * dx + dy * dy) * inv2s2);
      }
    }
  }
  return grid;
}
