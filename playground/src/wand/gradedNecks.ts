// PROTOTYPE: the opening's "piece the seed belongs to", graded. The binary opening keeps a lobe
// whole or drops it whole as the neck joining it crosses the opening's width, so a hover a pixel over
// can add or remove a lobe at once. Here each pixel is scored by the narrowest neck on its widest path
// from the seed, and a lobe fades out as its neck narrows past ρ instead of popping.

import { distanceTo } from "../../../src/spatial/regions";

export interface GradedCell {
  /** Membership in [0, 1]: the neck score through a smoothstep about ρ. */
  readonly weight: Float32Array;
  /** 1 where the weight is above zero: what the preview outlines, fading by `weight`. */
  readonly support: Float32Array;
  /** 1 where the weight is at least 0.5: what a press paints (the binary opening's piece, nearly). */
  readonly mask: Uint8Array;
}

/** A max-heap of pixel indices keyed by float, with lazy deletion (stale entries are skipped on pop). */
class MaxHeap {
  private keys = new Float32Array(1024);
  private ids = new Int32Array(1024);
  size = 0;
  push(id: number, key: number): void {
    if (this.size === this.keys.length) {
      const k = new Float32Array(this.size * 2);
      k.set(this.keys);
      this.keys = k;
      const d = new Int32Array(this.size * 2);
      d.set(this.ids);
      this.ids = d;
    }
    let c = this.size++;
    while (c > 0) {
      const p = (c - 1) >> 1;
      if (this.keys[p]! >= key) break;
      this.keys[c] = this.keys[p]!;
      this.ids[c] = this.ids[p]!;
      c = p;
    }
    this.keys[c] = key;
    this.ids[c] = id;
  }
  /** The top entry's id; its key is left in `top`. */
  top = 0;
  pop(): number {
    const id = this.ids[0]!;
    this.top = this.keys[0]!;
    const n = --this.size;
    const key = this.keys[n]!;
    const last = this.ids[n]!;
    let c = 0;
    for (;;) {
      const l = 2 * c + 1;
      if (l >= n) break;
      const m = l + 1 < n && this.keys[l + 1]! > this.keys[l]! ? l + 1 : l;
      if (this.keys[m]! <= key) break;
      this.keys[c] = this.keys[m]!;
      this.ids[c] = this.ids[m]!;
      c = m;
    }
    this.keys[c] = key;
    this.ids[c] = last;
    return id;
  }
}

/**
 * The graded piece of `cell` (0/1, row-major) that the seed belongs to, for an opening of radius
 * `rho` pixels. E is each cell pixel's distance to the nearest pixel outside the cell (outside the
 * raster counts as outside, as in the opening), so a disk of radius E fits there. B is the widest-path
 * bottleneck of E from the seed: the widest disk that can slide from the seed to the pixel. A pixel's
 * score is the best B within ρ of it (the disks cover it), and its weight is that score through a
 * smoothstep over ρ(1 − soft) to ρ(1 + soft). Thresholded at 0.5 this is the binary opening's piece.
 */
export function gradedCell(
  cell: ArrayLike<number>,
  width: number,
  height: number,
  seed: { col: number; row: number },
  rho: number,
  soft = 0.5,
): GradedCell {
  const n = width * height;
  // E on a raster padded by one pixel of outside, so the window's edge bounds the disks too.
  const pw = width + 2;
  const ph = height + 2;
  const padded = distanceTo(
    (i) => {
      const x = (i % pw) - 1;
      const y = Math.floor(i / pw) - 1;
      return x < 0 || y < 0 || x >= width || y >= height || !cell[y * width + x];
    },
    pw,
    ph,
  );
  const E = new Float32Array(n);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) E[y * width + x] = padded[(y + 1) * pw + x + 1]!;

  // Widest path from every cell pixel within ρ of the seed (the seed's own disk).
  const B = new Float32Array(n);
  const heap = new MaxHeap();
  const R = Math.ceil(rho);
  const sc = Math.round(seed.col);
  const sr = Math.round(seed.row);
  for (let dy = -R; dy <= R; dy++) {
    for (let dx = -R; dx <= R; dx++) {
      const x = sc + dx;
      const y = sr + dy;
      if (dx * dx + dy * dy > rho * rho || x < 0 || y < 0 || x >= width || y >= height) continue;
      const i = y * width + x;
      if (cell[i] && E[i]! > B[i]!) {
        B[i] = E[i]!;
        heap.push(i, E[i]!);
      }
    }
  }
  while (heap.size > 0) {
    const i = heap.pop();
    const v = heap.top;
    if (v < B[i]!) continue; // stale
    const x = i % width;
    for (const j of [x > 0 ? i - 1 : -1, x < width - 1 ? i + 1 : -1, i - width, i + width]) {
      if (j < 0 || j >= n || !cell[j]) continue;
      const nv = Math.min(v, E[j]!);
      if (nv > B[j]!) {
        B[j] = nv;
        heap.push(j, nv);
      }
    }
  }

  // Score: the best B within ρ; weight: a smoothstep of it about ρ.
  const disk: number[] = [];
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) if (dx * dx + dy * dy <= rho * rho) disk.push(dx, dy);
  const lo = rho * (1 - soft);
  const hi = rho * (1 + soft);
  const weight = new Float32Array(n);
  const support = new Float32Array(n);
  const mask = new Uint8Array(n);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (!cell[i]) continue;
      let s = 0;
      for (let k = 0; k < disk.length; k += 2) {
        const xx = x + disk[k]!;
        const yy = y + disk[k + 1]!;
        if (xx < 0 || yy < 0 || xx >= width || yy >= height) continue;
        const b = B[yy * width + xx]!;
        if (b > s) s = b;
      }
      const t = Math.max(0, Math.min(1, (s - lo) / (hi - lo)));
      const w = t * t * (3 - 2 * t);
      weight[i] = w;
      if (w > 0) support[i] = 1;
      if (w >= 0.5) mask[i] = 1;
    }
  }
  return { weight, support, mask };
}
