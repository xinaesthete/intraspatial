import { describe, expect, it } from "vitest";
import { buildBucketGrid } from "./bucketGrid";

// mulberry32 — a lattice-free PRNG (a plain LCG's consecutive values lie on 2D planes, which
// fabricates spatial structure; that already broke one CSR test in this front).
function rng(seed: number) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("buildBucketGrid", () => {
  const rnd = rng(0x1234);
  const n = 600;
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(rnd() * 200);
    ys.push(rnd() * 140);
  }

  it("is a partition: every point appears exactly once, in the bucket its coords imply", () => {
    const g = buildBucketGrid(xs, ys, 12);
    expect(g.items.length).toBe(n);
    expect(g.start[0]).toBe(0);
    expect(g.start[g.cols * g.rows]).toBe(n);

    const seen = new Uint8Array(n);
    let misplaced = 0;
    for (let b = 0; b < g.cols * g.rows; b++) {
      for (let k = g.start[b]!; k < g.start[b + 1]!; k++) {
        const i = g.items[k]!;
        seen[i]! += 1;
        const col = Math.floor((xs[i]! - g.minX) / g.cell);
        const row = Math.floor((ys[i]! - g.minY) / g.cell);
        if (row * g.cols + col !== b) misplaced++;
      }
    }
    expect(misplaced).toBe(0);
    expect(seen.every((c) => c === 1)).toBe(true);
  });

  it("the 3×3 neighbourhood of a cell-size-r grid finds every in-radius neighbour", () => {
    // This is the property the GPU kernels depend on; brute force is the oracle.
    const r = 9;
    const g = buildBucketGrid(xs, ys, r);
    let missing = 0;
    for (let i = 0; i < n; i++) {
      const brute = new Set<number>();
      for (let j = 0; j < n; j++) {
        const dx = xs[j]! - xs[i]!;
        const dy = ys[j]! - ys[i]!;
        if (j !== i && dx * dx + dy * dy < r * r) brute.add(j);
      }
      const c0 = Math.min(g.cols - 1, Math.max(0, Math.floor((xs[i]! - g.minX) / g.cell)));
      const r0 = Math.min(g.rows - 1, Math.max(0, Math.floor((ys[i]! - g.minY) / g.cell)));
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const rr = r0 + dr;
          const cc = c0 + dc;
          if (rr < 0 || rr >= g.rows || cc < 0 || cc >= g.cols) continue;
          const b = rr * g.cols + cc;
          for (let k = g.start[b]!; k < g.start[b + 1]!; k++) brute.delete(g.items[k]!);
        }
      }
      missing += brute.size;
    }
    expect(missing).toBe(0);
  });

  it("clamps out-of-bounds points into the edge buckets rather than dropping them", () => {
    const g = buildBucketGrid([-500, 0, 5, 900], [0, 0, 5, 900], 10, [0, 0, 10, 10]);
    expect(g.items.length).toBe(4);
    expect(g.start[g.cols * g.rows]).toBe(4);
  });

  it("survives an empty cloud", () => {
    const g = buildBucketGrid([], [], 5);
    expect(g.items.length).toBe(0);
    expect(g.start[g.cols * g.rows]).toBe(0);
  });
});
