import { describe, expect, it } from "vitest";
import { buildBucketGrid, latticeFor, numCells } from "./bucketGrid";

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
    expect(g.pointIds.length).toBe(n);
    expect(g.cellOffsets[0]).toBe(0);
    expect(g.cellOffsets[g.cols * g.rows]).toBe(n);

    const seen = new Uint8Array(n);
    let misplaced = 0;
    for (let b = 0; b < g.cols * g.rows; b++) {
      for (let k = g.cellOffsets[b]!; k < g.cellOffsets[b + 1]!; k++) {
        const i = g.pointIds[k]!;
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
          for (let k = g.cellOffsets[b]!; k < g.cellOffsets[b + 1]!; k++) brute.delete(g.pointIds[k]!);
        }
      }
      missing += brute.size;
    }
    expect(missing).toBe(0);
  });

  it("clamps out-of-bounds points into the edge buckets rather than dropping them", () => {
    const g = buildBucketGrid([-500, 0, 5, 900], [0, 0, 5, 900], 10, [0, 0, 10, 10]);
    expect(g.pointIds.length).toBe(4);
    expect(g.cellOffsets[g.cols * g.rows]).toBe(4);
  });

  it("survives an empty cloud", () => {
    const g = buildBucketGrid([], [], 5);
    expect(g.pointIds.length).toBe(0);
    expect(g.cellOffsets[g.cols * g.rows]).toBe(0);
  });

  describe("with a third axis", () => {
    const zs: number[] = [];
    for (let i = 0; i < n; i++) zs.push(rnd() * 90);

    it("has depth/minZ, x-fastest cell ids, and is a partition", () => {
      const g = buildBucketGrid(xs, ys, 12, undefined, zs);
      expect(g.depth).toBeGreaterThan(1);
      expect(g.minZ).toBe(Math.min(...zs));
      expect(g.cellOffsets.length).toBe(numCells(g) + 1);
      expect(g.cellOffsets[numCells(g)]).toBe(n);
      const seen = new Uint8Array(n);
      let misplaced = 0;
      for (let b = 0; b < numCells(g); b++) {
        for (let k = g.cellOffsets[b]!; k < g.cellOffsets[b + 1]!; k++) {
          const i = g.pointIds[k]!;
          seen[i]! += 1;
          const cx = Math.floor((xs[i]! - g.minX) / g.cell);
          const cy = Math.floor((ys[i]! - g.minY) / g.cell);
          const cz = Math.floor((zs[i]! - g.minZ!) / g.cell);
          if (cx + g.cols * (cy + g.rows * cz) !== b) misplaced++;
        }
      }
      expect(misplaced).toBe(0);
      expect(seen.every((c) => c === 1)).toBe(true);
    });

    it("the 3×3×3 neighbourhood finds every in-radius neighbour", () => {
      const r = 15;
      const g = buildBucketGrid(xs, ys, r, undefined, zs);
      const at = (v: number, lo: number, hi: number) => Math.min(hi - 1, Math.max(0, Math.floor((v - lo) / g.cell)));
      let missing = 0;
      for (let i = 0; i < n; i++) {
        const brute = new Set<number>();
        for (let j = 0; j < n; j++) {
          const dx = xs[j]! - xs[i]!;
          const dy = ys[j]! - ys[i]!;
          const dz = zs[j]! - zs[i]!;
          if (j !== i && dx * dx + dy * dy + dz * dz < r * r) brute.add(j);
        }
        const c0 = at(xs[i]!, g.minX, g.cols);
        const r0 = at(ys[i]!, g.minY, g.rows);
        const l0 = at(zs[i]!, g.minZ!, g.depth!);
        for (let dl = -1; dl <= 1; dl++) {
          for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
              const ll = l0 + dl;
              const rr = r0 + dr;
              const cc = c0 + dc;
              if (ll < 0 || ll >= g.depth! || rr < 0 || rr >= g.rows || cc < 0 || cc >= g.cols) continue;
              const b = cc + g.cols * (rr + g.rows * ll);
              for (let k = g.cellOffsets[b]!; k < g.cellOffsets[b + 1]!; k++) brute.delete(g.pointIds[k]!);
            }
          }
        }
        missing += brute.size;
      }
      expect(missing).toBe(0);
    });

    it("takes 6-number bounds, clamps on z, and a z-slab of one layer is the 2D grid", () => {
      const g = buildBucketGrid([1, 5, 13], [1, 5, 13], 4, [0, 0, 0, 10, 10, 10], [-50, 5, 500]);
      expect([g.cols, g.rows, g.depth]).toEqual([4, 4, 4]);
      expect(g.cellOffsets[numCells(g)]).toBe(3);
      // (1,1,-50) clamps to layer 0 → cell 0; (13,13,500) clamps on every axis → the last cell
      expect(g.pointIds[g.cellOffsets[0]!]).toBe(0);
      expect(g.pointIds[g.cellOffsets[numCells(g) - 1]!]).toBe(2);
      const flat = buildBucketGrid(xs, ys, 12, undefined, new Array(n).fill(3));
      const flat2 = buildBucketGrid(xs, ys, 12);
      expect(flat.depth).toBe(1);
      expect(Array.from(flat.cellOffsets)).toEqual(Array.from(flat2.cellOffsets));
    });

    it("rejects mismatched bounds arity and zs length", () => {
      expect(() => latticeFor([0], [0], 1, [0, 0, 1, 1], [0])).toThrow(/6 bounds/);
      expect(() => latticeFor([0], [0], 1, [0, 0, 0, 1, 1, 1])).toThrow(/4 bounds/);
      expect(() => latticeFor([0, 1], [0, 1], 1, undefined, [0])).toThrow(/zs.length/);
      expect(latticeFor([], [], 1, undefined, [])).toEqual({ cols: 1, rows: 1, cell: 1, minX: 0, minY: 0, depth: 1, minZ: 0 });
    });
  });
});
