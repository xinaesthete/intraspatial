import { describe, expect, it } from "vitest";
import { type BucketGrid, buildBucketGrid, latticeFor, numCells } from "../../spatial/bucketGrid";
import { buildGridIndexGpu, GRID_INDEX_WG, type GridIndexOptions } from "./gridIndex";

// Dawn-on-Node house rules, as in the neighbouring suites: assertions are AGGREGATED (one
// expect on a mismatch count), never a per-element expect() loop, and point counts stay small.

function rng(seed: number) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pack(xs: ArrayLike<number>, ys: ArrayLike<number>, zs?: ArrayLike<number>): Float32Array {
  const s = zs ? 3 : 2;
  const out = new Float32Array(s * xs.length);
  for (let i = 0; i < xs.length; i++) {
    out[s * i] = xs[i]!;
    out[s * i + 1] = ys[i]!;
    if (zs) out[s * i + 2] = zs[i]!;
  }
  return out;
}

/** The golden: same lattice, same `start`, and the same SET of items in every cell (order
 *  within a cell is not part of the contract). Returns a summary rather than asserting so the
 *  caller makes one `expect`. */
function compare(gpu: BucketGrid, cpu: BucketGrid) {
  const lattice =
    gpu.cols === cpu.cols &&
    gpu.rows === cpu.rows &&
    gpu.minX === cpu.minX &&
    gpu.minY === cpu.minY &&
    gpu.cell === cpu.cell &&
    gpu.depth === cpu.depth &&
    gpu.minZ === cpu.minZ;
  let startMismatch = 0;
  for (let b = 0; b < cpu.cellOffsets.length; b++) if (gpu.cellOffsets[b] !== cpu.cellOffsets[b]) startMismatch++;
  let cellMismatch = 0;
  const M = numCells(cpu);
  for (let b = 0; b < M; b++) {
    const lo = cpu.cellOffsets[b]!;
    const hi = cpu.cellOffsets[b + 1]!;
    const a = Array.from(cpu.pointIds.subarray(lo, hi)).sort((p, q) => p - q);
    const g = Array.from(gpu.pointIds.subarray(lo, hi)).sort((p, q) => p - q);
    if (a.length !== g.length || a.some((v, k) => v !== g[k])) cellMismatch++;
  }
  return { lattice, startMismatch, cellMismatch, cells: M };
}

/** Structural invariants that hold regardless of the CPU reference (3D note §5). */
function invariants(g: BucketGrid, n: number, xs: number[], ys: number[], zs?: number[]) {
  const M = numCells(g);
  const at = (v: number, lo: number, hi: number) => Math.min(hi - 1, Math.max(0, Math.floor((v - lo) / g.cell)));
  const cellOf = (i: number) => {
    const c = at(xs[i]!, g.minX, g.cols);
    const r = at(ys[i]!, g.minY, g.rows);
    const l = zs ? at(zs[i]!, g.minZ!, g.depth!) : 0;
    return c + g.cols * (r + g.rows * l);
  };
  let monotone = g.cellOffsets[0] === 0 && g.cellOffsets[M] === n;
  for (let b = 0; b < M; b++) if (g.cellOffsets[b + 1]! < g.cellOffsets[b]!) monotone = false;
  // every item lands in the cell its position implies, and every index appears exactly once
  const seen = new Uint8Array(n);
  let misplaced = 0;
  let dup = 0;
  for (let b = 0; b < M; b++) {
    for (let k = g.cellOffsets[b]!; k < g.cellOffsets[b + 1]!; k++) {
      const i = g.pointIds[k]!;
      if (cellOf(i) !== b) misplaced++;
      if (seen[i]) dup++;
      seen[i] = 1;
    }
  }
  return { monotone, misplaced, dup, covered: seen.reduce((s, v) => s + v, 0) };
}

/** Both builds from the SAME inputs: the GPU sees f32 coordinates, so the CPU golden gets
 *  them f32-rounded too — otherwise a bounds-free build derives `minX` from f64 on one side. */
async function both(xs: number[], ys: number[], cell: number, opts: Partial<GridIndexOptions> = {}) {
  const pts = pack(xs, ys);
  const fx = Array.from(xs, Math.fround);
  const fy = Array.from(ys, Math.fround);
  const cpu = buildBucketGrid(fx, fy, cell, opts.bounds);
  const gpu = await buildGridIndexGpu(pts, { cell, ...opts });
  return { cpu, gpu, fx, fy };
}

/** The 3D counterpart: `zs` goes to both builds, `dims: 3` + `stride: 3` to the GPU. */
async function both3(xs: number[], ys: number[], zs: number[], cell: number, opts: Partial<GridIndexOptions> = {}) {
  const pts = pack(xs, ys, zs);
  const fx = Array.from(xs, Math.fround);
  const fy = Array.from(ys, Math.fround);
  const fz = Array.from(zs, Math.fround);
  const cpu = buildBucketGrid(fx, fy, cell, opts.bounds, fz);
  const gpu = await buildGridIndexGpu(pts, { cell, ...opts, dims: 3, stride: 3 });
  return { cpu, gpu, fx, fy, fz };
}

describe("latticeFor", () => {
  it("is the lattice buildBucketGrid uses", () => {
    const xs = [0.5, 3.9, 7.1];
    const ys = [1, 1, 6.2];
    const cpu = buildBucketGrid(xs, ys, 2.5);
    const lat = latticeFor(xs, ys, 2.5);
    expect(lat).toEqual({ cols: cpu.cols, rows: cpu.rows, cell: cpu.cell, minX: cpu.minX, minY: cpu.minY });
    expect(latticeFor([], [], 1)).toEqual({ cols: 1, rows: 1, cell: 1, minX: 0, minY: 0 });
  });
});

describe("buildGridIndexGpu", () => {
  it("matches buildBucketGrid cell-for-cell on a random cloud", async () => {
    const rnd = rng(0x9e3779b9);
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < 1500; i++) {
      xs.push(rnd() * 400);
      ys.push(rnd() * 300);
    }
    const { cpu, gpu, fx, fy } = await both(xs, ys, 25);
    const c = compare(gpu, cpu);
    expect(c).toEqual({ lattice: true, startMismatch: 0, cellMismatch: 0, cells: cpu.cols * cpu.rows });
    expect(invariants(gpu, xs.length, fx, fy)).toEqual({ monotone: true, misplaced: 0, dup: 0, covered: xs.length });
    // a scan of zeros is zeros: make sure something actually ran
    expect(gpu.cellOffsets[cpu.cols * cpu.rows]).toBe(xs.length);
  });

  it("puts exactly one point in every cell, in lattice order", async () => {
    const cols = 7;
    const rows = 5;
    const xs: number[] = [];
    const ys: number[] = [];
    // visit cells in a scrambled order so items[k] == k would be a coincidence, not identity
    for (let b = 0; b < cols * rows; b++) {
      const c = (b * 11) % cols;
      const r = Math.floor((b * 11) / cols) % rows;
      xs.push(c * 10 + 5);
      ys.push(r * 10 + 5);
    }
    const { cpu, gpu } = await both(xs, ys, 10, { bounds: [0, 0, 60, 40] });
    expect(gpu.cols).toBe(cols);
    expect(gpu.rows).toBe(rows);
    expect(compare(gpu, cpu).cellMismatch).toBe(0);
    let wrong = 0;
    for (let b = 0; b < cols * rows; b++) if (gpu.cellOffsets[b] !== b) wrong++;
    expect(wrong).toBe(0);
  });

  it("floors and clamps boundary, corner and out-of-bounds points as the CPU does", async () => {
    const xs = [0, 10, 20, 30, 10, 29.999, 30.001, -5, 45, 0];
    const ys = [0, 10, 20, 20, 0, 19.999, 20.001, -5, 35, 20];
    const { cpu, gpu } = await both(xs, ys, 10, { bounds: [0, 0, 30, 20] });
    expect(compare(gpu, cpu)).toEqual({ lattice: true, startMismatch: 0, cellMismatch: 0, cells: cpu.cols * cpu.rows });
  });

  it("leaves an empty region as a run of equal offsets", async () => {
    const rnd = rng(7);
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < 200; i++) {
      xs.push(rnd() * 50); // left fifth only
      ys.push(rnd() * 100);
    }
    const { cpu, gpu } = await both(xs, ys, 10, { bounds: [0, 0, 250, 100] });
    expect(compare(gpu, cpu).cellMismatch).toBe(0);
    // cells at cols >= 6 are empty: start is flat across them, row by row
    let flat = true;
    for (let r = 0; r < gpu.rows; r++) {
      for (let c = 6; c < gpu.cols; c++) {
        const b = r * gpu.cols + c;
        if (gpu.cellOffsets[b + 1] !== gpu.cellOffsets[b]) flat = false;
      }
    }
    expect(flat).toBe(true);
  });

  it("builds an empty index for n = 0", async () => {
    const gpu = await buildGridIndexGpu(new Float32Array(0), { cell: 5, bounds: [0, 0, 20, 20] });
    expect(gpu.pointIds.length).toBe(0);
    expect(gpu.cellOffsets.length).toBe(gpu.cols * gpu.rows + 1);
    expect(gpu.cellOffsets.every((v) => v === 0)).toBe(true);
  });

  it("indexes the xy of xyz points with stride 3", async () => {
    const rnd = rng(99);
    const xs: number[] = [];
    const ys: number[] = [];
    const pts = new Float32Array(3 * 300);
    for (let i = 0; i < 300; i++) {
      xs.push(rnd() * 100);
      ys.push(rnd() * 100);
      pts[3 * i] = xs[i]!;
      pts[3 * i + 1] = ys[i]!;
      pts[3 * i + 2] = rnd() * 1000; // must be ignored
    }
    const cpu = buildBucketGrid(xs, ys, 12);
    const gpu = await buildGridIndexGpu(pts, { cell: 12, stride: 3 });
    expect(compare(gpu, cpu).cellMismatch).toBe(0);
  });

  it("survives the 2-D dispatch fold", async () => {
    const rnd = rng(3);
    const n = GRID_INDEX_WG * 5 + 17; // six workgroups, folded into a 3×2 grid
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < n; i++) {
      xs.push(rnd() * 100);
      ys.push(rnd() * 100);
    }
    const { cpu, gpu } = await both(xs, ys, 8, { maxWorkgroupsPerDim: 3 });
    expect(compare(gpu, cpu)).toEqual({ lattice: true, startMismatch: 0, cellMismatch: 0, cells: cpu.cols * cpu.rows });
    expect(gpu.cellOffsets[cpu.cols * cpu.rows]).toBe(n);
  });

  describe("dims: 3", () => {
    it("matches buildBucketGrid(…, zs) cell-for-cell on a random cloud", async () => {
      const rnd = rng(0xc0ffee);
      const xs: number[] = [];
      const ys: number[] = [];
      const zs: number[] = [];
      for (let i = 0; i < 1500; i++) {
        xs.push(rnd() * 120);
        ys.push(rnd() * 90);
        zs.push(rnd() * 60);
      }
      const { cpu, gpu, fx, fy, fz } = await both3(xs, ys, zs, 10);
      expect(gpu.depth).toBeGreaterThan(1);
      expect(compare(gpu, cpu)).toEqual({ lattice: true, startMismatch: 0, cellMismatch: 0, cells: numCells(cpu) });
      expect(invariants(gpu, xs.length, fx, fy, fz)).toEqual({ monotone: true, misplaced: 0, dup: 0, covered: xs.length });
      expect(gpu.cellOffsets[numCells(cpu)]).toBe(xs.length);
    });

    it("puts one point per cell of a 4×3×5 lattice, x-fastest", async () => {
      const [cols, rows, depth] = [4, 3, 5];
      const xs: number[] = [];
      const ys: number[] = [];
      const zs: number[] = [];
      const M = cols * rows * depth;
      for (let b = 0; b < M; b++) {
        const k = (b * 7) % M; // scrambled visiting order
        const cx = k % cols;
        const cy = Math.floor(k / cols) % rows;
        const cz = Math.floor(k / (cols * rows));
        xs.push(cx * 10 + 5);
        ys.push(cy * 10 + 5);
        zs.push(cz * 10 + 5);
      }
      const { cpu, gpu } = await both3(xs, ys, zs, 10, { bounds: [0, 0, 0, 30, 20, 40] });
      expect([gpu.cols, gpu.rows, gpu.depth]).toEqual([cols, rows, depth]);
      expect(compare(gpu, cpu).cellMismatch).toBe(0);
      // cell b holds the point visited at step k where (k*7) % M == b, i.e. items[b] == k
      let wrong = 0;
      for (let b = 0; b < M; b++) {
        if (gpu.cellOffsets[b] !== b) wrong++;
        const k = gpu.pointIds[b]!;
        if ((k * 7) % M !== b) wrong++;
      }
      expect(wrong).toBe(0);
    });

    it("floors and clamps boundary, corner and out-of-bounds points on all three axes", async () => {
      const xs = [0, 10, 20, 30, 10, 29.999, 30.001, -5, 45, 0, 15, 15];
      const ys = [0, 10, 20, 20, 0, 19.999, 20.001, -5, 35, 20, 15, 15];
      const zs = [0, 10, 10, 10, 0, 9.999, 10.001, -5, 25, 10, -1e6, 1e6];
      const { cpu, gpu, fx, fy, fz } = await both3(xs, ys, zs, 10, { bounds: [0, 0, 0, 30, 20, 10] });
      expect(compare(gpu, cpu)).toEqual({ lattice: true, startMismatch: 0, cellMismatch: 0, cells: numCells(cpu) });
      expect(invariants(gpu, xs.length, fx, fy, fz)).toEqual({ monotone: true, misplaced: 0, dup: 0, covered: xs.length });
    });

    it("leaves an empty z-slab as a run of equal offsets", async () => {
      const rnd = rng(11);
      const xs: number[] = [];
      const ys: number[] = [];
      const zs: number[] = [];
      for (let i = 0; i < 300; i++) {
        xs.push(rnd() * 50);
        ys.push(rnd() * 50);
        zs.push(rnd() * 20); // bottom two layers only
      }
      const { cpu, gpu } = await both3(xs, ys, zs, 10, { bounds: [0, 0, 0, 50, 50, 100] });
      expect(compare(gpu, cpu).cellMismatch).toBe(0);
      const plane = gpu.cols * gpu.rows;
      let flat = true;
      for (let b = 3 * plane; b < numCells(gpu); b++) if (gpu.cellOffsets[b + 1] !== gpu.cellOffsets[b]) flat = false;
      expect(flat).toBe(true);
      expect(gpu.cellOffsets[3 * plane]).toBe(xs.length);
    });

    it("builds an empty index for n = 0", async () => {
      const gpu = await buildGridIndexGpu(new Float32Array(0), { cell: 5, dims: 3, stride: 3, bounds: [0, 0, 0, 20, 20, 10] });
      expect([gpu.cols, gpu.rows, gpu.depth]).toEqual([5, 5, 3]);
      expect(gpu.pointIds.length).toBe(0);
      expect(gpu.cellOffsets.length).toBe(numCells(gpu) + 1);
      expect(gpu.cellOffsets.every((v) => v === 0)).toBe(true);
    });

    it("refuses a 3D lattice without stride 3", async () => {
      await expect(buildGridIndexGpu(new Float32Array(4), { cell: 1, dims: 3 })).rejects.toThrow(/stride 3/);
    });
  });
});
