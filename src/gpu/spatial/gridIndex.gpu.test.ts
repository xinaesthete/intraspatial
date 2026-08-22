import { describe, expect, it } from "vitest";
import { type BucketGrid, buildBucketGrid, latticeFor } from "../../spatial/bucketGrid";
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

function pack(xs: ArrayLike<number>, ys: ArrayLike<number>): Float32Array {
  const out = new Float32Array(2 * xs.length);
  for (let i = 0; i < xs.length; i++) {
    out[2 * i] = xs[i]!;
    out[2 * i + 1] = ys[i]!;
  }
  return out;
}

/** The golden: same lattice, same `start`, and the same SET of items in every cell (order
 *  within a cell is not part of the contract). Returns a summary rather than asserting so the
 *  caller makes one `expect`. */
function compare(gpu: BucketGrid, cpu: BucketGrid) {
  const lattice = gpu.cols === cpu.cols && gpu.rows === cpu.rows && gpu.minX === cpu.minX && gpu.minY === cpu.minY && gpu.cell === cpu.cell;
  let startMismatch = 0;
  for (let b = 0; b < cpu.start.length; b++) if (gpu.start[b] !== cpu.start[b]) startMismatch++;
  let cellMismatch = 0;
  const M = cpu.cols * cpu.rows;
  for (let b = 0; b < M; b++) {
    const lo = cpu.start[b]!;
    const hi = cpu.start[b + 1]!;
    const a = Array.from(cpu.items.subarray(lo, hi)).sort((p, q) => p - q);
    const g = Array.from(gpu.items.subarray(lo, hi)).sort((p, q) => p - q);
    if (a.length !== g.length || a.some((v, k) => v !== g[k])) cellMismatch++;
  }
  return { lattice, startMismatch, cellMismatch, cells: M };
}

/** Structural invariants that hold regardless of the CPU reference (3D note §5). */
function invariants(g: BucketGrid, n: number, xs: number[], ys: number[]) {
  const M = g.cols * g.rows;
  const cellOf = (i: number) => {
    const c = Math.min(g.cols - 1, Math.max(0, Math.floor((xs[i]! - g.minX) / g.cell)));
    const r = Math.min(g.rows - 1, Math.max(0, Math.floor((ys[i]! - g.minY) / g.cell)));
    return r * g.cols + c;
  };
  let monotone = g.start[0] === 0 && g.start[M] === n;
  for (let b = 0; b < M; b++) if (g.start[b + 1]! < g.start[b]!) monotone = false;
  // every item lands in the cell its position implies, and every index appears exactly once
  const seen = new Uint8Array(n);
  let misplaced = 0;
  let dup = 0;
  for (let b = 0; b < M; b++) {
    for (let k = g.start[b]!; k < g.start[b + 1]!; k++) {
      const i = g.items[k]!;
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
    expect(gpu.start[cpu.cols * cpu.rows]).toBe(xs.length);
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
    for (let b = 0; b < cols * rows; b++) if (gpu.start[b] !== b) wrong++;
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
        if (gpu.start[b + 1] !== gpu.start[b]) flat = false;
      }
    }
    expect(flat).toBe(true);
  });

  it("builds an empty index for n = 0", async () => {
    const gpu = await buildGridIndexGpu(new Float32Array(0), { cell: 5, bounds: [0, 0, 20, 20] });
    expect(gpu.items.length).toBe(0);
    expect(gpu.start.length).toBe(gpu.cols * gpu.rows + 1);
    expect(gpu.start.every((v) => v === 0)).toBe(true);
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
    expect(gpu.start[cpu.cols * cpu.rows]).toBe(n);
  });
});
