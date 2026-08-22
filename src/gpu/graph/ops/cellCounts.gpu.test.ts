import { describe, expect, it } from "vitest";
import { buildBucketGrid } from "../../../spatial/bucketGrid";
import { nodeBackend } from "../backend.node";
import { Graph, pull, pullResident, registerBuiltinOps } from "../index";

// The first consumer of a bundle (ADR-0023) — so as well as the arithmetic, this is where a
// bundle travelling along a resident edge is exercised end to end.
// House rules: aggregated assertions, small clouds.

registerBuiltinOps();

const BOUNDS = { minX: 0, minY: 0, maxX: 100, maxY: 80 };

function cloud(n: number, seed: number) {
  let a = seed;
  const rnd = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(Math.fround(rnd() * BOUNDS.maxX));
    ys.push(Math.fround(rnd() * BOUNDS.maxY));
  }
  return { xs, ys };
}

/** Counts straight off the CPU bucket grid — a different route to the same numbers than
 *  differencing the GPU's scanned offsets. */
function countsCpu(xs: number[], ys: number[], cell: number) {
  const g = buildBucketGrid(xs, ys, cell, [BOUNDS.minX, BOUNDS.minY, BOUNDS.maxX, BOUNDS.maxY]);
  const out = new Float32Array(g.cols * g.rows);
  for (let b = 0; b < out.length; b++) out[b] = g.cellOffsets[b + 1]! - g.cellOffsets[b]!;
  return { out, cols: g.cols, rows: g.rows };
}

function chain(xs: number[], ys: number[], cell: number) {
  const g = new Graph();
  const idx = g.op1("gridIndex", { points: g.points(xs, ys) }, { cell, ...BOUNDS });
  return { g, counts: g.op1("cellCounts", { buckets: idx }) };
}

describe("cellCountsOp", () => {
  it("counts every point exactly once, matching the CPU bucket grid", async () => {
    const { xs, ys } = cloud(800, 0xc0ffee);
    const want = countsCpu(xs, ys, 10);
    const { g, counts } = chain(xs, ys, 10);

    // A grid of f32 — so unlike the index's u32 parts, this one CAN come back to the host, which
    // is the point of having it: it is the previewable face of an index.
    const v = await pull(g, counts);
    expect(v.shape).toEqual({ kind: "grid", width: want.cols, height: want.rows });
    expect(v.shape).toEqual(counts.shape);

    const got = v.data as Float32Array;
    let mismatch = 0;
    let total = 0;
    for (let i = 0; i < want.out.length; i++) {
      if (got[i] !== want.out[i]) mismatch++;
      total += got[i]!;
    }
    expect({ mismatch, total }).toEqual({ mismatch: 0, total: xs.length });
  });

  it("agrees with its own cpuGolden", async () => {
    const { xs, ys } = cloud(300, 5);
    const { g, counts } = chain(xs, ys, 12);
    const gpu = (await pull(g, counts)).data as Float32Array;
    const cpu = (await pull(g, counts, { mode: "cpu" })).data as Float32Array;
    let mismatch = 0;
    for (let i = 0; i < cpu.length; i++) if (gpu[i] !== cpu[i]) mismatch++;
    expect({ mismatch, len: cpu.length }).toEqual({ mismatch: 0, len: cpu.length });
  });

  it("inherits the lattice's placement, not the points'", async () => {
    const { xs, ys } = cloud(50, 8);
    const g = new Graph();
    const placed = {
      shape: { kind: "points", n: xs.length } as const,
      dtype: "f32" as const,
      data: Float32Array.from(xs.flatMap((x, i) => [x, ys[i]!])),
      placement: {
        system: "global",
        worldFromArray: {
          origin: [0, 0, 0] as const,
          axes: [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 1],
          ] as const,
        },
      },
    };
    const idx = g.op1("gridIndex", { points: g.source(placed, "points") }, { cell: 10, ...BOUNDS });
    const v = await pullResident(g, g.op1("cellCounts", { buckets: idx }));
    // One counts cell spans 10 point-units, and each point-unit is 2 world units.
    expect(v.placement?.worldFromArray.axes[0]).toEqual([20, 0, 0]);
  });

  it("runs the whole chain resident, and returns every lease", async () => {
    const { xs, ys } = cloud(400, 21);
    const { g, counts } = chain(xs, ys, 20);
    const bridges: string[] = [];
    const before = nodeBackend.poolStats();
    const v = await pullResident(g, counts, { onBridge: (_k, d) => bridges.push(d) });
    const after = nodeBackend.poolStats();

    // The only transfer is uploading the host points source: the index and the counts never
    // touch the host, which is the ADR-0017 claim this chain is here to keep honest.
    expect(bridges.filter((d) => d === "download").length).toBe(0);
    expect(v.buffer).toBeTruthy();
    // The caller owns the counts grid; the index's own buffers went back to the pool.
    expect(after.live - before.live).toBe(1);
  });

  it("rejects an input that is not a grid-index bundle", () => {
    const g = new Graph();
    expect(() => g.op1("cellCounts", { buckets: g.grid(new Float32Array(16), 4, 4) })).toThrow(/expected a "bucketGrid" bundle/);
  });
});
