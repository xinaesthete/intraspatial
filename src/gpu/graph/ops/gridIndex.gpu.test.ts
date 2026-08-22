import { describe, expect, it } from "vitest";
import { buildBucketGrid } from "../../../spatial/bucketGrid";
import { getDevice } from "../../device";
import { readBack } from "../../scan/prefixSum";
import { nodeBackend } from "../backend.node";
import { Graph, pull, pullResident, registerBuiltinOps } from "../index";
import type { GridLatticePayload } from "./gridIndex";

// House rules, as in every GPU suite here: AGGREGATED assertions (a mismatch count, never a
// per-element expect loop, which kills the Dawn fork) and small point counts.

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

/** Read a u32 port off the device. The executor's host bridge is f32-only, so this is how a
 *  u32 resident output is inspected — `pull()` on these ports throws by design. */
async function readU32(buffer: GPUBuffer, n: number): Promise<Uint32Array> {
  const device = await getDevice();
  return new Uint32Array(await readBack(device, "test:gridIndexOp", buffer, 0, n * 4));
}

/** Per-cell SET comparison (order within a cell is not contractual) + `start` equality. */
function compare(start: Uint32Array, items: Uint32Array, cpu: ReturnType<typeof buildBucketGrid>) {
  let startMismatch = 0;
  for (let b = 0; b < cpu.start.length; b++) if (start[b] !== cpu.start[b]) startMismatch++;
  let cellMismatch = 0;
  for (let b = 0; b < cpu.cols * cpu.rows; b++) {
    const lo = cpu.start[b]!;
    const hi = cpu.start[b + 1]!;
    const want = Array.from(cpu.items.subarray(lo, hi)).sort((p, q) => p - q);
    const got = Array.from(items.subarray(lo, hi)).sort((p, q) => p - q);
    if (want.length !== got.length || want.some((v, k) => v !== got[k])) cellMismatch++;
  }
  return { startMismatch, cellMismatch };
}

describe("gridIndexOp", () => {
  it("builds the same index as buildBucketGrid, resident, with no host download", async () => {
    const { xs, ys } = cloud(900, 0x5eed);
    const g = new Graph();
    const pts = g.points(xs, ys);
    const idx = g.op1("gridIndex", { points: pts }, { cell: 10, ...BOUNDS });

    const bridges: string[] = [];
    const bundle = await pullResident(g, idx, { onBridge: (_k: string, d: string) => bridges.push(d) });
    const startV = bundle.parts!.start!;
    const itemsV = bundle.parts!.items!;
    const lat = bundle.parts!.lattice!.payload as GridLatticePayload;
    const cpu = buildBucketGrid(xs, ys, 10, [BOUNDS.minX, BOUNDS.minY, BOUNDS.maxX, BOUNDS.maxY]);
    expect({ cols: lat.cols, rows: lat.rows, cell: lat.cell, minX: lat.minX, minY: lat.minY }).toEqual({
      cols: cpu.cols,
      rows: cpu.rows,
      cell: cpu.cell,
      minX: cpu.minX,
      minY: cpu.minY,
    });
    expect(lat.cells).toBe(cpu.cols * cpu.rows);

    // The ports stay on the device: resident buffers, no host `data`, and the only bridging the
    // executor did was uploading the host points source.
    expect(startV.buffer).toBeTruthy();
    expect(itemsV.buffer).toBeTruthy();
    expect(startV.data).toBeUndefined();
    expect(bridges.filter((d) => d === "download").length).toBe(0);
    expect(bundle.shape.kind).toBe("bundle");

    const start = await readU32(startV.buffer!.buffer, lat.cells + 1);
    const items = await readU32(itemsV.buffer!.buffer, xs.length);
    expect(compare(start, items, cpu)).toEqual({ startMismatch: 0, cellMismatch: 0 });
    expect(start[lat.cells]).toBe(xs.length); // a scan of zeros is zeros — prove it ran
  });

  it("declares shapes and dtypes that match what it produces", async () => {
    const { xs, ys } = cloud(50, 7);
    const g = new Graph();
    const idx = g.op1("gridIndex", { points: g.points(xs, ys) }, { cell: 20, ...BOUNDS });
    const cpu = buildBucketGrid(xs, ys, 20, [BOUNDS.minX, BOUNDS.minY, BOUNDS.maxX, BOUNDS.maxY]);

    const v = await pullResident(g, idx);
    // What the graph promised at build time is what the device produced.
    expect(v.shape).toEqual(idx.shape);
    expect(v.shape).toEqual({
      kind: "bundle",
      name: "gridIndex",
      parts: {
        start: { kind: "points", n: cpu.cols * cpu.rows + 1 },
        items: { kind: "points", n: 50 },
        lattice: { kind: "grid", width: cpu.cols, height: cpu.rows },
      },
    });
    expect([v.parts!.start!.dtype, v.parts!.items!.dtype]).toEqual(["u32", "u32"]);
  });

  it("extracts a part by borrowing the very same buffer", async () => {
    const { xs, ys } = cloud(120, 3);
    const g = new Graph();
    const idx = g.op1("gridIndex", { points: g.points(xs, ys) }, { cell: 20, ...BOUNDS });
    const startF = g.op1("gridIndex.start", { bundle: idx });

    const bundle = await pullResident(g, idx);
    const start = await pullResident(g, startF);
    // Same shape, and — the point of ADR-0023 — the extract copied nothing. It cannot be the
    // identical object across two pulls (each tick rebuilds), so compare the contents instead.
    expect(start.shape).toEqual(bundle.parts!.start!.shape);
    const viaBundle = await readU32(bundle.parts!.start!.buffer!.buffer, 8);
    const viaPart = await readU32(start.buffer!.buffer, 8);
    expect(Array.from(viaPart)).toEqual(Array.from(viaBundle));
  });

  it("returns every lease it took", async () => {
    const { xs, ys } = cloud(200, 11);
    const g = new Graph();
    const idx = g.op1("gridIndex", { points: g.points(xs, ys) }, { cell: 25, ...BOUNDS });
    const before = nodeBackend.poolStats();
    // `pullResident` hands the sink to the caller, so the bundle's two buffers stay live;
    // everything else the tick leased (the uploaded points) comes back.
    const v = await pullResident(g, idx);
    const after = nodeBackend.poolStats();
    expect(v.parts!.start!.buffer).toBeTruthy();
    expect(after.live - before.live).toBe(2);
  });

  it("releases the bundle's other parts when only one is pulled (borrow, then detach)", async () => {
    // The lifetime case ADR-0023 exists to get right: the sink BORROWS `items` from the bundle,
    // so the bundle cannot be released while the pull is in flight — but once it lands, the
    // caller holds exactly one buffer and `start` must not be stranded.
    const { xs, ys } = cloud(200, 12);
    const g = new Graph();
    const idx = g.op1("gridIndex", { points: g.points(xs, ys) }, { cell: 25, ...BOUNDS });
    const itemsF = g.op1("gridIndex.items", { bundle: idx });
    const before = nodeBackend.poolStats();
    const v = await pullResident(g, itemsF);
    const after = nodeBackend.poolStats();
    expect(v.buffer).toBeTruthy();
    // Exactly one live lease more than we started with: the one the caller now owns.
    expect(after.live - before.live).toBe(1);
    // And it is still readable — the borrow kept it alive rather than recycling it underneath us.
    const got = await readU32(v.buffer!.buffer, xs.length);
    expect(new Set(got).size).toBe(xs.length);
  });

  it("refuses a host pull, because the bridge is f32-only", async () => {
    // Not a defect to fix here: `residentF32Count` would reinterpret u32 as f32 element-wise and
    // hand back mangled numbers, so ADR-0017 stage 1 throws instead. These ports feed resident
    // consumers; `mode: "cpu"` is the host-shaped path.
    const g = new Graph();
    const idx = g.op1("gridIndex", { points: g.points([1, 2], [3, 4]) }, { cell: 10, ...BOUNDS });
    // The message names the part, so it is obvious WHICH member of the bundle cannot cross.
    await expect(pull(g, idx)).rejects.toThrow(/bundle part ".*\.start" .*f32-only.*u32/);
    const golden = await pull(g, idx, { mode: "cpu" });
    expect(golden.parts!.start!.data).toBeInstanceOf(Uint32Array);
  });

  it("agrees with its own cpuGolden", async () => {
    const { xs, ys } = cloud(300, 99);
    const g = new Graph();
    const idx = g.op1("gridIndex", { points: g.points(xs, ys) }, { cell: 15, ...BOUNDS });
    const gpu = await pullResident(g, idx);
    const lat = gpu.parts!.lattice!.payload as GridLatticePayload;
    const goldenBundle = await pullResident(g, idx, { mode: "cpu" });

    const start = await readU32(gpu.parts!.start!.buffer!.buffer, lat.cells + 1);
    const items = await readU32(gpu.parts!.items!.buffer!.buffer, xs.length);

    let startMismatch = 0;
    const gs = goldenBundle.parts!.start!.data as Uint32Array;
    for (let i = 0; i < gs.length; i++) if (gs[i] !== start[i]) startMismatch++;
    expect({ startMismatch, len: gs.length }).toEqual({ startMismatch: 0, len: lat.cells + 1 });
    // Cell sets, not order.
    const gi = goldenBundle.parts!.items!.data as Uint32Array;
    let cellMismatch = 0;
    for (let b = 0; b < lat.cells; b++) {
      const want = Array.from(gi.subarray(gs[b]!, gs[b + 1]!)).sort((p, q) => p - q);
      const got = Array.from(items.subarray(gs[b]!, gs[b + 1]!)).sort((p, q) => p - q);
      if (want.some((v, k) => v !== got[k])) cellMismatch++;
    }
    expect(cellMismatch).toBe(0);
  });

  it("indexes an empty cloud", async () => {
    const g = new Graph();
    const v = await pullResident(g, g.op1("gridIndex", { points: g.points([], []) }, { cell: 10, ...BOUNDS }));
    const lat = v.parts!.lattice!.payload as GridLatticePayload;
    const start = await readU32(v.parts!.start!.buffer!.buffer, lat.cells + 1);
    expect(lat.n).toBe(0);
    expect(start.every((v) => v === 0)).toBe(true);
  });
});
