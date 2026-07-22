import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { getDevice } from "../device";
import { nodeBackend } from "./backend.node";
import { advance, createSimState, disposeSimState, Graph, pull, pullResident, simStateBytes } from "./index";
import { installReadbackCounter, measureReadbacks, uninstallReadbackCounter } from "./instrument";
import { BufferPool, residentUsage } from "./pool";
import { FieldRing } from "./ringBuffer";

// ADR-0017 stages 1-3. `readbackBudget.gpu.test.ts` proves the per-op transfers stopped happening
// along a realistic chain; this file proves the pool underneath is sound, the leases come back,
// and a resident feedback loop ping-pongs at constant cost. Numeric agreement with the CPU
// goldens is in residentValues.gpu.test.ts.
//
// Everything here is built on a plain host `grid` source rather than `splatDensity`. Three
// reasons: the subject is the two *converted* ops, not splat; splat's own GPU and CPU paths
// differ by ~1.6e-2, which would swamp the numeric checks; and a fully-resident chain lets the
// download count reach the numbers the ADR actually claims — 1 for a host sink, 0 for a device
// sink. Splat is a whole render pipeline, and keeping it out also keeps this fork's GPU work
// low: Dawn-on-Node segfaults a fork once enough device churn accumulates, and when it fires
// it takes the not-yet-reported results with it (see vitest.gpu.config.ts).

const W = 16;
const H = 16;

/** A smooth, non-degenerate field — no flat regions, so a convolution actually changes it. */
function ramp(): Float32Array {
  const g = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) g[y * W + x] = Math.sin(x * 0.4) * Math.cos(y * 0.3) + 1.5;
  }
  return g;
}

/** grid -> convolve -> threshold: a host source, then both converted ops. Every edge after the
 *  source is resident, so the only transfers possible are the source upload and the sink. */
function chain() {
  const g = new Graph();
  const src = g.grid(ramp(), W, H);
  // Gaussian, not box: `boxKernel` is all ones, i.e. a local *sum*, which for radius 1 scales
  // the field by ~9 and drives any threshold straight into saturation. The gaussian is
  // normalised, so the convolved values stay in the input's range and the threshold discriminates.
  const smooth = g.op1("convolveSeparable", { grid: src }, { kernel: "gaussian", radius: 2 });
  const mask = g.op1("threshold", { in: smooth }, { thresh: 1.5, soft: true, softness: 8 });
  return { g, mask };
}

describe("BufferPool — lease/release by liveness", () => {
  let device: GPUDevice;
  beforeAll(async () => {
    device = await getDevice();
  });

  it("reuses a released buffer instead of allocating a second one", () => {
    const pool = new BufferPool(device);
    const a = pool.lease(1024);
    expect(pool.stats()).toMatchObject({ live: 1, free: 0, created: 1 });

    pool.release(a);
    expect(pool.stats()).toMatchObject({ live: 0, free: 1, created: 1 });

    const b = pool.lease(1024);
    // The whole point of invariant 3: non-overlapping lifetimes share one allocation.
    expect(b.buffer).toBe(a.buffer);
    expect(pool.stats().created).toBe(1);
  });

  it("keeps two simultaneously-live leases on distinct buffers", () => {
    const pool = new BufferPool(device);
    const a = pool.lease(1024);
    const b = pool.lease(1024);
    expect(b.buffer).not.toBe(a.buffer);
    expect(pool.stats()).toMatchObject({ live: 2, created: 2 });
  });

  it("buckets by power of two, and does not mix sizes or usage classes", () => {
    const pool = new BufferPool(device);
    const a = pool.lease(1000); // -> 1024
    expect(a.lease.capacity).toBe(1024);
    expect(a.byteLength).toBe(1000); // logical length is preserved, not rounded
    pool.release(a);
    expect(pool.lease(1024).buffer).toBe(a.buffer);

    // Tiny leases floor at the minimum bucket.
    expect(pool.lease(4).lease.capacity).toBe(256);

    // A larger lease must not be served from a smaller bucket's free list.
    const small = pool.lease(256);
    pool.release(small);
    expect(pool.lease(4096).buffer).not.toBe(small.buffer);

    // Nor may a lease of a DIFFERENT usage class reuse a resident-class buffer — it physically
    // lacks the extra flag. VERTEX stands in for any class the pool doesn't name; callers pass
    // such flags explicitly (see `residentUsage`'s note), and the free lists must keep them apart.
    const VERTEX = (globalThis as unknown as { GPUBufferUsage: Record<string, number> }).GPUBufferUsage.VERTEX!;
    const plain = pool.lease(2048, residentUsage());
    pool.release(plain);
    expect(pool.lease(2048, residentUsage() | VERTEX).buffer).not.toBe(plain.buffer);
  });

  it("throws on double release rather than corrupting the free list", () => {
    const pool = new BufferPool(device);
    const a = pool.lease(512);
    pool.release(a);
    expect(() => pool.release(a)).toThrow(/not live/);
  });

  it("converges: a lease/release cycle allocates exactly once", () => {
    const pool = new BufferPool(device);
    for (let i = 0; i < 20; i++) pool.release(pool.lease(2048));
    expect(pool.stats()).toMatchObject({ created: 1, live: 0, free: 1 });
  });
});

describe("Tier-2 resident edges", () => {
  beforeAll(async () => {
    await getDevice();
    installReadbackCounter();
  });
  afterAll(() => uninstallReadbackCounter());

  it("downloads once for a host sink, not at all for a device sink, and says so", async () => {
    const { g, mask } = chain();

    const bridges: string[] = [];
    const host = await measureReadbacks(() => pull(g, mask, { onBridge: (k, d) => bridges.push(`${d} ${k}`) }));
    expect(host.result.data).toBeInstanceOf(Float32Array);
    // Invariant 4 exactly: the source uploads, the interior stays on-GPU, the sink downloads once.
    expect(host.stats.downloads).toBe(1);

    // And the bridge log agrees: the source upload and the sink download, nothing in between.
    // A resident op quietly falling back to a host round-trip would show up right here.
    expect(bridges.filter((b) => b.startsWith("upload"))).toHaveLength(1);
    expect(bridges.filter((b) => b.startsWith("download"))).toHaveLength(1);
    expect(bridges.some((b) => b.includes("convolveSeparable"))).toBe(false);

    const device = await measureReadbacks(() => pullResident(g, mask));
    expect(device.result.buffer).toBeDefined();
    expect(device.result.data).toBeUndefined();
    expect(device.result.buffer?.byteLength).toBe(W * H * 4);
    // ADR-0017 §5: a render-terminated graph performs NO downloads at all.
    expect(device.stats.downloads).toBe(0);

    // The caller owns a pullResident sink; hand it back so this test doesn't leak a lease.
    if (device.result.buffer) nodeBackend.release(device.result.buffer);
  });

  it("ping-pongs a resident feedback state, at constant cost per tick", async () => {
    // ADR-0017 stage 3 / invariant 1. A feedback loop whose `next` is produced by a resident op
    // keeps its state on the GPU between ticks. The store adopts each new lease and returns the
    // superseded one, so the pool alternates between two buffers rather than handing out a fresh
    // one every tick. The falsifiable claim is that the footprint is CONSTANT in tick count —
    // under Tier-1 this loop round-tripped the whole field to the host on every single tick.
    const g = new Graph();
    const init = g.grid(ramp(), W, H);
    const fb = g.feedback(init, "loop");
    const smooth = g.op1("convolveSeparable", { grid: fb.state }, { kernel: "gaussian", radius: 1 });
    fb.close(smooth);

    const state = createSimState();
    await advance(g, smooth, { steps: 2, state }); // settle: allocate the ping-pong pair
    const settled = nodeBackend.poolStats().created;

    await advance(g, smooth, { steps: 20, state });
    // Twenty more ticks, not one more buffer.
    expect(nodeBackend.poolStats().created).toBe(settled);

    // The state really is on the GPU, not quietly living on the host.
    const held = [...state.values()][0];
    expect(held && !(held instanceof FieldRing) && held.buffer).toBeTruthy();
    expect(simStateBytes(state)).toBe(W * H * 4);

    // And disposing returns the lease rather than stranding it.
    const liveBefore = nodeBackend.poolStats().live;
    disposeSimState(state, nodeBackend);
    expect(nodeBackend.poolStats().live).toBe(liveBefore - 1);
  });

  it("recycles buffers across pulls instead of allocating without bound", async () => {
    const { g, mask } = chain();
    await pull(g, mask); // warm up: the first pull populates the pool
    const before = nodeBackend.poolStats().created;
    await pull(g, mask);
    await pull(g, mask);

    // Invariant 3 in one line: a repeated identical pull is served entirely from the free list.
    expect(nodeBackend.poolStats().created).toBe(before);
    // And a completed `pull` leaves nothing checked out — the sink was downloaded and returned.
    expect(nodeBackend.poolStats().live).toBe(0);
  });
});
