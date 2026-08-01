// Re-test the Dawn limits recorded in ADR-0003 (2026-06-28). All three predate
// the Instance-lifetime fix (4e326b0, 2026-07-29) and all three have the shape
// that bug had: "a process that does enough GPU work segfaults its teardown".
//
// Runs INSIDE vitest deliberately — two of the claims are specifically about the
// vitest worker's teardown, so a standalone process would not test them.
// Correctness is asserted alongside liveness, because the failure mode this repo
// has hit before is silent zeroes rather than an error.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { describe, expect, it } from "vitest";
import { getDevice } from "../src/gpu/device";
import { nearestNeighborDistancesGpu } from "../src/gpu/spatial/nnDistance";

function nnCpu(xs: number[], ys: number[]): Float32Array {
  const n = xs.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let best = Number.POSITIVE_INFINITY;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dd = Math.hypot(xs[j]! - xs[i]!, ys[j]! - ys[i]!);
      if (dd < best) best = dd;
    }
    out[i] = best;
  }
  return out;
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ADR-0003: "For nnDistance this showed up past ~256 brute-force points."
describe("ADR-0003: nnDistance past ~256 points", () => {
  for (const n of [256, 512, 1024, 2048, 4096, 8192]) {
    it(`n=${n} runs, exits, and matches the CPU golden`, async () => {
      const rnd = mulberry32(0xc0ffee + n);
      const xs = Array.from({ length: n }, () => rnd() * 100);
      const ys = Array.from({ length: n }, () => rnd() * 100);
      const gpu = await nearestNeighborDistancesGpu(xs, ys);
      const cpu = nnCpu(xs, ys);
      let maxErr = 0;
      let nonZero = 0;
      for (let i = 0; i < n; i++) {
        maxErr = Math.max(maxErr, Math.abs(gpu[i]! - cpu[i]!));
        if (gpu[i]! !== 0) nonZero++;
      }
      // a silently-killed dispatch returns all zeroes and would pass a loose check
      expect(nonZero).toBe(n);
      expect(maxErr).toBeLessThan(1e-3);
    });
  }
});

// ADR-0003: "creating a second guarded pipeline in one process **segfaulted
// Dawn-on-Node's exit teardown**." Four here, each dispatched and read back.
describe("ADR-0003: multiple guarded compute pipelines in one process", () => {
  it("builds and dispatches four, and the worker survives", async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const N = 64;
    for (let k = 0; k < 4; k++) {
      const buf = root.createBuffer(d.arrayOf(d.u32, N)).$usage("storage");
      const mutable = buf.as("mutable");
      const bump = k;
      // one parameter = n threads in parallel, x is the global invocation id
      const pipeline = root.createGuardedComputePipeline((x: number) => {
        "use gpu";
        mutable.value[x] = x * 2 + bump;
      });
      pipeline.dispatchThreads(N);
      const got = (await buf.read()) as ArrayLike<number>;
      expect(got[0]).toBe(bump);
      expect(got[N - 1]).toBe((N - 1) * 2 + bump);
    }
  });
});

// ADR-0017 is the one claim in this sweep that does NOT get retired. Wrapping a
// pooled buffer as a TypeGPU buffer makes the root a second owner of the same
// Dawn handle, so exit double-frees it — a genuine ownership bug that merely
// produces the same symptom as the lifetime one. The fix (one cached wrapper per
// raw buffer, `backend.node.ts`) is what this pins.
describe("ADR-0017: pooled-buffer wrapper ownership", () => {
  it("one cached wrapper per raw buffer reads correctly across reuse", async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const N = 256;
    const raw = device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // the fix: ONE wrapper, reused — exactly what backend.node.ts's WeakMap gives
    const wrapper = root.createBuffer(d.arrayOf(d.f32, N), raw);
    for (let cycle = 0; cycle < 8; cycle++) {
      const data = new Float32Array(N);
      for (let i = 0; i < N; i++) data[i] = i + cycle;
      device.queue.writeBuffer(raw, 0, data);
      const got = (await wrapper.read()) as ArrayLike<number>;
      expect(got[0]).toBeCloseTo(cycle, 5);
      expect(got[N - 1]).toBeCloseTo(N - 1 + cycle, 5);
    }
    // `raw` belongs to the pool, so the pool frees it — the wrapper must not have
    // taken ownership, or this is the double free ADR-0017 describes
    raw.destroy();
  });
});

// ADR-0003: "a raw `mapAsync` on a pooled `MAP_READ` buffer **segfaulted the
// vitest worker on teardown**, even though the same render+readback exits
// cleanly outside vitest". Pooled = reused across calls, then grown.
describe("ADR-0003: raw mapAsync on a pooled MAP_READ buffer", () => {
  it("reads through a reused and then grown staging buffer", async () => {
    const device = await getDevice();
    let cap = 0;
    let src: GPUBuffer | undefined;
    let staging: GPUBuffer | undefined;

    async function roundTrip(n: number) {
      const bytes = n * 4;
      if (bytes > cap) {
        src?.destroy();
        staging?.destroy();
        cap = bytes;
        src = device.createBuffer({ size: cap, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        staging = device.createBuffer({ size: cap, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      }
      const data = new Uint32Array(n);
      for (let i = 0; i < n; i++) data[i] = i + 1;
      device.queue.writeBuffer(src!, 0, data);
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(src!, 0, staging!, 0, bytes);
      device.queue.submit([enc.finish()]);
      await staging!.mapAsync(GPUMapMode.READ, 0, bytes);
      const view = new Uint32Array(staging!.getMappedRange(0, bytes).slice(0));
      staging!.unmap();
      return view;
    }

    // reuse the same pooled buffer, then force a growth
    for (const n of [1024, 1024, 1024, 65536]) {
      const got = await roundTrip(n);
      expect(got[0]).toBe(1);
      expect(got[n - 1]).toBe(n);
    }
    src?.destroy();
    staging?.destroy();
  });
});
