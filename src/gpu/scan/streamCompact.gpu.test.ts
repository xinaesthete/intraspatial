import { describe, expect, it } from "vitest";
import { getDevice } from "../device";
import { ensureBuf, readBack, SCAN_BLOCK } from "./prefixSum";
import {
  COMPACT_WG,
  type CompactOptions,
  type EncodeCompactOptions,
  type EncodedCompact,
  encodeCompact,
  getCompactCtx,
  type MaskArray,
  maskEncodingOf,
  maskWords,
  streamCompactGpu,
  uploadMask,
} from "./streamCompact";

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** CPU golden — a plain host filter, which is the definition of the answer and shares no
 *  machinery with the scan/scatter the GPU does. Returns a checksum rather than the list
 *  for the big cases: comparing 20M indices element-by-element in a GPU test is exactly the
 *  per-element `expect()` loop that kills the Dawn fork. */
function compactCpu(mask: MaskArray, opts: CompactOptions = {}) {
  const gt = opts.pass !== "eq";
  const value = opts.value ?? 0;
  let count = 0;
  // Sum of the passing indices. A cheap independent witness: any dropped, duplicated or
  // zeroed index moves it, and unlike `count` alone it also pins the *identities*.
  let checksum = 0;
  const first: number[] = [];
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i]!;
    if (gt ? v > value : v === value) {
      count++;
      checksum += i;
      if (first.length < 8) first.push(i);
    }
  }
  return { count, checksum, first };
}

function checksumOf(indices: Uint32Array) {
  let s = 0;
  let ascending = true;
  for (let i = 0; i < indices.length; i++) {
    s += indices[i]!;
    if (i > 0 && indices[i]! <= indices[i - 1]!) ascending = false;
  }
  return { checksum: s, ascending };
}

describe("streamCompactGpu", () => {
  it("compacts a hand-worked 0/1 mask", async () => {
    // positions of the 1s in [0,1,0,0,1,1,0,1], read off by eye
    const { indices, count } = await streamCompactGpu(new Uint32Array([0, 1, 0, 0, 1, 1, 0, 1]));
    expect(Array.from(indices)).toEqual([1, 4, 5, 7]);
    expect(count).toBe(4);
  });

  it("returns nothing for a mask that selects nothing", async () => {
    const { indices, count } = await streamCompactGpu(new Uint32Array(5000));
    expect(count).toBe(0);
    expect(indices.length).toBe(0);
  });

  it("returns every index for an all-pass mask", async () => {
    const n = 5000;
    const { indices, count } = await streamCompactGpu(new Uint32Array(n).fill(1));
    expect(count).toBe(n);
    // reduced to one assertion: identity means indices[i] === i everywhere
    let wrong = 0;
    for (let i = 0; i < n; i++) if (indices[i] !== i) wrong++;
    expect(wrong).toBe(0);
  });

  it("handles a single passing element and a single failing one", async () => {
    expect(Array.from((await streamCompactGpu(new Uint32Array([1]))).indices)).toEqual([0]);
    expect((await streamCompactGpu(new Uint32Array([0]))).count).toBe(0);
  });

  it("compacts an empty input", async () => {
    const { indices, count } = await streamCompactGpu(new Uint32Array(0));
    expect(count).toBe(0);
    expect(indices.length).toBe(0);
  });

  // Sizes either side of the workgroup, the scan block, and the point where the scan gains
  // a level — the same list `prefixSum.gpu.test.ts` uses, because compaction inherits every
  // one of the scan's boundaries and adds the scatter's own.
  const SIZES = [255, 256, 257, SCAN_BLOCK - 1, SCAN_BLOCK, SCAN_BLOCK + 1, 3 * SCAN_BLOCK + 7, 1_200_000];

  for (const n of SIZES) {
    it(`matches the CPU golden at n=${n}`, async () => {
      const rnd = mulberry32(0xbee5 + n);
      const mask = Uint32Array.from({ length: n }, () => (rnd() < 0.3 ? 1 : 0));
      const want = compactCpu(mask);
      const { indices, count } = await streamCompactGpu(mask);
      const got = checksumOf(indices);

      expect(count, `n=${n} count`).toBe(want.count);
      expect(indices.length, `n=${n} length`).toBe(want.count);
      expect(got.checksum, `n=${n} checksum`).toBe(want.checksum);
      expect(got.ascending, `n=${n} ascending`).toBe(true);
      expect(Array.from(indices.subarray(0, 8)), `n=${n} head`).toEqual(want.first);
    });
  }

  it("reads MDV's byte-per-row filterArray with no host conversion", async () => {
    // MDV's `Dimension.filterArray` is one byte per row over a SharedArrayBuffer and a row
    // passes iff the byte is 0 — 1 is the local filter, 2 the background one, 3 both. The
    // length is deliberately not a multiple of 4, because that is where a packed-byte path
    // loses its tail: the last 1-3 rows go up in a separate padded write and would
    // otherwise be read as whatever was in the buffer before.
    const n = 4 * SCAN_BLOCK + 3;
    const rnd = mulberry32(0x11dd);
    const mask = Uint8Array.from({ length: n }, () => [0, 1, 2, 3][Math.floor(rnd() * 4)]!);
    mask[n - 1] = 0; // the very last row passes, so a dropped tail cannot go unnoticed
    const opts: CompactOptions = { pass: "eq", value: 0 };
    const want = compactCpu(mask, opts);
    const { indices, count } = await streamCompactGpu(mask, opts);
    const got = checksumOf(indices);

    expect(count).toBe(want.count);
    expect(got.checksum).toBe(want.checksum);
    expect(indices[count - 1]).toBe(n - 1);
  });

  it("thresholds ADR-0005's soft f32 weight mask", async () => {
    // The soft `support` mask is one weight per row in [0,1]; a hard filter is the boxcar
    // case. `gt` with a threshold is how a soft mask becomes a compact index list.
    const n = 2 * SCAN_BLOCK + 61;
    const rnd = mulberry32(0x50f7);
    const mask = Float32Array.from({ length: n }, () => rnd());
    const opts: CompactOptions = { pass: "gt", value: 0.5 };
    const want = compactCpu(mask, opts);
    const { indices, count } = await streamCompactGpu(mask, opts);

    expect(count).toBe(want.count);
    expect(checksumOf(indices).checksum).toBe(want.checksum);
    // guards against a degenerate threshold making this vacuous
    expect(count).toBeGreaterThan(n / 4);
    expect(count).toBeLessThan((3 * n) / 4);
  });

  it("is unchanged when the dispatch is forced into a 2-D grid", async () => {
    const n = 7000; // 28 workgroups at 256 threads
    const rnd = mulberry32(0x9a11);
    const mask = Uint32Array.from({ length: n }, () => (rnd() < 0.4 ? 1 : 0));
    const want = compactCpu(mask);
    // 28 workgroups folded into 6x5 — the shape a 16.8M-row mask takes for real
    const { indices, count } = await streamCompactGpu(mask, { maxWorkgroupsPerDim: 6 });

    expect(count).toBe(want.count);
    expect(checksumOf(indices).checksum).toBe(want.checksum);
  });

  it("keeps a small run correct after a large one has grown the pool", async () => {
    // The buffer pool is grow-only, so after a big call every binding covers a buffer far
    // larger than the next call needs. Correctness must not depend on the pool's history:
    // the descending order here is the one that catches a stale tail being read as data.
    const runs = [40_000, 900, 40_001, 7];
    const results: number[] = [];
    const wants: number[] = [];
    for (const n of runs) {
      const rnd = mulberry32(0xd0e5 + n);
      const mask = Uint32Array.from({ length: n }, () => (rnd() < 0.25 ? 1 : 0));
      wants.push(compactCpu(mask).checksum);
      results.push(checksumOf((await streamCompactGpu(mask)).indices).checksum);
    }
    expect(results).toEqual(wants);
  });

  it("refuses an n past the device's storage-binding limit instead of answering wrongly", async () => {
    // Found by `pnpm bench:scan`, and it is the nastiest failure this module can have: a
    // binding over `maxStorageBufferBindingSize` is a validation error, and a validation
    // error is silence — the bind group is invalid, the dispatch does nothing, and the
    // readback returns whatever the pooled buffer held from the PREVIOUS call. Measured
    // before the fix: a 33M-row compaction after a 20M one reported 3,913,615 passing rows
    // (the 20M answer) in 7 ms, with the right shape and the wrong number.
    const device = await getDevice();
    const maxRows = Math.floor(device.limits.maxStorageBufferBindingSize / 4);
    // Host-side only — this must throw before a single device buffer is allocated.
    await expect(streamCompactGpu(new Uint8Array(maxRows + 1))).rejects.toThrow(/maxStorageBufferBindingSize/);
  });

  it("crosses 65535 workgroups for real", async () => {
    // The tests above drive the 2-D fold with an injected limit, which proves the
    // arithmetic but not that the real ceiling is where we think it is. This one actually
    // goes past it: one thread per row means ceil(n / 256) workgroups, so 20M rows is
    // 78,125 — comfortably over. Under a 1-D dispatch Dawn would invalidate the command
    // buffer, the passes would never run, and the only outward sign would be a count of 0
    // arriving suspiciously fast.
    //
    // 20M is chosen for the binding limit, not for the workgroup one: the flags and offsets
    // buffers are 4 bytes per row, so 128 MiB per binding caps this at 33,554,432 rows
    // (measured — `pnpm bench:scan` runs 33M fine and the test below pins the throw just
    // past it). MDV's 35M target is on the WRONG side of that and needs `getDevice()` to
    // request a higher `maxStorageBufferBindingSize`; this adapter reports 4 GiB available.
    const n = 20_000_000;
    expect(Math.ceil(n / COMPACT_WG), "n must actually exceed the limit or this test proves nothing").toBeGreaterThan(65535);

    const rnd = mulberry32(0x65535);
    // A byte mask with MDV's semantics, which is also the cheapest to upload at this size.
    const mask = new Uint8Array(n);
    for (let i = 0; i < n; i++) mask[i] = rnd() < 0.05 ? 0 : 1;
    const opts: CompactOptions = { pass: "eq", value: 0 };
    const want = compactCpu(mask, opts);
    const { indices, count } = await streamCompactGpu(mask, opts);
    const got = checksumOf(indices);

    expect(count).toBe(want.count);
    // ~1M passing rows, and a checksum over indices spread across the whole 20M range —
    // every silent-failure mode this repo has hit lands on zeroes, which none of these are.
    expect(count).toBeGreaterThan(900_000);
    expect(got.checksum).toBe(want.checksum);
    expect(got.ascending).toBe(true);
    expect(indices[count - 1]).toBeGreaterThan(19_900_000);
  }, 120_000);
});

describe("encodeCompact", () => {
  function upload(device: GPUDevice, key: string, mask: MaskArray): GPUBuffer {
    const bytes = Math.max(4, maskWords(maskEncodingOf(mask), mask.length) * 4);
    const buf = ensureBuf(device, key, bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    uploadMask(device, buf, mask);
    return buf;
  }
  async function readResult(device: GPUDevice, key: string, r: EncodedCompact) {
    const count = new Uint32Array(await readBack(device, `${key}:count`, r.countBuf, 0, 4))[0]!;
    const indices = count > 0 ? new Uint32Array(await readBack(device, `${key}:idx`, r.indices, 0, count * 4)) : new Uint32Array(0);
    return { count, indices };
  }
  /** One compaction recorded into a fresh encoder, submitted once, read back — what a fused
   *  consumer does, minus its own passes. */
  async function compactViaEncoder(mask: MaskArray, opts: EncodeCompactOptions = {}) {
    const ctx = await getCompactCtx();
    const { device } = ctx.scan;
    const maskBuf = upload(device, "test:encMask", mask);
    const enc = device.createCommandEncoder();
    const r = encodeCompact(ctx, maskBuf, mask.length, enc, { ...opts, mask: maskEncodingOf(mask) });
    device.queue.submit([enc.finish()]);
    return readResult(device, "test:enc", r);
  }

  // Every encoding x both comparisons, each against the CPU golden AND the two-submit path,
  // so a disagreement localises: golden differs → the kernels; only the two paths differ →
  // the sequencing that is new here.
  const CASES: Array<{ name: string; mk: (n: number, rnd: () => number) => MaskArray; opts: CompactOptions }> = [
    { name: "u32 gt 0", mk: (n, rnd) => Uint32Array.from({ length: n }, () => (rnd() < 0.3 ? 1 : 0)), opts: {} },
    { name: "u32 eq 2", mk: (n, rnd) => Uint32Array.from({ length: n }, () => Math.floor(rnd() * 4)), opts: { pass: "eq", value: 2 } },
    { name: "u8 eq 0 (MDV)", mk: (n, rnd) => Uint8Array.from({ length: n }, () => Math.floor(rnd() * 4)), opts: { pass: "eq", value: 0 } },
    { name: "u8 gt 1", mk: (n, rnd) => Uint8Array.from({ length: n }, () => Math.floor(rnd() * 4)), opts: { pass: "gt", value: 1 } },
    { name: "f32 gt 0.5", mk: (n, rnd) => Float32Array.from({ length: n }, () => rnd()), opts: { pass: "gt", value: 0.5 } },
    { name: "f32 eq 1", mk: (n, rnd) => Float32Array.from({ length: n }, () => (rnd() < 0.2 ? 1 : rnd())), opts: { pass: "eq", value: 1 } },
  ];
  for (const [ci, c] of CASES.entries()) {
    it(`matches streamCompactGpu and the golden: ${c.name}`, async () => {
      const n = 3 * SCAN_BLOCK + 5 + ci; // off every boundary, and a u8 tail of 1-3 bytes
      const mask = c.mk(n, mulberry32(0xe0c0 + ci));
      const want = compactCpu(mask, c.opts);
      const two = await streamCompactGpu(mask, c.opts);
      const one = await compactViaEncoder(mask, c.opts);
      const got = checksumOf(one.indices);

      expect(one.count, "count vs golden").toBe(want.count);
      expect(want.count, "degenerate case").toBeGreaterThan(n / 10);
      expect(got.checksum, "checksum vs golden").toBe(want.checksum);
      expect(got.ascending).toBe(true);
      let mismatch = 0;
      for (let i = 0; i < two.count; i++) if (two.indices[i] !== one.indices[i]) mismatch++;
      expect(mismatch, "mismatches vs streamCompactGpu").toBe(0);
    });
  }

  it("leaves the count on the device as a u32 the caller reads back", async () => {
    const { count, indices } = await compactViaEncoder(new Uint32Array([0, 1, 0, 0, 1, 1, 0, 1]));
    expect(count).toBe(4);
    expect(Array.from(indices)).toEqual([1, 4, 5, 7]);
  });

  it("records only a zero count for n=0", async () => {
    const { count } = await compactViaEncoder(new Uint32Array(0));
    expect(count).toBe(0);
  });

  it("reports zero for a mask that selects nothing", async () => {
    const { count } = await compactViaEncoder(new Uint32Array(5000));
    expect(count).toBe(0);
  });

  it("keeps two compactions in one submit apart under distinct keyPrefixes", async () => {
    // The regression this guards: pool keys are global, so two compactions under one prefix
    // in one command buffer share flags/offsets/out and the second silently wins both.
    const ctx = await getCompactCtx();
    const { device } = ctx.scan;
    const nA = 2 * SCAN_BLOCK + 9;
    const nB = 5 * SCAN_BLOCK + 3;
    const rA0 = mulberry32(0xa);
    const rB0 = mulberry32(0xb);
    const a = Uint32Array.from({ length: nA }, () => (rA0() < 0.5 ? 1 : 0));
    const b = Float32Array.from({ length: nB }, () => rB0());
    const optsB: CompactOptions = { pass: "gt", value: 0.25 };
    const wantA = compactCpu(a);
    const wantB = compactCpu(b, optsB);

    const enc = device.createCommandEncoder();
    const rA = encodeCompact(ctx, upload(device, "test:maskA", a), nA, enc, { mask: "u32", keyPrefix: "test:A" });
    const rB = encodeCompact(ctx, upload(device, "test:maskB", b), nB, enc, { ...optsB, mask: "f32", keyPrefix: "test:B" });
    device.queue.submit([enc.finish()]);
    const gotA = await readResult(device, "test:A", rA);
    const gotB = await readResult(device, "test:B", rB);

    expect(rA.indices, "distinct output buffers").not.toBe(rB.indices);
    expect([gotA.count, gotB.count]).toEqual([wantA.count, wantB.count]);
    expect([checksumOf(gotA.indices).checksum, checksumOf(gotB.indices).checksum]).toEqual([wantA.checksum, wantB.checksum]);
  });

  it("is unchanged when the dispatch is forced into a 2-D grid", async () => {
    const n = 7000; // 28 workgroups at 256 threads, folded into 6x5
    const rnd = mulberry32(0x2d);
    const mask = Uint32Array.from({ length: n }, () => (rnd() < 0.4 ? 1 : 0));
    const want = compactCpu(mask);
    const { count, indices } = await compactViaEncoder(mask, { maxWorkgroupsPerDim: 6 });
    expect(count).toBe(want.count);
    expect(checksumOf(indices).checksum).toBe(want.checksum);
  });
});
