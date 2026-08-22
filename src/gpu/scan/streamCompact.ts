// GPU stream compaction — turn a per-row mask into the compact list of passing row
// indices, plus the count. This is ADR-0005's `materializeSupport` in its mask → index
// direction, and MDV already ships the host twin of it (`getFilteredIndices` →
// `filteredIndexWorker.ts`, off-thread and promise-cached), so the shape of the answer is
// borrowed from something proven rather than invented here.
//
// Three passes over `src/gpu/scan/prefixSum.ts`:
//   predicate  mask → 0/1 flags
//   scan       flags → the offset each passing row writes to, and the total
//   scatter    passing row i writes its own index at offsets[i]
//
// ## Mask encodings, and why there are three
//
// The point of compaction is to delete an O(N) main-thread pass, so an API that needs an
// O(N) main-thread pass to convert the mask first would give the cost straight back. The
// two real callers disagree about the encoding:
//   - MDV's `Dimension.filterArray` is a byte per row over a SharedArrayBuffer, and a row
//     passes iff the byte is **0** (1/2/3 are the local / background / both exclusions).
//     Uploaded as raw words and unpacked 4 rows per word, that is a 35 MB transfer at 35M
//     rows instead of 140 MB, with no host conversion at all.
//   - ADR-0005's soft `support` mask is one f32 weight per row in [0,1], where a hard
//     filter is the boxcar case. Read through `bitcast`, so it shares the one binding.
// Hence `enc` (u8 / u32 / f32) x `cmp` (== / >) in the uniform. `> 0` is the default and
// is the sensible reading of both a 0/1 mask and a soft weight with no threshold.
//
// ## Two submits, not one
//
// The output buffer is sized to the count, which is only known after the scan, so this
// reads back the 4-byte total, sizes the buffer, then scatters. The alternative — one
// submit into a worst-case buffer — costs 140 MB at 35M rows to return however few indices
// actually pass, which is the wrong trade for the sparse case the compact encoding exists
// to serve. A zero count skips the scatter dispatch entirely.
//
// ## MDV's 35M rows does not fit yet, and this is where that is recorded
//
// The flags and offsets buffers are 4 bytes per row, so one storage binding caps N at
// `maxStorageBufferBindingSize / 4`. Measured on this machine: the DEVICE reports
// 134,217,728 (the spec default) = **33,554,432 rows**, while the ADAPTER reports
// 4,294,967,295 = 1.07B rows. So the ceiling is not the hardware, it is that `getDevice()`
// requests default limits — and 35M is just past it. Raising it is a device-wide decision
// (every other kernel shares that device) and belongs with whoever wires MDV up, not here.
// Until then `checkBindingSize` throws with the number and the remedy, because the
// alternative is the silent wrong answer described on the bind groups below.
import { compileShader } from "../device";
import { checkBindingSize, dispatchGrid, encodeScan, ensureBuf, getScanCtx, MAX_WORKGROUPS_PER_DIM, type ScanCtx } from "./prefixSum";

/** One thread per row in the predicate and scatter passes. Exported so a test can state
 *  the size at which `ceil(n / WG)` crosses 65535 without hardcoding it. */
export const COMPACT_WG = 256;
const WG = COMPACT_WG;

/** How a row's mask entry is laid out in the uploaded words. */
const ENC = { u8: 0, u32: 1, f32: 2 } as const;
/** How it is compared against `value`. */
const CMP = { eq: 0, gt: 1 } as const;

export type MaskArray = Uint8Array | Uint32Array | Float32Array;

export interface CompactOptions {
  /** `"gt"` (default) selects rows whose mask entry is greater than `value`; `"eq"` selects
   *  rows equal to it. `{ pass: "eq", value: 0 }` is MDV's `filterArray[row] === 0`. */
  readonly pass?: "gt" | "eq";
  /** Threshold or exact value. Default 0, so a 0/1 mask and a soft weight both mean
   *  "anything non-zero passes". */
  readonly value?: number;
  /** Test seam — see `ScanOptions.maxWorkgroupsPerDim` in `prefixSum.ts`. */
  readonly maxWorkgroupsPerDim?: number;
}

export interface CompactResult {
  /** Ascending row indices of the passing rows. Exactly `count` long. */
  indices: Uint32Array;
  /** Number of passing rows. */
  count: number;
}

const SHADER = /* wgsl */ `
struct Uni {
  n: u32,
  gridX: u32,
  enc: u32,
  cmp: u32,
  // Bit pattern of the comparison value: an integer for u8/u32, an f32's bits for f32.
  // One field rather than two so the struct stays 32 bytes and the two paths cannot
  // disagree about which one was written.
  refBits: u32,
  pad0: u32, pad1: u32, pad2: u32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> outIdx: array<u32>;

// WebGPU buffers are little-endian, so byte i of an uploaded Uint8Array lands in word
// i >> 2 at bit (i & 3) * 8. That is what makes the u8 path a plain reinterpretation of
// MDV's SharedArrayBuffer rather than a repack.
fn passes(i: u32) -> bool {
  if (U.enc == ${ENC.u8}u) {
    let v = (mask[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
    return select(v == U.refBits, v > U.refBits, U.cmp == ${CMP.gt}u);
  }
  if (U.enc == ${ENC.u32}u) {
    let v = mask[i];
    return select(v == U.refBits, v > U.refBits, U.cmp == ${CMP.gt}u);
  }
  let v = bitcast<f32>(mask[i]);
  let r = bitcast<f32>(U.refBits);
  return select(v == r, v > r, U.cmp == ${CMP.gt}u);
}

// The 2-D fold, for the same reason as in prefixSum.ts: one thread per row means
// ceil(n / ${WG}) workgroups, which passes 65535 at 16.8M rows — a size MDV is well past.
// Over the limit the command buffer is invalidated and the pass silently does nothing.
fn rowOf(lid: vec3u, wid: vec3u) -> u32 {
  return (wid.x + wid.y * U.gridX) * ${WG}u + lid.x;
}

@compute @workgroup_size(${WG})
fn predicate(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let i = rowOf(lid, wid);
  if (i >= U.n) { return; }
  flags[i] = select(0u, 1u, passes(i));
}

@compute @workgroup_size(${WG})
fn scatter(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let i = rowOf(lid, wid);
  if (i >= U.n) { return; }
  // 'flags' rather than a second call to 'passes': the two must agree exactly or rows are
  // dropped or double-written, and reading back the value the scan actually consumed is
  // the only way to guarantee that.
  if (flags[i] == 1u) { outIdx[offsets[i]] = i; }
}
`;

interface CompactCtx {
  scan: ScanCtx;
  layout: GPUBindGroupLayout;
  predicate: GPUComputePipeline;
  scatter: GPUComputePipeline;
}
let ctxCache: Promise<CompactCtx> | undefined;

function getCtx(): Promise<CompactCtx> {
  ctxCache ??= (async () => {
    const scan = await getScanCtx();
    const { device } = scan;
    const module = await compileShader(device, SHADER, "streamCompact");
    // 4 storage bindings. Nine would make the layout invalid and every dispatch a silent
    // no-op — the failure that reads as a maths bug.
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const mk = (entryPoint: string) => device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
    return { scan, layout, predicate: mk("predicate"), scatter: mk("scatter") };
  })();
  return ctxCache;
}

/** Words the mask occupies on the device, and its encoding. */
function encodingOf(mask: MaskArray): { enc: number; words: number } {
  if (mask instanceof Uint8Array) return { enc: ENC.u8, words: Math.ceil(mask.length / 4) };
  return { enc: mask instanceof Float32Array ? ENC.f32 : ENC.u32, words: mask.length };
}

/** Upload the mask verbatim. `writeBuffer` needs a size that is a multiple of 4, so a byte
 *  mask whose length is not writes its aligned prefix as a view (no copy — the point of
 *  the u8 path) and its 1-3 byte tail separately. */
function uploadMask(device: GPUDevice, buf: GPUBuffer, mask: MaskArray): void {
  if (!(mask instanceof Uint8Array)) {
    device.queue.writeBuffer(buf, 0, mask as unknown as BufferSource, 0, mask.length);
    return;
  }
  const aligned = mask.length & ~3;
  if (aligned > 0) device.queue.writeBuffer(buf, 0, mask as unknown as BufferSource, 0, aligned);
  if (aligned < mask.length) {
    const tail = new Uint8Array(4);
    tail.set(mask.subarray(aligned));
    device.queue.writeBuffer(buf, aligned, tail);
  }
}

/**
 * Compact a per-row mask into the ascending list of passing row indices.
 *
 * The result is stable and ordered — index `k` of the output is the `k`th passing row —
 * because the scan gives every row its own slot rather than an atomic bump. That is what
 * makes it usable as a `support` encoding: two runs over the same mask give byte-identical
 * lists, so a content-addressed memo over it is meaningful.
 */
export async function streamCompactGpu(mask: MaskArray, opts: CompactOptions = {}): Promise<CompactResult> {
  const n = mask.length;
  if (n === 0) return { indices: new Uint32Array(0), count: 0 };

  const ctx = await getCtx();
  const { device } = ctx.scan;
  // Before anything is allocated: the flags and offsets buffers are 4 bytes per row, and
  // over the binding limit every dispatch below silently does nothing.
  checkBindingSize(device, "streamCompact", n);
  const maxPerDim = opts.maxWorkgroupsPerDim ?? MAX_WORKGROUPS_PER_DIM;
  const { enc, words } = encodingOf(mask);
  const cmp = opts.pass === "eq" ? CMP.eq : CMP.gt;
  const value = opts.value ?? 0;
  // f32 compares against the bits of the threshold; u8/u32 against the integer itself.
  const refBits = enc === ENC.f32 ? new Uint32Array(Float32Array.of(value).buffer)[0]! : value >>> 0;

  const maskBuf = ensureBuf(device, "compact:mask", words * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  uploadMask(device, maskBuf, mask);

  const flags = ensureBuf(device, "compact:flags", n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  const uni = ensureBuf(device, "compact:uni", 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const grid = dispatchGrid(Math.ceil(n / WG), maxPerDim);
  device.queue.writeBuffer(uni, 0, new Uint32Array([n, grid.x, enc, cmp, refBits, 0, 0, 0]));

  // --- pass 1 + 2: predicate, then scan the flags in the same command buffer ------------
  const enc1 = device.createCommandEncoder();
  {
    // `offsets` and `outIdx` do not exist yet — the scan has not run and the output is not
    // yet sized, which is the point of the two submits — but every binding in the layout
    // must still be filled. They get their own scratch buffers rather than a spare pointer
    // at `flags`: WebGPU rejects a bind group that aliases one buffer across two writable
    // bindings (or a writable and a readable one), and the rejection invalidates the whole
    // command buffer, so the predicate would simply never run.
    const scratchRo = ensureBuf(device, "compact:scratchRo", 256, GPUBufferUsage.STORAGE);
    const scratchRw = ensureBuf(device, "compact:scratchRw", 256, GPUBufferUsage.STORAGE);
    // Explicit `size` on every binding — see the same note in `prefixSum.ts`. The pool is
    // grow-only, so binding a whole buffer binds however big a previous call made it.
    const bind = device.createBindGroup({
      layout: ctx.layout,
      entries: [
        { binding: 0, resource: { buffer: uni, size: 32 } },
        { binding: 1, resource: { buffer: maskBuf, size: words * 4 } },
        { binding: 2, resource: { buffer: flags, size: n * 4 } },
        { binding: 3, resource: { buffer: scratchRo, size: 256 } },
        { binding: 4, resource: { buffer: scratchRw, size: 256 } },
      ],
    });
    const pass = enc1.beginComputePass({ label: "compact:predicate" });
    pass.setPipeline(ctx.predicate);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(grid.x, grid.y);
    pass.end();
  }
  const { dst: offsets, totalBuf } = encodeScan(ctx.scan, "u32", flags, n, enc1, maxPerDim);
  device.queue.submit([enc1.finish()]);

  const count = new Uint32Array(await readWord(device, totalBuf))[0]!;
  if (count === 0) return { indices: new Uint32Array(0), count: 0 };

  // --- pass 3: scatter, into a buffer sized to the answer -------------------------------
  const outBuf = ensureBuf(device, "compact:out", count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const enc2 = device.createCommandEncoder();
  const bind = device.createBindGroup({
    layout: ctx.layout,
    entries: [
      { binding: 0, resource: { buffer: uni, size: 32 } },
      { binding: 1, resource: { buffer: maskBuf, size: words * 4 } },
      { binding: 2, resource: { buffer: flags, size: n * 4 } },
      { binding: 3, resource: { buffer: offsets, size: n * 4 } },
      { binding: 4, resource: { buffer: outBuf, size: count * 4 } },
    ],
  });
  const pass = enc2.beginComputePass({ label: "compact:scatter" });
  pass.setPipeline(ctx.scatter);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(grid.x, grid.y);
  pass.end();
  device.queue.submit([enc2.finish()]);

  const staging = ensureBuf(device, "compact:staging", count * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const copy = device.createCommandEncoder();
  copy.copyBufferToBuffer(outBuf, 0, staging, 0, count * 4);
  device.queue.submit([copy.finish()]);
  await staging.mapAsync(GPUMapMode.READ, 0, count * 4);
  const indices = new Uint32Array(staging.getMappedRange(0, count * 4).slice(0));
  staging.unmap();
  return { indices, count };
}

async function readWord(device: GPUDevice, src: GPUBuffer): Promise<ArrayBuffer> {
  const staging = ensureBuf(device, "compact:total", 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ, 0, 4);
  const copy = staging.getMappedRange(0, 4).slice(0);
  staging.unmap();
  return copy;
}
