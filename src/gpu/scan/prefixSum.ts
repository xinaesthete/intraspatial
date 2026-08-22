// GPU exclusive prefix sum (scan) — the blocking primitive for ADR-0005's `support`
// facet (dense mask → compact index list) and ADR-0004's 3D uniform-grid spatial index
// (scan + counting sort). MDV wants the same kernel from the other side; see
// `docs/mdv-dimension-vs-support-facet.md` §8.
//
// **WGSL template, not `"use gpu"` TGSL.** A multi-workgroup scan needs workgroup shared
// memory and barriers, which ADR-0003 explicitly routes to WGSL templates. The rest of
// that ADR's pattern is followed exactly: one explicit bind-group layout, pipelines built
// ONCE and bound to the layout (so buffer growth never rebuilds them), pooled grow-only
// buffers that are never `.destroy()`ed, bind groups recreated per call.
//
// ## Shape: two-level (recursive) scan, not decoupled lookback
//
// Three passes per level: scan each block into `dst` while writing the block's total to
// `blockSums`; scan `blockSums` (the same kernel, one level up) until a level has a single
// block; then walk back down adding each level's scanned block offsets into the level
// below. At 1024 elements per block that is 3 levels for 35M rows and 2 for anything under
// a million, so the recursion is shallow and every level after the first is tiny.
//
// Decoupled lookback would be one pass instead of ~2n reads, but it needs forward progress
// guarantees between workgroups that WebGPU does not give — there is no spec guarantee that
// workgroup k is scheduled while workgroup k+1 spins waiting for it, so a lookback spin can
// deadlock on a device that serialises workgroups. The recursive form has no such
// dependency and its extra bandwidth is not the bottleneck at these sizes.
//
// ## Silent-failure hazards this file is built around
//
// Every one of these returns zeroes and looks fast and valid — a scan of zeros IS zeros, so
// a tolerance check alone cannot see any of them. Tests assert non-zero as well as correct.
//   - `maxComputeWorkgroupsPerDimension` is 65535. A 1-D dispatch past it does not clamp,
//     it invalidates the command buffer and the kernel never runs. Every dispatch here is a
//     2-D grid folded back to a linear block index in the shader (`dispatchGrid`).
//   - More than 8 storage bindings per stage makes the LAYOUT invalid and every dispatch a
//     silent no-op. This layout uses 3.
//   - A dispatch past ~2s is killed with no error. Measured, not assumed: `pnpm bench:scan`
//     on this machine gives 46.6 ms for a 33M-element scan and 10.4 ms for a 33M-row
//     compaction at ~6% selectivity, and both of those figures are dominated by the
//     readback rather than the dispatch. Two orders of magnitude of headroom.
//   - A storage binding past `maxStorageBufferBindingSize` (128 MiB by default, i.e. ~33.5M
//     rows) is a validation error, i.e. silence. Every binding here carries an explicit
//     `size` and `checkBindingSize` throws above the limit — see the note on the bind
//     groups for the wrong answer that cost.
import { checkBindingSize, compileShader, dispatchGrid, getDevice, MAX_WORKGROUPS_PER_DIM, sized } from "../device";

// Re-exported: they live in `device.ts` because every kernel with a per-element dispatch needs
// them, but this module and its tests were their first caller.
export { dispatchGrid, MAX_WORKGROUPS_PER_DIM };

/** Threads per workgroup. */
export const SCAN_WG = 256;
/** Elements each thread scans serially before the workgroup-wide step. Serial work is far
 *  cheaper than tree work, so widening the block this way costs one register array and
 *  divides the block count (and hence the recursion depth) by 4. */
export const SCAN_PER_THREAD = 4;
/** Elements per workgroup. */
export const SCAN_BLOCK = SCAN_WG * SCAN_PER_THREAD;

export type ScanElement = "u32" | "f32";

export interface ScanOptions {
  /** Test seam, as in `pcaGpu.ts`'s `rowsPerTile`: pretend the device's
   *  `maxComputeWorkgroupsPerDimension` is this, so the 2-D fold can be exercised without
   *  allocating the ~67M elements it takes to cross the real limit. */
  readonly maxWorkgroupsPerDim?: number;
}

// ---------------------------------------------------------------------------------------
// The kernels
// ---------------------------------------------------------------------------------------

/** One shader per element type. The two differ only in the element type and its zero, so
 *  they come from one template rather than two hand-kept copies. */
function scanShader(elem: ScanElement): string {
  const zero = elem === "u32" ? "0u" : "0.0";
  return /* wgsl */ `
struct Uni {
  // Elements at THIS level of the recursion, not the caller's n.
  n: u32,
  numBlocks: u32,
  gridX: u32,
  pad: u32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> src: array<${elem}>;
@group(0) @binding(2) var<storage, read_write> dst: array<${elem}>;
@group(0) @binding(3) var<storage, read_write> blockSums: array<${elem}>;

var<workgroup> sdata: array<${elem}, ${SCAN_WG}>;

// Exclusive scan of one block of ${SCAN_BLOCK} elements, plus the block's total.
//
// The early return is on 'workgroup_id', which is uniform within a workgroup, so the
// barriers below still sit in uniform control flow. (A return on 'global_invocation_id'
// would not, and Tint would reject it.)
@compute @workgroup_size(${SCAN_WG})
fn scanBlocks(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let block = wid.x + wid.y * U.gridX;
  if (block >= U.numBlocks) { return; }
  let base = block * ${SCAN_BLOCK}u;
  let t = lid.x;

  // 1. Serial exclusive scan over this thread's ${SCAN_PER_THREAD} contiguous elements.
  //    Reads past n contribute zero, which is what makes a partial final block correct
  //    without a separate epilogue.
  var pre: array<${elem}, ${SCAN_PER_THREAD}>;
  var sum = ${zero};
  for (var k = 0u; k < ${SCAN_PER_THREAD}u; k = k + 1u) {
    let idx = base + t * ${SCAN_PER_THREAD}u + k;
    var v = ${zero};
    if (idx < U.n) { v = src[idx]; }
    pre[k] = sum;
    sum = sum + v;
  }

  // 2. Workgroup-wide scan over the per-thread totals: Hillis-Steele inclusive, then
  //    shifted by one to make it exclusive. Every barrier is unconditional.
  sdata[t] = sum;
  workgroupBarrier();
  var offset = 1u;
  loop {
    if (offset >= ${SCAN_WG}u) { break; }
    var add = ${zero};
    if (t >= offset) { add = sdata[t - offset]; }
    workgroupBarrier();
    if (t >= offset) { sdata[t] = sdata[t] + add; }
    workgroupBarrier();
    offset = offset << 1u;
  }
  var threadOffset = ${zero};
  if (t > 0u) { threadOffset = sdata[t - 1u]; }
  let blockTotal = sdata[${SCAN_WG}u - 1u];

  // 3. Write the block's slice, and hand its total to the level above.
  for (var k = 0u; k < ${SCAN_PER_THREAD}u; k = k + 1u) {
    let idx = base + t * ${SCAN_PER_THREAD}u + k;
    if (idx < U.n) { dst[idx] = threadOffset + pre[k]; }
  }
  if (t == 0u) { blockSums[block] = blockTotal; }
}

// Add the scanned block offsets from the level above into this level's partial scan.
// 'src' is bound to the parent level's scanned output; 'blockSums' is unused here and is
// bound only because one layout serves both entry points.
@compute @workgroup_size(${SCAN_WG})
fn addOffsets(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let block = wid.x + wid.y * U.gridX;
  if (block >= U.numBlocks) { return; }
  let add = src[block];
  let base = block * ${SCAN_BLOCK}u;
  for (var k = 0u; k < ${SCAN_PER_THREAD}u; k = k + 1u) {
    let idx = base + lid.x * ${SCAN_PER_THREAD}u + k;
    if (idx < U.n) { dst[idx] = dst[idx] + add; }
  }
}
`;
}

interface ScanPipes {
  scanBlocks: GPUComputePipeline;
  addOffsets: GPUComputePipeline;
}

export interface ScanCtx {
  device: GPUDevice;
  layout: GPUBindGroupLayout;
  pipes: Record<ScanElement, ScanPipes>;
}

let ctxCache: Promise<ScanCtx> | undefined;

export function getScanCtx(): Promise<ScanCtx> {
  ctxCache ??= (async () => {
    const device = await getDevice();
    // One explicit layout shared by both entry points and both element types. `layout:
    // "auto"` derives a different layout per entry point — only the bindings that entry
    // point happens to touch — so no single bind group could satisfy `addOffsets`, which
    // never mentions `blockSums`.
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const build = async (elem: ScanElement): Promise<ScanPipes> => {
      const module = await compileShader(device, scanShader(elem), `prefixSum:${elem}`);
      const mk = (entryPoint: string) => device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
      return { scanBlocks: mk("scanBlocks"), addOffsets: mk("addOffsets") };
    };
    return { device, layout, pipes: { u32: await build("u32"), f32: await build("f32") } };
  })();
  return ctxCache;
}

// ---------------------------------------------------------------------------------------
// Pooled buffers
// ---------------------------------------------------------------------------------------

/** Grow-only, never destroyed — `.destroy()` on a pooled buffer is what ADR-0017's
 *  double-free was about, and the pool outlives every call by design. Shared with
 *  `streamCompact.ts`, which is why it is exported. */
const pool = new Map<string, { buf: GPUBuffer; cap: number }>();

export function ensureBuf(device: GPUDevice, key: string, bytes: number, usage: number): GPUBuffer {
  const got = pool.get(key);
  if (got && got.cap >= bytes) return got.buf;
  const cap = Math.max(bytes, (got?.cap ?? 0) * 2, 256);
  const buf = device.createBuffer({ size: cap, usage, label: key });
  pool.set(key, { buf, cap });
  return buf;
}

/** Not a module-level constant: `GPUBufferUsage` and friends are installed as globals by
 *  `getDevice()`, so anything evaluated at import time sees `undefined`. */
const storageRw = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

/** Copy `bytes` from `src` at `srcOffset` into a pooled staging buffer and map it.
 *
 *  Raw `mapAsync` on a pooled `MAP_READ` buffer used to kill the vitest worker; that was
 *  the Dawn Instance-lifetime bug in `device.ts`, fixed 2026-07-29 and re-tested in
 *  `test/dawn-limits-sweep.gpu.test.ts`, which pins exactly this pattern. It is used here
 *  rather than TypeGPU's `.read()` because the compacted index list has a length that is
 *  only known at run time, and `.read()` reads the whole wrapper — a pooled wrapper sized
 *  for the worst case would move 140 MB to return a thousand indices. */
export async function readBack(device: GPUDevice, key: string, src: GPUBuffer, srcOffset: number, bytes: number): Promise<ArrayBuffer> {
  const staging = ensureBuf(device, key, bytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, srcOffset, staging, 0, bytes);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ, 0, bytes);
  const copy = staging.getMappedRange(0, bytes).slice(0);
  staging.unmap();
  return copy;
}

// ---------------------------------------------------------------------------------------
// Encoding the scan
// ---------------------------------------------------------------------------------------

interface Level {
  src: GPUBuffer;
  dst: GPUBuffer;
  blockSums: GPUBuffer;
  n: number;
  numBlocks: number;
  gridX: number;
  gridY: number;
}

export interface EncodedScan {
  /** Exclusive scan of the input, `n` elements. */
  dst: GPUBuffer;
  /** One element: the sum of every input element (the value an inclusive scan would have
   *  put after the last position). Falls out of the top level's block sums for free. */
  totalBuf: GPUBuffer;
}

/**
 * Record an exclusive scan of `src` (`n` elements) into `enc`. The caller owns submission,
 * so a consumer can fuse the scan with its own passes in one command buffer.
 *
 * `src` must be a distinct buffer from the returned `dst`: the scan is not in-place. It
 * could be — each block reads its own range before the barrier and writes it after — but
 * that needs the same buffer bound as both `read-only-storage` and `storage` in one bind
 * group, which is aliasing WebGPU does not define.
 *
 * The pool keys are global, so two scans recorded into one command buffer under the same
 * `keyPrefix` alias each other's `dst` and the second silently overwrites the first. A caller
 * that fuses more than one scan per submit (an index build plus an occupied-cell compaction,
 * say) gives each its own prefix; the default serves the one-scan-per-submit case.
 */
export function encodeScan(
  ctx: ScanCtx,
  elem: ScanElement,
  src: GPUBuffer,
  n: number,
  enc: GPUCommandEncoder,
  maxWorkgroupsPerDim: number = MAX_WORKGROUPS_PER_DIM,
  keyPrefix = "scan",
): EncodedScan {
  const { device } = ctx;
  checkBindingSize(device, `prefixSum: ${n} elements`, n * 4);

  const levels: Level[] = [];
  let cur = src;
  let curN = n;
  for (let li = 0; ; li++) {
    const numBlocks = Math.ceil(curN / SCAN_BLOCK);
    const dst = ensureBuf(device, `${keyPrefix}:${elem}:dst${li}`, curN * 4, storageRw());
    const blockSums = ensureBuf(device, `${keyPrefix}:${elem}:bs${li}`, numBlocks * 4, storageRw());
    const { x: gridX, y: gridY } = dispatchGrid(numBlocks, maxWorkgroupsPerDim);
    levels.push({ src: cur, dst, blockSums, n: curN, numBlocks, gridX, gridY });
    if (numBlocks <= 1) break;
    cur = blockSums;
    curN = numBlocks;
  }

  // A uniform buffer PER LEVEL, not one rewritten between passes. `queue.writeBuffer` is
  // ordered against submits, not against commands within a submit, so a single reused
  // uniform would leave every pass reading the LAST level's n — the exact bug
  // `umapLayoutGpu.ts` documents, where the work still happens and the answer is merely
  // wrong. Per-level buffers make the ordering question disappear.
  const binds = levels.map((lv, li) => {
    const uni = ensureBuf(device, `${keyPrefix}:${elem}:uni${li}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(uni, 0, new Uint32Array([lv.n, lv.numBlocks, lv.gridX, 0]));
    // Every binding carries an explicit `size`. Without it the binding covers the whole
    // POOLED buffer, which is grow-only and routinely larger than this call needs — and a
    // binding past `maxStorageBufferBindingSize` invalidates the bind group, which
    // invalidates the command buffer, which makes the whole scan a silent no-op returning
    // whatever the buffer held before. Measured: a 33M-row compaction after a 20M one bound
    // a 160 MB pooled buffer for 132 MB of data, sailed past the 128 MiB limit, and
    // reported a plausible wrong count in 7 ms.
    const group = (srcBuf: GPUBuffer, srcBytes: number) =>
      device.createBindGroup({
        layout: ctx.layout,
        entries: [
          { binding: 0, resource: sized(uni, 16) },
          { binding: 1, resource: sized(srcBuf, srcBytes) },
          { binding: 2, resource: sized(lv.dst, lv.n * 4) },
          { binding: 3, resource: sized(lv.blockSums, lv.numBlocks * 4) },
        ],
      });
    // `scan` reads this level's input; `add` reads the level above's scanned output, which
    // has exactly one element per block of this level.
    const parent = levels[li + 1];
    return { scan: group(lv.src, lv.n * 4), add: parent ? group(parent.dst, lv.numBlocks * 4) : undefined };
  });

  const pipes = ctx.pipes[elem];
  const pass = enc.beginComputePass({ label: `scan:${elem}` });
  // WebGPU orders dispatches within a compute pass and inserts the memory barriers between
  // them, so no explicit synchronisation is needed between levels.
  for (let li = 0; li < levels.length; li++) {
    pass.setPipeline(pipes.scanBlocks);
    pass.setBindGroup(0, binds[li]!.scan);
    pass.dispatchWorkgroups(levels[li]!.gridX, levels[li]!.gridY);
  }
  for (let li = levels.length - 2; li >= 0; li--) {
    pass.setPipeline(pipes.addOffsets);
    pass.setBindGroup(0, binds[li]!.add!);
    pass.dispatchWorkgroups(levels[li]!.gridX, levels[li]!.gridY);
  }
  pass.end();

  const top = levels[levels.length - 1]!;
  return { dst: levels[0]!.dst, totalBuf: top.blockSums };
}

// ---------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------

export interface ScanResult<T> {
  /** Exclusive prefix sum: `scan[i]` is the sum of `values[0..i)`, so `scan[0]` is 0. */
  scan: T;
  /** Sum of every element — one past the end of the inclusive scan. */
  total: number;
}

/**
 * Exclusive prefix sum of `values` on the GPU.
 *
 * `u32` sums wrap at 2^32 exactly as the type says; `f32` accumulates in tree order, which
 * is a different (generally better-conditioned) rounding than a serial host loop, so f32
 * results should be compared with a tolerance rather than for equality.
 */
export function exclusiveScanGpu(values: Uint32Array, opts?: ScanOptions): Promise<ScanResult<Uint32Array>>;
export function exclusiveScanGpu(values: Float32Array, opts?: ScanOptions): Promise<ScanResult<Float32Array>>;
export async function exclusiveScanGpu(
  values: Uint32Array | Float32Array,
  opts: ScanOptions = {},
): Promise<ScanResult<Uint32Array | Float32Array>> {
  const elem: ScanElement = values instanceof Float32Array ? "f32" : "u32";
  const n = values.length;
  if (n === 0) {
    return { scan: elem === "f32" ? new Float32Array(0) : new Uint32Array(0), total: 0 };
  }

  const ctx = await getScanCtx();
  const { device } = ctx;
  const input = ensureBuf(device, `scan:${elem}:in`, n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(input, 0, values.buffer, values.byteOffset, n * 4);

  const enc = device.createCommandEncoder();
  const { dst, totalBuf } = encodeScan(ctx, elem, input, n, enc, opts.maxWorkgroupsPerDim);
  device.queue.submit([enc.finish()]);

  const scanBytes = await readBack(device, "scan:staging", dst, 0, n * 4);
  const totalBytes = await readBack(device, "scan:total", totalBuf, 0, 4);
  const view = elem === "f32" ? new Float32Array(scanBytes) : new Uint32Array(scanBytes);
  const total = (elem === "f32" ? new Float32Array(totalBytes) : new Uint32Array(totalBytes))[0]!;
  return { scan: view, total };
}
