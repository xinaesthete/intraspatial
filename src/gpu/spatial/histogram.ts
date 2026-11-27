// A value histogram of one GPU-resident raster plane, left on the device: counts, per-row CDF,
// edges, totals and moments, from one submit and no readback. The base for per-mode gram
// histograms, the wand's tolerance histogram and zonal (per-region) statistics.
//
// The f64 golden, the binning rule both sides share, the layout and the quantile helper (with its
// error bound) are in `src/spatial/histogram.ts` — read its header first.
//
// ## Passes (one command buffer)
//
// 1. `reduce`: every workgroup grid-strides the plane, keeping per-row finite min / max / sum in
//    registers, tree-reduces them in workgroup memory and writes one partial per workgroup. This is
//    the signed twin of `fieldMax.ts`, which only handles non-negative max (u32 `atomicMax` on the
//    f32 bits); a tree reduce needs no key mapping and gives the sum as well.
// 2. `finalize` (one workgroup): reduces the partials, picks the edges — fixed, or auto from the
//    finite [min, max] — and writes `edges` and the moments. Pass 3 reads the edges from the
//    buffer, so auto-range never comes back to the host.
// 3. `bin`: `atomicAdd` into a workgroup-private histogram, then one global `atomicAdd` per
//    non-empty bin per workgroup — far less contention than one global atomic per texel. Over
//    `PRIVATE_CAP` bins×rows the workgroup copy doesn't fit in 16 KiB, so it adds globally.
// 4. The CDF: the existing exclusive scan (`encodeScan`) over the flat `rows × bins` counts, then
//    `cdfFix` turns it into a per-row inclusive CDF and writes the mean (sum / finite count).
//
// Raw WGSL, not "use gpu": workgroup memory, barriers, atomics, and u32 division (`q / width`),
// which TGSL compiles as float division (ADR-0003; AGENTS.md).
//
// ## Input
//
// A `RasterPlane`: `height` rows of `rowWords` 4-byte words, texels f32 (one per word) or f16
// (two per word, `unpack2x16float`) — the `ResidentRasters` layout from `gramMatrix.ts`, read the
// way its `at()` reads it. A dense f32 grid is the same thing with `rowWords = width`. The optional
// mask is a dense `width × height` u32 or f32 grid; non-zero is inside (row 0), zero is outside
// (row 1). For an f32 mask the sign bit is ignored, so −0 is outside, like `=== 0` on the host.
import type { HistogramOptions, HistogramResult } from "#src/spatial/histogram";
import { HISTOGRAM_MOMENTS, HISTOGRAM_TOTALS, histogramBins, histogramFixedRange } from "#src/spatial/histogram";
import { checkBindingSize, compileShader, deviceScoped, perDevice, readBackBytes, sized } from "../device";
import type { ResidentBuffer } from "../graph/handle";
import { BufferPool } from "../graph/pool";
import { encodeScan, ensureBuf, getScanCtx } from "../scan/prefixSum";
import type { RasterPrecision, ResidentRasters } from "./gramMatrix";

const WG = 256;
/** Workgroups for the two grid-stride passes. Each thread strides, so this bounds the partials
 *  and the global merge traffic for any raster size, well under the per-dimension cap. */
const MAX_GROUPS = 256;
/** Rows supported today: one, or inside/outside with a mask. R3 (zonal) raises this. */
const MAX_ROWS = 2;
/** Workgroup-private bins: 4088 + 8 totals = 4096 words = 16 KiB, the default
 *  `maxComputeWorkgroupStorageSize`. */
const PRIVATE_CAP = 4088;

/** One plane of a raster on the device. `byteOffset` must be a multiple of 256
 *  (`minStorageBufferOffsetAlignment`); a `ResidentRasters` plane always is. */
export interface RasterPlane {
  readonly buffer: GPUBuffer;
  readonly byteOffset?: number;
  readonly width: number;
  readonly height: number;
  /** Row stride in 4-byte words (≥ width at f32, ≥ ceil(width/2) at f16). */
  readonly rowWords: number;
  readonly precision: RasterPrecision;
}

/** Plane `k` of the rasters `gramMatrixGpu` left on the device (`width`/`height` from its result). */
export function residentRasterPlane(r: ResidentRasters, k: number, width: number, height: number): RasterPlane {
  return { buffer: r.buffer, byteOffset: k * height * r.rowWords * 4, width, height, rowWords: r.rowWords, precision: r.precision };
}

/** A dense row-major f32 grid as a plane. */
export function denseRasterPlane(buffer: GPUBuffer, width: number, height: number, byteOffset = 0): RasterPlane {
  return { buffer, byteOffset, width, height, rowWords: width, precision: "f32" };
}

/** A dense `width × height` mask: non-zero inside. */
export interface RasterMask {
  readonly buffer: GPUBuffer;
  readonly byteOffset?: number;
  readonly dtype: "u32" | "f32";
}

/** Caller-owned outputs, each bound whole from offset 0; STORAGE | COPY_DST usage (plus COPY_SRC
 *  to read back). Sizes from `histogramByteSizes`. */
export interface HistogramTargets {
  /** u32 `rows × bins`. Must not alias `cdf` (the scan reads it while `cdfFix` writes `cdf`). */
  readonly counts: GPUBuffer;
  /** u32 `rows × bins`, inclusive, per row. */
  readonly cdf: GPUBuffer;
  /** u32 `rows × 4`: `[inRange, under, over, nonFinite]`. */
  readonly totals: GPUBuffer;
  /** f32 `[lo, hi]`. */
  readonly edges: GPUBuffer;
  /** f32 `rows × 4`: `[min, max, sum, mean]`; NaN for an empty row. */
  readonly moments: GPUBuffer;
}

export function histogramByteSizes(bins: number, rows: number): Record<keyof HistogramTargets, number> {
  return {
    counts: rows * bins * 4,
    cdf: rows * bins * 4,
    totals: rows * HISTOGRAM_TOTALS * 4,
    edges: 8,
    moments: rows * HISTOGRAM_MOMENTS * 4,
  };
}

// The uniform is shared by every kernel: one write, one submit. `nanBits` carries a quiet NaN's bit
// pattern because Tint rejects a NaN constant (`bitcast<f32>(0x7fc00000u)` fails const-eval).
const UNI = /* wgsl */ `
struct Uni {
  n: u32, width: u32, rowWords: u32, half: u32,
  rows: u32, hasMask: u32, maskF32: u32, bins: u32,
  fixedRange: u32, groups: u32, stride: u32, privatized: u32,
  lo: f32, hi: f32, nanBits: u32, pad0: u32,
};
@group(0) @binding(0) var<uniform> U: Uni;
`;

// Reading a texel and its row. Mirrors gramMatrix.ts's at(): the row padding is skipped by indexing.
const READ = /* wgsl */ `
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;

fn texel(q: u32) -> f32 {
  let row = q / U.width;
  let col = q - row * U.width;
  let base = row * U.rowWords;
  if (U.half != 0u) {
    let pair = unpack2x16float(src[base + (col >> 1u)]);
    return select(pair.y, pair.x, (col & 1u) == 0u);
  }
  return bitcast<f32>(src[base + col]);
}

fn rowOf(q: u32) -> u32 {
  if (U.hasMask == 0u) { return 0u; }
  var m = mask[q];
  if (U.maskF32 != 0u) { m = m & 0x7fffffffu; }
  return select(1u, 0u, m != 0u);
}

// By the exponent bits: a compiler may assume no NaNs and fold v != v away.
fn isFinite(v: f32) -> bool {
  return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}
`;

const REDUCE = /* wgsl */ `${UNI}${READ}
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;   // groups x rows x 3

var<workgroup> wMin: array<f32, ${WG}>;
var<workgroup> wMax: array<f32, ${WG}>;
var<workgroup> wSum: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn reduce(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32,
          @builtin(workgroup_id) wid: vec3u) {
  // Largest finite f32. A const +Inf is rejected by Tint, and an empty row is detected as
  // min > max rather than by the sentinel's value.
  let FMAX = bitcast<f32>(0x7f7fffffu);
  var mn = array<f32, ${MAX_ROWS}>(FMAX, FMAX);
  var mx = array<f32, ${MAX_ROWS}>(-FMAX, -FMAX);
  var sm = array<f32, ${MAX_ROWS}>(0.0, 0.0);
  for (var q = gid.x; q < U.n; q = q + U.stride) {
    let v = texel(q);
    if (isFinite(v)) {
      let r = rowOf(q);
      mn[r] = min(mn[r], v);
      mx[r] = max(mx[r], v);
      sm[r] = sm[r] + v;
    }
  }
  for (var r = 0u; r < U.rows; r = r + 1u) {
    wMin[li] = mn[r];
    wMax[li] = mx[r];
    wSum[li] = sm[r];
    workgroupBarrier();
    for (var s = ${WG / 2}u; s > 0u; s = s >> 1u) {
      if (li < s) {
        wMin[li] = min(wMin[li], wMin[li + s]);
        wMax[li] = max(wMax[li], wMax[li + s]);
        wSum[li] = wSum[li] + wSum[li + s];
      }
      workgroupBarrier();
    }
    if (li == 0u) {
      let o = (wid.x * U.rows + r) * 3u;
      partials[o] = wMin[0];
      partials[o + 1u] = wMax[0];
      partials[o + 2u] = wSum[0];
    }
    workgroupBarrier();
  }
}
`;

const FINALIZE = /* wgsl */ `${UNI}
@group(0) @binding(1) var<storage, read> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> edges: array<f32>;      // [lo, hi]
@group(0) @binding(3) var<storage, read_write> moments: array<f32>;    // rows x [min, max, sum, mean]

var<workgroup> wMin: array<f32, ${WG}>;
var<workgroup> wMax: array<f32, ${WG}>;
var<workgroup> wSum: array<f32, ${WG}>;
var<workgroup> rowMin: array<f32, ${MAX_ROWS}>;
var<workgroup> rowMax: array<f32, ${MAX_ROWS}>;

@compute @workgroup_size(${WG})
fn finalize(@builtin(local_invocation_index) li: u32) {
  let FMAX = bitcast<f32>(0x7f7fffffu);
  let NAN = bitcast<f32>(U.nanBits);
  for (var r = 0u; r < U.rows; r = r + 1u) {
    var mn = FMAX;
    var mx = -FMAX;
    var sm = 0.0;
    for (var g = li; g < U.groups; g = g + ${WG}u) {
      let o = (g * U.rows + r) * 3u;
      mn = min(mn, partials[o]);
      mx = max(mx, partials[o + 1u]);
      sm = sm + partials[o + 2u];
    }
    wMin[li] = mn;
    wMax[li] = mx;
    wSum[li] = sm;
    workgroupBarrier();
    for (var s = ${WG / 2}u; s > 0u; s = s >> 1u) {
      if (li < s) {
        wMin[li] = min(wMin[li], wMin[li + s]);
        wMax[li] = max(wMax[li], wMax[li + s]);
        wSum[li] = wSum[li] + wSum[li + s];
      }
      workgroupBarrier();
    }
    if (li == 0u) {
      rowMin[r] = wMin[0];
      rowMax[r] = wMax[0];
      let empty = wMin[0] > wMax[0];
      moments[r * 4u] = select(wMin[0], NAN, empty);
      moments[r * 4u + 1u] = select(wMax[0], NAN, empty);
      moments[r * 4u + 2u] = select(wSum[0], NAN, empty);
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    var lo = U.lo;
    var hi = U.hi;
    if (U.fixedRange == 0u) {
      var mn = FMAX;
      var mx = -FMAX;
      for (var r = 0u; r < U.rows; r = r + 1u) {
        mn = min(mn, rowMin[r]);
        mx = max(mx, rowMax[r]);
      }
      // histogramEdges() in src/spatial/histogram.ts, exactly.
      if (mn > mx) { lo = 0.0; hi = 1.0; }
      else if (mn == mx) {
        lo = mn - 0.5;
        hi = mn + 0.5;
        // |v| >= 2^24: both round to v. Step the edge nearer zero one ulp toward zero instead.
        if (!(lo < hi)) {
          let tz = bitcast<f32>(bitcast<u32>(mn) - 1u);
          if (mn > 0.0) { lo = tz; hi = mn; } else { lo = mn; hi = tz; }
        }
      }
      else { lo = mn; hi = mx; }
    }
    edges[0] = lo;
    edges[1] = hi;
  }
}
`;

const BIN = /* wgsl */ `${UNI}${READ}
@group(0) @binding(3) var<storage, read> edges: array<f32>;
@group(0) @binding(4) var<storage, read_write> counts: array<atomic<u32>>;   // rows x bins
@group(0) @binding(5) var<storage, read_write> totals: array<atomic<u32>>;   // rows x 4

var<workgroup> local: array<atomic<u32>, ${PRIVATE_CAP}>;
var<workgroup> localTot: array<atomic<u32>, ${MAX_ROWS * 4}>;

@compute @workgroup_size(${WG})
fn bin(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let lo = edges[0];
  let hi = edges[1];
  // histogramScale(): an exact 2^-32 prescale near f32 max, so range and t cannot overflow.
  let s = select(1.0, 2.3283064365386963e-10, max(abs(lo), abs(hi)) >= 7.922816251426434e28);
  let los = lo * s;
  let range = hi * s - los;
  let nb = U.bins;
  let fb = f32(nb);
  let cells = U.rows * nb;
  let usePrivate = U.privatized != 0u;
  if (usePrivate) {
    for (var i = li; i < cells; i = i + ${WG}u) { atomicStore(&local[i], 0u); }
  }
  if (li < ${MAX_ROWS * 4}u) { atomicStore(&localTot[li], 0u); }
  workgroupBarrier();

  for (var q = gid.x; q < U.n; q = q + U.stride) {
    let v = texel(q);
    let r = rowOf(q);
    var slot = 3u;                       // non-finite
    if (isFinite(v)) {
      if (v < lo) { slot = 1u; }         // under (fixed range only)
      else if (v > hi) { slot = 2u; }    // over (fixed range only)
      else {
        slot = 0u;
        // histogramBinOf() in src/spatial/histogram.ts: the division only seeds b; the two
        // correctly-rounded multiplies decide it, so the host golden agrees bit for bit.
        let t = (v * s - los) * fb;
        var b = min(nb - 1u, u32(max(0.0, floor(t / range))));
        if (b > 0u && f32(b) * range > t) { b = b - 1u; }
        if (b + 1u < nb && f32(b + 1u) * range <= t) { b = b + 1u; }
        if (usePrivate) { atomicAdd(&local[r * nb + b], 1u); }
        else { atomicAdd(&counts[r * nb + b], 1u); }
      }
    }
    atomicAdd(&localTot[r * 4u + slot], 1u);
  }
  workgroupBarrier();

  if (usePrivate) {
    for (var i = li; i < cells; i = i + ${WG}u) {
      let c = atomicLoad(&local[i]);
      if (c > 0u) { atomicAdd(&counts[i], c); }
    }
  }
  if (li < U.rows * 4u) {
    let c = atomicLoad(&localTot[li]);
    if (c > 0u) { atomicAdd(&totals[li], c); }
  }
}
`;

const CDF_FIX = /* wgsl */ `${UNI}
@group(0) @binding(1) var<storage, read> scan: array<u32>;      // exclusive, flat rows x bins
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<storage, read> totals: array<u32>;
@group(0) @binding(4) var<storage, read_write> cdf: array<u32>;
@group(0) @binding(5) var<storage, read_write> moments: array<f32>;

@compute @workgroup_size(${WG})
fn cdfFix(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let nb = U.bins;
  if (i < U.rows * nb) {
    let r = i / nb;
    cdf[i] = scan[i] + counts[i] - scan[r * nb];
  }
  if (i < U.rows) {
    let finite = totals[i * 4u] + totals[i * 4u + 1u] + totals[i * 4u + 2u];
    if (finite > 0u) { moments[i * 4u + 3u] = moments[i * 4u + 2u] / f32(finite); }
    else { moments[i * 4u + 3u] = bitcast<f32>(U.nanBits); }
  }
}
`;

interface Ctx {
  device: GPUDevice;
  reduce: GPUComputePipeline;
  finalize: GPUComputePipeline;
  bin: GPUComputePipeline;
  cdfFix: GPUComputePipeline;
}

const getCtx = deviceScoped(async (device: GPUDevice): Promise<Ctx> => {
  const build = async (code: string, entry: string) =>
    device.createComputePipeline({
      label: `histogram:${entry}`,
      layout: "auto",
      compute: { module: await compileShader(device, code, `histogram:${entry}`), entryPoint: entry },
    });
  const [reduce, finalize, bin, cdfFix] = await Promise.all([
    build(REDUCE, "reduce"),
    build(FINALIZE, "finalize"),
    build(BIN, "bin"),
    build(CDF_FIX, "cdfFix"),
  ]);
  return { device, reduce, finalize, bin, cdfFix };
});

function binding(buffer: GPUBuffer, offset: number, size: number, who: string): GPUBufferBinding {
  if (offset % 256 !== 0) throw new Error(`histogram: ${who} byteOffset ${offset} is not a multiple of 256`);
  if (offset + size > buffer.size) throw new Error(`histogram: ${who} needs ${offset + size} bytes of a ${buffer.size} byte buffer`);
  return { buffer, offset, size };
}

export interface HistogramGpuOptions extends HistogramOptions {
  readonly mask?: RasterMask;
}

/**
 * Histogram `plane` into `out` (see `HistogramTargets`), all on the device, in one submit. `rows`
 * is 2 with a mask, else 1; size `out` with `histogramByteSizes(bins, rows)`. Same result as
 * `histogramCpu` over the plane's texels: counts, CDF, totals and edges exactly. min/max are exact;
 * `sum` and `mean` are ACCUMULATED in f32 (a tree reduce), so they differ from the f64 golden by
 * f32 accumulation error, and by more for cancelling inputs (`[2²⁴, −2²⁴, 1]` can sum to 0) or
 * overflow to ±Inf for values near f32 max.
 * `device` is the one the buffers live on; default: the library's own.
 */
export async function histogramResident(
  plane: RasterPlane,
  out: HistogramTargets,
  opts: HistogramGpuOptions = {},
  device?: GPUDevice,
): Promise<void> {
  const bins = histogramBins(opts);
  const rows = opts.mask ? 2 : 1;
  const { width, height, rowWords } = plane;
  const n = width * height;
  const half = plane.precision === "f16";
  if (rowWords < (half ? Math.ceil(width / 2) : width)) throw new Error(`histogram: rowWords ${rowWords} is too short for width ${width}`);
  const fixedRange = opts.range ? 1 : 0;
  // Validated after f32 rounding, as the uniform will hold them.
  const [lo, hi] = opts.range ? histogramFixedRange(opts.range) : [0, 0];

  const ctx = await getCtx(device);
  const scanCtx = await getScanCtx(ctx.device);
  const dev = ctx.device;
  // No await below this line: the uniform, partials and scan scratch are shared pools, written and
  // consumed in one submit, so a concurrent caller cannot interleave (see fieldMax.ts).
  const planeBytes = Math.max(4, height * rowWords * 4);
  checkBindingSize(dev, `histogram: ${width}x${height} plane`, planeBytes);
  const srcBind = binding(plane.buffer, plane.byteOffset ?? 0, planeBytes, "plane");
  if (opts.mask) checkBindingSize(dev, `histogram: ${width}x${height} mask`, n * 4);
  const maskBind = opts.mask
    ? binding(opts.mask.buffer, opts.mask.byteOffset ?? 0, Math.max(4, n * 4), "mask")
    : sized(ensureBuf(dev, "histogram:noMask", 4, GPUBufferUsage.STORAGE), 4);

  const groups = Math.max(1, Math.min(MAX_GROUPS, Math.ceil(n / WG)));
  const privatized = rows * bins <= PRIVATE_CAP ? 1 : 0;
  const uni = ensureBuf(dev, "histogram:uni", 64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const words = new Uint32Array(16);
  words.set([n, width, rowWords, half ? 1 : 0, rows, opts.mask ? 1 : 0, opts.mask?.dtype === "f32" ? 1 : 0, bins]);
  words.set([fixedRange, groups, groups * WG, privatized], 8);
  new Float32Array(words.buffer).set([lo, hi], 12);
  words[14] = 0x7fc00000;
  dev.queue.writeBuffer(uni, 0, words);
  const partialBytes = groups * rows * 3 * 4;
  const partials = ensureBuf(dev, "histogram:partials", partialBytes, GPUBufferUsage.STORAGE);

  const sizes = histogramByteSizes(bins, rows);
  const whole = (b: GPUBuffer, key: keyof HistogramTargets) => sized(b, sizes[key]);
  const group = (pipeline: GPUComputePipeline, resources: GPUBufferBinding[]) =>
    dev.createBindGroup({
      label: pipeline.label,
      layout: pipeline.getBindGroupLayout(0),
      entries: resources.map((resource, binding) => ({ binding, resource })),
    });
  const uniBind = sized(uni, 64);

  const enc = dev.createCommandEncoder({ label: "histogram" });
  enc.clearBuffer(out.counts, 0, sizes.counts);
  enc.clearBuffer(out.totals, 0, sizes.totals);
  const pass = enc.beginComputePass({ label: "histogram" });
  pass.setPipeline(ctx.reduce);
  pass.setBindGroup(0, group(ctx.reduce, [uniBind, srcBind, maskBind, sized(partials, partialBytes)]));
  pass.dispatchWorkgroups(groups);
  pass.setPipeline(ctx.finalize);
  pass.setBindGroup(
    0,
    group(ctx.finalize, [uniBind, sized(partials, partialBytes), whole(out.edges, "edges"), whole(out.moments, "moments")]),
  );
  pass.dispatchWorkgroups(1);
  pass.setPipeline(ctx.bin);
  pass.setBindGroup(
    0,
    group(ctx.bin, [uniBind, srcBind, maskBind, whole(out.edges, "edges"), whole(out.counts, "counts"), whole(out.totals, "totals")]),
  );
  pass.dispatchWorkgroups(groups);
  pass.end();

  const { dst: scanned } = encodeScan(scanCtx, "u32", out.counts, rows * bins, enc, undefined, "histogram");
  const fix = enc.beginComputePass({ label: "histogram:cdf" });
  fix.setPipeline(ctx.cdfFix);
  fix.setBindGroup(
    0,
    group(ctx.cdfFix, [
      uniBind,
      sized(scanned, sizes.counts),
      whole(out.counts, "counts"),
      whole(out.totals, "totals"),
      whole(out.cdf, "cdf"),
      whole(out.moments, "moments"),
    ]),
  );
  fix.dispatchWorkgroups(Math.ceil((rows * bins) / WG));
  fix.end();
  dev.queue.submit([enc.finish()]);
}

/**
 * `histogramResident` into leased targets, read back to the host as a `HistogramResult` — the
 * shape `histogramCpu` returns, so `histogramRow`/`histogramSummary` read either. For a consumer
 * that wants the numbers (a legend, a panel); a GPU consumer uses `histogramResident` or the graph
 * op and never reads back. Moments come back as f32 values.
 *
 * The five targets are LEASED per call, not pooled scratch: they outlive the submit (the readback
 * copies come after it), so a shared buffer would let a concurrent call's submit overwrite them
 * before they were read.
 */
export async function histogramGpu(plane: RasterPlane, opts: HistogramGpuOptions = {}, device?: GPUDevice): Promise<HistogramResult> {
  const dev = device ?? (await getCtx()).device;
  const bins = histogramBins(opts);
  const rows = opts.mask ? 2 : 1;
  const sizes = histogramByteSizes(bins, rows);
  const pool = targetPool(dev);
  const keys = ["counts", "cdf", "totals", "edges", "moments"] as const;
  const leased: ResidentBuffer[] = [];
  try {
    for (const key of keys) leased.push(pool.lease(sizes[key]));
    const [cb, fb, tb, eb, mb] = leased.map((b) => b.buffer) as [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer];
    await histogramResident(plane, { counts: cb, cdf: fb, totals: tb, edges: eb, moments: mb }, opts, dev);
    const [counts, cdf, totals, edges, moments] = await Promise.all(
      keys.map((key, i) => readBackBytes(dev, `histogram:rb:${key}`, leased[i]!.buffer, 0, sizes[key])),
    );
    return toResult(bins, rows, counts!, cdf!, totals!, edges!, moments!);
  } finally {
    for (const b of leased) pool.release(b);
  }
}

/** Leases for `histogramGpu`'s targets, per device (resident usage: STORAGE | COPY_SRC | COPY_DST). */
const targetPool = perDevice((device) => new BufferPool(device));

function toResult(
  bins: number,
  rows: number,
  counts: ArrayBuffer,
  cdf: ArrayBuffer,
  totals: ArrayBuffer,
  edges: ArrayBuffer,
  moments: ArrayBuffer,
): HistogramResult {
  const e = new Float32Array(edges);
  return {
    bins,
    rows,
    lo: e[0]!,
    hi: e[1]!,
    counts: new Uint32Array(counts),
    cdf: new Uint32Array(cdf),
    totals: new Uint32Array(totals),
    moments: Float64Array.from(new Float32Array(moments)),
  };
}
