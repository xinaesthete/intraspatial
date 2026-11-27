// Zonal statistics on the GPU: polygons → a u32 label raster, and a labelled histogram — rows =
// regions × channels, in R1's layout — accumulated tile by tile into one set of counts. The f64
// golden, the pixel classification and the exact-quantile plan are in `src/spatial/zonal.ts`
// (read its header first); the tiled driver over a pyramid level is `zonalTiled.ts`; the design
// note is docs/zonal-stats.md.
//
// ## Kernels (raw WGSL: atomics, workgroup memory, u32 division)
//
// - `rasterise`: per pixel, even–odd crossing parity against the ring buffers of every polygon
//   whose box holds the pixel centre; the highest label wins. Vertices arrive in the raster's
//   pixel frame (`polygonRingBuffers`), so centres are exact `(i + ½, j + ½)` in f32.
// - `range` (auto edges only): per-channel finite min/max over every region's values, as ordered
//   u32 keys with `atomicMin`/`atomicMax`, so it accumulates across tiles. `finalize` then applies
//   R1's edge rule per auto channel.
// - `bin`: one dispatch per channel. Each thread walks RUNs of consecutive pixels; regions are
//   contiguous, so it keeps the run's totals, min, max and sum in registers and flushes them once
//   per label change rather than once per pixel. Bins go to a workgroup-private histogram when
//   regions × bins fits (R1's PRIVATE_CAP), else straight to global atomics. The bin index is R1's
//   `histBin`, bit for bit.
// - `fold` (once per tile): the tile's f32 row sums (CAS-added) are added into a compensated
//   (hi, lo) pair per row with TwoSum and reset, so the sum's error does not grow with tile count.
// - `finish` (graph op): per-row inclusive CDF and `[min, max, sum, mean]` on the device.
// - `collect` (exact pass): values in each row's target bins, appended at the plan's offsets.
//
// Every binding is ≤ 7 storage buffers (the 8-buffer limit; a 9th silently no-ops a dispatch).
import { HISTOGRAM_MOMENTS, histogramCdf, histogramFixedRange } from "#src/spatial/histogram";
import {
  type PolygonRingBuffers,
  POLY_STRIDE,
  ZONAL_TOTALS,
  type ZonalExactPlan,
  type ZonalHistogram,
  type ZonalOptions,
  checkZonalOptions,
  zonalBins,
} from "#src/spatial/zonal";
import { checkBindingSize, compileShader, deviceScoped, perDevice, readBackBytes, sized, writeView } from "../device";
import type { ResidentBuffer } from "../graph/handle";
import { BufferPool } from "../graph/pool";
import { HIST_BIN_WGSL, type RasterMask, type RasterPlane } from "./histogram";

const WG = 256;
/** Consecutive pixels one thread of `bin` walks before striding. */
const RUN = 16;
const MAX_GROUPS = 1024;
/** Workgroup-private bins (16 KiB), as R1. */
const PRIVATE_CAP = 4088;
/** One uniform slot per dispatch in a submit (minUniformBufferOffsetAlignment). */
const SLOT = 256;

/** A dense label raster: `width × height` words, u32 ids or f32 (truncated) ids. Label L in
 *  1..regions is region L − 1; anything else is no region. */
export interface LabelPlane {
  readonly buffer: GPUBuffer;
  readonly byteOffset?: number;
  readonly width: number;
  readonly height: number;
  readonly dtype: "u32" | "f32";
}

/** One channel of one tile: its plane (R1's `RasterPlane`: f32 or f16-packed, any rowWords — a
 *  `ResidentRasters` plane works as is) and an optional support mask (non-zero ⇒ supported). */
export interface ZonalChannelPlane {
  readonly plane: RasterPlane;
  readonly support?: RasterMask;
}

/** One tile: labels and every channel, all the same `width × height`. */
export interface ZonalTile {
  readonly labels: LabelPlane;
  readonly channels: readonly ZonalChannelPlane[];
}

const UNI = /* wgsl */ `
struct Uni {
  n: u32, width: u32, rowWords: u32, half: u32,
  labelF32: u32, hasSupport: u32, supportF32: u32, positive: u32,
  regions: u32, channels: u32, channel: u32, bins: u32,
  runs: u32, stride: u32, privatized: u32, perRow: u32,
  autoMask: u32, rows: u32, nanBits: u32, pad0: u32,
};
@group(0) @binding(0) var<uniform> U: Uni;

fn isFinite(v: f32) -> bool {
  return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}
// Ordered keys: u32 order = f32 order, so atomicMin/atomicMax give a float min/max.
fn fkey(v: f32) -> u32 {
  let b = bitcast<u32>(v);
  return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}
fn unkey(k: u32) -> f32 {
  return bitcast<f32>(select(~k, k & 0x7fffffffu, (k & 0x80000000u) != 0u));
}
`;

// Reading a tile: labels, one channel's texel (R1's texel(), padding skipped by indexing), support.
const READ = /* wgsl */ `
@group(0) @binding(1) var<storage, read> labels: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<u32>;
@group(0) @binding(3) var<storage, read> support: array<u32>;

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

// 1..regions, or 0 for no region. An f32 label truncates; NaN, < 1 and huge are no region.
fn regionOf(q: u32) -> u32 {
  var l = labels[q];
  if (U.labelF32 != 0u) {
    let v = bitcast<f32>(l);
    if (!isFinite(v) || v < 1.0 || v >= 4294967040.0) { return 0u; }
    l = u32(v);
  }
  return select(0u, l, l <= U.regions);
}

// Non-zero is supported; for an f32 mask the sign bit is ignored (−0 is unsupported, NaN is
// supported — numpy's img != 0).
fn supported(q: u32) -> bool {
  if (U.hasSupport == 0u) { return true; }
  var m = support[q];
  if (U.supportF32 != 0u) { m = m & 0x7fffffffu; }
  return m != 0u;
}
`;

const RANGE = /* wgsl */ `${UNI}${READ}
@group(0) @binding(4) var<storage, read_write> rangeKeys: array<atomic<u32>>;   // channels x [min, max]

@compute @workgroup_size(${WG})
fn range(@builtin(global_invocation_id) gid: vec3u) {
  let FMAX = bitcast<f32>(0x7f7fffffu);
  var mn = FMAX;
  var mx = -FMAX;
  var seen = false;
  for (var q = gid.x; q < U.n; q = q + U.stride) {
    if (regionOf(q) != 0u && supported(q)) {
      let v = texel(q);
      if (isFinite(v) && (U.positive == 0u || v > 0.0)) {
        mn = min(mn, v);
        mx = max(mx, v);
        seen = true;
      }
    }
  }
  if (seen) {
    atomicMin(&rangeKeys[U.channel * 2u], fkey(mn));
    atomicMax(&rangeKeys[U.channel * 2u + 1u], fkey(mx));
  }
}
`;

const FINALIZE = /* wgsl */ `${UNI}
@group(0) @binding(1) var<storage, read> rangeKeys: array<u32>;
@group(0) @binding(2) var<storage, read_write> edges: array<f32>;

@compute @workgroup_size(32)
fn finalize(@builtin(global_invocation_id) gid: vec3u) {
  let c = gid.x;
  if (c >= U.channels || ((U.autoMask >> c) & 1u) == 0u) { return; }
  let kmn = rangeKeys[c * 2u];
  let kmx = rangeKeys[c * 2u + 1u];
  var lo = 0.0;
  var hi = 1.0;
  // histogramEdges() in src/spatial/histogram.ts, as R1's finalize.
  if (kmn <= kmx) {
    let mn = unkey(kmn);
    let mx = unkey(kmx);
    if (mn == mx) {
      lo = mn - 0.5;
      hi = mn + 0.5;
      if (!(lo < hi)) {
        let tz = bitcast<f32>(bitcast<u32>(mn) - 1u);
        if (mn > 0.0) { lo = tz; hi = mn; } else { lo = mn; hi = tz; }
      }
    } else { lo = mn; hi = mx; }
  }
  edges[c * 2u] = lo;
  edges[c * 2u + 1u] = hi;
}
`;

const BIN = /* wgsl */ `${UNI}${READ}${HIST_BIN_WGSL}
@group(0) @binding(4) var<storage, read> edges: array<f32>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;   // rows x bins
@group(0) @binding(6) var<storage, read_write> totals: array<atomic<u32>>;   // rows x ${ZONAL_TOTALS}
@group(0) @binding(7) var<storage, read_write> rowAcc: array<atomic<u32>>;   // rows x [minKey, maxKey, tileSum, 0]

var<workgroup> local: array<atomic<u32>, ${PRIVATE_CAP}>;

fn addTot(i: u32, k: u32) {
  if (k > 0u) { atomicAdd(&totals[i], k); }
}

fn addSum(i: u32, v: f32) {
  var old = atomicLoad(&rowAcc[i]);
  loop {
    let r = atomicCompareExchangeWeak(&rowAcc[i], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

fn flush(g: u32, inR: u32, und: u32, ovr: u32, nf: u32, exc: u32, uns: u32, pix: u32,
         mn: f32, mx: f32, sm: f32, cnt: u32) {
  let row = (g - 1u) * U.channels + U.channel;
  let t = row * ${ZONAL_TOTALS}u;
  addTot(t, inR);
  addTot(t + 1u, und);
  addTot(t + 2u, ovr);
  addTot(t + 3u, nf);
  addTot(t + 4u, exc);
  addTot(t + 5u, uns);
  addTot(t + 6u, pix);
  if (cnt > 0u) {
    atomicMin(&rowAcc[row * 4u], fkey(mn));
    atomicMax(&rowAcc[row * 4u + 1u], fkey(mx));
    addSum(row * 4u + 2u, sm);
  }
}

@compute @workgroup_size(${WG})
fn bin(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let FMAX = bitcast<f32>(0x7f7fffffu);
  let lo = edges[U.channel * 2u];
  let hi = edges[U.channel * 2u + 1u];
  let nb = U.bins;
  let cells = U.regions * nb;
  let usePrivate = U.privatized != 0u;
  if (usePrivate) {
    for (var i = li; i < cells; i = i + ${WG}u) { atomicStore(&local[i], 0u); }
  }
  workgroupBarrier();

  for (var run = gid.x; run < U.runs; run = run + U.stride) {
    let q0 = run * ${RUN}u;
    let q1 = min(q0 + ${RUN}u, U.n);
    var cur = 0u;
    var inR = 0u; var und = 0u; var ovr = 0u; var nf = 0u; var exc = 0u; var uns = 0u; var pix = 0u;
    var mn = FMAX; var mx = -FMAX; var sm = 0.0; var cnt = 0u;
    for (var q = q0; q < q1; q = q + 1u) {
      let g = regionOf(q);
      if (g != cur) {
        if (cur != 0u) { flush(cur, inR, und, ovr, nf, exc, uns, pix, mn, mx, sm, cnt); }
        cur = g;
        inR = 0u; und = 0u; ovr = 0u; nf = 0u; exc = 0u; uns = 0u; pix = 0u;
        mn = FMAX; mx = -FMAX; sm = 0.0; cnt = 0u;
      }
      if (g == 0u) { continue; }
      pix = pix + 1u;
      if (!supported(q)) { uns = uns + 1u; continue; }
      let v = texel(q);
      if (!isFinite(v)) { nf = nf + 1u; continue; }
      if (U.positive != 0u && v <= 0.0) { exc = exc + 1u; continue; }
      mn = min(mn, v);
      mx = max(mx, v);
      sm = sm + v;
      cnt = cnt + 1u;
      if (v < lo) { und = und + 1u; }
      else if (v > hi) { ovr = ovr + 1u; }
      else {
        inR = inR + 1u;
        let b = histBin(v, lo, hi, nb);
        if (usePrivate) { atomicAdd(&local[(g - 1u) * nb + b], 1u); }
        else { atomicAdd(&counts[((g - 1u) * U.channels + U.channel) * nb + b], 1u); }
      }
    }
    if (cur != 0u) { flush(cur, inR, und, ovr, nf, exc, uns, pix, mn, mx, sm, cnt); }
  }
  workgroupBarrier();

  if (usePrivate) {
    for (var i = li; i < cells; i = i + ${WG}u) {
      let c = atomicLoad(&local[i]);
      if (c > 0u) {
        let g = i / nb;
        atomicAdd(&counts[(g * U.channels + U.channel) * nb + (i - g * nb)], c);
      }
    }
  }
}
`;

const FOLD = /* wgsl */ `${UNI}
@group(0) @binding(1) var<storage, read_write> rowAcc: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> sums: array<f32>;   // rows x [hi, lo]

@compute @workgroup_size(${WG})
fn fold(@builtin(global_invocation_id) gid: vec3u) {
  for (var r = gid.x; r < U.rows; r = r + U.stride) {
    let x = bitcast<f32>(atomicExchange(&rowAcc[r * 4u + 2u], 0u));
    // TwoSum: s + err == a + x exactly.
    let a = sums[r * 2u];
    let s = a + x;
    let bb = s - a;
    let err = (a - (s - bb)) + (x - bb);
    sums[r * 2u] = s;
    sums[r * 2u + 1u] = sums[r * 2u + 1u] + err;
  }
}
`;

const FINISH = /* wgsl */ `${UNI}
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> totals: array<u32>;
@group(0) @binding(3) var<storage, read> rowAcc: array<u32>;
@group(0) @binding(4) var<storage, read> sums: array<f32>;
@group(0) @binding(5) var<storage, read_write> cdf: array<u32>;
@group(0) @binding(6) var<storage, read_write> moments: array<f32>;

@compute @workgroup_size(${WG})
fn finish(@builtin(global_invocation_id) gid: vec3u) {
  let NAN = bitcast<f32>(U.nanBits);
  let nb = U.bins;
  for (var r = gid.x; r < U.rows; r = r + U.stride) {
    var acc = 0u;
    for (var b = 0u; b < nb; b = b + 1u) {
      acc = acc + counts[r * nb + b];
      cdf[r * nb + b] = acc;
    }
    let t = r * ${ZONAL_TOTALS}u;
    let n = totals[t] + totals[t + 1u] + totals[t + 2u];
    let sum = sums[r * 2u] + sums[r * 2u + 1u];
    let o = r * 4u;
    if (n > 0u) {
      moments[o] = unkey(rowAcc[o]);
      moments[o + 1u] = unkey(rowAcc[o + 1u]);
      moments[o + 2u] = sum;
      moments[o + 3u] = sum / f32(n);
    } else {
      moments[o] = NAN; moments[o + 1u] = NAN; moments[o + 2u] = NAN; moments[o + 3u] = NAN;
    }
  }
}
`;

const COLLECT = /* wgsl */ `${UNI}${READ}${HIST_BIN_WGSL}
@group(0) @binding(4) var<storage, read> edges: array<f32>;
@group(0) @binding(5) var<storage, read> table: array<u32>;               // rows x perRow x [bin, offset]
@group(0) @binding(6) var<storage, read_write> cursors: array<atomic<u32>>; // rows x perRow
@group(0) @binding(7) var<storage, read_write> collected: array<f32>;

@compute @workgroup_size(${WG})
fn collect(@builtin(global_invocation_id) gid: vec3u) {
  let lo = edges[U.channel * 2u];
  let hi = edges[U.channel * 2u + 1u];
  for (var q = gid.x; q < U.n; q = q + U.stride) {
    let g = regionOf(q);
    if (g == 0u || !supported(q)) { continue; }
    let v = texel(q);
    if (!isFinite(v) || (U.positive != 0u && v <= 0.0) || v < lo || v > hi) { continue; }
    let b = histBin(v, lo, hi, U.bins);
    let base = ((g - 1u) * U.channels + U.channel) * U.perRow;
    var done = false;
    for (var t = 0u; t < U.perRow && !done; t = t + 1u) {
      let tb = table[(base + t) * 2u];
      if (tb == 0xffffffffu) { done = true; }
      else if (tb == b) {
        let k = atomicAdd(&cursors[base + t], 1u);
        collected[table[(base + t) * 2u + 1u] + k] = v;
        done = true;
      }
    }
  }
}
`;

const RASTERISE = /* wgsl */ `
struct RUni { width: u32, n: u32, polys: u32, stride: u32 };
@group(0) @binding(0) var<uniform> U: RUni;
@group(0) @binding(1) var<storage, read> edges: array<f32>;   // E x [x0, y0, x1, y1], pixel frame
@group(0) @binding(2) var<storage, read> polys: array<f32>;   // P x [first, count, label, 0, minX, minY, maxX, maxY]
@group(0) @binding(3) var<storage, read_write> labelsOut: array<u32>;

@compute @workgroup_size(${WG})
fn rasterise(@builtin(global_invocation_id) gid: vec3u) {
  for (var q = gid.x; q < U.n; q = q + U.stride) {
    let j = q / U.width;
    let x = f32(q - j * U.width) + 0.5;
    let y = f32(j) + 0.5;
    var best = 0u;
    for (var p = 0u; p < U.polys; p = p + 1u) {
      let o = p * ${POLY_STRIDE}u;
      let id = u32(polys[o + 2u]);
      if (id > best && x >= polys[o + 4u] && x <= polys[o + 6u] && y >= polys[o + 5u] && y <= polys[o + 7u]) {
        let e0 = u32(polys[o]);
        let e1 = e0 + u32(polys[o + 1u]);
        var inside = false;
        for (var e = e0; e < e1; e = e + 1u) {
          let x0 = edges[e * 4u];
          let y0 = edges[e * 4u + 1u];
          let x1 = edges[e * 4u + 2u];
          let y1 = edges[e * 4u + 3u];
          if ((y0 > y) != (y1 > y) && x < (x1 - x0) * (y - y0) / (y1 - y0) + x0) { inside = !inside; }
        }
        if (inside) { best = id; }
      }
    }
    labelsOut[q] = best;
  }
}
`;

interface Ctx {
  device: GPUDevice;
  range: GPUComputePipeline;
  finalize: GPUComputePipeline;
  bin: GPUComputePipeline;
  fold: GPUComputePipeline;
  finish: GPUComputePipeline;
  collect: GPUComputePipeline;
  rasterise: GPUComputePipeline;
}

const getCtx = deviceScoped(async (device: GPUDevice): Promise<Ctx> => {
  const build = async (code: string, entry: string) =>
    device.createComputePipeline({
      label: `zonal:${entry}`,
      layout: "auto",
      compute: { module: await compileShader(device, code, `zonal:${entry}`), entryPoint: entry },
    });
  const [range, finalize, bin, fold, finish, collect, rasterise] = await Promise.all([
    build(RANGE, "range"),
    build(FINALIZE, "finalize"),
    build(BIN, "bin"),
    build(FOLD, "fold"),
    build(FINISH, "finish"),
    build(COLLECT, "collect"),
    build(RASTERISE, "rasterise"),
  ]);
  return { device, range, finalize, bin, fold, finish, collect, rasterise };
});

/** Leases for accumulators and scratch, per device. */
const pool = perDevice((device) => new BufferPool(device));

function bindGroup(dev: GPUDevice, pipeline: GPUComputePipeline, resources: GPUBufferBinding[]): GPUBindGroup {
  return dev.createBindGroup({
    label: pipeline.label,
    layout: pipeline.getBindGroupLayout(0),
    entries: resources.map((resource, binding) => ({ binding, resource })),
  });
}

function aligned(buffer: GPUBuffer, offset: number, size: number, who: string): GPUBufferBinding {
  if (offset % 256 !== 0) throw new Error(`zonal: ${who} byteOffset ${offset} is not a multiple of 256`);
  if (offset + size > buffer.size) throw new Error(`zonal: ${who} needs ${offset + size} bytes of a ${buffer.size} byte buffer`);
  return { buffer, offset, size };
}

const groupsFor = (items: number) => Math.max(1, Math.min(MAX_GROUPS, Math.ceil(items / WG)));

/**
 * Rasterise polygons into `out` (a dense u32 `width × height` buffer, from `byteOffset`): each
 * pixel gets the highest label of the polygons containing its centre, else 0 — every word is
 * written. `rings` come from `polygonRingBuffers(polygons, frame)` for this raster. One submit.
 */
export async function rasterisePolygonsResident(rings: PolygonRingBuffers, out: LabelTarget, device?: GPUDevice): Promise<void> {
  const ctx = await getCtx(device);
  const dev = ctx.device;
  const p = pool(dev);
  const leases = [p.lease(rings.edges.byteLength), p.lease(rings.polys.byteLength)];
  const [eb, pb] = leases.map((l) => l.buffer) as [GPUBuffer, GPUBuffer];
  try {
    writeView(dev.queue, eb, rings.edges);
    writeView(dev.queue, pb, rings.polys);
    encodeRasterise(ctx, sized(eb, rings.edges.byteLength), sized(pb, rings.polys.byteLength), rings.count, out);
  } finally {
    // Safe once submitted: a later lease writes through the queue, which orders after this submit.
    for (const l of leases) p.release(l);
  }
}

/** A dense u32 label raster to write: `width × height` words from `byteOffset`. */
export interface LabelTarget {
  readonly buffer: GPUBuffer;
  readonly byteOffset?: number;
  readonly width: number;
  readonly height: number;
}

/** `rasterisePolygonsResident` over ring buffers already on the device (the graph op's inputs):
 *  `edges` `E × 4` and `polys` `count × 8` f32, laid out as `polygonRingBuffers` lays them out. */
export async function rasterisePolygonBuffers(
  edges: GPUBufferBinding,
  polys: GPUBufferBinding,
  count: number,
  out: LabelTarget,
  device?: GPUDevice,
): Promise<void> {
  encodeRasterise(await getCtx(device), edges, polys, count, out);
}

function encodeRasterise(ctx: Ctx, edges: GPUBufferBinding, polys: GPUBufferBinding, count: number, out: LabelTarget): void {
  const dev = ctx.device;
  const p = pool(dev);
  const n = out.width * out.height;
  checkBindingSize(dev, `zonal: ${out.width}x${out.height} labels`, n * 4);
  const uni = p.lease(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  try {
    const groups = groupsFor(n);
    writeView(dev.queue, uni.buffer, new Uint32Array([out.width, n, count, groups * WG]));
    const enc = dev.createCommandEncoder({ label: "zonal:rasterise" });
    const pass = enc.beginComputePass({ label: "zonal:rasterise" });
    pass.setPipeline(ctx.rasterise);
    pass.setBindGroup(
      0,
      bindGroup(dev, ctx.rasterise, [
        sized(uni.buffer, 16),
        edges,
        polys,
        aligned(out.buffer, out.byteOffset ?? 0, Math.max(4, n * 4), "labels"),
      ]),
    );
    pass.dispatchWorkgroups(groups);
    pass.end();
    dev.queue.submit([enc.finish()]);
  } finally {
    p.release(uni);
  }
}

/** Output buffers of the graph op's `finish`: each STORAGE | COPY_DST, bound whole from 0. */
export interface ZonalTargets {
  readonly counts: GPUBuffer;
  readonly cdf: GPUBuffer;
  readonly edges: GPUBuffer;
  readonly totals: GPUBuffer;
  readonly moments: GPUBuffer;
}

export function zonalByteSizes(bins: number, rows: number, channels: number): Record<keyof ZonalTargets, number> {
  return {
    counts: rows * bins * 4,
    cdf: rows * bins * 4,
    edges: channels * 8,
    totals: rows * ZONAL_TOTALS * 4,
    moments: rows * HISTOGRAM_MOMENTS * 4,
  };
}

type Phase = "range" | "bin" | "collect";

/**
 * A labelled histogram being accumulated on the device, tile by tile.
 *
 *     const acc = await ZonalAccumulator.create(opts);
 *     if (acc.needsRange) { for (const t of tiles) acc.rangeTile(t); acc.finalizeEdges(); }
 *     for (const t of tiles) acc.binTile(t);
 *     const h = await acc.read();          // or acc.finish(targets) to stay on the device
 *     acc.release();
 *
 * Each `*Tile` call writes its uniforms and submits at once (no await inside), so a caller may
 * reuse or rewrite the tile's buffers right after it returns: queue writes order after the submit.
 * The tiles of the range pass and the bin pass must be the same pixels.
 */
export class ZonalAccumulator {
  readonly regions: number;
  readonly channels: number;
  readonly bins: number;
  readonly rows: number;
  private readonly autoMask: number;
  private readonly leases: ResidentBuffer[] = [];
  private readonly b: Record<"counts" | "totals" | "rowAcc" | "sums" | "edges" | "rangeKeys" | "noSupport" | "uni", GPUBuffer>;
  private phase: Phase = "range";
  private edgesReady: boolean;
  private collecting?: { plan: ZonalExactPlan; table: ResidentBuffer; cursors: ResidentBuffer; out: ResidentBuffer };

  private constructor(
    private readonly ctx: Ctx,
    private readonly opts: ZonalOptions,
  ) {
    checkZonalOptions(opts);
    const dev = ctx.device;
    this.regions = opts.regions;
    this.channels = opts.channels.length;
    this.bins = zonalBins(opts);
    this.rows = this.regions * this.channels;
    this.autoMask = opts.channels.reduce((m, c, i) => (c.range ? m : m | (1 << i)) >>> 0, 0);
    this.edgesReady = this.autoMask === 0;
    const { rows, bins, channels } = this;
    checkBindingSize(dev, `zonal: ${rows} rows × ${bins} bins`, rows * bins * 4);
    const p = pool(dev);
    const lease = (bytes: number, usage?: number) => {
      const l = p.lease(bytes, usage);
      this.leases.push(l);
      return l.buffer;
    };
    this.b = {
      counts: lease(rows * bins * 4),
      totals: lease(rows * ZONAL_TOTALS * 4),
      rowAcc: lease(rows * 16),
      sums: lease(rows * 8),
      edges: lease(channels * 8),
      rangeKeys: lease(channels * 8),
      noSupport: lease(4),
      uni: lease((channels + 1) * SLOT, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
    };
    const q = dev.queue;
    const rowAcc = new Uint32Array(rows * 4);
    for (let r = 0; r < rows; r++) rowAcc[r * 4] = 0xffffffff;
    writeView(q, this.b.rowAcc, rowAcc);
    const keys = new Uint32Array(channels * 2);
    const edges = new Float32Array(channels * 2);
    opts.channels.forEach((c, i) => {
      keys[i * 2] = 0xffffffff;
      if (c.range) edges.set(histogramFixedRange(c.range), i * 2);
    });
    writeView(q, this.b.rangeKeys, keys);
    writeView(q, this.b.edges, edges);
    const enc = dev.createCommandEncoder({ label: "zonal:init" });
    enc.clearBuffer(this.b.counts, 0, rows * bins * 4);
    enc.clearBuffer(this.b.totals, 0, rows * ZONAL_TOTALS * 4);
    enc.clearBuffer(this.b.sums, 0, rows * 8);
    q.submit([enc.finish()]);
  }

  static async create(opts: ZonalOptions, device?: GPUDevice): Promise<ZonalAccumulator> {
    return new ZonalAccumulator(await getCtx(device), opts);
  }

  get device(): GPUDevice {
    return this.ctx.device;
  }

  /** Whether any channel has auto edges, so a range pass must run before binning. */
  get needsRange(): boolean {
    return !this.edgesReady && this.phase === "range";
  }

  private uniform(
    slot: number,
    tile: ZonalTile | undefined,
    c: number,
    extra: { runs?: number; stride: number; privatized?: number; perRow?: number },
  ): void {
    const w = new Uint32Array(SLOT / 4);
    if (tile) {
      const { plane, support } = tile.channels[c]!;
      const half = plane.precision === "f16";
      w.set([plane.width * plane.height, plane.width, plane.rowWords, half ? 1 : 0]);
      w.set(
        [
          tile.labels.dtype === "f32" ? 1 : 0,
          support ? 1 : 0,
          support?.dtype === "f32" ? 1 : 0,
          this.opts.channels[c]!.valid === "positive" ? 1 : 0,
        ],
        4,
      );
    }
    w.set([this.regions, this.channels, c, this.bins], 8);
    w.set([extra.runs ?? 0, extra.stride, extra.privatized ?? 0, extra.perRow ?? 0], 12);
    w.set([this.autoMask, this.rows, 0x7fc00000, 0], 16);
    writeView(this.ctx.device.queue, this.b.uni, w, slot * SLOT);
  }

  private tileBindings(tile: ZonalTile, c: number): GPUBufferBinding[] {
    const dev = this.ctx.device;
    const { labels } = tile;
    const { plane, support } = tile.channels[c]!;
    const n = plane.width * plane.height;
    if (plane.width !== labels.width || plane.height !== labels.height) {
      throw new Error(`zonal: channel ${c} is ${plane.width}x${plane.height}, labels ${labels.width}x${labels.height}`);
    }
    const half = plane.precision === "f16";
    if (plane.rowWords < (half ? Math.ceil(plane.width / 2) : plane.width))
      throw new Error(`zonal: rowWords ${plane.rowWords} is too short for width ${plane.width}`);
    const planeBytes = Math.max(4, plane.height * plane.rowWords * 4);
    checkBindingSize(dev, `zonal: ${plane.width}x${plane.height} plane`, planeBytes);
    // Never read when hasSupport = 0, but must not alias a buffer the same dispatch writes.
    const noSupport = sized(this.b.noSupport, 4);
    return [
      { buffer: this.b.uni, offset: c * SLOT, size: 80 },
      aligned(labels.buffer, labels.byteOffset ?? 0, Math.max(4, n * 4), "labels"),
      aligned(plane.buffer, plane.byteOffset ?? 0, planeBytes, "plane"),
      support ? aligned(support.buffer, support.byteOffset ?? 0, Math.max(4, n * 4), "support") : noSupport,
    ];
  }

  private checkTile(tile: ZonalTile): void {
    if (tile.channels.length !== this.channels) throw new Error(`zonal: tile has ${tile.channels.length} channels, want ${this.channels}`);
  }

  /** Auto-range pass over one tile: folds its in-region values into each auto channel's min/max. */
  rangeTile(tile: ZonalTile): void {
    if (this.phase !== "range" || this.edgesReady) throw new Error("zonal: rangeTile after the edges were fixed");
    this.checkTile(tile);
    const { device: dev } = this.ctx;
    const enc = dev.createCommandEncoder({ label: "zonal:range" });
    const pass = enc.beginComputePass({ label: "zonal:range" });
    pass.setPipeline(this.ctx.range);
    for (let c = 0; c < this.channels; c++) {
      if (((this.autoMask >>> c) & 1) === 0) continue;
      const { plane } = tile.channels[c]!;
      const groups = groupsFor(plane.width * plane.height);
      this.uniform(c, tile, c, { stride: groups * WG });
      pass.setBindGroup(0, bindGroup(dev, this.ctx.range, [...this.tileBindings(tile, c), sized(this.b.rangeKeys, this.channels * 8)]));
      pass.dispatchWorkgroups(groups);
    }
    pass.end();
    dev.queue.submit([enc.finish()]);
  }

  /** Fix the auto edges from the range pass (an empty channel gets [0, 1], as R1). */
  finalizeEdges(): void {
    if (this.edgesReady) return;
    const dev = this.ctx.device;
    this.uniform(this.channels, undefined, 0, { stride: 0 });
    const enc = dev.createCommandEncoder({ label: "zonal:finalize" });
    const pass = enc.beginComputePass({ label: "zonal:finalize" });
    pass.setPipeline(this.ctx.finalize);
    pass.setBindGroup(
      0,
      bindGroup(dev, this.ctx.finalize, [
        { buffer: this.b.uni, offset: this.channels * SLOT, size: 80 },
        sized(this.b.rangeKeys, this.channels * 8),
        sized(this.b.edges, this.channels * 8),
      ]),
    );
    pass.dispatchWorkgroups(1);
    pass.end();
    dev.queue.submit([enc.finish()]);
    this.edgesReady = true;
  }

  /** Bin one tile into the counts (every channel, then the per-tile sum fold), one submit. */
  binTile(tile: ZonalTile): void {
    if (!this.edgesReady) throw new Error("zonal: binTile before finalizeEdges (a channel has auto edges)");
    if (this.phase === "collect") throw new Error("zonal: binTile after the exact pass began");
    this.phase = "bin";
    this.checkTile(tile);
    const dev = this.ctx.device;
    const privatized = this.regions * this.bins <= PRIVATE_CAP ? 1 : 0;
    const enc = dev.createCommandEncoder({ label: "zonal:bin" });
    const pass = enc.beginComputePass({ label: "zonal:bin" });
    pass.setPipeline(this.ctx.bin);
    for (let c = 0; c < this.channels; c++) {
      const { plane } = tile.channels[c]!;
      const runs = Math.ceil((plane.width * plane.height) / RUN);
      const groups = groupsFor(runs);
      this.uniform(c, tile, c, { runs, stride: groups * WG, privatized });
      pass.setBindGroup(
        0,
        bindGroup(dev, this.ctx.bin, [
          ...this.tileBindings(tile, c),
          sized(this.b.edges, this.channels * 8),
          sized(this.b.counts, this.rows * this.bins * 4),
          sized(this.b.totals, this.rows * ZONAL_TOTALS * 4),
          sized(this.b.rowAcc, this.rows * 16),
        ]),
      );
      pass.dispatchWorkgroups(groups);
    }
    const groups = groupsFor(this.rows);
    this.uniform(this.channels, undefined, 0, { stride: groups * WG });
    pass.setPipeline(this.ctx.fold);
    pass.setBindGroup(
      0,
      bindGroup(dev, this.ctx.fold, [
        { buffer: this.b.uni, offset: this.channels * SLOT, size: 80 },
        sized(this.b.rowAcc, this.rows * 16),
        sized(this.b.sums, this.rows * 8),
      ]),
    );
    pass.dispatchWorkgroups(groups);
    pass.end();
    dev.queue.submit([enc.finish()]);
  }

  /** Write the result into device targets (`zonalByteSizes`): counts, totals and edges copied,
   *  CDF and moments computed. One submit; nothing comes back to the host. */
  finish(out: ZonalTargets): void {
    const dev = this.ctx.device;
    const sizes = zonalByteSizes(this.bins, this.rows, this.channels);
    const groups = groupsFor(this.rows);
    this.uniform(this.channels, undefined, 0, { stride: groups * WG });
    const enc = dev.createCommandEncoder({ label: "zonal:finish" });
    enc.copyBufferToBuffer(this.b.counts, 0, out.counts, 0, sizes.counts);
    enc.copyBufferToBuffer(this.b.totals, 0, out.totals, 0, sizes.totals);
    enc.copyBufferToBuffer(this.b.edges, 0, out.edges, 0, sizes.edges);
    const pass = enc.beginComputePass({ label: "zonal:finish" });
    pass.setPipeline(this.ctx.finish);
    pass.setBindGroup(
      0,
      bindGroup(dev, this.ctx.finish, [
        { buffer: this.b.uni, offset: this.channels * SLOT, size: 80 },
        sized(this.b.counts, sizes.counts),
        sized(this.b.totals, sizes.totals),
        sized(this.b.rowAcc, this.rows * 16),
        sized(this.b.sums, this.rows * 8),
        sized(out.cdf, sizes.cdf),
        sized(out.moments, sizes.moments),
      ]),
    );
    pass.dispatchWorkgroups(groups);
    pass.end();
    dev.queue.submit([enc.finish()]);
  }

  /** Read the histogram back. Sums come back as their compensated (hi, lo) pair, added in f64. */
  async read(): Promise<ZonalHistogram> {
    const dev = this.ctx.device;
    const { rows, bins, channels } = this;
    const [counts, totals, rowAcc, sums, edges] = await Promise.all([
      readBackBytes(dev, "zonal:rb:counts", this.b.counts, 0, rows * bins * 4),
      readBackBytes(dev, "zonal:rb:totals", this.b.totals, 0, rows * ZONAL_TOTALS * 4),
      readBackBytes(dev, "zonal:rb:rowAcc", this.b.rowAcc, 0, rows * 16),
      readBackBytes(dev, "zonal:rb:sums", this.b.sums, 0, rows * 8),
      readBackBytes(dev, "zonal:rb:edges", this.b.edges, 0, channels * 8),
    ]);
    const c = new Uint32Array(counts);
    const t = new Uint32Array(totals);
    const acc = new Uint32Array(rowAcc);
    const s = new Float32Array(sums);
    const moments = new Float64Array(rows * HISTOGRAM_MOMENTS);
    for (let r = 0; r < rows; r++) {
      const n = t[r * ZONAL_TOTALS]! + t[r * ZONAL_TOTALS + 1]! + t[r * ZONAL_TOTALS + 2]!;
      if (n === 0) {
        moments.fill(Number.NaN, r * 4, r * 4 + 4);
        continue;
      }
      const sum = s[r * 2]! + s[r * 2 + 1]!;
      moments.set([unkey(acc[r * 4]!), unkey(acc[r * 4 + 1]!), sum, sum / n], r * 4);
    }
    return {
      regions: this.regions,
      channels,
      bins,
      edges: new Float32Array(edges),
      counts: c,
      cdf: histogramCdf(c, bins),
      totals: t,
      moments,
    };
  }

  /** Start the exact pass with `zonalExactPlan(await acc.read(), quantiles)`. */
  beginCollect(plan: ZonalExactPlan): void {
    if (!this.edgesReady) throw new Error("zonal: beginCollect before the edges were fixed");
    if (this.collecting) throw new Error("zonal: the exact pass already began");
    if (plan.table.length !== this.rows * plan.perRow * 2) throw new Error("zonal: the plan is for a different histogram");
    const dev = this.ctx.device;
    const p = pool(dev);
    checkBindingSize(dev, `zonal: exact pass collecting ${plan.total} values`, plan.total * 4);
    const table = p.lease(plan.table.byteLength);
    const cursors = p.lease(plan.table.byteLength / 2);
    const out = p.lease(Math.max(4, plan.total * 4));
    this.leases.push(table, cursors, out);
    writeView(dev.queue, table.buffer, plan.table);
    const enc = dev.createCommandEncoder({ label: "zonal:collect:init" });
    enc.clearBuffer(cursors.buffer, 0, plan.table.byteLength / 2);
    dev.queue.submit([enc.finish()]);
    this.collecting = { plan, table, cursors, out };
    this.phase = "collect";
  }

  /** Collect one tile's target-bin values (the same tiles as the bin pass). */
  collectTile(tile: ZonalTile): void {
    const k = this.collecting;
    if (!k) throw new Error("zonal: collectTile before beginCollect");
    this.checkTile(tile);
    const dev = this.ctx.device;
    const enc = dev.createCommandEncoder({ label: "zonal:collect" });
    const pass = enc.beginComputePass({ label: "zonal:collect" });
    pass.setPipeline(this.ctx.collect);
    for (let c = 0; c < this.channels; c++) {
      const { plane } = tile.channels[c]!;
      const groups = groupsFor(plane.width * plane.height);
      this.uniform(c, tile, c, { stride: groups * WG, perRow: k.plan.perRow });
      pass.setBindGroup(
        0,
        bindGroup(dev, this.ctx.collect, [
          ...this.tileBindings(tile, c),
          sized(this.b.edges, this.channels * 8),
          sized(k.table.buffer, k.plan.table.byteLength),
          sized(k.cursors.buffer, k.plan.table.byteLength / 2),
          sized(k.out.buffer, Math.max(4, k.plan.total * 4)),
        ]),
      );
      pass.dispatchWorkgroups(groups);
    }
    pass.end();
    dev.queue.submit([enc.finish()]);
  }

  /** The collected values, for `zonalExactQuantiles`. */
  async readCollected(): Promise<Float32Array> {
    const k = this.collecting;
    if (!k) throw new Error("zonal: readCollected before beginCollect");
    if (k.plan.total === 0) return new Float32Array(1);
    return new Float32Array(await readBackBytes(this.ctx.device, "zonal:rb:collected", k.out.buffer, 0, k.plan.total * 4));
  }

  /** Return every buffer to the pool. Safe right after the last submit or read. */
  release(): void {
    const p = pool(this.ctx.device);
    for (const l of this.leases.splice(0)) p.release(l);
  }
}

const keyView = new DataView(new ArrayBuffer(4));
/** The f32 an ordered key encodes (the WGSL `unkey`). */
function unkey(k: number): number {
  keyView.setUint32(0, k & 0x80000000 ? k & 0x7fffffff : ~k >>> 0);
  return keyView.getFloat32(0);
}

/** Labels → histogram for one resident image (one "tile"), read back. With `quantiles`, also the
 *  exact second pass. For tiled input see `zonalStatsTiled`; for a graph, the `zonalHistogram` op. */
export async function zonalHistogramGpu(tile: ZonalTile, opts: ZonalOptions, device?: GPUDevice): Promise<ZonalHistogram> {
  const acc = await ZonalAccumulator.create(opts, device);
  try {
    if (acc.needsRange) {
      acc.rangeTile(tile);
      acc.finalizeEdges();
    }
    acc.binTile(tile);
    return await acc.read();
  } finally {
    acc.release();
  }
}
