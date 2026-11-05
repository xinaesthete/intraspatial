// The wand's selection as a region rather than a level set: "everywhere within tol of the sample"
// can be many fragments scattered over the window, and often the question is only about the one
// the sample sits in.
//
// Two halves. `wandDistance` evaluates the wand's whitened distance at every raster pixel on the GPU
// — the very `sampleWhitened` the map draws its outline from, so the region and the outline cannot
// disagree — and reads it back. `connectedRegion` flood-fills that field from the sample on the host:
// a breadth-first fill is one pass over the pixels, and keeping it on the host means moving the
// tolerance re-fills without another dispatch or readback.

import { compileShader, getDevice } from "../device";
import type { GramMatrixGpuResult } from "./gramMatrix";
import type { ModeParams } from "./gramModes";
import { SIMILARITY_WGSL } from "./similarityWgsl";

const DISTANCE_SHADER = /* wgsl */ `
struct Uni { K: f32, m: f32, width: f32, height: f32, rowWords: f32, half: f32, pad0: f32, pad1: f32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> rasters: array<u32>;
@group(0) @binding(2) var<storage, read> chan: array<f32>;
@group(0) @binding(3) var<storage, read> wand: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

fn fetch(a: u32, col: u32, row: u32) -> f32 {
  let base = a * u32(U.height) * u32(U.rowWords) + row * u32(U.rowWords);
  if (U.half != 0.0) {
    let pair = unpack2x16float(rasters[base + (col >> 1u)]);
    return select(pair.y, pair.x, (col & 1u) == 0u);
  }
  return bitcast<f32>(rasters[base + col]);
}

${SIMILARITY_WGSL}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let w = u32(U.width);
  if (gid.x >= w || gid.y >= u32(U.height)) { return; }
  out[gid.y * w + gid.x] = sampleWhitened(gid.x, gid.y).w;
}
`;

let ctx: Promise<{ device: GPUDevice; pipeline: GPUComputePipeline }> | undefined;
let bufs: { uni: GPUBuffer; chan: GPUBuffer; wand: GPUBuffer; out: GPUBuffer; staging: GPUBuffer } | undefined;

function getCtx() {
  ctx ??= (async () => {
    const device = await getDevice();
    const module = await compileShader(device, DISTANCE_SHADER, "wandRegion.distance");
    const pipeline = device.createComputePipeline({
      label: "wandRegion:distance",
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    return { device, pipeline };
  })();
  return ctx;
}

/** A buffer at least `size` bytes, kept from the last call when it is big enough. */
function fit(device: GPUDevice, old: GPUBuffer | undefined, size: number, usage: number, label: string): GPUBuffer {
  if (old && old.size >= size) return old;
  old?.destroy();
  return device.createBuffer({ label, size: Math.max(size, 16), usage });
}

/**
 * The wand's whitened distance at every pixel centre of `res`, row-major (row 0 at the bbox's top).
 *
 * `params` is the `modeParams` the map is drawn with, so the field is the one the outline is a level
 * set of. `rasters` is a buffer in `res.resident`'s layout to read instead of `res.resident.buffer`.
 * One call at a time: the output buffers are shared, so callers coalesce rather than overlap.
 */
export async function wandDistance(
  res: GramMatrixGpuResult,
  params: Pick<ModeParams, "chan" | "wand" | "m">,
  rasters: GPUBuffer = res.resident.buffer,
): Promise<Float32Array> {
  const { device, pipeline } = await getCtx();
  const { width, height } = res;
  const n = width * height;
  const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  bufs = {
    uni: bufs?.uni ?? device.createBuffer({ label: "wandRegion:uni", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    chan: fit(device, bufs?.chan, params.chan.byteLength, STORAGE, "wandRegion:chan"),
    wand: fit(device, bufs?.wand, params.wand.byteLength, STORAGE, "wandRegion:wand"),
    out: fit(device, bufs?.out, n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "wandRegion:out"),
    staging: fit(device, bufs?.staging, n * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, "wandRegion:staging"),
  };
  const { uni, chan, wand, out, staging } = bufs;
  const half = res.resident.precision === "f16" ? 1 : 0;
  device.queue.writeBuffer(uni, 0, new Float32Array([res.labels.length, params.m, width, height, res.resident.rowWords, half, 0, 0]));
  device.queue.writeBuffer(chan, 0, params.chan);
  device.queue.writeBuffer(wand, 0, params.wand);
  const bind = device.createBindGroup({
    label: "wandRegion:distance",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uni } },
      { binding: 1, resource: { buffer: rasters } },
      { binding: 2, resource: { buffer: chan } },
      { binding: 3, resource: { buffer: wand } },
      { binding: 4, resource: { buffer: out } },
    ],
  });
  const enc = device.createCommandEncoder({ label: "wandRegion:distance" });
  const pass = enc.beginComputePass({ label: "wandRegion:distance" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
  pass.end();
  enc.copyBufferToBuffer(out, 0, staging, 0, n * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ, 0, n * 4);
  const got = new Float32Array(staging.getMappedRange(0, n * 4).slice(0));
  staging.unmap();
  return got;
}

/**
 * The pixels within `tol` of the wand that can be reached from `seed` without leaving that range,
 * stepping between edge-sharing pixels: 1 inside, 0 outside, row-major like `d`.
 *
 * `seed` is in raster coordinates with pixel centres at whole numbers, as the marker is; the fill
 * starts from the nearest pixel. A fractional seed is interpolated from the four pixels around it, so
 * when the nearest is out of range the fill starts from the closest of those four that is in range.
 * Only when none is does the region come back empty.
 */
export function connectedRegion(
  d: ArrayLike<number>,
  width: number,
  height: number,
  seed: { readonly col: number; readonly row: number },
  tol: number,
): Uint8Array {
  const region = new Uint8Array(width * height);
  const start = seedPixel(d, width, height, seed, tol);
  if (start < 0) return region;
  // Each pixel is queued at most once (marked as it is queued), so the queue never exceeds the raster.
  const queue = new Int32Array(width * height);
  let head = 0;
  let tail = 0;
  region[start] = 1;
  queue[tail++] = start;
  const visit = (i: number) => {
    if (region[i] === 0 && d[i]! <= tol) {
      region[i] = 1;
      queue[tail++] = i;
    }
  };
  while (head < tail) {
    const i = queue[head++]!;
    const x = i % width;
    if (x > 0) visit(i - 1);
    if (x < width - 1) visit(i + 1);
    if (i >= width) visit(i - width);
    if (i < width * (height - 1)) visit(i + width);
  }
  return region;
}

/** The pixel the fill starts from: the nearest to `seed`, else the closest in-range corner of the cell
 *  `seed` sits in, else -1. */
function seedPixel(
  d: ArrayLike<number>,
  width: number,
  height: number,
  seed: { readonly col: number; readonly row: number },
  tol: number,
): number {
  const clampCol = (c: number) => Math.max(0, Math.min(width - 1, c));
  const clampRow = (r: number) => Math.max(0, Math.min(height - 1, r));
  const nearest = clampRow(Math.round(seed.row)) * width + clampCol(Math.round(seed.col));
  if (d[nearest]! <= tol) return nearest;
  let best = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const r of [Math.floor(seed.row), Math.ceil(seed.row)])
    for (const c of [Math.floor(seed.col), Math.ceil(seed.col)]) {
      const i = clampRow(r) * width + clampCol(c);
      const dist = Math.hypot(clampCol(c) - seed.col, clampRow(r) - seed.row);
      if (d[i]! <= tol && dist < bestDist) {
        best = i;
        bestDist = dist;
      }
    }
  return best;
}
