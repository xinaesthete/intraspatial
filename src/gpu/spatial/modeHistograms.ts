// Value histograms of the three painted modes over the current Gram window, for the mode legend
// and for the map's contrast (R2 of the region/raster statistics plan; docs/histogram-op.md).
//
// The rasters hold CHANNELS; a mode's value at a pixel is the projection y_k = Σ_a z_a v_ka of the
// standardised channels, which the map computes in its fragment shader and never stores. So one
// dispatch writes y₁..y₃ for every pixel centre into three dense planes — with `sampleWhitened`,
// the very expression the map and the wand draw from — and the R1 `histogram` op bins each plane.
// Nothing comes back but the three small histograms.
//
// The planes describe the window the rasters cover (the current view, at its raster resolution),
// not the slide: a legend reading them should say so.

import { histogramQuantile, histogramRow, type HistogramOptions, type HistogramResult } from "#src/spatial/histogram";
import { compileShader, getDevice } from "../device";
import type { GramMatrixGpuResult } from "./gramMatrix";
import type { ModeParams } from "./gramModes";
import { denseRasterPlane, histogramGpu } from "./histogram";
import { SIMILARITY_WGSL } from "./similarityWgsl";

const PROJECT_SHADER = /* wgsl */ `
struct Uni { K: f32, m: f32, width: f32, height: f32, rowWords: f32, half: f32, planeWords: f32, pad: f32 };
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
  let y = sampleWhitened(gid.x, gid.y).xyz;
  let i = gid.y * w + gid.x;
  let stride = u32(U.planeWords);
  out[i] = y.x;
  out[stride + i] = y.y;
  out[2u * stride + i] = y.z;
}
`;

let ctx: Promise<{ device: GPUDevice; pipeline: GPUComputePipeline }> | undefined;
let bufs: { uni: GPUBuffer; chan: GPUBuffer; wand: GPUBuffer; out: GPUBuffer } | undefined;
/** Calls run one at a time: the planes are one pooled buffer, read by the histogram submits. */
let chain: Promise<unknown> = Promise.resolve();

function getCtx() {
  ctx ??= (async () => {
    const device = await getDevice();
    const module = await compileShader(device, PROJECT_SHADER, "modeHistograms.project");
    const pipeline = device.createComputePipeline({
      label: "modeHistograms:project",
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    return { device, pipeline };
  })();
  return ctx;
}

function fit(device: GPUDevice, old: GPUBuffer | undefined, size: number, usage: number, label: string): GPUBuffer {
  if (old && old.size >= size) return old;
  old?.destroy();
  return device.createBuffer({ label, size: Math.max(size, 16), usage });
}

/** Words per plane: the pixel count rounded up to 256 bytes, so each plane binds at a legal offset. */
const planeWordsFor = (n: number) => Math.ceil(n / 64) * 64;

/** Project `res`'s rasters onto the three painted modes, into the pooled planes. Returns their buffer. */
async function project(res: GramMatrixGpuResult, params: Pick<ModeParams, "chan" | "wand" | "m">, rasters: GPUBuffer) {
  const { device, pipeline } = await getCtx();
  const { width, height } = res;
  const planeWords = planeWordsFor(width * height);
  const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  bufs = {
    uni:
      bufs?.uni ?? device.createBuffer({ label: "modeHistograms:uni", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    chan: fit(device, bufs?.chan, params.chan.byteLength, STORAGE, "modeHistograms:chan"),
    wand: fit(device, bufs?.wand, params.wand.byteLength, STORAGE, "modeHistograms:wand"),
    out: fit(device, bufs?.out, 3 * planeWords * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "modeHistograms:planes"),
  };
  const { uni, chan, wand, out } = bufs;
  const half = res.resident.precision === "f16" ? 1 : 0;
  device.queue.writeBuffer(
    uni,
    0,
    new Float32Array([res.labels.length, params.m, width, height, res.resident.rowWords, half, planeWords, 0]),
  );
  device.queue.writeBuffer(chan, 0, params.chan);
  device.queue.writeBuffer(wand, 0, params.wand);
  const bind = device.createBindGroup({
    label: "modeHistograms:project",
    layout: pipeline.getBindGroupLayout(0),
    entries: [uni, rasters, chan, wand, out].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const enc = device.createCommandEncoder({ label: "modeHistograms:project" });
  const pass = enc.beginComputePass({ label: "modeHistograms:project" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
  pass.end();
  device.queue.submit([enc.finish()]);
  return { device, out, planeWords };
}

function serial<T>(task: () => Promise<T>): Promise<T> {
  const run = chain.then(task, task);
  chain = run.catch(() => undefined);
  return run;
}

export interface ModeHistograms {
  /** One histogram per painted mode (1 → lightness, 2–3 → hue), each over the window's pixels. */
  readonly modes: readonly HistogramResult[];
  /** Per mode, max(|q_lo|, |q_hi|) of its values: the half-range the ramp saturates at when
   *  `modeParams` is given these as `ranges`. 0 for a mode that does not exist (K < 3). */
  readonly halfRanges: [number, number, number];
  /** With `readback`: the three planes on the host, as `modeProjection` returns them. */
  readonly planes?: Float32Array;
}

export interface ModeHistogramOptions extends Pick<HistogramOptions, "bins"> {
  /** The quantile the contrast clips at, each side. Default 0.01 (the 1st and 99th percentiles). */
  readonly clip?: number;
  /** Also read the planes back (12 bytes a pixel), for a consumer that looks values up by pixel. */
  readonly readback?: boolean;
}

/**
 * Histograms of the three painted modes over the window `res` covers. `params` is the map's
 * `modeParams` (only the loadings and standardisation in `chan` are used, so a wand need not be
 * set); `rasters` is a buffer in `res.resident`'s layout to read instead of `res.resident.buffer`.
 * Calls are serialised internally.
 */
export function modeHistograms(
  res: GramMatrixGpuResult,
  params: Pick<ModeParams, "chan" | "wand" | "m">,
  rasters: GPUBuffer = res.resident.buffer,
  opts: ModeHistogramOptions = {},
): Promise<ModeHistograms> {
  return serial(async () => {
    const { device, out, planeWords } = await project(res, params, rasters);
    const K = res.labels.length;
    const modes = await Promise.all(
      [0, 1, 2]
        .filter((k) => k < K)
        .map((k) => histogramGpu(denseRasterPlane(out, res.width, res.height, k * planeWords * 4), { bins: opts.bins ?? 64 }, device)),
    );
    const clip = opts.clip ?? 0.01;
    const halfRanges: [number, number, number] = [0, 0, 0];
    modes.forEach((h, k) => {
      const row = histogramRow(h);
      halfRanges[k] = Math.max(Math.abs(histogramQuantile(row, clip)), Math.abs(histogramQuantile(row, 1 - clip)));
    });
    if (!opts.readback) return { modes, halfRanges };
    return { modes, halfRanges, planes: await readPlanes(device, out, planeWords, res.width * res.height) };
  });
}

/** The three mode planes, read back (row-major, plane after plane, each `width × height`) — for
 *  tests and for a consumer that wants the values themselves. */
export function modeProjection(
  res: GramMatrixGpuResult,
  params: Pick<ModeParams, "chan" | "wand" | "m">,
  rasters: GPUBuffer = res.resident.buffer,
): Promise<Float32Array> {
  return serial(async () => {
    const { device, out, planeWords } = await project(res, params, rasters);
    return readPlanes(device, out, planeWords, res.width * res.height);
  });
}

/** The pooled planes, copied to the host and packed (plane after plane, `n` values each). */
async function readPlanes(device: GPUDevice, out: GPUBuffer, planeWords: number, n: number): Promise<Float32Array> {
  const staging = device.createBuffer({
    label: "modeHistograms:rb",
    size: 3 * planeWords * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  try {
    const enc = device.createCommandEncoder({ label: "modeHistograms:rb" });
    enc.copyBufferToBuffer(out, 0, staging, 0, 3 * planeWords * 4);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(staging.getMappedRange());
    const got = new Float32Array(3 * n);
    for (let k = 0; k < 3; k++) got.set(all.subarray(k * planeWords, k * planeWords + n), k * n);
    staging.unmap();
    return got;
  } finally {
    staging.destroy();
  }
}
