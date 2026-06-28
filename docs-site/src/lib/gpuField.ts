// Browser WebGPU field generator — the GPU path for the fuzzy-filtration demo.
//
// Produces the SAME width*height scalar grid as the CPU library functions in
// `src/spatial/scalarField.ts` (gaussianKdeField / distanceField / dtmField), so a
// caller can swap backends without changing the persistence diagram. One compute
// shader, one thread per grid cell, brute force over the points:
//   • mode 0 (KDE)      sum_i exp(-|c-p_i|^2 / 2sigma^2), truncated at `support`
//   • mode 1 (distance) min_i |c - p_i|
//   • mode 2 (DTM)      sqrt(mean of the k smallest |c - p_i|^2)
//
// This mirrors the additive-Gaussian splat / gather of the toolbox's `splatDensity`
// primitive, but as a browser compute pass (`navigator.gpu`) rather than the
// Node/Dawn render path, so it bundles for the docs site. The persistence reduction
// that consumes this field is inherently sequential and always stays on the CPU.
//
// Robustness (different WebGPU backends behave differently — Safari/WebKit on Metal
// is stricter and has been crash-prone on iOS):
//   • every call validates its output (finite, plausible range); a bad result
//     returns null so the caller falls back to the CPU library — never garbage.
//   • calls are single-flighted (serialised) and buffers are pooled/reused rather
//     than created+destroyed per call, to avoid rapid alloc/free races.
//   • any throw → null → CPU fallback; a device-loss drops the cached context.

const K_MAX = 16; // max DTM neighbours the shader's local array holds

const SHADER = /* wgsl */ `
struct P {
  W: u32, H: u32, n: u32, mode: u32,
  minX: f32, minY: f32, maxX: f32, maxY: f32,
  sigma: f32, support: f32, k: u32, _pad: u32,
};
@group(0) @binding(0) var<storage, read> params: P;
@group(0) @binding(1) var<storage, read> pts: array<f32>;       // x, y per point (stride 2)
@group(0) @binding(2) var<storage, read_write> field: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.W * params.H;
  if (idx >= total) { return; }
  let c = idx % params.W;
  let r = idx / params.W;
  let spanX = max(params.maxX - params.minX, 1e-6);
  let spanY = max(params.maxY - params.minY, 1e-6);
  let cx = params.minX + (f32(c) + 0.5) / f32(params.W) * spanX;
  let cy = params.maxY - (f32(r) + 0.5) / f32(params.H) * spanY;

  if (params.mode == 0u) {                                  // KDE
    let inv2s2 = 1.0 / (2.0 * params.sigma * params.sigma);
    let sup2 = params.support * params.support;
    var acc = 0.0;
    for (var i = 0u; i < params.n; i = i + 1u) {
      let dx = cx - pts[2u * i];
      let dy = cy - pts[2u * i + 1u];
      let d2 = dx * dx + dy * dy;
      if (d2 <= sup2) { acc = acc + exp(-d2 * inv2s2); }
    }
    field[idx] = acc;
  } else if (params.mode == 1u) {                           // distance to nearest
    var best = 3.0e38;
    for (var i = 0u; i < params.n; i = i + 1u) {
      let dx = cx - pts[2u * i];
      let dy = cy - pts[2u * i + 1u];
      let d2 = dx * dx + dy * dy;
      best = min(best, d2);
    }
    field[idx] = sqrt(best);
  } else {                                                  // DTM: sqrt(mean of k smallest d^2)
    let k = min(params.k, ${K_MAX}u);
    var small: array<f32, ${K_MAX}>;
    for (var s = 0u; s < ${K_MAX}u; s = s + 1u) { small[s] = 3.0e38; }
    for (var i = 0u; i < params.n; i = i + 1u) {
      let dx = cx - pts[2u * i];
      let dy = cy - pts[2u * i + 1u];
      let d2 = dx * dx + dy * dy;
      // Insert d2 into the sorted (ascending) k-array if it beats the current max.
      if (d2 < small[k - 1u]) {
        var j = k - 1u;
        loop {
          if (j > 0u && small[j - 1u] > d2) {
            small[j] = small[j - 1u];
            j = j - 1u;
          } else { break; }
        }
        small[j] = d2;
      }
    }
    var sum = 0.0;
    for (var s = 0u; s < k; s = s + 1u) { sum = sum + small[s]; }
    field[idx] = sqrt(sum / f32(k));
  }
}
`;

export type FieldMode = "kde" | "distance" | "dtm";

export interface GpuFieldParams {
  width: number;
  height: number;
  bbox: [number, number, number, number];
  mode: FieldMode;
  sigma?: number;        // KDE bandwidth (world units)
  radiusSigma?: number;  // KDE support cutoff in sigmas (default 4)
  k?: number;            // DTM neighbours (default 5)
}

interface Ctx {
  device: GPUDevice;
  pipeline: GPUComputePipeline;
}

let ctxPromise: Promise<Ctx | null> | undefined;

// Pooled buffers (reused/grown, reset on device loss) — avoids per-call alloc/free,
// which some backends (notably WebKit/Metal) handle poorly under rapid rebuilds.
let pBuf: GPUBuffer | undefined;
let ptsBuf: GPUBuffer | undefined, ptsCap = 0;
let outBuf: GPUBuffer | undefined, readBuf: GPUBuffer | undefined, cellCap = 0;
function resetPool(): void {
  pBuf = ptsBuf = outBuf = readBuf = undefined;
  ptsCap = 0; cellCap = 0;
}

/** Acquire (once) a browser GPU device + compute pipeline, or null if unavailable. */
export function getGpuFieldContext(): Promise<Ctx | null> {
  ctxPromise ??= (async () => {
    try {
      const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
      if (!gpu) return null;
      const adapter = await gpu.requestAdapter();
      if (!adapter) return null;
      const device = await adapter.requestDevice();
      const module = device.createShaderModule({ code: SHADER });
      const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
      // If the device is ever lost, drop the cache + pool so the next call re-probes.
      device.lost.then(() => { ctxPromise = undefined; resetPool(); });
      return { device, pipeline };
    } catch {
      return null;
    }
  })();
  return ctxPromise;
}

/** True if the browser exposes a usable WebGPU device for the field compute. */
export async function gpuFieldAvailable(): Promise<boolean> {
  return (await getGpuFieldContext()) !== null;
}

/** Heuristic: WebKit (Safari / all iOS browsers). Its WebGPU backend is stricter and
 *  has been crash-prone, so the demo defaults the GPU path OFF there (still toggleable). */
export function isAppleWebKit(): boolean {
  try {
    return /Apple/.test((navigator as Navigator).vendor || "");
  } catch {
    return false;
  }
}

const MODE_CODE: Record<FieldMode, number> = { kde: 0, distance: 1, dtm: 2 };

/** A returned field is trusted only if every sample is finite and it carries some
 *  signal (max > 0) — guards against a backend silently emitting NaN/Inf/zeros. */
function looksValid(out: Float32Array): boolean {
  let max = 0;
  for (let i = 0; i < out.length; i++) {
    const v = out[i]!;
    if (!Number.isFinite(v)) return false;
    if (v > max) max = v;
  }
  return max > 0;
}

// Serialise calls: overlapping compute/readback on pooled buffers races on some
// backends. Each call waits for the previous to finish.
let inFlight: Promise<unknown> = Promise.resolve();

async function doCompute(
  ctx: Ctx,
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  params: GpuFieldParams,
): Promise<Float32Array | null> {
  const { device, pipeline } = ctx;
  const n = xs.length;
  const { width: W, height: H, bbox, mode } = params;
  const sigma = params.sigma ?? 1.5;
  const support = sigma * (params.radiusSigma ?? 4);
  const k = Math.max(1, Math.min(params.k ?? 5, n || 1, K_MAX));
  const cells = W * H;
  const outBytes = cells * 4;

  try {
    // Params (std430-packed scalars) — write via a typed-array view (stricter
    // backends reject a bare ArrayBuffer).
    pBuf ??= device.createBuffer({ size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const pData = new ArrayBuffer(48);
    const dv = new DataView(pData);
    dv.setUint32(0, W, true); dv.setUint32(4, H, true);
    dv.setUint32(8, n, true); dv.setUint32(12, MODE_CODE[mode], true);
    dv.setFloat32(16, bbox[0], true); dv.setFloat32(20, bbox[1], true);
    dv.setFloat32(24, bbox[2], true); dv.setFloat32(28, bbox[3], true);
    dv.setFloat32(32, sigma, true); dv.setFloat32(36, support, true);
    dv.setUint32(40, k, true); dv.setUint32(44, 0, true);
    device.queue.writeBuffer(pBuf, 0, new Uint8Array(pData));

    // Points (x,y interleaved), pooled + grown.
    const ptsFloats = Math.max(n, 1) * 2;
    if (!ptsBuf || ptsCap < ptsFloats) {
      ptsCap = Math.max(ptsFloats, ptsCap * 2, 2);
      ptsBuf = device.createBuffer({ size: ptsCap * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    const pts = new Float32Array(ptsFloats);
    for (let i = 0; i < n; i++) { pts[2 * i] = xs[i]!; pts[2 * i + 1] = ys[i]!; }
    device.queue.writeBuffer(ptsBuf, 0, pts);

    if (!outBuf || !readBuf || cellCap < cells) {
      cellCap = Math.max(cells, cellCap * 2, 1);
      outBuf = device.createBuffer({ size: cellCap * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      readBuf = device.createBuffer({ size: cellCap * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }

    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: pBuf } },
        { binding: 1, resource: { buffer: ptsBuf } },
        { binding: 2, resource: { buffer: outBuf } },
      ],
    });

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(cells / 64));
    pass.end();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBytes);
    device.queue.submit([enc.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ, 0, outBytes);
    const out = new Float32Array(readBuf.getMappedRange(0, outBytes).slice(0));
    readBuf.unmap();
    return looksValid(out) ? out : null;
  } catch {
    return null;
  }
}

/**
 * Compute the field on the GPU. Returns the row-major width*height Float32Array, or
 * null if WebGPU is unavailable / the result fails validation / anything throws — in
 * every such case the caller should fall back to the CPU library.
 */
export async function computeFieldGpu(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  params: GpuFieldParams,
): Promise<Float32Array | null> {
  const ctx = await getGpuFieldContext();
  if (!ctx) return null;
  // Chain onto the previous call so only one compute/readback is in flight.
  const run = inFlight.then(() => doCompute(ctx, xs, ys, params), () => doCompute(ctx, xs, ys, params));
  inFlight = run.catch(() => undefined);
  return run;
}
