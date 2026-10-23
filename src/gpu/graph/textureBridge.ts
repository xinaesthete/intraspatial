// The texture → storage-buffer adapter the executor inserts when a texture-resident value meets an
// op that wants a buffer (ADR-0017).
//
// A render-producing op leaves its output in a texture, which is free for a consumer that also
// renders. Most compute ops instead bind `array<f32>`, and a texture cannot be bound that way — so
// somewhere a copy has to happen. Putting it HERE, in the executor's bridge, rather than at the end
// of every render-producing op, is the point of the whole change: the copy is paid only when a
// buffer consumer actually exists. A splat feeding a paint pass pays nothing.
//
// `copyTextureToBuffer` requires `bytesPerRow` to be a multiple of 256, so the texture always lands
// in a row-PADDED staging buffer; this strips that padding on-device into a tightly packed w*h.
// (The host path strips it in a JS loop, which is exactly what a resident value must not do.)
//
// Raw WGSL rather than TGSL deliberately: `i / w` on u32 operands is honest integer division here,
// where the TGSL transpiler turns it into float division and silently scrambles the row index (see
// splatDensity's own de-pad for the bug that caused).
import { getDevice, sized } from "../device";
import type { ResidentTexture } from "./handle";

const WG = 64;

// Two entry points, one per source format. `U.rowWords` is the padded row stride in u32 WORDS,
// which is the unit both read as — an `r16float` row is half the bytes but the same kind of word,
// holding two texels each.
//
// The f16 texels are widened to f32 HERE, on the device, rather than handed to the host as halves.
// Downstream is f32 either way, so the halving is a storage decision that stops at this boundary:
// no consumer, no test and no readback has to know the field was accumulated in f16. (`Float16Array`
// exists in Node 24 but not in every browser this ships to, so converting on the host would have
// been the fragile half of the same trade.)
const DEPAD = /* wgsl */ `
struct Uni { w: u32, h: u32, rowWords: u32, pad: u32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(${WG})
fn depad(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.w * U.h) { return; }
  let row = i / U.w;
  let col = i - row * U.w;
  dst[i] = bitcast<f32>(src[row * U.rowWords + col]);
}

@compute @workgroup_size(${WG})
fn depadHalf(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.w * U.h) { return; }
  let row = i / U.w;
  let col = i - row * U.w;
  // Two texels per word, little-endian: x is the even column, y the odd one.
  let word = src[row * U.rowWords + (col >> 1u)];
  let pair = unpack2x16float(word);
  dst[i] = select(pair.y, pair.x, (col & 1u) == 0u);
}
`;

interface Ctx {
  device: GPUDevice;
  pipeline: GPUComputePipeline;
  pipelineHalf: GPUComputePipeline;
  uni: GPUBuffer;
}
let ctxCache: Promise<Ctx> | undefined;

function getCtx(): Promise<Ctx> {
  ctxCache ??= (async () => {
    const device = await getDevice();
    const module = device.createShaderModule({ code: DEPAD });
    return {
      device,
      pipeline: device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "depad" } }),
      pipelineHalf: device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "depadHalf" } }),
      uni: device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    };
  })();
  return ctxCache;
}

// Row-padded staging, pooled and grown, never destroyed (mid-process destruction segfaults
// Dawn-on-Node — ADR-0002/0003).
let staging: GPUBuffer | undefined;
let stagingBytes = 0;
function ensureStaging(device: GPUDevice, bytes: number): GPUBuffer {
  if (staging && stagingBytes >= bytes) return staging;
  stagingBytes = Math.max(bytes, stagingBytes * 2, 256);
  staging = device.createBuffer({
    size: stagingBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  return staging;
}

const align256 = (n: number) => Math.ceil(n / 256) * 256;

/** Copy a resident texture's red channel into a tightly packed `width*height` **f32** buffer,
 *  whether the texture itself is `r32float` or `r16float`.
 *
 *  Submits and returns; the copy is ordered before anything the caller submits afterwards, so no
 *  fence is needed for a GPU-side consumer. */
export async function textureToBuffer(tex: ResidentTexture, dst: GPUBuffer): Promise<void> {
  const ctx = await getCtx();
  const { device, uni } = ctx;
  const { width: w, height: h } = tex;
  const half = tex.format === "r16float";
  const pipeline = half ? ctx.pipelineHalf : ctx.pipeline;
  const bytesPerRow = align256(w * (half ? 2 : 4));
  const src = ensureStaging(device, bytesPerRow * h);

  device.queue.writeBuffer(uni, 0, new Uint32Array([w, h, bytesPerRow / 4, 0]));
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex.texture }, { buffer: src, bytesPerRow }, { width: w, height: h });
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uni } },
        // `src` comes from the doubling staging pool, so it must be sized to this texture or a
        // larger earlier copy pushes the binding past maxStorageBufferBindingSize and the depad
        // silently writes nothing. `dst` is a Tier-2 pool buffer, which buckets to powers of two
        // and so cannot exceed a power-of-two limit — but it is sized here anyway rather than
        // relying on that arithmetic coincidence.
        { binding: 1, resource: sized(src, bytesPerRow * h) },
        { binding: 2, resource: sized(dst, w * h * 4) },
      ],
    }),
  );
  pass.dispatchWorkgroups(Math.ceil((w * h) / WG));
  pass.end();
  device.queue.submit([enc.finish()]);
}
