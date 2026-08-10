// Tile assembly on the device — the pass that lets a `Loader` hand back a device-resident `Tile`
// (docs/gpu-resident-loader.md §5).
//
// What it replaces: every decoded chunk currently makes two full passes over its samples on the
// host before it can be drawn — normalise-and-interleave into a `Float32Array` (spatialDataLoader,
// spatialDataVolume) and then quantise/half-convert into the texture's storage type (tileRenderer,
// naiveVolumeRenderer). Both are pure format conversion, not decode, so both belong here. For a
// [1,1,32,512,512] uint16 brick that is ~16.8 M host loop iterations and ~42 MB of intermediates
// per brick, on the thread that is also trying to render.
//
// This is deliberately CODEC-INDEPENDENT. It takes packed unsigned integer samples in a GPUBuffer —
// whoever produced them, a worker-decoded plane uploaded verbatim or a future in-GPU HTJ2K decode
// writing its output straight here — and produces the two things consumers actually want:
//
//   • `"f32"` — tightly packed, lane-interleaved f32. The layout `FieldValue.data` already
//     specifies, so the result is a legal `Tile.buffer` / resident field payload.
//   • `"f16"` — half-floats packed two per u32 with rows padded to 256 B, i.e. exactly what
//     `copyBufferToTexture` wants, so the render backend's fp16 texture is filled with no host
//     involvement at all.
//
// RAW WGSL, not TGSL, for the same reason `textureBridge.ts` is: these kernels do integer
// division and shifting on u32 indices, and the TGSL transpiler turns `i / w` into FLOAT division
// and silently scrambles the index (see the splatDensity de-pad bug, and the note in AGENTS.md).
//
// Two device limits are load-bearing here and are handled explicitly rather than assumed away:
// the 65535-per-dimension workgroup cap (a 8.4 M-voxel brick needs 131 072 workgroups at 64
// threads, which is over the cap and fails SILENTLY as a no-op), and `copyBufferToTexture`'s
// 256-byte `bytesPerRow` alignment.

import type { LeaseToken, ResidentBuffer } from "../graph/handle";

/** Bit width of one stored sample in the source planes. Unsigned integers only — the dtypes
 *  OME-Zarr imagery actually uses. */
export type SampleBits = 8 | 16 | 32;

export interface AssembleOpts {
  device: GPUDevice;
  /** One source buffer per output lane, each holding `width*height*depth` packed samples of
   *  `bits` bits, in x-fastest order. Lane order is the output's interleave order. */
  planes: readonly GPUBuffer[];
  width: number;
  height: number;
  /** 1 for a plane tile, >1 for a volumetric brick — the same kernel serves both. */
  depth: number;
  bits: SampleBits;
  /** Multiplied into every sample, i.e. the dtype normalisation (`1/65535` for uint16). */
  scale: number;
  /** `"f32"`: tight, lane-interleaved, a legal resident field payload.
   *  `"f16"`: half-float pairs packed 2-per-u32, rows padded to 256 B for `copyBufferToTexture`. */
  out: "f32" | "f16";
}

export interface AssembledTile {
  /** The assembled samples, owned (see `ownedBuffer`). */
  readonly payload: ResidentBuffer;
  /** Bytes per row of the result — 256-aligned in `"f16"` mode, tight (`width*lanes*4`) in
   *  `"f32"` mode. Pass straight to `copyBufferToTexture`. */
  readonly bytesPerRow: number;
  /** Rows per image slice (= `height`), for the same copy. */
  readonly rowsPerImage: number;
  readonly lanes: number;
}

const WG = 64;
const align256 = (n: number): number => Math.ceil(n / 256) * 256;

/** Unpack one integer plane, normalise it, and scatter it into lane `lane` of a tight
 *  interleaved f32 destination. One dispatch per plane: N planes cost N dispatches rather than
 *  one kernel with N bindings, which keeps us clear of the 8-storage-buffer limit that makes
 *  every dispatch a silent no-op when exceeded. */
const SCATTER = /* wgsl */ `
struct Uni {
  count: u32,     // samples in this plane (w*h*d)
  lanes: u32,     // interleave stride in the destination
  lane: u32,      // which lane this plane writes
  bits: u32,      // 8 | 16 | 32
  scale: f32,     // dtype normalisation
  strideX: u32,   // invocations per dispatch row, for the 2-D dispatch below
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(${WG})
fn scatter(@builtin(global_invocation_id) gid: vec3u) {
  // The dispatch is 2-D purely to clear the 65535-workgroup-per-dimension cap; this folds it
  // back into one linear sample index.
  let i = gid.y * U.strideX + gid.x;
  if (i >= U.count) { return; }
  var v: u32;
  if (U.bits == 8u) {
    v = (src[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
  } else if (U.bits == 16u) {
    v = (src[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu;
  } else {
    v = src[i];
  }
  dst[i * U.lanes + U.lane] = f32(v) * U.scale;
}
`;

/** Pack a tight interleaved f32 tile into half-float pairs with 256-byte-aligned rows — the
 *  layout `copyBufferToTexture` requires. One invocation per destination u32 word, so no two
 *  invocations touch the same word and the pad is written (not left stale). */
const PACK_HALF = /* wgsl */ `
struct Uni {
  srcRowSamples: u32,  // width * lanes
  dstRowWords: u32,    // 256-aligned row, in u32 words
  rows: u32,           // height * depth
  pad0: u32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(${WG})
fn packHalf(@builtin(global_invocation_id) gid: vec3u) {
  let word = gid.x;
  let row = gid.y;
  if (word >= U.dstRowWords || row >= U.rows) { return; }
  let s0 = word * 2u;
  var v = vec2f(0.0, 0.0);
  if (s0 < U.srcRowSamples) {
    let base = row * U.srcRowSamples;
    v.x = src[base + s0];
    if (s0 + 1u < U.srcRowSamples) { v.y = src[base + s0 + 1u]; }
  }
  dst[row * U.dstRowWords + word] = pack2x16float(v);
}
`;

interface Ctx {
  scatter: GPUComputePipeline;
  pack: GPUComputePipeline;
}
const ctxByDevice = new WeakMap<GPUDevice, Ctx>();

function getCtx(device: GPUDevice): Ctx {
  let ctx = ctxByDevice.get(device);
  if (!ctx) {
    const pipe = (code: string, entryPoint: string): GPUComputePipeline =>
      device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint } });
    ctx = { scatter: pipe(SCATTER, "scatter"), pack: pipe(PACK_HALF, "packHalf") };
    ctxByDevice.set(device, ctx);
  }
  return ctx;
}

/** WebGPU caps `dispatchWorkgroups` at 65535 per dimension, and exceeding it is a SILENT no-op
 *  (the tile simply comes back zeroed, which reads as a maths bug). Split the linear invocation
 *  count across x and y; `strideX` tells the kernel how to fold them back. */
function dispatch2D(invocations: number): { x: number; y: number; strideX: number } {
  const groups = Math.ceil(invocations / WG);
  if (groups <= 65535) return { x: groups, y: 1, strideX: groups * WG };
  const x = 65535;
  return { x, y: Math.ceil(groups / x), strideX: x * WG };
}

let ownedSeq = 0;
/** A tile payload is **owned**, not leased from the Tier-2 pool: it lives in the `TileCache` for
 *  as long as the camera wants it, which is not a liveness the pool models, and recycling it
 *  underneath a resident texture is exactly the bug the pool's rules exist to prevent. The lease
 *  id is negative so an owned payload can never be mistaken for (or released into) a pool. */
function ownedBuffer(device: GPUDevice, byteLength: number, usage: number): ResidentBuffer {
  const size = Math.max(4, Math.ceil(byteLength / 4) * 4);
  const lease: LeaseToken = { id: -++ownedSeq, usage, capacity: size };
  return { buffer: device.createBuffer({ size, usage }), byteLength, lease };
}

// Scratch for the f32 intermediate in "f16" mode. Grow-only and never destroyed — destroying a
// buffer mid-process segfaults Dawn-on-Node (ADR-0002/0003), the same rule the Tier-2 pool follows.
const scratchByDevice = new WeakMap<GPUDevice, { buffer: GPUBuffer; bytes: number }>();
function scratch(device: GPUDevice, bytes: number): GPUBuffer {
  const held = scratchByDevice.get(device);
  if (held && held.bytes >= bytes) return held.buffer;
  const size = Math.max(bytes, (held?.bytes ?? 0) * 2, 256);
  const buffer = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  scratchByDevice.set(device, { buffer, bytes: size });
  return buffer;
}

/** Upload one decoded integer plane verbatim. No conversion, no copy — `writeBuffer` straight
 *  from the codec's own output, which is the whole point: the u16 samples the decoder produced
 *  are the bytes that reach the device. */
export function uploadPlane(device: GPUDevice, samples: ArrayBufferView): GPUBuffer {
  const bytes = samples.byteLength;
  const size = Math.ceil(bytes / 4) * 4;
  const buffer = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  if (size === bytes) {
    device.queue.writeBuffer(buffer, 0, samples as BufferSource);
  } else {
    // `writeBuffer` requires the SOURCE size to be a multiple of 4, not just the destination —
    // so an odd-length 8/16-bit plane (only ever a tiny edge chunk) needs one padded copy. Real
    // chunks are even, so this branch is not on the hot path.
    const padded = new Uint8Array(size);
    padded.set(new Uint8Array(samples.buffer, samples.byteOffset, bytes));
    device.queue.writeBuffer(buffer, 0, padded);
  }
  return buffer;
}

/**
 * Assemble `planes` into one device-resident tile payload.
 *
 * Submits and resolves when the GPU has finished, so the caller may immediately copy the result
 * into a texture or bind it. The `planes` buffers belong to the caller and are untouched.
 */
export async function assembleTile(opts: AssembleOpts): Promise<AssembledTile> {
  const { device, planes, width, height, depth, bits, scale, out } = opts;
  const lanes = planes.length;
  if (lanes < 1) throw new Error("assembleTile: no planes");
  if (width < 1 || height < 1 || depth < 1) throw new Error(`assembleTile: empty extent ${width}×${height}×${depth}`);

  const ctx = getCtx(device);
  const voxels = width * height * depth;
  const samples = voxels * lanes;
  const tightBytes = samples * 4;

  // Stage 1 — every plane scatters into the tight f32 interleave. In "f32" mode that buffer IS
  // the result; in "f16" mode it is the scratch the pack stage reads.
  const f32Target =
    out === "f32" ? ownedBuffer(device, tightBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) : null;
  const f32Buffer = f32Target?.buffer ?? scratch(device, tightBytes);

  const encoder = device.createCommandEncoder({ label: "assembleTile" });
  const pass = encoder.beginComputePass();
  pass.setPipeline(ctx.scatter);
  const d = dispatch2D(voxels);
  const uniforms: GPUBuffer[] = [];
  for (let lane = 0; lane < lanes; lane++) {
    const uni = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    uniforms.push(uni);
    const words = new ArrayBuffer(32);
    new Uint32Array(words, 0, 4).set([voxels, lanes, lane, bits]);
    new Float32Array(words, 16, 1)[0] = scale;
    new Uint32Array(words, 20, 1)[0] = d.strideX;
    device.queue.writeBuffer(uni, 0, words);
    pass.setBindGroup(
      0,
      device.createBindGroup({
        layout: ctx.scatter.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uni } },
          { binding: 1, resource: { buffer: planes[lane] as GPUBuffer } },
          { binding: 2, resource: { buffer: f32Buffer } },
        ],
      }),
    );
    pass.dispatchWorkgroups(d.x, d.y);
  }

  // Stage 2 — pack to half-floats with texture-ready row padding.
  let result: ResidentBuffer;
  let bytesPerRow: number;
  if (out === "f32") {
    result = f32Target as ResidentBuffer;
    bytesPerRow = width * lanes * 4;
  } else {
    bytesPerRow = align256(width * lanes * 2);
    const dstRowWords = bytesPerRow / 4;
    const rows = height * depth;
    result = ownedBuffer(device, bytesPerRow * rows, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    uniforms.push(uni);
    device.queue.writeBuffer(uni, 0, new Uint32Array([width * lanes, dstRowWords, rows, 0]));
    pass.setPipeline(ctx.pack);
    pass.setBindGroup(
      0,
      device.createBindGroup({
        layout: ctx.pack.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uni } },
          { binding: 1, resource: { buffer: f32Buffer } },
          { binding: 2, resource: { buffer: result.buffer } },
        ],
      }),
    );
    // rows ≤ 65535 for every chunk shape in practice (a 512×512×32 brick is 16 384); assert
    // rather than silently produce a partly-written tile if that ever stops being true.
    if (rows > 65535) throw new Error(`assembleTile: ${rows} rows exceeds the 65535 workgroup cap`);
    pass.dispatchWorkgroups(Math.ceil(dstRowWords / WG), rows);
  }

  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  return { payload: result, bytesPerRow, rowsPerImage: height, lanes };
}

/** Copy an assembled `"f16"` payload into a texture. The row padding `assembleTile` applied is
 *  exactly what makes this legal for any width. */
export function copyAssembledToTexture(device: GPUDevice, tile: AssembledTile, texture: GPUTexture, size: GPUExtent3DStrict): void {
  const encoder = device.createCommandEncoder({ label: "assembledToTexture" });
  encoder.copyBufferToTexture(
    { buffer: tile.payload.buffer, bytesPerRow: tile.bytesPerRow, rowsPerImage: tile.rowsPerImage },
    { texture },
    size,
  );
  device.queue.submit([encoder.finish()]);
}
