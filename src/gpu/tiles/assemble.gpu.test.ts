// The device-side tile assembly pass (docs/gpu-resident-loader.md §5) against the host loops it
// replaces. The reference in each case is the exact JS the playground runs today, so a pass here
// means the two paths are interchangeable, not merely "close".
import { describe, expect, it } from "vitest";
import { getDevice } from "../device";
import { adoptDevice } from "../interop/adoptDevice";
import { assembleTile, assembleTileTexture, copyAssembledToTexture, destroyTileTexture, type SampleBits, uploadPlane } from "./assemble";

/** The host path being replaced: normalise each plane and interleave into lane-major f32
 *  (spatialDataLoader.getChunk / spatialDataVolume.getChunk). */
function interleaveOnHost(planes: readonly ArrayLike<number>[], voxels: number, scale: number): Float32Array {
  const lanes = planes.length;
  const out = new Float32Array(voxels * lanes);
  for (let c = 0; c < lanes; c++) {
    const p = planes[c] as ArrayLike<number>;
    for (let i = 0; i < voxels; i++) out[i * lanes + c] = (p[i] ?? 0) * scale;
  }
  return out;
}

function ramp(n: number, bits: SampleBits): Uint8Array | Uint16Array | Uint32Array {
  const max = bits === 8 ? 255 : bits === 16 ? 65535 : 4294967295;
  const Ctor = bits === 8 ? Uint8Array : bits === 16 ? Uint16Array : Uint32Array;
  const out = new Ctor(n);
  // A non-trivial pattern: distinct per index, and hitting the top of the range.
  for (let i = 0; i < n; i++) out[i] = (i * 2654435761) % (max + 1);
  return out;
}

/** Unpack a half-float-packed buffer back to f32 on the device, so the pack stage can be verified
 *  through the same f32 readback the rest of the stack uses (a raw u32 readback would have its
 *  bit patterns mangled by the float conversion). */
const UNPACK = /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn unpack(@builtin(global_invocation_id) gid: vec3u) {
  let w = gid.x;
  if (w * 2u >= arrayLength(&dst)) { return; }
  let v = unpack2x16float(src[w]);
  dst[w * 2u] = v.x;
  dst[w * 2u + 1u] = v.y;
}
`;

/** Read a texture's texels back through a storage buffer, to prove the copyBufferToTexture leg. */
const SAMPLE_TEX = /* wgsl */ `
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(8, 8)
fn sample(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(tex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let t = textureLoad(tex, vec2u(gid.x, gid.y), 0);
  let base = (gid.y * dims.x + gid.x) * 4u;
  dst[base] = t.r;
  dst[base + 1u] = t.g;
  dst[base + 2u] = t.b;
  dst[base + 3u] = t.a;
}
`;

/** Read a 3-D texture back through textureLoad — the volume raymarch's access, minus the march. */
const LOAD_3D = /* wgsl */ `
@group(0) @binding(0) var tex: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(4, 4, 4)
fn load3d(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(tex);
  if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) { return; }
  dst[(gid.z * dims.y + gid.y) * dims.x + gid.x] = textureLoad(tex, vec3u(gid.x, gid.y, gid.z), 0).r;
}
`;

async function runKernel(
  device: GPUDevice,
  code: string,
  entryPoint: string,
  entries: GPUBindGroupEntry[],
  groups: [number, number] | [number, number, number],
): Promise<void> {
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint } });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
  pass.dispatchWorkgroups(groups[0], groups[1], groups[2] ?? 1);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

describe("assembleTile", () => {
  it.each([8, 16, 32] as SampleBits[])("matches the host interleave loop for %i-bit samples", async (bits) => {
    const device = await getDevice();
    const backend = adoptDevice(device, "assemble-test");
    // Deliberately awkward extent: width is not a multiple of 4 (so 8-bit unpacking straddles
    // words), and depth > 1 exercises the volumetric case in the same kernel.
    const [w, h, d] = [5, 3, 2];
    const voxels = w * h * d;
    const lanes = 3;
    const max = bits === 8 ? 255 : bits === 16 ? 65535 : 4294967295;
    const scale = 1 / max;

    const host = Array.from({ length: lanes }, () => ramp(voxels, bits));
    const planes = host.map((p) => uploadPlane(device, p));
    const tile = await assembleTile({ device, planes, width: w, height: h, depth: d, bits, scale, out: "f32" });

    const got = await backend.readbackF32(tile.payload.buffer, voxels * lanes);
    const want = interleaveOnHost(host, voxels, scale);
    expect(tile.lanes).toBe(lanes);
    // NOT bit-identical, and shouldn't be asserted as such: JS multiplies `v * scale` in f64 and
    // rounds once on the store to Float32Array, while the GPU multiplies two f32s. For 8-bit that
    // shows up in the last ulp (13/255 lands on either side). Agreement to f32 epsilon is the
    // honest contract — the paths are interchangeable, not bitwise equal.
    let maxDiff = 0;
    for (let i = 0; i < want.length; i++) maxDiff = Math.max(maxDiff, Math.abs((got[i] ?? 0) - (want[i] ?? 0)));
    expect(maxDiff).toBeLessThan(1e-6);
  });

  it("packs half-floats with texture-ready row padding", async () => {
    const device = await getDevice();
    const backend = adoptDevice(device, "assemble-test");
    const [w, h, d] = [5, 3, 1];
    const voxels = w * h * d;
    const lanes = 4;
    const scale = 1 / 65535;

    const host = Array.from({ length: lanes }, () => ramp(voxels, 16));
    const planes = host.map((p) => uploadPlane(device, p));
    const tile = await assembleTile({ device, planes, width: w, height: h, depth: d, bits: 16, scale, out: "f16" });

    // 5 px × 4 lanes × 2 B = 40 B, padded up to one 256 B row.
    expect(tile.bytesPerRow).toBe(256);
    const words = (tile.bytesPerRow / 4) * h;
    const unpacked = device.createBuffer({ size: words * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    await runKernel(
      device,
      UNPACK,
      "unpack",
      [
        { binding: 0, resource: { buffer: tile.payload.buffer } },
        { binding: 1, resource: { buffer: unpacked } },
      ],
      [Math.ceil(words / 64), 1],
    );
    const got = await backend.readbackF32(unpacked, words * 2);
    const want = interleaveOnHost(host, voxels, scale);

    const rowSamples = (tile.bytesPerRow / 4) * 2; // f32-equivalent samples per padded row
    for (let y = 0; y < h; y++) {
      for (let i = 0; i < w * lanes; i++) {
        // fp16 has ~3 decimal digits; the values are normalised into [0,1].
        expect(got[y * rowSamples + i]).toBeCloseTo(want[y * w * lanes + i] as number, 3);
      }
    }
  });

  it("fills a texture end-to-end, with no host pass over the samples", async () => {
    const device = await getDevice();
    const backend = adoptDevice(device, "assemble-test");
    const [w, h] = [5, 3];
    const voxels = w * h;
    const lanes = 4;
    const scale = 1 / 65535;

    const host = Array.from({ length: lanes }, () => ramp(voxels, 16));
    const planes = host.map((p) => uploadPlane(device, p));
    const tile = await assembleTile({ device, planes, width: w, height: h, depth: 1, bits: 16, scale, out: "f16" });

    const texture = device.createTexture({
      size: { width: w, height: h },
      format: "rgba16float",
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    });
    copyAssembledToTexture(device, tile, texture, { width: w, height: h });

    const out = device.createBuffer({ size: voxels * 4 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    await runKernel(
      device,
      SAMPLE_TEX,
      "sample",
      [
        { binding: 0, resource: texture.createView() },
        { binding: 1, resource: { buffer: out } },
      ],
      [Math.ceil(w / 8), Math.ceil(h / 8)],
    );
    const got = await backend.readbackF32(out, voxels * 4);
    const want = interleaveOnHost(host, voxels, scale);
    for (let i = 0; i < voxels * lanes; i++) expect(got[i]).toBeCloseTo(want[i] as number, 3);
  });

  it("reads a strided sub-box, as a fill-padded edge chunk needs", async () => {
    // The real case this exists for: a zarr chunk is stored FULL-SIZE (32×512×512) even at the
    // volume's border, and the tile is the sub-box that actually holds data. Plus a chunked `c`
    // axis, which offsets the wanted channel inside the same buffer. Both were host loops.
    const device = await getDevice();
    const backend = adoptDevice(device, "assemble-test");
    const [cw, ch, cd] = [8, 6, 4]; // stored chunk
    const [w, h, d] = [5, 4, 3]; // real extent of the tile
    const channels = 2;
    const channel = 1;

    // Stored layout: (c, z, y, x), x fastest — the C order zarr gives us.
    const stored = new Uint16Array(channels * cd * ch * cw);
    for (let c = 0; c < channels; c++) {
      for (let z = 0; z < cd; z++) {
        for (let y = 0; y < ch; y++) {
          for (let x = 0; x < cw; x++) {
            // Distinct per (c,z,y,x) so any mis-stride lands on a different value.
            stored[((c * cd + z) * ch + y) * cw + x] = (c * 10000 + z * 700 + y * 37 + x) % 65536;
          }
        }
      }
    }

    const tile = await assembleTile({
      device,
      planes: [uploadPlane(device, stored)],
      width: w,
      height: h,
      depth: d,
      bits: 16,
      scale: 1 / 65535,
      out: "f32",
      layout: { offset: channel * cd * ch * cw, x: 1, y: cw, z: cw * ch },
    });
    const got = await backend.readbackF32(tile.payload.buffer, w * h * d);

    let maxErr = 0;
    for (let z = 0; z < d; z++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const want = ((channel * 10000 + z * 700 + y * 37 + x) % 65536) / 65535;
          maxErr = Math.max(maxErr, Math.abs((got[(z * h + y) * w + x] ?? -1) - want));
        }
      }
    }
    expect(maxErr).toBeLessThan(1e-6);
  });

  it("assembleTileTexture builds a sampled 3-D texture in one call", async () => {
    const device = await getDevice();
    const backend = adoptDevice(device, "assemble-test");
    const [w, h, d] = [6, 5, 3];
    const voxels = w * h * d;
    const src = new Uint16Array(voxels);
    for (let i = 0; i < voxels; i++) src[i] = (i * 977) % 65536;

    const res = await assembleTileTexture({
      device,
      planes: [uploadPlane(device, src)],
      width: w,
      height: h,
      depth: d,
      bits: 16,
      scale: 1 / 65535,
    });
    expect(res.format).toBe("r16float");
    expect(res.depth).toBe(d);
    expect(res.texture.dimension).toBe("3d");

    // Read the texture back through textureLoad, which is how the raymarch will reach it.
    const out = device.createBuffer({ size: voxels * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    await runKernel(
      device,
      LOAD_3D,
      "load3d",
      [
        { binding: 0, resource: res.texture.createView() },
        { binding: 1, resource: { buffer: out } },
      ],
      [Math.ceil(w / 4), Math.ceil(h / 4), Math.ceil(d / 4)],
    );
    const got = await backend.readbackF32(out, voxels);
    for (let i = 0; i < voxels; i++) expect(got[i]).toBeCloseTo(((i * 977) % 65536) / 65535, 3);

    destroyTileTexture(res);
  });

  it("survives a dispatch past the 65535-workgroup cap", async () => {
    // 4 194 304 voxels needs 65 536 workgroups at 64 threads — one over the per-dimension cap,
    // which WebGPU fails SILENTLY (the tile comes back zeroed and reads as a maths bug). This is
    // the smallest extent that would catch losing the 2-D dispatch.
    const device = await getDevice();
    const backend = adoptDevice(device, "assemble-test");
    const [w, h] = [2048, 2048];
    const voxels = w * h;
    const src = new Uint16Array(voxels);
    for (let i = 0; i < voxels; i++) src[i] = i % 65536;

    const tile = await assembleTile({
      device,
      planes: [uploadPlane(device, src)],
      width: w,
      height: h,
      depth: 1,
      bits: 16,
      scale: 1 / 65535,
      out: "f32",
    });
    const got = await backend.readbackF32(tile.payload.buffer, voxels);

    // The tail is what a lost second dispatch dimension would leave at zero.
    expect(got[voxels - 1]).toBeCloseTo(((voxels - 1) % 65536) / 65535, 6);
    expect(got[voxels - 65537]).toBeCloseTo(((voxels - 65537) % 65536) / 65535, 6);
    expect(got[0]).toBe(0);
    // Sampling every 4096th voxel: the ramp wraps every 65536, so exactly one sample in 16 is a
    // true zero — 1024 samples, 64 of them zero.
    let nonZero = 0;
    for (let i = 0; i < voxels; i += 4096) if ((got[i] ?? 0) > 0) nonZero++;
    expect(nonZero).toBe(960);
  });
});
