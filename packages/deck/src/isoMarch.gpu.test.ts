// The ray march IsoSurfaceLayer draws with (`isoMarchWgsl.ts`), run in a compute shader against
// fields whose surface is known exactly. The layer's own projection and lighting are not covered.
import { compileShader, getDevice, readBackBytes } from "../../../src/gpu/device";
import { splatVolumeResident, volumeBytes } from "../../../src/gpu/spatial/splatVolume";
import { describe, expect, it } from "vite-plus/test";
import { ISO_MARCH_WGSL } from "./isoMarchWgsl";

const SHADER = /* wgsl */ `
struct IsoUniforms {
  dims: vec4<f32>,
  origin: vec4<f32>,
  tint: vec4<f32>,
  march: vec4<f32>,
};
/** In voxels, with the box from 0 to dims. origin.w is 1 when the ray starts at a near plane at its
 *  origin, as when the camera is inside the box. */
struct Ray {
  origin: vec4<f32>,
  dir: vec4<f32>,
};
@group(0) @binding(0) var<uniform> iso: IsoUniforms;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read> rays: array<Ray>;
/** Per ray: where along it the surface is met, in voxels (-1 for none), and 1 when met from inside. */
@group(0) @binding(3) var<storage, read_write> hits: array<vec2<f32>>;

${ISO_MARCH_WGSL}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= arrayLength(&rays)) {
    return;
  }
  let ray = rays[i];
  let u0 = ray.origin.xyz / iso.dims.xyz;
  let du = ray.dir.xyz / iso.dims.xyz;
  let safe = select(du, vec3<f32>(1e-9), abs(du) < vec3<f32>(1e-9));
  let ta = (vec3<f32>(0.0) - u0) / safe;
  let tb = (vec3<f32>(1.0) - u0) / safe;
  let lo = min(ta, tb);
  let hi = max(ta, tb);
  var enter = max(lo.x, max(lo.y, lo.z));
  let leave = min(hi.x, min(hi.y, hi.z));
  var fromNear = false;
  if (ray.origin.w > 0.5 && enter < 0.0) {
    enter = 0.0;
    fromNear = true;
  }
  var out = vec2<f32>(-1.0, 0.0);
  if (enter < leave) {
    let found = iso_march(u0, du, enter, leave, fromNear);
    if (found.hit) {
      out = vec2<f32>(found.at, select(0.0, 1.0, found.exiting));
    }
  }
  hits[i] = out;
}
`;

type Dims = readonly [number, number, number];

async function march(field: GPUBuffer, dims: Dims, level: number, rays: Float32Array): Promise<Float32Array> {
  const device = await getDevice();
  const module = await compileShader(device, SHADER, "test:isoMarch");
  const pipeline = device.createComputePipeline({ label: "test:isoMarch", layout: "auto", compute: { module, entryPoint: "main" } });
  const n = rays.length / 8;
  const uniforms = new Float32Array([dims[0], dims[1], dims[2], level, 0, 0, 0, 1, 0, 0, 0, 0, 512, 1, 1, 0]);
  const make = (label: string, data: Float32Array, usage: number) => {
    const b = device.createBuffer({ label, size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  };
  const uni = make("test:isoMarch:uniforms", uniforms, GPUBufferUsage.UNIFORM);
  const rayBuf = make("test:isoMarch:rays", rays, GPUBufferUsage.STORAGE);
  const hits = device.createBuffer({ label: "test:isoMarch:hits", size: n * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const bindGroup = device.createBindGroup({
    label: "test:isoMarch",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uni } },
      { binding: 1, resource: { buffer: field } },
      { binding: 2, resource: { buffer: rayBuf } },
      { binding: 3, resource: { buffer: hits } },
    ],
  });
  const enc = device.createCommandEncoder({ label: "test:isoMarch" });
  const pass = enc.beginComputePass({ label: "test:isoMarch" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(n / 64));
  pass.end();
  device.queue.submit([enc.finish()]);
  const out = new Float32Array(await readBackBytes(device, "test:isoMarch:hits", hits, 0, n * 8));
  for (const b of [uni, rayBuf, hits]) b.destroy();
  return out;
}

/** The density of one point at `at`, `radius` reaching, on a `dims` grid of unit voxels. */
async function sphereField(dims: Dims, at: readonly [number, number, number], radius: number): Promise<GPUBuffer> {
  const device = await getDevice();
  const points = device.createBuffer({ label: "test:isoMarch:point", size: 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(points, 0, new Float32Array(at));
  const field = device.createBuffer({ label: "test:isoMarch:field", size: volumeBytes(dims), usage: GPUBufferUsage.STORAGE });
  await splatVolumeResident(points, 1, field, { dims, origin: [0, 0, 0], voxel: 1, radius, stride: 3 });
  return field;
}

const N = 32;
const DIMS: Dims = [N, N, N];
const R = 12;
const LEVEL = 0.25;
/** Where (1 - d²/R²)² = LEVEL. */
const SURFACE = R * Math.sqrt(1 - Math.sqrt(LEVEL));

describe("iso_march", () => {
  it("finds a sphere's silhouette and front from outside the box", async () => {
    const field = await sphereField(DIMS, [16, 16, 16], R);
    // A 64×64 grid of parallel rays along +z, starting a voxel below the box.
    const side = 64;
    const rays = new Float32Array(side * side * 8);
    for (let j = 0; j < side; j++) {
      for (let i = 0; i < side; i++) rays.set([((i + 0.5) * N) / side, ((j + 0.5) * N) / side, -1, 0, 0, 0, 1, 0], (i + side * j) * 8);
    }
    const hits = await march(field, DIMS, LEVEL, rays);
    field.destroy();
    let wrongSide = 0;
    let worstDepth = 0;
    let inside = 0;
    for (let k = 0; k < side * side; k++) {
      const d = Math.hypot(rays[k * 8]! - 16, rays[k * 8 + 1]! - 16);
      const hit = hits[k * 2]! >= 0;
      // Half a voxel either side of the outline is left to interpolation.
      if (Math.abs(d - SURFACE) > 0.5 && hit !== d < SURFACE) wrongSide++;
      if (d < SURFACE - 1) {
        inside++;
        const want = 16 - Math.sqrt(SURFACE ** 2 - d * d) + 1;
        worstDepth = Math.max(worstDepth, Math.abs(hits[k * 2]! - want));
      }
    }
    expect(inside).toBeGreaterThan(500);
    expect(wrongSide).toBe(0);
    expect(worstDepth).toBeLessThan(0.15);
  });

  it("sees the inside of the wall from a camera within the surface", async () => {
    const field = await sphereField(DIMS, [16, 16, 16], R);
    // 64 directions spread over the sphere, all from its centre, each starting at a near plane there.
    const count = 64;
    const rays = new Float32Array(count * 8);
    for (let k = 0; k < count; k++) {
      const z = 1 - (2 * (k + 0.5)) / count;
      const ring = Math.sqrt(1 - z * z);
      const turn = k * Math.PI * (3 - Math.sqrt(5));
      rays.set([16, 16, 16, 1, ring * Math.cos(turn), ring * Math.sin(turn), z, 0], k * 8);
    }
    const hits = await march(field, DIMS, LEVEL, rays);
    field.destroy();
    let worst = 0;
    let exits = 0;
    for (let k = 0; k < count; k++) {
      worst = Math.max(worst, Math.abs(hits[k * 2]! - SURFACE));
      exits += hits[k * 2 + 1]!;
    }
    expect(exits).toBe(count);
    expect(worst).toBeLessThan(0.15);
  });

  it("draws the cut where the box face crosses the surface", async () => {
    // The sphere's centre on the box's low z face: rays along +z meet the field above the level at
    // that face, where the cut is drawn, not inside.
    const field = await sphereField(DIMS, [16, 16, 0], R);
    const rays = new Float32Array([16, 16, -1, 0, 0, 0, 1, 0, 20, 16, -1, 0, 0, 0, 1, 0, 16 + SURFACE + 2, 16, -1, 0, 0, 0, 1, 0]);
    const hits = await march(field, DIMS, LEVEL, rays);
    field.destroy();
    expect([hits[0], hits[1], hits[2], hits[3]]).toEqual([1, 0, 1, 0]);
    expect(hits[4]).toBe(-1);
  });

  it("does not draw at the camera when it starts inside (CodeRabbit's four slices)", async () => {
    // Four z slices [0, 2, 2, 0] at level 1, seen from between the 2s looking down z: the wall is
    // where the field falls through 1, halfway between the slices at z 0.5 and 1.5.
    const device = await getDevice();
    const dims: Dims = [1, 1, 4];
    const field = device.createBuffer({ label: "test:isoMarch:slices", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(field, 0, new Float32Array([0, 2, 2, 0]));
    const hits = await march(field, dims, 1, new Float32Array([0.5, 0.5, 2, 1, 0, 0, -1, 0]));
    field.destroy();
    expect(hits[0]).toBeCloseTo(1, 2);
    expect(hits[1]).toBe(1);
  });

  it("finds a wall it grazes on the way out (CodeRabbit's diagonal)", async () => {
    // A 2×2 checkerboard [1, 0, 0, 1] at level 0.51, seen from a corner voxel's centre along the
    // diagonal. The field dips to about 0.5 between two half-voxel steps that both stay above the
    // level, so only the closer search finds the wall.
    const device = await getDevice();
    const dims: Dims = [2, 2, 1];
    const values = [1, 0, 0, 1];
    const level = 0.51;
    const field = device.createBuffer({
      label: "test:isoMarch:checker",
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(field, 0, new Float32Array(values));
    // In box coordinates the direction is (1, 1, 0) / √2; rays here are in voxels.
    const hits = await march(field, dims, level, new Float32Array([0.5, 0.5, 0.5, 1, Math.SQRT2, Math.SQRT2, 0, 0]));
    field.destroy();
    // The field where the march stopped, interpolated as the shader does.
    const at = (v: number) => Math.min(Math.max(v, 0), 1);
    const p = 0.5 + Math.SQRT2 * hits[0]! - 0.5;
    const f = at(p);
    const there = (1 - f) * (1 - f) * values[0]! + f * (1 - f) * (values[1]! + values[2]!) + f * f * values[3]!;
    expect(hits[1]).toBe(1);
    expect(Math.abs(there - level)).toBeLessThan(0.005);
  });
});
