// The maximum of a GPU-resident non-negative f32 buffer, left on the GPU — so a renderer can scale
// colour to a field it has never seen, with no readback.
//
// `paintField.ts` does the same for a texture, one atomic per texel. This is the buffer form, and it
// reduces inside each workgroup first: one `atomicMax` per workgroup rather than per element, since
// a million threads contending on one word is the slow part of an atomic reduction.
//
// The result is written as the u32 BIT PATTERN of the f32 maximum: for non-negative floats IEEE-754
// bit order is numeric order, which is what makes `atomicMax` on u32 an exact float max. A consumer
// reads it back with `bitcast<f32>(m[0])`. Negative values (which a density never has) count as 0.
import { compileShader, getDevice } from "../device";

const WG = 256;
/** Workgroups dispatched at most; each thread strides over the buffer, so this bounds the dispatch
 *  well under the per-dimension cap for any buffer size. */
const MAX_GROUPS = 1024;

const SHADER = /* wgsl */ `
struct Uni { n: u32, stride: u32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<atomic<u32>>;

var<workgroup> part: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn fieldMax(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  var m = 0.0;
  for (var i = gid.x; i < U.n; i = i + U.stride) {
    m = max(m, src[i]);
  }
  part[li] = m;
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s = s >> 1u) {
    if (li < s) {
      part[li] = max(part[li], part[li + s]);
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    atomicMax(&dst[0], bitcast<u32>(part[0]));
  }
}
`;

interface Ctx {
  device: GPUDevice;
  pipeline: GPUComputePipeline;
  uni: GPUBuffer;
}

/** Per device, held weakly — see `splatDensity.ts`'s `Gpu` for why. */
const perDevice = new WeakMap<GPUDevice, Promise<Ctx>>();

async function getCtx(device?: GPUDevice): Promise<Ctx> {
  const dev = device ?? (await getDevice());
  let ctx = perDevice.get(dev);
  if (!ctx) {
    ctx = (async () => {
      const module = await compileShader(dev, SHADER, "fieldMax");
      const pipeline = dev.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "fieldMax" } });
      const uni = dev.createBuffer({ label: "fieldMax:uniforms", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      return { device: dev, pipeline, uni };
    })();
    perDevice.set(dev, ctx);
  }
  return ctx;
}

/** Write max(0, max(src[0..n))) into `dst[0]` as its f32 bit pattern (a u32). `dst` needs STORAGE
 *  and COPY_DST usage and at least 4 bytes; it is cleared here, so any previous value is gone.
 *  `device` is the one the buffers live on; default: the library's own. */
export async function fieldMaxResident(src: GPUBuffer, n: number, dst: GPUBuffer, device?: GPUDevice): Promise<void> {
  const { device: dev, pipeline, uni } = await getCtx(device);
  // No await below this line: the shared uniform is written and consumed in one submit, so a second
  // caller cannot slip its `n` in between (see splatDensityResident).
  const groups = Math.max(1, Math.min(MAX_GROUPS, Math.ceil(n / WG)));
  dev.queue.writeBuffer(uni, 0, new Uint32Array([n, groups * WG]));
  dev.queue.writeBuffer(dst, 0, new Uint32Array([0]));
  const enc = dev.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    dev.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: src, size: Math.max(4, n * 4) } },
        { binding: 2, resource: { buffer: dst, size: 4 } },
      ],
    }),
  );
  pass.dispatchWorkgroups(groups);
  pass.end();
  dev.queue.submit([enc.finish()]);
}
