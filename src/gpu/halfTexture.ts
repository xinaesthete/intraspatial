// Narrowing an f32 field to f16 for storage — the one rounding the half-precision paths spend.
//
// Shared by `splatDensity` (a density texture) and `gramMatrix` (each channel's raster on its way
// into the stacked buffer). Both accumulate additively in f32 first, and they must: f16 carries 10
// mantissa bits, so once a texel's running total exceeds ~1024x one contribution, `total + c ==
// total` and the contribution is simply gone. `docs/field-precision.md` has the measurements —
// blending straight into `r16float` lost three quarters of a dense field's mass.
//
// Narrowing here instead costs one rounding, flat at ~0.1% whatever the overlap.
//
// A render pass rather than a compute one: `r16float` with `STORAGE_BINDING` needs the
// `texture-formats-tier2` feature, while rendering to it is core. `textureLoad` rather than a
// sampler: this is a 1:1 copy that wants no filtering, and an `r32float` source is not filterable
// without `float32-filterable`, which `getDevice()` does not request.
import { compileShader } from "./device";

const NARROW = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  // One oversized triangle rather than a quad: no index buffer, no seam down the diagonal.
  let p = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) f32 {
  return textureLoad(src, vec2i(pos.xy), 0).r;
}
`;

interface NarrowPipe {
  pipeline: GPURenderPipeline;
  layout: GPUBindGroupLayout;
}
let pipeP: Promise<NarrowPipe> | undefined;

function getPipe(device: GPUDevice): Promise<NarrowPipe> {
  pipeP ??= (async () => {
    const module = await compileShader(device, NARROW, "halfTexture:narrow");
    const layout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } }],
    });
    return {
      layout,
      pipeline: device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module, entryPoint: "vs" },
        primitive: { topology: "triangle-list" },
        fragment: { module, entryPoint: "fs", targets: [{ format: "r16float" }] },
      }),
    };
  })();
  return pipeP;
}

/**
 * Record a copy of the top-left `w × h` of an f32 texture into an `r16float` one.
 *
 * The source may be larger than the region — both callers keep a grown-and-reused scratch — so the
 * region is taken by position rather than by scaling.
 */
export async function narrowToHalf(
  device: GPUDevice,
  enc: GPUCommandEncoder,
  src: GPUTexture,
  dst: GPUTexture,
  w: number,
  h: number,
): Promise<void> {
  const { pipeline, layout } = await getPipe(device);
  const pass = enc.beginRenderPass({
    colorAttachments: [{ view: dst.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout, entries: [{ binding: 0, resource: src.createView() }] }));
  pass.setViewport(0, 0, w, h, 0, 1);
  pass.draw(3);
  pass.end();
}

/** A grown-and-reused `r16float` scratch target, one per caller key.
 *
 *  Reused rather than reallocated because reuse is cheaper, NOT because destroying it is unsafe:
 *  `test/destroy-mid-process.gpu.test.ts` destroys hundreds of textures and buffers mid-process
 *  and exits clean. The repo's older "destroying segfaults Dawn-on-Node" comments predate the
 *  Instance-lifetime fix in `device.ts` (2026-07-29), which is what those crashes actually were. */
const halfTex = new Map<string, { tex: GPUTexture; w: number; h: number }>();

export function ensureHalfTex(device: GPUDevice, key: string, w: number, h: number, extraUsage = 0): GPUTexture {
  const got = halfTex.get(key);
  if (got && got.w >= w && got.h >= h) return got.tex;
  const width = Math.max(w, got?.w ?? 0);
  const height = Math.max(h, got?.h ?? 0);
  const tex = device.createTexture({
    size: { width, height },
    format: "r16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | extraUsage,
  });
  halfTex.set(key, { tex, w: width, h: height });
  return tex;
}
