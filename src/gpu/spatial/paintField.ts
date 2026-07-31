// Paint a GPU-resident scalar field straight to a canvas, through a colour LUT — no readback.
//
// This is ADR-0017 invariant 4 taken seriously for the display path: "a render-terminated graph
// should perform ZERO downloads". A density or correlation map computed on the GPU and then LOOKED
// AT never needs to reach the host. Routing it through one costs, per panel per frame:
//
//   • the GPU→CPU round trip itself (a pipeline flush, not just a memcpy),
//   • a JS loop over every pixel to strip the 256-byte row padding, and
//   • another JS loop over every pixel to write an ImageData.
//
// Measured on the cell-stats demo, that was ~55 ms for a pair of KDE panels against ~4 ms for the
// Γ render that produced them — i.e. the display path cost more than ten times the statistics.
//
// Instead: a fullscreen-triangle pipeline reads the field where it already lives — the render
// target the producing pass left it in — and writes colour into the canvas's own swapchain texture.
// ONE path for every field on screen: whatever produced it, it arrives here as a texture and is
// painted the same way. Auto-scaling stays on-device too, as an atomic max over |v| — the
// bitcast trick is exact, because for non-negative floats IEEE-754 bit order IS numeric order.
//
// The LUT is a 256×1 rgba8unorm texture. Building the ramp (OKLCh, with a gamut bisection) stays on
// the host where it belongs; only the finished 256 entries cross.
import { getDevice } from "../device";

const UNI_FLOATS = 8; // w, h, n, mode, flipY — padded to 32 bytes

const TEX_REDUCE = /* wgsl */ `
struct Uni { w: f32, h: f32, n: f32, mode: f32, flipY: f32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> scaleRW: atomic<u32>;

@compute @workgroup_size(8, 8)
fn reduceTex(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u32(U.w) || gid.y >= u32(U.h)) { return; }
  let v = textureLoad(src, vec2i(i32(gid.x), i32(gid.y)), 0).r;
  atomicMax(&scaleRW, bitcast<u32>(abs(v)));
}
`;

const PAINT_TEX = /* wgsl */ `
struct Uni { w: f32, h: f32, n: f32, mode: f32, flipY: f32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(3) var<storage, read> scaleRO: array<u32>;
@group(0) @binding(4) var lut: texture_2d<f32>;
@group(0) @binding(1) var srcTex: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
  let y = f32(vi & 2u) * 2.0 - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}

fn ramp(v: f32) -> vec4f {
  let scale = max(bitcast<f32>(scaleRO[0]), 1e-30);
  var t: f32;
  if (U.mode > 0.5) { t = (v / scale + 1.0) * 0.5; } else { t = sqrt(max(v, 0.0) / scale); }
  return textureLoad(lut, vec2i(i32(clamp(t, 0.0, 1.0) * 255.0), 0), 0);
}

@fragment
fn fsTex(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let x = i32(pos.x);
  let py = i32(pos.y);
  if (x >= i32(U.w) || py >= i32(U.h)) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  // pos.y is 0 at the TOP of the target and the raster's row 0 is at maxY, so the default maps maxY
  // to the top: a y-UP plot. flipY maps row 0 to the bottom instead, which is what data in image
  // coordinates needs -- there y is a row index growing downward, and drawing it y-up shows the
  // tissue upside down relative to its own source image. (No backticks in here: this is inside a
  // template literal.)
  let y = select(py, i32(U.h) - 1 - py, U.flipY > 0.5);
  return ramp(textureLoad(srcTex, vec2i(x, y), 0).r);
}
`;

interface Ctx {
  device: GPUDevice;
  format: GPUTextureFormat;
  reduceTex: GPUComputePipeline;
  paintTex: GPURenderPipeline;
  uni: GPUBuffer;
  scale: GPUBuffer;
}
let ctxCache: Promise<Ctx> | undefined;

function getCtx(): Promise<Ctx> {
  ctxCache ??= (async () => {
    const device = await getDevice();
    const gpu = (globalThis as { navigator?: { gpu?: GPU } }).navigator?.gpu;
    const format = gpu?.getPreferredCanvasFormat?.() ?? ("bgra8unorm" as GPUTextureFormat);
    const compute = (code: string, entryPoint: string) =>
      device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint } });
    const render = (code: string, fs: string) => {
      const module = device.createShaderModule({ code });
      return device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        primitive: { topology: "triangle-list" },
        fragment: { module, entryPoint: fs, targets: [{ format }] },
      });
    };
    return {
      device,
      format,
      reduceTex: compute(TEX_REDUCE, "reduceTex"),
      paintTex: render(PAINT_TEX, "fsTex"),
      uni: device.createBuffer({ size: UNI_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      scale: device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
    };
  })();
  return ctxCache;
}

// --- canvas contexts, LUT textures: both cached, both keyed on identity ---

interface Surface {
  ctx: GPUCanvasContext;
  w: number;
  h: number;
}
const surfaces = new WeakMap<HTMLCanvasElement, Surface>();

function ensureSurface(device: GPUDevice, format: GPUTextureFormat, canvas: HTMLCanvasElement, w: number, h: number): Surface {
  const got = surfaces.get(canvas);
  if (got && got.w === w && got.h === h) return got;
  canvas.width = w;
  canvas.height = h;
  const ctx = (got?.ctx ?? canvas.getContext("webgpu")) as GPUCanvasContext | null;
  if (!ctx) {
    throw new Error("paintField: canvas has no WebGPU context (a canvas cannot switch away from 2d once used)");
  }
  // Resizing invalidates the swapchain, so reconfigure rather than assuming the resize took.
  ctx.configure({ device, format, alphaMode: "opaque" });
  const made = { ctx, w, h };
  surfaces.set(canvas, made);
  return made;
}

const luts = new WeakMap<Uint8Array, GPUTexture>();

/** Upload a 256-entry RGB LUT as a 256×1 rgba8unorm texture, once per LUT array. */
function ensureLut(device: GPUDevice, lut: Uint8Array): GPUTexture {
  const got = luts.get(lut);
  if (got) return got;
  const n = Math.floor(lut.length / 3);
  const tex = device.createTexture({
    size: { width: n, height: 1 },
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const rgba = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    rgba[i * 4] = lut[i * 3]!;
    rgba[i * 4 + 1] = lut[i * 3 + 1]!;
    rgba[i * 4 + 2] = lut[i * 3 + 2]!;
    rgba[i * 4 + 3] = 255;
  }
  device.queue.writeTexture({ texture: tex }, rgba, { bytesPerRow: n * 4 }, { width: n, height: 1 });
  luts.set(lut, tex);
  return tex;
}

export interface PaintOptions {
  /** 256-entry RGB byte LUT (as built by the demo's `rampLut`). */
  lut: Uint8Array;
  /** `true` for a signed field (Γ): the ramp is centred, and |v| sets the scale symmetrically.
   *  `false` for a density: the ramp runs 0 → max with a sqrt gamma. */
  signed?: boolean;
  /** Draw raster row 0 at the BOTTOM of the canvas instead of the top.
   *
   *  The raster convention here is row 0 = maxY (see `splatDensity.ts`), and `@builtin(position).y`
   *  is 0 at the top, so the default puts maxY at the top: correct for a plot read y-up. Pass `true`
   *  when the world y axis points DOWN — imaging data, where y indexes a row — so that a field panel
   *  and a scatter of the same points agree, and both match the source image. */
  flipY?: boolean;
}

const ZERO = new Uint32Array(1);

/** Shared tail: reduce for the scale, then paint the canvas. `bindSource` supplies binding 1. */
function encodePaint(
  c: Ctx,
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
  opts: PaintOptions,
  reducePipe: GPUComputePipeline,
  paintPipe: GPURenderPipeline,
  source: GPUBindingResource,
  dispatch: (pass: GPUComputePassEncoder) => void,
): void {
  const { device } = c;
  const surface = ensureSurface(device, c.format, canvas, width, height);
  device.queue.writeBuffer(c.uni, 0, new Float32Array([width, height, width * height, opts.signed ? 1 : 0, opts.flipY ? 1 : 0, 0, 0, 0]));
  device.queue.writeBuffer(c.scale, 0, ZERO); // atomicMax needs a clean floor each frame

  const enc = device.createCommandEncoder();
  const rpass = enc.beginComputePass();
  rpass.setPipeline(reducePipe);
  rpass.setBindGroup(
    0,
    device.createBindGroup({
      layout: reducePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: c.uni } },
        { binding: 1, resource: source },
        { binding: 2, resource: { buffer: c.scale } },
      ],
    }),
  );
  dispatch(rpass);
  rpass.end();

  const view = surface.ctx.getCurrentTexture().createView();
  const pass = enc.beginRenderPass({
    colorAttachments: [{ view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
  });
  pass.setPipeline(paintPipe);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      layout: paintPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: c.uni } },
        { binding: 1, resource: source },
        { binding: 3, resource: { buffer: c.scale } },
        { binding: 4, resource: ensureLut(device, opts.lut).createView() },
      ],
    }),
  );
  pass.draw(3);
  pass.end();
  device.queue.submit([enc.finish()]);
}

/** Paint an r32float texture (a previous pass's render target) to a canvas. Same, with no copy at
 *  all — the field is read where the pass that produced it left it. */
export async function paintFieldTexture(
  canvas: HTMLCanvasElement,
  texture: GPUTexture,
  width: number,
  height: number,
  opts: PaintOptions,
): Promise<void> {
  const c = await getCtx();
  encodePaint(c, canvas, width, height, opts, c.reduceTex, c.paintTex, texture.createView(), (pass) =>
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8)),
  );
}
