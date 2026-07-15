// GPU-resident renderer for an HsPf field (ADR-0011, decision 6): a full-screen fragment pass that
// samples the sim's state buffer directly — no per-frame CPU readback — and, in one shader, does
// nodata→transparent (drawing the Africa coastline), a palette LUT (sequential for genotype
// frequencies, diverging-about-0 for LD), and `fwidth`-antialiased iso-line contours. This is the
// codebase's first canvas WebGPU render pass; it follows splatDensity's cached-pipeline /
// vertexless-quad pattern and the Dawn-on-Node rules (build the pipeline once, never `.destroy()`).
//
// The state buffer is 5 layers × n f32, layer-major (layer c starts at c*n). Ocean cells carry a
// negative sentinel in every layer, so layer 0 (< 0) is the coastline mask regardless of which
// channel is shown.
import type { HspfSim } from "../../../src/gpu/sim/hspf/kernel";

const SHADER = /* wgsl */ `
struct RenderParams {
  gridW: u32, gridH: u32, channel: u32, diverging: u32,
  vmin: f32, vmax: f32, contourInterval: f32, contourOn: u32,
};
@group(0) @binding(0) var<uniform> P: RenderParams;
@group(0) @binding(1) var<storage, read> state: array<f32>;

struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  let c = corners[vi];
  var o: VSOut;
  o.pos = vec4<f32>(c, 0.0, 1.0);
  o.uv = c * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5); // uv.y = 0 at top
  return o;
}

fn n() -> u32 { return P.gridW * P.gridH; }
fn cellIndex(col: i32, row: i32) -> u32 {
  let c = u32(clamp(col, 0, i32(P.gridW) - 1));
  let r = u32(clamp(row, 0, i32(P.gridH) - 1));
  return r * P.gridW + c;
}
fn isLand(col: i32, row: i32) -> bool { return state[cellIndex(col, row)] >= 0.0; } // layer 0 mask
fn channelAt(col: i32, row: i32) -> f32 { return state[P.channel * n() + cellIndex(col, row)]; }

// Bilinear sample of the selected channel, clamping ocean neighbours to the base cell (no bleed).
fn sampleValue(gx: f32, gy: f32) -> f32 {
  let col0 = i32(floor(gx));
  let row0 = i32(floor(gy));
  let fx = gx - f32(col0);
  let fy = gy - f32(row0);
  let base = channelAt(col0, row0);
  let v00 = select(base, channelAt(col0, row0), isLand(col0, row0));
  let v10 = select(base, channelAt(col0 + 1, row0), isLand(col0 + 1, row0));
  let v01 = select(base, channelAt(col0, row0 + 1), isLand(col0, row0 + 1));
  let v11 = select(base, channelAt(col0 + 1, row0 + 1), isLand(col0 + 1, row0 + 1));
  return mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
}

fn sequential(t: f32) -> vec3<f32> {
  // viridis-ish 5-stop ramp
  let c0 = vec3<f32>(0.267, 0.005, 0.329);
  let c1 = vec3<f32>(0.188, 0.407, 0.556);
  let c2 = vec3<f32>(0.208, 0.718, 0.472);
  let c3 = vec3<f32>(0.565, 0.866, 0.196);
  let c4 = vec3<f32>(0.993, 0.906, 0.144);
  let u = clamp(t, 0.0, 1.0) * 4.0;
  if (u < 1.0) { return mix(c0, c1, u); }
  if (u < 2.0) { return mix(c1, c2, u - 1.0); }
  if (u < 3.0) { return mix(c2, c3, u - 2.0); }
  return mix(c3, c4, u - 3.0);
}

fn diverging(t: f32) -> vec3<f32> {
  // blue → white → red, centred at t = 0.5
  let blue = vec3<f32>(0.230, 0.299, 0.754);
  let white = vec3<f32>(0.95, 0.95, 0.95);
  let red = vec3<f32>(0.706, 0.016, 0.150);
  let u = clamp(t, 0.0, 1.0);
  if (u < 0.5) { return mix(blue, white, u * 2.0); }
  return mix(white, red, (u - 0.5) * 2.0);
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let gx = in.uv.x * f32(P.gridW) - 0.5;
  let gy = in.uv.y * f32(P.gridH) - 0.5;

  let v = sampleValue(gx, gy);
  let t = (v - P.vmin) / max(P.vmax - P.vmin, 1e-6);
  var col = select(sequential(t), diverging(t), P.diverging == 1u);

  // Contours. fwidth() reads screen-space derivatives, so it MUST run in uniform control flow —
  // i.e. before any nodata early-return (an ocean fragment exiting the quad makes the derivative
  // undefined → NaN → the whole field vanishes). So we mask nodata via alpha at the very end.
  let f = v / max(P.contourInterval, 1e-6);
  let aa = fwidth(f);
  if (P.contourOn == 1u && aa > 1e-6) {
    let line = abs(fract(f - 0.5) - 0.5); // 0 exactly on an iso-line
    let strength = 1.0 - smoothstep(0.0, aa * 1.5, line);
    col = mix(col, vec3<f32>(0.0, 0.0, 0.0), strength * 0.55);
  }

  // nodata (ocean) → transparent, drawing the coastline. Premultiplied output.
  let land = select(0.0, 1.0, isLand(i32(round(gx)), i32(round(gy))));
  return vec4<f32>(col * land, land);
}
`;

/** How to colour a channel: value range and whether it's diverging (LD) with optional contours. */
export interface ChannelStyle {
  channel: number;
  vmin: number;
  vmax: number;
  diverging: boolean;
  contourInterval: number;
  contours: boolean;
}

/** Default styling per state layer: 0..3 genotype frequencies (sequential 0..1), 4 = LD (diverging). */
export function channelStyle(channel: number): ChannelStyle {
  if (channel === 4) return { channel, vmin: -1, vmax: 1, diverging: true, contourInterval: 0.25, contours: true };
  return { channel, vmin: 0, vmax: 1, diverging: false, contourInterval: 0.1, contours: true };
}

interface Pipe {
  device: GPUDevice;
  pipeline: GPURenderPipeline;
  format: GPUTextureFormat;
}
let pipeCache: Promise<Pipe> | undefined;

function getPipe(device: GPUDevice): Promise<Pipe> {
  pipeCache ??= (async () => {
    const module = device.createShaderModule({ label: "hspfFieldRender", code: SHADER });
    const format = navigator.gpu.getPreferredCanvasFormat();
    const pipeline = device.createRenderPipeline({
      label: "hspfFieldRender",
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      primitive: { topology: "triangle-list" },
      fragment: { module, entryPoint: "fs", targets: [{ format }] },
    });
    return { device, pipeline, format };
  })();
  return pipeCache;
}

/** Renders an `HspfSim`'s current field to a canvas, GPU-resident (no readback). */
export class FieldRenderer {
  private readonly uniform: GPUBuffer;

  private constructor(
    private readonly device: GPUDevice,
    private readonly ctx: GPUCanvasContext,
    private readonly pipeline: GPURenderPipeline,
  ) {
    this.uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  static async create(device: GPUDevice, canvas: HTMLCanvasElement): Promise<FieldRenderer> {
    const { pipeline, format } = await getPipe(device);
    const ctx = canvas.getContext("webgpu");
    if (!ctx) throw new Error("hspf: canvas WebGPU context unavailable");
    ctx.configure({ device, format, alphaMode: "premultiplied" });
    return new FieldRenderer(device, ctx, pipeline);
  }

  render(sim: HspfSim, style: ChannelStyle): void {
    const u = new ArrayBuffer(32);
    const dv = new DataView(u);
    dv.setUint32(0, sim.width, true);
    dv.setUint32(4, sim.height, true);
    dv.setUint32(8, style.channel, true);
    dv.setUint32(12, style.diverging ? 1 : 0, true);
    dv.setFloat32(16, style.vmin, true);
    dv.setFloat32(20, style.vmax, true);
    dv.setFloat32(24, style.contourInterval, true);
    dv.setUint32(28, style.contours ? 1 : 0, true);
    this.device.queue.writeBuffer(this.uniform, 0, u);

    const bind = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: { buffer: sim.currentStateBuffer() } },
      ],
    });

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: this.ctx.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bind);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}
