// The pair matrix, drawn on the GPU from the simulation's own buffers: one fragment shader over a
// canvas, raw WebGPU, on the device the simulation runs on. Nothing is read back and it is live
// every frame.
//
// A pixel is a pair of particles (a, b). Its colour is how far apart they are now; a pair joined by
// a spring is marked, straight from the bond table, so what is drawn is what the model holds; and
// the editor's lines are drawn over the top with a handle at each end.
import { compileShader, perDevice } from "../../../src/gpu/device";
import type { MatrixLine } from "../../../src/gpu/sim/springs/index";

/** Lines the shader can draw. More are still applied to the model; they are just not outlined. */
const MAX_LINES = 32;

const WGSL = /* wgsl */ `
struct Params {
  n: f32, K: f32, slotWords: f32, positionStride: f32,
  maxDistance: f32, lineCount: f32, activeLine: f32, pad: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> slots: array<u32>;
// The editor's lines, (a0, b0, a1, b1) each. A storage buffer, not an array in the uniform block:
// indexing a uniform array in a loop made Metal copy the whole array per pixel (40 ms a frame).
@group(0) @binding(3) var<storage, read> lines: array<vec4<f32>>;

struct Varyings {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vertexMain(@builtin(vertex_index) i: u32) -> Varyings {
  // One triangle covering the canvas.
  var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var output: Varyings;
  output.position = vec4<f32>(corners[i], 0.0, 1.0);
  // uv.y runs down the canvas, as the matrix's rows do.
  output.uv = vec2<f32>(corners[i].x * 0.5 + 0.5, 0.5 - corners[i].y * 0.5);
  return output;
}

fn particle(i: u32) -> vec3<f32> {
  let p = i * u32(P.positionStride);
  return vec3<f32>(positions[p], positions[p + 1u], positions[p + 2u]);
}

/** Distance from point q to the segment a..b. */
fn toSegment(q: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  let ab = b - a;
  let t = clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
  return distance(q, a + ab * t);
}

@fragment
fn fragmentMain(input: Varyings) -> @location(0) vec4<f32> {
  let n = u32(P.n);
  let cell = input.uv * P.n;
  let a = min(u32(cell.x), n - 1u);
  let b = min(u32(cell.y), n - 1u);

  // Distance now: dark is near, bright is far.
  let t = clamp(distance(particle(a), particle(b)) / P.maxDistance, 0.0, 1.0);
  var rgb = vec3<f32>(20.0 + 235.0 * t, 30.0 + 190.0 * t * t, 80.0 + 100.0 * (1.0 - t)) / 255.0;

  // A spring between them, read from a's row of the bond table. Accumulated, not returned early.
  var tied = 0.0;
  let K = u32(P.K);
  for (var k = 0u; k < K; k++) {
    let word = slots[(a * K + k) * u32(P.slotWords)];
    if (word != 0xFFFFFFFFu && (word & 0xFFFFFFu) == b) {
      tied = 1.0;
    }
  }
  let backbone = abs(i32(a) - i32(b)) <= 1;
  rgb = mix(rgb, select(vec3<f32>(0.97, 0.98, 1.0), vec3<f32>(0.25, 0.3, 0.9), backbone), tied * select(0.85, 0.6, backbone));

  // The editor's lines, in cell units, with a square handle at each end.
  let px = fwidth(cell.x);
  var stroke = 0.0;
  var handle = 0.0;
  var lit = 0.0;
  for (var k = 0u; k < u32(P.lineCount); k++) {
    let line = lines[k] + vec4<f32>(0.5);
    let d = toSegment(cell, line.xy, line.zw);
    let isActive = select(0.0, 1.0, f32(k) == P.activeLine);
    stroke = max(stroke, 1.0 - smoothstep(0.3, 0.3 + px, d));
    let ends = min(max(abs(cell.x - line.x), abs(cell.y - line.y)), max(abs(cell.x - line.z), abs(cell.y - line.w)));
    handle = max(handle, (1.0 - smoothstep(1.0, 1.0 + px, ends)) * smoothstep(0.6 - px, 0.6, ends));
    lit = max(lit, isActive * max(1.0 - smoothstep(0.3, 0.3 + px, d), 1.0 - smoothstep(1.0, 1.0 + px, ends)));
  }
  let pink = vec3<f32>(0.96, 0.45, 0.71);
  rgb = mix(rgb, pink, max(stroke, handle) * 0.9);
  rgb = mix(rgb, vec3<f32>(1.0), lit * 0.35);
  return vec4<f32>(rgb, 1.0);
}
`;

const UNIFORM_BYTES = 32;

const getPipeline = perDevice(async (device: GPUDevice) => {
  const module = await compileShader(device, WGSL, "pairMatrix");
  const format = navigator.gpu.getPreferredCanvasFormat();
  const pipeline = await device.createRenderPipelineAsync({
    label: "pairMatrix",
    layout: "auto",
    vertex: { module, entryPoint: "vertexMain" },
    fragment: { module, entryPoint: "fragmentMain", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
  return { pipeline, format };
});

export interface PairMatrixFrame {
  /** `positionStride` f32 per particle. */
  readonly positions: GPUBuffer;
  /** `K` slots of `slotWords` u32 per particle. */
  readonly slots: GPUBuffer;
  readonly n: number;
  readonly K: number;
  readonly slotWords: number;
  readonly positionStride?: number;
  /** The distance shown as brightest. */
  readonly maxDistance: number;
  readonly lines: readonly MatrixLine[];
  /** Index of the line under the pointer or being dragged, or -1. */
  readonly active: number;
}

export interface PairMatrixRenderer {
  draw(frame: PairMatrixFrame): void;
  destroy(): void;
}

/** A renderer for `canvas` on `device`. The canvas's pixel size is the caller's to set. */
export async function createPairMatrixRenderer(device: GPUDevice, canvas: HTMLCanvasElement): Promise<PairMatrixRenderer> {
  const { pipeline, format } = await getPipeline(device);
  const context = canvas.getContext("webgpu");
  if (!context) throw new Error("pairMatrix: this canvas cannot give a WebGPU context");
  context.configure({ device, format, alphaMode: "opaque" });
  const uniforms = device.createBuffer({
    label: "pairMatrix:params",
    size: UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const lineBuffer = device.createBuffer({
    label: "pairMatrix:lines",
    size: MAX_LINES * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const data = new Float32Array(UNIFORM_BYTES / 4);
  const lineData = new Float32Array(MAX_LINES * 4);
  return {
    draw(f) {
      if (canvas.width === 0 || canvas.height === 0) return;
      const count = Math.min(f.lines.length, MAX_LINES);
      data.set([f.n, f.K, f.slotWords, f.positionStride ?? 4, f.maxDistance, count, f.active < count ? f.active : -1, 0]);
      for (let k = 0; k < count; k++) {
        const l = f.lines[k]!;
        lineData.set([l.a0, l.b0, l.a1, l.b1], k * 4);
      }
      device.queue.writeBuffer(uniforms, 0, data);
      device.queue.writeBuffer(lineBuffer, 0, lineData);
      // A new bind group each frame: the positions buffer changes whenever the model has stepped.
      const bind = device.createBindGroup({
        label: "pairMatrix",
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniforms } },
          { binding: 1, resource: { buffer: f.positions } },
          { binding: 2, resource: { buffer: f.slots } },
          { binding: 3, resource: { buffer: lineBuffer } },
        ],
      });
      const encoder = device.createCommandEncoder({ label: "pairMatrix" });
      const pass = encoder.beginRenderPass({
        label: "pairMatrix",
        colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.draw(3);
      pass.end();
      device.queue.submit([encoder.finish()]);
    },
    destroy() {
      context.unconfigure();
      uniforms.destroy();
      lineBuffer.destroy();
    },
  };
}
