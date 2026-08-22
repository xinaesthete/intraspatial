// The fused HsPf step on the GPU. One kernel does the whole per-cell update: the seeded
// mosquito-bite neighbourhood gather, the reaction F (fitness blend + single/two-bite
// recombination), normalisation, LD, and a dormant barrier segment-crossing test (ADR-0011,
// decision 2). The arithmetic is identical to the CPU-tested `math.ts`; the GPU test
// cross-checks a small grid against `gatherCell`.
//
// Authored in raw WGSL rather than `"use gpu"` TGSL (a deliberate, documented deviation from
// ADR-0003): the two-bite recombination needs dynamic vector/array indexing (`pf[g1]`,
// `offspring[r]`) that is clumsy to express through the transpiler, and the original is already
// WGSL. `pf` is a `var` so the runtime index is legal.
//
// State layout: `pfsa` is a flat f32 array of 5 layers × width × height, layer-major
// (`layer*n + cell`). Layers 0..3 are the genotype vector [--, -+, +-, ++]; layer 4 is LD `r`.
// Ocean/missing cells carry a negative sentinel in every layer.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import { writeView } from "../../device";
import { DEFAULT_FITNESS, type FitnessMatrix, ld, OFFSPRING, type Vec4 } from "./math";
import type { Neighbourhood } from "./neighbourhood";

const WGSL = /* wgsl */ `
struct Params {
  w: u32, h: u32, nbhdCount: u32, numBarriers: u32,
  twoBiteRate: f32, pad0: u32, pad1: u32, pad2: u32,
};
struct Fitness { A: vec4<f32>, S: vec4<f32> };
struct NbhdPoint { dx: f32, dy: f32, weight: f32 };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> fitness: Fitness;
@group(0) @binding(2) var<uniform> offspring: array<vec4<f32>, 16>;
@group(0) @binding(3) var<storage, read> nbhd: array<NbhdPoint>;
@group(0) @binding(4) var<storage, read> hbs: array<f32>;
@group(0) @binding(5) var<storage, read> weights: array<f32>;
@group(0) @binding(6) var<uniform> barriers: array<vec4<f32>, 16>;
@group(1) @binding(0) var<storage, read> pfsaIn: array<f32>;
@group(1) @binding(1) var<storage, read_write> pfsaOut: array<f32>;

fn between(x: f32, a: f32, b: f32) -> bool { return sign(x - a) != sign(x - b); }
fn segments_overlap(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>, q: vec2<f32>) -> bool {
  let ma = (b.y - a.y) / (b.x - a.x); let ca = a.y - a.x * ma;
  let mp = (q.y - p.y) / (q.x - p.x); let cp = p.y - p.x * mp;
  let xs = (cp - ca) / (ma - mp);
  return between(xs, a.x, b.x) && between(xs, p.x, q.x);
}

@compute @workgroup_size(64)
fn hspfStep(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = params.w * params.h;
  if (i >= n) { return; }
  let col = i % params.w;
  let row = i / params.w;

  let fs = hbs[i];
  if (fs < 0.0) {
    pfsaOut[0u * n + i] = fs; pfsaOut[1u * n + i] = fs; pfsaOut[2u * n + i] = fs;
    pfsaOut[3u * n + i] = fs; pfsaOut[4u * n + i] = fs;
    return;
  }

  let s = fs * fs + 2.0 * fs * (1.0 - fs);
  let a = 1.0 - s;
  let fit = a * fitness.A + s * fitness.S;
  let tbr = params.twoBiteRate;

  var acc = vec4<f32>(0.0);
  var denom = 0.0;
  for (var k: u32 = 0u; k < params.nbhdCount; k = k + 1u) {
    let np = nbhd[k];
    let bx = i32(col) + i32(np.dx);
    let by = i32(row) + i32(np.dy);
    if (bx < 0 || bx >= i32(params.w) || by < 0 || by >= i32(params.h)) { continue; }
    let bidx = u32(by) * params.w + u32(bx);
    if (hbs[bidx] < 0.0) { continue; }

    var weight = np.weight * weights[bidx];
    for (var j: u32 = 0u; j < params.numBarriers; j = j + 1u) {
      if (segments_overlap(barriers[j].xy, barriers[j].zw,
                           vec2<f32>(f32(col), f32(row)), vec2<f32>(f32(bx), f32(by)))) {
        weight = weight * 0.1;
      }
    }

    var pf = vec4<f32>(pfsaIn[0u * n + bidx], pfsaIn[1u * n + bidx],
                       pfsaIn[2u * n + bidx], pfsaIn[3u * n + bidx]);
    let single = pf * fit;
    // Two-bite recombination: sum_{g1,g2} pf[g1]*pf[g2]*offspring[g1*4+g2]. twoBiteRate (tbr) is a
    // uniform, so this branch is uniform control flow — skip the 16-iteration term entirely when
    // the two-bite rate is 0 (the default), a ~16x saving on the hot per-bite path.
    var bite = single;
    if (tbr > 0.0) {
      var two = vec4<f32>(0.0);
      for (var r: u32 = 0u; r < 16u; r = r + 1u) {
        let g1 = r / 4u;
        let g2 = r % 4u;
        two = two + (pf[g1] * pf[g2]) * offspring[r];
      }
      bite = (1.0 - tbr) * single + tbr * (two * fit);
    }
    acc = acc + weight * bite;
    denom = denom + weight * (bite.x + bite.y + bite.z + bite.w);
  }

  var value = vec4<f32>(0.0);
  if (denom > 0.0) { value = acc / denom; }
  pfsaOut[0u * n + i] = value.x; pfsaOut[1u * n + i] = value.y;
  pfsaOut[2u * n + i] = value.z; pfsaOut[3u * n + i] = value.w;

  let f1_ = value.z + value.w;
  let f_1 = value.y + value.w;
  let dd = value.w - f1_ * f_1;
  let den = sqrt(f1_ * (1.0 - f1_) * f_1 * (1.0 - f_1));
  var rr = 0.0;
  if (den > 0.0) { rr = clamp(dd / den, -1.0, 1.0); }
  pfsaOut[4u * n + i] = rr;
}
`;

const NUM_LAYERS = 5;
const MAX_BARRIERS = 16;

/** The fixed background a run is built on. `hbs`/`weights` are row-major width×height; ocean or
 *  missing cells carry a negative sentinel in `hbs`. */
export interface HspfScaffold {
  hbs: Float32Array;
  weights: Float32Array;
  width: number;
  height: number;
}

export interface HspfParams {
  /** Fraction of transmission that is two-bite recombination, in [0, 1]. */
  twoBiteRate?: number;
  /** Fitness matrix (defaults to the original's). */
  fitness?: FitnessMatrix;
  /** Barrier segments as [x0, y0, x1, y1] in grid coordinates (≤ 16; dormant/empty in phase 1). */
  barriers?: ReadonlyArray<readonly [number, number, number, number]>;
}

interface Pipe {
  device: GPUDevice;
  root: ReturnType<typeof tgpu.initFromDevice>;
  pipeline: GPUComputePipeline;
  bg0Layout: GPUBindGroupLayout;
  bg1Layout: GPUBindGroupLayout;
}
let pipeCache: Promise<Pipe> | undefined;

async function getPipe(device: GPUDevice): Promise<Pipe> {
  pipeCache ??= (async () => {
    const root = tgpu.initFromDevice({ device });
    const module = device.createShaderModule({ label: "hspfStep", code: WGSL });
    const ro = { type: "read-only-storage" } as const;
    const st = { type: "storage" } as const;
    const un = { type: "uniform" } as const;
    const bg0Layout = device.createBindGroupLayout({
      entries: [0, 1, 2, 3, 4, 5, 6].map((binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: binding === 3 || binding === 4 || binding === 5 ? ro.type : un.type },
      })),
    });
    const bg1Layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: ro },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: st },
      ],
    });
    const pipeline = device.createComputePipeline({
      label: "hspf",
      layout: device.createPipelineLayout({ bindGroupLayouts: [bg0Layout, bg1Layout] }),
      compute: { module, entryPoint: "hspfStep" },
    });
    return { device, root, pipeline, bg0Layout, bg1Layout };
  })();
  return pipeCache;
}

function packParams(width: number, height: number, nbhdCount: number, numBarriers: number, tbr: number): ArrayBuffer {
  const buf = new ArrayBuffer(32);
  const dv = new DataView(buf);
  dv.setUint32(0, width, true);
  dv.setUint32(4, height, true);
  dv.setUint32(8, nbhdCount, true);
  dv.setUint32(12, numBarriers, true);
  dv.setFloat32(16, tbr, true);
  return buf;
}

function packVec4Array(rows: ReadonlyArray<Vec4 | readonly number[]>, count: number): Float32Array {
  const out = new Float32Array(count * 4);
  for (let r = 0; r < count; r++) {
    const row = rows[r];
    if (!row) continue;
    out[r * 4 + 0] = row[0] ?? 0;
    out[r * 4 + 1] = row[1] ?? 0;
    out[r * 4 + 2] = row[2] ?? 0;
    out[r * 4 + 3] = row[3] ?? 0;
  }
  return out;
}

/** Build the initial `pfsa` (5 layers × n) from a single starting genotype vector, copying the
 *  scaffold's sentinel into every layer of ocean/missing cells and seeding LD from the start
 *  vector. Mirrors the original's `resetPfsa`. */
export function seedPfsa(scaffold: HspfScaffold, start: Vec4): Float32Array {
  const n = scaffold.width * scaffold.height;
  const out = new Float32Array(NUM_LAYERS * n);
  const r0 = ld(start);
  for (let i = 0; i < n; i++) {
    const fs = scaffold.hbs[i] ?? -2;
    if (fs < 0) {
      for (let l = 0; l < NUM_LAYERS; l++) out[l * n + i] = fs;
    } else {
      for (let g = 0; g < 4; g++) out[g * n + i] = start[g] ?? 0;
      out[4 * n + i] = r0;
    }
  }
  return out;
}

type Root = Pipe["root"];
type StateBuffer = ReturnType<Root["createBuffer"]>;

function clampBarriers(params: HspfParams): { list: ReadonlyArray<readonly [number, number, number, number]>; numBarriers: number } {
  const list = params.barriers ?? [];
  return { list, numBarriers: Math.min(list.length, MAX_BARRIERS) };
}

/** A stateful HsPf simulation whose genotype-frequency field stays resident on the GPU and advances
 *  without reading back — the form the browser render loop wants (its state buffer is handed
 *  straight to the renderer). `hspfStepsGpu` wraps this for one-shot, read-back use (tests). */
export class HspfSim {
  /** Cells per layer (width×height); layer `c` of the state buffer starts at index `c*n`. */
  readonly n: number;
  iteration = 0;

  private nbhdBuf: GPUBuffer;
  private nbhdCount: number;
  private numBarriers: number;
  private bg0: GPUBindGroup;
  private src: StateBuffer;
  private dst: StateBuffer;

  private constructor(
    readonly width: number,
    readonly height: number,
    private readonly device: GPUDevice,
    private readonly root: Root,
    private readonly pipeline: GPUComputePipeline,
    private readonly bg0Layout: GPUBindGroupLayout,
    private readonly bg1Layout: GPUBindGroupLayout,
    private readonly paramsBuf: GPUBuffer,
    private readonly fitnessBuf: GPUBuffer,
    private readonly offspringBuf: GPUBuffer,
    private readonly barriersBuf: GPUBuffer,
    private readonly hbsBuf: GPUBuffer,
    private readonly weightsBuf: GPUBuffer,
    private readonly a: StateBuffer,
    private readonly b: StateBuffer,
    nbhdBuf: GPUBuffer,
    nbhdCount: number,
    numBarriers: number,
  ) {
    this.n = width * height;
    this.nbhdBuf = nbhdBuf;
    this.nbhdCount = nbhdCount;
    this.numBarriers = numBarriers;
    this.src = a;
    this.dst = b;
    this.bg0 = this.buildBg0();
  }

  static async create(
    device: GPUDevice,
    scaffold: HspfScaffold,
    neighbourhood: Neighbourhood,
    pfsa: Float32Array,
    params: HspfParams = {},
  ): Promise<HspfSim> {
    const { width: w, height: h } = scaffold;
    const n = w * h;
    if (scaffold.hbs.length !== n || scaffold.weights.length !== n) throw new Error("hspf: scaffold length != w*h");
    if (pfsa.length !== NUM_LAYERS * n) throw new Error("hspf: pfsa length != 5*w*h");

    const fit = params.fitness ?? DEFAULT_FITNESS;
    const { numBarriers } = clampBarriers(params);
    const tbr = Math.min(1, Math.max(0, params.twoBiteRate ?? 0));
    const { root, pipeline, bg0Layout, bg1Layout } = await getPipe(device);

    const uni = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    const sto = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const mk = (data: ArrayBufferView, use: number) => {
      const buf = device.createBuffer({ size: data.byteLength, usage: use });
      writeView(device.queue, buf, data);
      return buf;
    };

    const paramsBuf = device.createBuffer({ size: 32, usage: uni });
    device.queue.writeBuffer(paramsBuf, 0, packParams(w, h, neighbourhood.count, numBarriers, tbr));
    const fitnessBuf = mk(new Float32Array([...fit.A, ...fit.S]), uni);
    const offspringBuf = mk(packVec4Array(OFFSPRING, 16), uni);
    const barriersBuf = mk(packVec4Array(params.barriers ?? [], MAX_BARRIERS), uni);
    const nbhdBuf = mk(neighbourhood.data, sto);
    const hbsBuf = mk(scaffold.hbs, sto);
    const weightsBuf = mk(scaffold.weights, sto);
    // State (ping-pong) buffers are tgpu-managed so readback uses tgpu's `.read()` — a raw staging
    // `mapAsync` here trips Dawn-on-Node's exit-teardown segfault (see splatDensity notes).
    const a = root.createBuffer(d.arrayOf(d.f32, NUM_LAYERS * n)).$usage("storage");
    const b = root.createBuffer(d.arrayOf(d.f32, NUM_LAYERS * n)).$usage("storage");
    writeView(device.queue, root.unwrap(a), pfsa);

    return new HspfSim(
      w,
      h,
      device,
      root,
      pipeline,
      bg0Layout,
      bg1Layout,
      paramsBuf,
      fitnessBuf,
      offspringBuf,
      barriersBuf,
      hbsBuf,
      weightsBuf,
      a,
      b,
      nbhdBuf,
      neighbourhood.count,
      numBarriers,
    );
  }

  private buildBg0(): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.bg0Layout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: this.fitnessBuf } },
        { binding: 2, resource: { buffer: this.offspringBuf } },
        { binding: 3, resource: { buffer: this.nbhdBuf } },
        { binding: 4, resource: { buffer: this.hbsBuf } },
        { binding: 5, resource: { buffer: this.weightsBuf } },
        { binding: 6, resource: { buffer: this.barriersBuf } },
      ],
    });
  }

  /** Advance `steps` fused steps on the GPU. No readback — the state stays resident. */
  step(steps = 1): void {
    const groups = Math.ceil(this.n / 64);
    for (let s = 0; s < steps; s++) {
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bg0);
      pass.setBindGroup(
        1,
        this.device.createBindGroup({
          layout: this.bg1Layout,
          entries: [
            { binding: 0, resource: { buffer: this.root.unwrap(this.src) } },
            { binding: 1, resource: { buffer: this.root.unwrap(this.dst) } },
          ],
        }),
      );
      pass.dispatchWorkgroups(groups);
      pass.end();
      this.device.queue.submit([enc.finish()]);
      [this.src, this.dst] = [this.dst, this.src];
      this.iteration++;
    }
  }

  /** The GPUBuffer holding the latest state (5 layers × n, layer-major) — for the renderer. */
  currentStateBuffer(): GPUBuffer {
    return this.root.unwrap(this.src);
  }

  /** Read the latest state back to the CPU (tgpu `.read()`, the Dawn-on-Node-safe path). */
  async readback(): Promise<Float32Array> {
    const read = (await this.src.read()) as ArrayLike<number>;
    return Float32Array.from({ length: NUM_LAYERS * this.n }, (_, i) => read[i] ?? 0);
  }

  /** Reset the field to a fresh `pfsa` (5 layers × n) and restart the iteration count. */
  reset(pfsa: Float32Array): void {
    if (pfsa.length !== NUM_LAYERS * this.n) throw new Error("hspf: pfsa length != 5*w*h");
    writeView(this.device.queue, this.root.unwrap(this.a), pfsa);
    this.src = this.a;
    this.dst = this.b;
    this.iteration = 0;
  }

  /** Update the fitness matrix, two-bite rate, and barriers (uniform rewrites; no realloc). */
  setParams(params: HspfParams): void {
    const fit = params.fitness ?? DEFAULT_FITNESS;
    const { numBarriers } = clampBarriers(params);
    const tbr = Math.min(1, Math.max(0, params.twoBiteRate ?? 0));
    this.numBarriers = numBarriers;
    writeView(this.device.queue, this.fitnessBuf, new Float32Array([...fit.A, ...fit.S]));
    writeView(this.device.queue, this.barriersBuf, packVec4Array(params.barriers ?? [], MAX_BARRIERS));
    this.device.queue.writeBuffer(this.paramsBuf, 0, packParams(this.width, this.height, this.nbhdCount, numBarriers, tbr));
  }

  /** Replace the neighbourhood (e.g. after a spread-param change). Grows the buffer if needed and
   *  rebuilds the background bind group. Preserve `twoBiteRate` by passing it in `params`. */
  setNeighbourhood(neighbourhood: Neighbourhood, params: HspfParams = {}): void {
    this.nbhdBuf = this.device.createBuffer({
      size: neighbourhood.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    writeView(this.device.queue, this.nbhdBuf, neighbourhood.data);
    this.nbhdCount = neighbourhood.count;
    const tbr = Math.min(1, Math.max(0, params.twoBiteRate ?? 0));
    this.device.queue.writeBuffer(this.paramsBuf, 0, packParams(this.width, this.height, this.nbhdCount, this.numBarriers, tbr));
    this.bg0 = this.buildBg0();
  }
}

/** Advance an HsPf `pfsa` state by `steps` fused steps on the GPU and read the result back. Thin
 *  one-shot wrapper over `HspfSim` for tests / batch use; the final `pfsa` (5 layers × n) returns. */
export async function hspfStepsGpu(
  device: GPUDevice,
  scaffold: HspfScaffold,
  neighbourhood: Neighbourhood,
  pfsa: Float32Array,
  steps: number,
  params: HspfParams = {},
): Promise<Float32Array> {
  if (steps < 1) return pfsa;
  const sim = await HspfSim.create(device, scaffold, neighbourhood, pfsa, params);
  sim.step(steps);
  return sim.readback();
}
