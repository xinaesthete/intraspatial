// UMAP's layout SGD on the GPU, over a **resident** embedding.
//
// The host implementation (`src/spatial/umapLayout.ts`) walks the edge list in JS. At
// 100k cells that list is ~1.5M directed edges and an epoch costs tens of milliseconds
// on the host — enough that the interactive page stops being interactive well before the
// k-NN becomes the bottleneck. Here every edge is a thread and the embedding never
// leaves the device between epochs, so a frame costs one dispatch and (only when
// drawing) one readback of `n * dim` floats.
//
// **This is a stateful handle, not a pure function, and that is the point.** The whole
// animation model is one embedding that keeps being optimised while the graph underneath
// changes (docs/umap-on-anndata.md §4). A `layout(graph) -> coords` signature would have
// to upload and download every frame, which is exactly the cost this exists to remove.
//
// **On races.** Threads for two edges sharing an endpoint read-modify-write the same
// position with no synchronisation. That is deliberate — it is the Hogwild! regime every
// GPU UMAP uses (cuML included), and the alternative (atomics on fixed-point positions)
// costs more than the noise is worth when the objective is stochastic anyway. Two
// consequences, both real:
//
//   • Output is NOT reproducible run to run, even at a fixed seed, and is NOT comparable
//     elementwise with the host implementation. Tests assert `trustworthiness` — the
//     structure preserved — not coordinates. See `umapLayoutGpu.gpu.test.ts`.
//   • The offline `obsm` path deliberately keeps using the host SGD, because a written
//     `obsm` should be reproducible. The GPU path is for the interactive view.
//
// The per-edge sampling schedule is NOT raced: each thread owns its own edge's slot.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import type { AbParams, LayoutOptions } from "../../spatial/umapLayout";
import { fitAB, makeEpochsPerSample, mulberry32 } from "../../spatial/umapLayout";
import { getDevice } from "../device";

const WG = 64;
/** Embedding dimensions supported; the inner loops are bounded by this. */
const MAX_DIM = 3;
/** Ceiling on negative samples drawn for one edge in one epoch. Without it a long-idle
 *  edge (one whose schedule fell far behind) would spin for thousands of iterations in a
 *  single thread and stall the whole dispatch. */
const MAX_NEG = 16;
/** Matches `GRAD_CLIP` in the host implementation — the two must agree or the layouts
 *  drift apart in a way that has nothing to do with the races. */
const GRAD_CLIP = 4;

const Params = d.struct({
  n: d.u32,
  nEdges: d.u32,
  dim: d.u32,
  epoch: d.f32,
  alpha: d.f32,
  a: d.f32,
  b: d.f32,
  gamma: d.f32,
  seed: d.u32,
  _pad: d.u32,
});

const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  emb: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" }, // n * dim
  head: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
  tail: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
  epochsPerSample: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  epochsPerNeg: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  nextSample: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
  nextNeg: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
});

// WGSL template rather than `"use gpu"`: the kernel needs an integer hash RNG and
// unbounded-ish control flow, which this TypeGPU version does not express in TGSL
// (ADR-0003, same reason as `knn.ts` and `kthNeighborDistance.ts`).
const TEMPLATE = /* wgsl */ `
// murmur-style integer finaliser — cheap, well-distributed, and reproducible per
// (edge, epoch, draw) so a given seed gives the same NEGATIVE SAMPLES even though the
// position updates themselves race.
fn hashU32(x: u32) -> u32 {
  var h = x;
  h = h ^ (h >> 16u);
  h = h * 0x7feb352du;
  h = h ^ (h >> 15u);
  h = h * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return h;
}

fn clipGrad(v: f32) -> f32 {
  return clamp(v, -${GRAD_CLIP}.0, ${GRAD_CLIP}.0);
}

@compute @workgroup_size(${WG})
fn layoutStep(@builtin(global_invocation_id) gid: vec3u) {
  let e = gid.x;
  if (e >= params.nEdges) { return; }
  if (nextSample[e] > params.epoch) { return; }

  let dim = params.dim;
  let j = head[e];
  let k = tail[e];
  let jb = j * dim;
  let kb = k * dim;
  let alpha = params.alpha;

  // --- attractive term, along the edge -------------------------------------------
  var d2: f32 = 0.0;
  for (var c: u32 = 0u; c < dim; c = c + 1u) {
    let delta = emb[jb + c] - emb[kb + c];
    d2 = d2 + delta * delta;
  }

  var coeff: f32 = 0.0;
  if (d2 > 0.0) {
    coeff = (-2.0 * params.a * params.b * pow(d2, params.b - 1.0)) / (params.a * pow(d2, params.b) + 1.0);
  }
  for (var c: u32 = 0u; c < dim; c = c + 1u) {
    let g = clipGrad(coeff * (emb[jb + c] - emb[kb + c])) * alpha;
    // Hogwild: unsynchronised read-modify-write, by design (see the module header).
    emb[jb + c] = emb[jb + c] + g;
    emb[kb + c] = emb[kb + c] - g;
  }

  nextSample[e] = nextSample[e] + epochsPerSample[e];

  // --- repulsive term, against uniformly sampled non-neighbours -------------------
  let epn = epochsPerNeg[e];
  var nNeg: i32 = 0;
  if (epn > 0.0) {
    nNeg = i32(floor((params.epoch - nextNeg[e]) / epn));
  }
  if (nNeg > ${MAX_NEG}) { nNeg = ${MAX_NEG}; }

  var rng = hashU32(params.seed ^ (e * 0x9e3779b9u) ^ (u32(params.epoch) * 0x85ebca6bu));
  for (var p: i32 = 0; p < nNeg; p = p + 1) {
    rng = hashU32(rng);
    let other = rng % params.n;
    if (other == j) { continue; }
    let ob = other * dim;

    var nd2: f32 = 0.0;
    for (var c: u32 = 0u; c < dim; c = c + 1u) {
      let delta = emb[jb + c] - emb[ob + c];
      nd2 = nd2 + delta * delta;
    }

    var ncoeff: f32 = 0.0;
    if (nd2 > 0.0) {
      ncoeff = (2.0 * params.gamma * params.b) / ((0.001 + nd2) * (params.a * pow(nd2, params.b) + 1.0));
    }
    for (var c: u32 = 0u; c < dim; c = c + 1u) {
      // A coincident pair has no direction to separate along; nudge it by the clip
      // amount rather than dividing by zero — matching the host implementation.
      var g: f32 = ${GRAD_CLIP}.0;
      if (ncoeff > 0.0) { g = clipGrad(ncoeff * (emb[jb + c] - emb[ob + c])); }
      emb[jb + c] = emb[jb + c] + g * alpha;
    }
  }
  if (nNeg > 0) {
    nextNeg[e] = nextNeg[e] + f32(nNeg) * epn;
  }
}
`;

interface Pipe {
  device: GPUDevice;
  root: ReturnType<typeof tgpu.initFromDevice>;
  pipeline: GPUComputePipeline;
}
let pipeCache: Promise<Pipe> | undefined;
function getPipe(): Promise<Pipe> {
  pipeCache ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext({
      template: TEMPLATE,
      externals: { ...layout.bound },
      names: "strict",
    });
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
    const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "layoutStep" } });
    return { device, root, pipeline };
  })();
  return pipeCache;
}

export interface GpuLayoutGraph {
  readonly n: number;
  readonly head: Uint32Array;
  readonly tail: Uint32Array;
  readonly weight: Float32Array;
  readonly nEdges: number;
}

/**
 * A live GPU layout: buffers stay allocated and the embedding stays on the device
 * between calls.
 *
 * Not pooled module-side, unlike the other primitives here, because there can
 * legitimately be more than one alive (two panels, or a cross-fade between parameter
 * settings) and its lifetime is owned by the caller. Call `destroy()` when done.
 */
export class GpuUmapLayout {
  private constructor(
    private readonly pipe: Pipe,
    private readonly buffers: ReturnType<typeof makeBuffers>,
    readonly n: number,
    readonly dim: number,
    readonly nEdges: number,
    private readonly ab: AbParams,
    private readonly opts: Required<Pick<LayoutOptions, "nEpochs" | "negativeSampleRate" | "repulsionStrength" | "initialAlpha" | "seed">>,
    /** The per-edge periods, kept host-side so `reheat` can re-base the schedule
     *  without reading the device back. `nEdges` floats — trivial next to the graph. */
    private readonly epochsPerSample: Float32Array,
    private readonly epochsPerNeg: Float32Array,
  ) {}

  /** Epochs completed. Mirrors `LayoutState.epoch` so the two paths read alike. */
  epoch = 0;

  /**
   * Upload a graph and (optionally) an existing embedding.
   *
   * Passing `embedding` is the continuation path — the same one `initLayout` offers on
   * the host, and the reason a gene-subset change relaxes instead of restarting.
   */
  static async create(
    graph: GpuLayoutGraph,
    opts: LayoutOptions & { minDist?: number; spread?: number } = {},
    embedding?: Float32Array,
  ): Promise<GpuUmapLayout> {
    const dim = opts.dim ?? 2;
    if (dim < 1 || dim > MAX_DIM) throw new Error(`umapLayoutGpu: dim must be 1..${MAX_DIM}`);
    if (graph.nEdges === 0) throw new Error("umapLayoutGpu: graph has no edges");

    const nEpochs = opts.nEpochs ?? 200;
    const negativeSampleRate = opts.negativeSampleRate ?? 5;
    const seed = opts.seed ?? 42;

    let emb = embedding;
    if (!emb) {
      const rnd = mulberry32(seed);
      emb = new Float32Array(graph.n * dim);
      for (let t = 0; t < emb.length; t++) emb[t] = rnd() * 20 - 10;
    } else if (emb.length !== graph.n * dim) {
      throw new Error(`umapLayoutGpu: embedding has ${emb.length} entries, expected ${graph.n * dim}`);
    }

    // f32 from the start: these live on the device, and `reheat` must re-derive exactly
    // the values the device holds. `Infinity` (an edge too weak to ever be sampled)
    // survives the narrowing, and `nextSample[e] > epoch` then rejects it forever — the
    // intended behaviour, matching the host path.
    const epochsPerSample = Float32Array.from(makeEpochsPerSample(graph.weight, nEpochs));
    const epochsPerNeg = Float32Array.from(epochsPerSample, (v) => v / negativeSampleRate);

    const pipe = await getPipe();
    const buffers = makeBuffers(pipe.root, graph.n * dim, graph.nEdges);
    const q = pipe.device.queue;
    const unwrap = pipe.root.unwrap.bind(pipe.root);
    q.writeBuffer(unwrap(buffers.emb), 0, emb as BufferSource);
    q.writeBuffer(unwrap(buffers.head), 0, graph.head as BufferSource);
    q.writeBuffer(unwrap(buffers.tail), 0, graph.tail as BufferSource);
    q.writeBuffer(unwrap(buffers.epochsPerSample), 0, epochsPerSample as BufferSource);
    q.writeBuffer(unwrap(buffers.epochsPerNeg), 0, epochsPerNeg as BufferSource);
    // First due epoch = the period itself, matching `initLayout`.
    q.writeBuffer(unwrap(buffers.nextSample), 0, epochsPerSample as BufferSource);
    q.writeBuffer(unwrap(buffers.nextNeg), 0, epochsPerNeg as BufferSource);

    return new GpuUmapLayout(
      pipe,
      buffers,
      graph.n,
      dim,
      graph.nEdges,
      opts.ab ?? fitAB(opts.minDist ?? 0.1, opts.spread ?? 1),
      {
        nEpochs,
        negativeSampleRate,
        repulsionStrength: opts.repulsionStrength ?? 1,
        initialAlpha: opts.initialAlpha ?? 1,
        seed,
      },
      epochsPerSample,
      epochsPerNeg,
    );
  }

  /** The learning rate this epoch would use — decays linearly to zero over `nEpochs`. */
  alphaAt(epoch: number): number {
    return this.opts.initialAlpha * Math.max(0, 1 - epoch / this.opts.nEpochs);
  }

  /**
   * Run `epochs` optimisation epochs. No readback — the embedding stays on the device.
   *
   * **One submit per epoch, deliberately.** The obvious optimisation — encode every
   * epoch's pass into a single command buffer — is WRONG here, and silently so: the
   * per-epoch uniform (`epoch`, and the decaying `alpha` derived from it) is written with
   * `queue.writeBuffer`, so all of those writes land before the single submit and every
   * pass then runs with the LAST epoch's parameters. The layout still moves and still
   * looks plausible; it just optimises at a constant learning rate against a constant
   * epoch, and trustworthiness collapses to ~0.49 — barely better than random
   * coordinates. Batching would need per-epoch uniform slices with dynamic offsets, and
   * the dispatch dominates the submit at any interesting edge count anyway.
   */
  step(epochs = 1): void {
    const { device, pipeline } = this.pipe;
    const bind = this.pipe.root.unwrap(this.pipe.root.createBindGroup(layout, this.buffers));
    for (let i = 0; i < epochs; i++) {
      const epoch = this.epoch + i;
      this.buffers.params.write({
        n: this.n,
        nEdges: this.nEdges,
        dim: this.dim,
        epoch,
        alpha: this.alphaAt(epoch),
        a: this.ab.a,
        b: this.ab.b,
        gamma: this.opts.repulsionStrength,
        seed: this.opts.seed,
        _pad: 0,
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(this.nEdges / WG));
      pass.end();
      device.queue.submit([enc.finish()]);
    }
    this.epoch += epochs;
  }

  /** Download the current embedding, row-major `[n, dim]`. */
  async read(): Promise<Float32Array> {
    const got = (await this.buffers.emb.read()) as ArrayLike<number>;
    const out = new Float32Array(this.n * this.dim);
    for (let t = 0; t < out.length; t++) out[t] = got[t]!;
    return out;
  }

  /** Re-base the anneal so a settled layout moves again — the GPU counterpart of
   *  `reheatLayout`, and required for the same reason: rewinding the epoch counter alone
   *  leaves every edge's next-sample epoch past the horizon and nothing is ever due. */
  reheat(epoch = 0): void {
    this.epoch = Math.max(0, epoch);
    const q = this.pipe.device.queue;
    const unwrap = this.pipe.root.unwrap.bind(this.pipe.root);
    // Re-based host-side: `nEdges` floats, not worth a kernel.
    const nextSample = Float32Array.from(this.epochsPerSample, (v) => this.epoch + v);
    const nextNeg = Float32Array.from(this.epochsPerNeg, (v) => this.epoch + v);
    q.writeBuffer(unwrap(this.buffers.nextSample), 0, nextSample as BufferSource);
    q.writeBuffer(unwrap(this.buffers.nextNeg), 0, nextNeg as BufferSource);
  }

  destroy(): void {
    for (const b of Object.values(this.buffers)) (b as { destroy?: () => void }).destroy?.();
  }
}

type Root = Pipe["root"];
function makeBuffers(root: Root, embLen: number, nEdges: number) {
  return {
    params: root.createBuffer(Params).$usage("uniform"),
    emb: root.createBuffer(d.arrayOf(d.f32, embLen)).$usage("storage"),
    head: root.createBuffer(d.arrayOf(d.u32, nEdges)).$usage("storage"),
    tail: root.createBuffer(d.arrayOf(d.u32, nEdges)).$usage("storage"),
    epochsPerSample: root.createBuffer(d.arrayOf(d.f32, nEdges)).$usage("storage"),
    epochsPerNeg: root.createBuffer(d.arrayOf(d.f32, nEdges)).$usage("storage"),
    nextSample: root.createBuffer(d.arrayOf(d.f32, nEdges)).$usage("storage"),
    nextNeg: root.createBuffer(d.arrayOf(d.f32, nEdges)).$usage("storage"),
  };
}
