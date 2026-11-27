// The anchor screen's shuffle null on the GPU — the GPU twin of `src/spatial/anchorScreen.ts`
// (`docs/anchor-screen.md`).
//
// **Only the null moves.** Observed ρ, n_finite and skip codes come from the f64 CPU
// `prepareAnchorBlock`, so they are the same numbers whichever path runs. What runs here is the
// N·m·k multiply-add of the null for the features whose finite mask is the whole finite-anchor set
// (in the TNBC data, every ST gene and nearly every patch feature). For those the shuffled anchor's
// centred ranks are just the observed ones permuted, so a shuffle is a gather:
//
//   1. `permuteAnchor`, one thread per (shuffle, region): `u_s[t] = a[π_s(t)]`;
//   2. `tally`, one thread per (feature, shuffle): one length-k dot product, and an `atomicAdd` on the
//      feature's exceedance count when it is extreme.
//
// Features with missing values of their own need the shuffled anchor re-ranked on their regions,
// a per-(shuffle, mask) sort; they stay on the CPU (`anchorNullExceedances` on those groups only),
// fed the same permutations.
//
// ## Where π comes from
//
// - **`"fisher-yates"` (default).** The host draws the CPU path's own permutations (mulberry32,
//   `randomPermutation`) and uploads them a batch at a time. That is O(N·k) host work against the
//   O(N·k·m) the GPU does, so it costs 1/m of the job, and it makes the GPU p **equal** to the CPU p,
//   draw for draw, rather than merely close.
// - **`"feistel"`.** π_s computed per thread by the shared Feistel permutation (`feistelWgsl.ts`):
//   no host shuffle and no upload, for large k with few features. Below `FEISTEL_MIN_REGIONS` it
//   falls back to Fisher–Yates, because with ≤ 2-bit Feistel halves the permutation is far from
//   uniform: the χ²/dof of (π(0), π(1)) over 200k keys is 91 at n = 5, 33 at n = 8 and 10 at
//   n = 16, against 1.0–1.2 from n = 20 up. ST ROIs of a resection sample can be 8.
//
// ## Ties are exact, in integers
//
// Ranks are scaled by 2, so centred ranks are integers, and the null and observed dot products are
// integers (×4). For the full-mask group both share the anchor norm, so `|ρ_null| ≥ |ρ_obs|` is
// `|dot_null| ≥ |dot_obs|` and ties are counted with a margin of ½, which is the CPU's 1e-12 tie
// margin restated for integers. f32 holds every value and partial sum exactly while they stay below
// 2^24 (k up to ~370 regions in the worst case, beyond that only in practice), so for the small-k
// tables where tied |ρ| are common (ST ROIs) the count is exactly the CPU rule. For large k the
// rounding is far below the null's spread and ties are rare.

import { mulberry32 } from "#src/evo/rng";
import {
  type AnchorBlock,
  type AnchorScreenInput,
  type AnchorScreenOptions,
  type AnchorScreenResult,
  ANCHOR_SCREEN_PERMUTATIONS,
  anchorEmpiricalP,
  anchorNullExceedances,
  anchorResultTable,
  anchorSampleSeed,
  anchorSamples,
  prepareAnchorBlock,
  sliceSample,
  writeAnchorRows,
} from "#src/spatial/anchorScreen";
import { randomPermutation } from "#src/spatial/permute";
import { checkBindingSize, compileShader, deviceScoped, getDevice, MAX_WORKGROUPS_PER_DIM } from "../device";
import { FEISTEL_WGSL, feistelDomain } from "./feistelWgsl";

const PERMUTE_WG = 256;
const TALLY_WG = 64;
/** Multiply-adds per submitted batch, so no single command nears the ~2 s OS watchdog. */
const WORK_PER_BATCH = 2 ** 26;
/** Bytes of shuffled anchors (and as many of uploaded permutations) resident at once. */
const SHUFFLED_BYTES = 32 * 2 ** 20;
/** The smallest finite-anchor count the Feistel shuffle is used for (see the header). */
export const FEISTEL_MIN_REGIONS = 20;

const SHADER = /* wgsl */ `
struct Uni { n: u32, m: u32, batch: u32, first: u32, seedBase: u32, halfBits: u32, halfMask: u32, feistel: u32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> anchor: array<f32>;                 // n, 2 x centred ranks
@group(0) @binding(2) var<storage, read> feats: array<f32>;                  // n*m, region-major
@group(0) @binding(3) var<storage, read> thr: array<f32>;                    // m
@group(0) @binding(4) var<storage, read> perms: array<u32>;                  // batch*n (host shuffles)
@group(0) @binding(5) var<storage, read_write> shuffled: array<f32>;         // batch*n
@group(0) @binding(6) var<storage, read_write> exceed: array<atomic<u32>>;   // m

${FEISTEL_WGSL}

@compute @workgroup_size(${PERMUTE_WG})
fn permuteAnchor(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  let s = gid.y;
  if (t >= U.n || s >= U.batch) { return; }
  var src = 0u;
  if (U.feistel != 0u) {
    src = feistelPermute(t, feistelKey(U.seedBase, U.first + s), U.n, U.halfBits, U.halfMask);
  } else {
    src = perms[s * U.n + t];
  }
  shuffled[s * U.n + t] = anchor[src];
}

@compute @workgroup_size(${TALLY_WG})
fn tally(@builtin(global_invocation_id) gid: vec3u) {
  let f = gid.x;
  let s = gid.y;
  if (f >= U.m || s >= U.batch) { return; }
  let row = s * U.n;
  var dot = 0.0;
  for (var t = 0u; t < U.n; t = t + 1u) {
    dot = dot + shuffled[row + t] * feats[t * U.m + f];
  }
  if (abs(dot) >= thr[f]) { atomicAdd(&exceed[f], 1u); }
}
`;

const getCtx = deviceScoped(async (device: GPUDevice) => {
  const module = await compileShader(device, SHADER, "anchorScreenGpu");
  const layout = device.createBindGroupLayout({
    label: "anchorScreenGpu",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ...[1, 2, 3, 4].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })),
      ...[5, 6].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ label: "anchorScreenGpu", bindGroupLayouts: [layout] });
  const pipe = (entryPoint: string) =>
    device.createComputePipeline({ label: `anchorScreenGpu:${entryPoint}`, layout: pipelineLayout, compute: { module, entryPoint } });
  return { layout, permuteAnchor: pipe("permuteAnchor"), tally: pipe("tally") };
});

export type AnchorShuffle = "fisher-yates" | "feistel";

export interface AnchorNullGpuOptions {
  /** Seeds the host shuffle stream (as the CPU path's `mulberry32(seed)`) and the Feistel keys. */
  readonly seed?: number;
  /** Default `"fisher-yates"`: the CPU path's permutations, so p is the CPU's exactly. */
  readonly shuffle?: AnchorShuffle;
  /** Default: the library's device. */
  readonly device?: GPUDevice;
}

/** Doubled centred ranks of the full-mask features, region-major, and each one's integer tie
 *  threshold `4·|dot_obs| − ½`. */
function fullGroupData(block: AnchorBlock, ranks: Float64Array, m: number) {
  const n = block.nAnchor;
  const feats = new Float32Array(n * m);
  const thr = new Float32Array(m);
  for (let f = 0; f < m; f++) {
    let dot = 0;
    for (let t = 0; t < n; t++) {
      const r = ranks[f * n + t]!;
      feats[t * m + f] = 2 * r;
      dot += block.anchorRanks[t]! * r;
    }
    thr[f] = 4 * Math.abs(dot) - 0.5;
  }
  return { anchor2: Float32Array.from(block.anchorRanks, (v) => 2 * v), feats, thr };
}

/** GPU twin of `anchorNullExceedances` driven by `mulberry32(seed)`, as `anchorScreen` drives the CPU
 *  one. No tie-tolerance option: ties are counted exactly (see the header). Partial-mask groups run
 *  on the CPU over the same permutations. */
export async function anchorNullExceedancesGpu(
  block: AnchorBlock,
  permutations: number,
  opts: AnchorNullGpuOptions = {},
): Promise<Int32Array> {
  const seed = (opts.seed ?? 0) >>> 0;
  const n = block.nAnchor;
  const full = block.groups.find((g) => g.full);
  const partial = { ...block, groups: block.groups.filter((g) => !g.full) };
  const exceed = new Int32Array(block.rho.length);
  if (permutations <= 0 || block.groups.length === 0) return exceed;
  const feistel = opts.shuffle === "feistel" && n >= FEISTEL_MIN_REGIONS;
  const hostPerms = !feistel || partial.groups.length > 0;
  const m = full?.features.length ?? 0;
  const batchCap = Math.max(
    1,
    Math.min(permutations, Math.floor(WORK_PER_BATCH / (n * Math.max(m, 1))), Math.floor(SHUFFLED_BYTES / (4 * n)), MAX_WORKGROUPS_PER_DIM),
  );

  const rnd = mulberry32(seed);
  const permBatch = new Uint32Array(hostPerms ? batchCap * n : 0);
  /** Fill the next `batch` host permutations and score the partial-mask groups on them. */
  const hostBatch = (batch: number) => {
    for (let s = 0; s < batch; s++) randomPermutation(n, rnd, permBatch.subarray(s * n, (s + 1) * n));
    if (partial.groups.length === 0) return;
    let s = 0;
    const counts = anchorNullExceedances(partial, batch, () => permBatch.subarray(n * s, n * ++s));
    for (const g of partial.groups) for (const j of g.features) exceed[j]! += counts[j]!;
  };
  if (!full) {
    for (let first = 0; first < permutations; first += batchCap) hostBatch(Math.min(batchCap, permutations - first));
    return exceed;
  }

  const device = opts.device ?? (await getDevice());
  const ctx = await getCtx(device);
  if (Math.ceil(n / PERMUTE_WG) > MAX_WORKGROUPS_PER_DIM || Math.ceil(m / TALLY_WG) > MAX_WORKGROUPS_PER_DIM) {
    throw new Error(`anchorNullExceedancesGpu: ${n} regions x ${m} features is past the dispatch limit`);
  }
  const { anchor2, feats, thr } = fullGroupData(block, full.ranks, m);
  const bytes = {
    anchor: 4 * n,
    feats: 4 * n * m,
    thr: 4 * m,
    perms: feistel ? 4 : 4 * n * batchCap,
    shuffled: 4 * n * batchCap,
    exceed: 4 * m,
  };
  checkBindingSize(device, `anchorNullExceedancesGpu: ${n} regions x ${m} features`, bytes.feats);
  checkBindingSize(device, `anchorNullExceedancesGpu: ${batchCap} shuffles x ${n} regions`, bytes.shuffled);

  const made: GPUBuffer[] = [];
  const buffer = (label: string, size: number, usage: number) => {
    const b = device.createBuffer({ label: `anchorScreenGpu:${label}`, size: Math.max(16, Math.ceil(size / 4) * 4), usage });
    made.push(b);
    return b;
  };
  try {
    const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const uni = buffer("uni", 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const anchorBuf = buffer("anchor", bytes.anchor, STORAGE_IN);
    const featBuf = buffer("feats", bytes.feats, STORAGE_IN);
    const thrBuf = buffer("thr", bytes.thr, STORAGE_IN);
    const permBuf = buffer("perms", bytes.perms, STORAGE_IN);
    const shufBuf = buffer("shuffled", bytes.shuffled, GPUBufferUsage.STORAGE);
    const exBuf = buffer("exceed", bytes.exceed, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const rb = buffer("readback", bytes.exceed, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(anchorBuf, 0, anchor2);
    device.queue.writeBuffer(featBuf, 0, feats);
    device.queue.writeBuffer(thrBuf, 0, thr);
    device.queue.writeBuffer(exBuf, 0, new Uint32Array(m));
    // Bound at their exact sizes, never the padded allocation.
    const bound: [GPUBuffer, number][] = [
      [uni, 32],
      [anchorBuf, bytes.anchor],
      [featBuf, bytes.feats],
      [thrBuf, bytes.thr],
      [permBuf, bytes.perms],
      [shufBuf, bytes.shuffled],
      [exBuf, bytes.exceed],
    ];
    const bind = device.createBindGroup({
      label: "anchorScreenGpu",
      layout: ctx.layout,
      entries: bound.map(([buffer, size], binding) => ({ binding, resource: { buffer, size } })),
    });
    const { halfBits, halfMask } = feistelDomain(n);
    for (let first = 0; first < permutations; first += batchCap) {
      const batch = Math.min(batchCap, permutations - first);
      if (hostPerms) hostBatch(batch);
      // Written before this batch's submit and after the previous one's: queue order keeps them apart.
      if (!feistel) device.queue.writeBuffer(permBuf, 0, permBatch.buffer, permBatch.byteOffset, batch * n * 4);
      device.queue.writeBuffer(uni, 0, new Uint32Array([n, m, batch, first, seed, halfBits, halfMask, feistel ? 1 : 0]));
      const enc = device.createCommandEncoder({ label: "anchorScreenGpu" });
      const pass = enc.beginComputePass({ label: "anchorScreenGpu" });
      pass.setBindGroup(0, bind);
      pass.setPipeline(ctx.permuteAnchor);
      pass.dispatchWorkgroups(Math.ceil(n / PERMUTE_WG), batch);
      pass.setPipeline(ctx.tally);
      pass.dispatchWorkgroups(Math.ceil(m / TALLY_WG), batch);
      pass.end();
      device.queue.submit([enc.finish()]);
    }
    const enc = device.createCommandEncoder({ label: "anchorScreenGpu:readback" });
    enc.copyBufferToBuffer(exBuf, 0, rb, 0, bytes.exceed);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const counts = new Uint32Array(rb.getMappedRange(0, bytes.exceed).slice(0));
    rb.unmap();
    full.features.forEach((j, f) => (exceed[j] = counts[f]!));
    return exceed;
  } finally {
    for (const b of made) b.destroy();
  }
}

/** `anchorScreen` with the full-mask null on the GPU. Same input, options (bar `tieTolerance`) and
 *  result; ρ, n_finite and skip are the CPU's values exactly, and with the default shuffle so is p
 *  (up to f32 rounding of near-ties at large k). */
export async function anchorScreenGpu(
  input: AnchorScreenInput,
  opts: Omit<AnchorScreenOptions, "tieTolerance"> & { readonly device?: GPUDevice; readonly shuffle?: AnchorShuffle } = {},
): Promise<AnchorScreenResult> {
  const permutations = opts.permutations ?? ANCHOR_SCREEN_PERMUTATIONS;
  const names = [...input.features.keys()];
  const bySample = anchorSamples(input);
  const out = anchorResultTable(names.length, bySample, permutations);
  let row = 0;
  for (const [key, regions] of bySample) {
    const { anchor, features } = sliceSample(input, regions);
    const block = prepareAnchorBlock(anchor, features, opts.minFinite);
    const exceed = await anchorNullExceedancesGpu(block, permutations, {
      seed: anchorSampleSeed(opts.seed ?? 0, key),
      device: opts.device,
      shuffle: opts.shuffle,
    });
    writeAnchorRows(out, row, key, names, block, anchorEmpiricalP(block.rho, exceed, permutations));
    row += names.length;
  }
  return out;
}
