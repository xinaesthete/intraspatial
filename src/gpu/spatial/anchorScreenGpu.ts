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
// In `"feistel"` mode the partial-mask groups (and a block with no full-mask group) use the host
// twin of the same permutations (`feistelPermuteCpu`, same keys), so one mode is one null.
//
// ## The count is the CPU's, in exact integers
//
// Ranks are scaled by 2, so centred ranks are integers and so are the null and observed dot
// products (×4 the single-unit ones). `tally` accumulates them exactly, as a two-word (64-bit)
// integer: each product is below 2^31 for k ≤ `EXACT_MAX_REGIONS` (46,340), and the sum below
// 2^63. For the full-mask group both ρ share the anchor norm, so the CPU's
// `|ρ_null| ≥ |ρ_obs| − tol` is `|dot_null| ≥ |dot_obs| − 4·tol·‖a‖‖f‖` in doubled units. For an
// integer `dot_null` that is `|dot_null| ≥ T`, `T = ceil(|dot_obs| − 4·tol·‖a‖‖f‖)`, computed on the
// host in f64 (exact up to the CPU comparison's own f64 rounding). So the GPU applies the CPU rule
// at every k: an identity and a near-identity shuffle (one central swap) stay distinguishable.
// Blocks with more than `EXACT_MAX_REGIONS` finite anchors run entirely on the CPU.

import { mulberry32 } from "#src/evo/rng";
import {
  type AnchorBlock,
  type AnchorMaskGroup,
  type AnchorScreenInput,
  type AnchorScreenOptions,
  type AnchorScreenResult,
  ANCHOR_SCREEN_PERMUTATIONS,
  ANCHOR_SCREEN_TIE_TOLERANCE,
  anchorEmpiricalP,
  anchorNullExceedances,
  anchorResultTable,
  anchorSampleSeed,
  anchorSamples,
  checkAnchorPermutations,
  prepareAnchorBlock,
  sliceSample,
  writeAnchorRows,
} from "#src/spatial/anchorScreen";
import { randomPermutation } from "#src/spatial/permute";
import { checkBindingSize, compileShader, deviceScoped, getDevice, MAX_WORKGROUPS_PER_DIM } from "../device";
import { FEISTEL_WGSL, feistelDomain, feistelKeyCpu, feistelPermuteCpu } from "./feistelWgsl";

const PERMUTE_WG = 256;
const TALLY_WG = 64;
/** Multiply-adds per submitted batch, so no single command nears the ~2 s OS watchdog. */
const WORK_PER_BATCH = 2 ** 26;
/** Bytes of shuffled anchors (and as many of uploaded permutations) resident at once. */
const SHUFFLED_BYTES = 32 * 2 ** 20;
/** The smallest finite-anchor count the Feistel shuffle is used for (see the header). */
export const FEISTEL_MIN_REGIONS = 20;
/** The largest finite-anchor count the exact integer tally takes: products of doubled centred ranks
 *  are at most (k−1)², which must stay below 2^31. */
export const EXACT_MAX_REGIONS = 46_340;

const SHADER = /* wgsl */ `
struct Uni { n: u32, m: u32, batch: u32, first: u32, seedBase: u32, halfBits: u32, halfMask: u32, feistel: u32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> anchor: array<i32>;                 // n, 2 x centred ranks
@group(0) @binding(2) var<storage, read> feats: array<i32>;                  // n*m, region-major
@group(0) @binding(3) var<storage, read> thr: array<vec2<u32>>;              // m, (lo, hi) of T
@group(0) @binding(4) var<storage, read> perms: array<u32>;                  // batch*n (host shuffles)
@group(0) @binding(5) var<storage, read_write> shuffled: array<i32>;         // batch*n
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
  // The dot product as a 64-bit two's-complement integer (lo, hi): exact, see the header.
  var lo = 0u;
  var hi = 0u;
  for (var t = 0u; t < U.n; t = t + 1u) {
    let p = shuffled[row + t] * feats[t * U.m + f];
    let next = lo + bitcast<u32>(p);
    hi = hi + select(0u, 1u, next < lo) + select(0u, 0xffffffffu, p < 0);
    lo = next;
  }
  if ((hi >> 31u) != 0u) {
    lo = ~lo + 1u;
    hi = ~hi + select(0u, 1u, lo == 0u);
  }
  let T = thr[f];
  if (hi > T.y || (hi == T.y && lo >= T.x)) { atomicAdd(&exceed[f], 1u); }
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
  /** The CPU path's tie margin on |ρ| (default 1e-12), applied exactly. */
  readonly tieTolerance?: number;
  /** Seeds the host shuffle stream (as the CPU path's `mulberry32(seed)`) and the Feistel keys. */
  readonly seed?: number;
  /** Default `"fisher-yates"`: the CPU path's permutations, so p is the CPU's exactly. */
  readonly shuffle?: AnchorShuffle;
  /** Default: the library's device. */
  readonly device?: GPUDevice;
}

/** Doubled centred ranks of the full-mask group, region-major, and each feature's integer
 *  threshold `T = ceil(|dot_obs| − 4·tol·‖a‖‖f‖)` (doubled units) as (lo, hi) words. */
function fullGroupData(block: AnchorBlock, group: AnchorMaskGroup, tieTolerance: number) {
  const n = block.nAnchor;
  const m = group.features.length;
  const feats = new Int32Array(n * m);
  const thr = new Uint32Array(2 * m);
  for (let f = 0; f < m; f++) {
    let dot = 0; // exact: integer products of doubled ranks, summed below 2^53
    for (let t = 0; t < n; t++) {
      const r2 = 2 * group.ranks[f * n + t]!;
      feats[t * m + f] = r2;
      dot += 2 * block.anchorRanks[t]! * r2;
    }
    const T = Math.max(0, Math.ceil(Math.abs(dot) - 4 * tieTolerance * block.anchorNorm * group.norms[f]!));
    thr[2 * f] = T % 2 ** 32;
    thr[2 * f + 1] = Math.floor(T / 2 ** 32);
  }
  return { anchor2: Int32Array.from(block.anchorRanks, (v) => 2 * v), feats, thr };
}

/** GPU twin of `anchorNullExceedances` driven by `mulberry32(seed)` (as `anchorScreen` drives the CPU
 *  one), or by the Feistel permutations. Partial-mask groups run on the CPU over the same
 *  permutations. */
export async function anchorNullExceedancesGpu(
  block: AnchorBlock,
  permutations: number,
  opts: AnchorNullGpuOptions = {},
): Promise<Int32Array> {
  checkAnchorPermutations(permutations);
  const seed = (opts.seed ?? 0) >>> 0;
  const tieTolerance = opts.tieTolerance ?? ANCHOR_SCREEN_TIE_TOLERANCE;
  const n = block.nAnchor;
  const exceed = new Int32Array(block.rho.length);
  if (permutations === 0 || block.groups.length === 0) return exceed;
  const candidate = block.groups.find((g) => g.full);
  const full = candidate && n <= EXACT_MAX_REGIONS ? candidate : undefined;
  const partial = { ...block, groups: block.groups.filter((g) => g !== full) };
  const feistel = opts.shuffle === "feistel" && n >= FEISTEL_MIN_REGIONS;
  const hostPerms = !feistel || partial.groups.length > 0;
  const m = full?.features.length ?? 0;
  const batchCap = Math.max(
    1,
    Math.min(permutations, Math.floor(WORK_PER_BATCH / (n * Math.max(m, 1))), Math.floor(SHUFFLED_BYTES / (4 * n)), MAX_WORKGROUPS_PER_DIM),
  );

  const rnd = mulberry32(seed);
  const { halfBits, halfMask } = feistelDomain(n);
  const permBatch = new Uint32Array(hostPerms ? batchCap * n : 0);
  /** Fill host permutations `first .. first+batch-1` and score the CPU-side groups on them. */
  const hostBatch = (first: number, batch: number) => {
    for (let s = 0; s < batch; s++) {
      const out = permBatch.subarray(s * n, (s + 1) * n);
      if (!feistel) randomPermutation(n, rnd, out);
      else {
        const key = feistelKeyCpu(seed, first + s);
        for (let t = 0; t < n; t++) out[t] = feistelPermuteCpu(t, key, n, halfBits, halfMask);
      }
    }
    if (partial.groups.length === 0) return;
    let s = 0;
    const counts = anchorNullExceedances(partial, batch, () => permBatch.subarray(n * s, n * ++s), tieTolerance);
    for (const g of partial.groups) for (const j of g.features) exceed[j]! += counts[j]!;
  };
  if (!full) {
    for (let first = 0; first < permutations; first += batchCap) hostBatch(first, Math.min(batchCap, permutations - first));
    return exceed;
  }

  const device = opts.device ?? (await getDevice());
  const ctx = await getCtx(device);
  if (Math.ceil(n / PERMUTE_WG) > MAX_WORKGROUPS_PER_DIM || Math.ceil(m / TALLY_WG) > MAX_WORKGROUPS_PER_DIM) {
    throw new Error(`anchorNullExceedancesGpu: ${n} regions x ${m} features is past the dispatch limit`);
  }
  const { anchor2, feats, thr } = fullGroupData(block, full, tieTolerance);
  const bytes = {
    anchor: 4 * n,
    feats: 4 * n * m,
    thr: 8 * m,
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
    for (let first = 0; first < permutations; first += batchCap) {
      const batch = Math.min(batchCap, permutations - first);
      if (hostPerms) hostBatch(first, batch);
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

/** `anchorScreen` with the full-mask null on the GPU. Same input, options and result; ρ, n_finite
 *  and skip are the CPU's values exactly, and with the default shuffle so is p. */
export async function anchorScreenGpu(
  input: AnchorScreenInput,
  opts: AnchorScreenOptions & { readonly device?: GPUDevice; readonly shuffle?: AnchorShuffle } = {},
): Promise<AnchorScreenResult> {
  const permutations = checkAnchorPermutations(opts.permutations ?? ANCHOR_SCREEN_PERMUTATIONS);
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
      tieTolerance: opts.tieTolerance,
    });
    writeAnchorRows(out, row, key, names, block, anchorEmpiricalP(block.rho, exceed, permutations));
    row += names.length;
  }
  return out;
}
