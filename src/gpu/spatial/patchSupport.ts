// Patch support on the GPU: how many support pixels of a resident mask fall in each rectangle.
// The device half of R4's patch grid (docs/patch-grid.md); `maskCountsInRects` in
// `src/spatial/patchGrid.ts` is the golden and gives the same integers.
//
// Three passes, two of them borrowed:
//   predicate  mask → 0/1 flags (f32: finite and > 0; u32 / u8: non-zero)
//   scan       flags → exclusive prefix P (`encodeScan`, src/gpu/scan/prefixSum.ts)
//   rectSum    rectangle i sums, over its rows y, P[y·W + x1] − P[y·W + x0]
//
// A flat scan over the row-major mask is a summed-area table along x for free: the support in
// row y left of column x is P[y·W + x] − P[y·W]. So a rectangle costs one subtraction per row
// (its height), not one read per pixel, and no second (column) scan is needed — at 1 µm/px a
// 300 µm patch is 300 reads instead of 90,000. P[n] (x1 = W on the last row) is the scan's total.
// u32 offsets wrap past 2^32 support pixels; a difference within one row never does, so the
// counts stay exact (mod 2^32 arithmetic cancels the wrap).
//
// One storage binding holds the whole mask's flags, so `maxStorageBufferBindingSize / 4` pixels
// (33.5M at the default 128 MiB: about 5,800² ) is the ceiling; `checkBindingSize` throws past
// it rather than returning a plausible wrong count. Full-resolution canvases go through the run
// form on the host.
import { checkBindingSize, compileShader, deviceScoped, dispatchGrid, perDevice, sized } from "../device";
import { encodeScan, ensureBuf, getScanCtx, readBack, type ScanCtx } from "../scan/prefixSum";

const WG = 64;

/** How mask words are read: f32 (support = finite and > 0), u32 or u8 (support = non-zero;
 *  u8 packs four pixels per word, little-endian, as an uploaded `Uint8Array`). */
export type SupportMaskEncoding = "f32" | "u32" | "u8";
const ENC: Record<SupportMaskEncoding, number> = { f32: 0, u32: 1, u8: 2 };

const SHADER = /* wgsl */ `
struct Uni {
  width: u32,
  n: u32,
  nRects: u32,
  enc: u32,
  gridX: u32,
  pad0: u32, pad1: u32, pad2: u32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read> total: array<u32>;
@group(0) @binding(5) var<storage, read> rects: array<u32>;
@group(0) @binding(6) var<storage, read_write> counts: array<u32>;

fn idOf(lid: vec3u, wid: vec3u) -> u32 {
  return (wid.x + wid.y * U.gridX) * ${WG}u + lid.x;
}

@compute @workgroup_size(${WG})
fn predicate(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let i = idOf(lid, wid);
  if (i >= U.n) { return; }
  var on = false;
  if (U.enc == ${ENC.u8}u) {
    on = ((mask[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu) != 0u;
  } else if (U.enc == ${ENC.u32}u) {
    on = mask[i] != 0u;
  } else {
    // Finite: the exponent is not all ones (WGSL has no isnan / isinf).
    let bits = mask[i];
    on = (bits & 0x7f800000u) != 0x7f800000u && bitcast<f32>(bits) > 0.0;
  }
  flags[i] = select(0u, 1u, on);
}

// P[k], with P[n] = the total. No early return inside the caller's loop (Dawn/Metal miscompiles
// returns from a helper inside a for loop): select instead.
fn prefix(k: u32) -> u32 {
  var v = total[0];
  if (k < U.n) { v = offsets[k]; }
  return v;
}

@compute @workgroup_size(${WG})
fn rectSum(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let i = idOf(lid, wid);
  if (i >= U.nRects) { return; }
  let x0 = rects[i * 4u];
  let y0 = rects[i * 4u + 1u];
  let x1 = rects[i * 4u + 2u];
  let y1 = rects[i * 4u + 3u];
  var s = 0u;
  for (var y = y0; y < y1; y = y + 1u) {
    let base = y * U.width;
    s = s + (prefix(base + x1) - prefix(base + x0));
  }
  counts[i] = s;
}
`;

interface Ctx {
  scan: ScanCtx;
  predLayout: GPUBindGroupLayout;
  sumLayout: GPUBindGroupLayout;
  predicate: GPUComputePipeline;
  rectSum: GPUComputePipeline;
}

const getCtx = deviceScoped(async (device: GPUDevice): Promise<Ctx> => {
  const scan = await getScanCtx(device);
  const module = await compileShader(device, SHADER, "patchSupport");
  const entry = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type },
  });
  // One layout per entry point, each naming only the bindings that entry uses: a shared layout
  // would need scratch buffers in the unused slots (WebGPU rejects aliasing a writable binding).
  const predLayout = device.createBindGroupLayout({
    label: "patchSupport:predicate",
    entries: [entry(0, "uniform"), entry(1, "read-only-storage"), entry(2, "storage")],
  });
  const sumLayout = device.createBindGroupLayout({
    label: "patchSupport:rectSum",
    entries: [
      entry(0, "uniform"),
      entry(3, "read-only-storage"),
      entry(4, "read-only-storage"),
      entry(5, "read-only-storage"),
      entry(6, "storage"),
    ],
  });
  const mk = (entryPoint: string, layout: GPUBindGroupLayout) =>
    device.createComputePipeline({
      label: `patchSupport:${entryPoint}`,
      layout: device.createPipelineLayout({ label: `patchSupport:${entryPoint}`, bindGroupLayouts: [layout] }),
      compute: { module, entryPoint },
    });
  return { scan, predLayout, sumLayout, predicate: mk("predicate", predLayout), rectSum: mk("rectSum", sumLayout) };
});

/** Words a `width × height` mask occupies on the device. */
export function supportMaskWords(encoding: SupportMaskEncoding, pixels: number): number {
  return encoding === "u8" ? Math.ceil(pixels / 4) : pixels;
}

export interface PatchSupportResidentArgs {
  /** The mask, row-major, at `maskOffset` bytes (default 0). */
  readonly mask: GPUBuffer;
  readonly maskOffset?: number;
  readonly encoding: SupportMaskEncoding;
  readonly width: number;
  readonly height: number;
  /** `nRects × 4` u32 `[x0, y0, x1, y1]`, half-open, inside the canvas (not checked on the device). */
  readonly rects: GPUBuffer;
  readonly rectsOffset?: number;
  readonly nRects: number;
  /** Receives one u32 count per rectangle. */
  readonly counts: GPUBuffer;
  readonly countsOffset?: number;
}

/** Tier-2 form: count support pixels per rectangle, every buffer caller-owned and resident. */
export async function patchSupportResident(a: PatchSupportResidentArgs, device?: GPUDevice): Promise<void> {
  const ctx = await getCtx(device);
  const dev = ctx.scan.device;
  const n = a.width * a.height;
  if (a.nRects === 0) return;
  if (n === 0) throw new Error("patchSupport: empty canvas");
  checkBindingSize(dev, `patchSupport: ${n} pixels`, n * 4);
  const flags = ensureBuf(dev, "patchSupport:flags", n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const enc = dev.createCommandEncoder({ label: "patchSupport" });
  const uniFor = (key: string, items: number) => {
    const grid = dispatchGrid(Math.ceil(items / WG));
    const uni = ensureBuf(dev, `patchSupport:uni:${key}`, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    dev.queue.writeBuffer(uni, 0, new Uint32Array([a.width, n, a.nRects, ENC[a.encoding], grid.x, 0, 0, 0]));
    return { grid, uni };
  };
  const pass = (label: string, pipeline: GPUComputePipeline, bind: GPUBindGroup, grid: { x: number; y: number }) => {
    const p = enc.beginComputePass({ label });
    p.setPipeline(pipeline);
    p.setBindGroup(0, bind);
    p.dispatchWorkgroups(grid.x, grid.y);
    p.end();
  };
  const pred = uniFor("predicate", n);
  const maskBytes = supportMaskWords(a.encoding, n) * 4;
  pass(
    "patchSupport:predicate",
    ctx.predicate,
    dev.createBindGroup({
      label: "patchSupport:predicate",
      layout: ctx.predLayout,
      entries: [
        { binding: 0, resource: sized(pred.uni, 32) },
        { binding: 1, resource: { buffer: a.mask, offset: a.maskOffset ?? 0, size: maskBytes } },
        { binding: 2, resource: sized(flags, n * 4) },
      ],
    }),
    pred.grid,
  );
  const { dst: offsets, totalBuf } = encodeScan(ctx.scan, "u32", flags, n, enc, undefined, "patchSupport:scan");
  const sum = uniFor("rectSum", a.nRects);
  pass(
    "patchSupport:rectSum",
    ctx.rectSum,
    dev.createBindGroup({
      label: "patchSupport:rectSum",
      layout: ctx.sumLayout,
      entries: [
        { binding: 0, resource: sized(sum.uni, 32) },
        { binding: 3, resource: sized(offsets, n * 4) },
        { binding: 4, resource: sized(totalBuf, 4) },
        { binding: 5, resource: { buffer: a.rects, offset: a.rectsOffset ?? 0, size: a.nRects * 16 } },
        { binding: 6, resource: { buffer: a.counts, offset: a.countsOffset ?? 0, size: a.nRects * 4 } },
      ],
    }),
    sum.grid,
  );
  dev.queue.submit([enc.finish()]);
}

/** The host form's tail of calls per device: its mask, rectangle, count and staging buffers are
 *  pooled per device, so a second call uploading while the first awaits would be counted by the
 *  first, and two readbacks would map one staging buffer at once (`serial` in modeHistograms.ts).
 *  The resident form needs none: it writes its pooled scratch and submits without awaiting. */
const hostChain = perDevice((): { chain: Promise<unknown> } => ({ chain: Promise.resolve() }));

function serial<T>(device: GPUDevice, task: () => Promise<T>): Promise<T> {
  const tail = hostChain(device);
  const run = tail.chain.then(task, task);
  // A failed call must not stall the ones queued behind it.
  tail.chain = run.catch(() => undefined);
  return run;
}

/** Host-array form: support pixels of `mask` in each `[x0, y0, x1, y1]` rectangle. A
 *  `Float32Array` mask counts finite values > 0, a `Uint32Array` / `Uint8Array` non-zero ones.
 *  Concurrent calls on one device run one at a time. */
export async function patchSupportGpu(
  mask: Float32Array | Uint32Array | Uint8Array,
  width: number,
  height: number,
  rects: Uint32Array,
  device?: GPUDevice,
): Promise<Uint32Array> {
  const n = width * height;
  if (mask.length !== n) throw new Error(`patchSupport: mask has ${mask.length} values for ${width}x${height}`);
  if (rects.length % 4 !== 0) throw new Error("patchSupport: rects must be n × 4");
  const nRects = rects.length / 4;
  for (let i = 0; i < nRects; i++) {
    const [x0, y0, x1, y1] = [rects[i * 4]!, rects[i * 4 + 1]!, rects[i * 4 + 2]!, rects[i * 4 + 3]!];
    if (x1 > width || y1 > height || x0 > x1 || y0 > y1) throw new Error(`patchSupport: rectangle ${i} leaves the canvas or is inverted`);
  }
  if (nRects === 0) return new Uint32Array(0);
  const encoding: SupportMaskEncoding = mask instanceof Float32Array ? "f32" : mask instanceof Uint8Array ? "u8" : "u32";
  const dev = (await getCtx(device)).scan.device;
  return serial(dev, async () => {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const maskBytes = supportMaskWords(encoding, n) * 4;
    const maskBuf = ensureBuf(dev, "patchSupport:mask", maskBytes, usage);
    // writeBuffer needs a multiple of 4 bytes: a byte mask's 1-3 byte tail goes in its own word.
    const aligned = mask.byteLength & ~3;
    if (aligned > 0) dev.queue.writeBuffer(maskBuf, 0, mask.buffer, mask.byteOffset, aligned);
    if (aligned < mask.byteLength) {
      const tail = new Uint8Array(4);
      tail.set(new Uint8Array(mask.buffer, mask.byteOffset + aligned, mask.byteLength - aligned));
      dev.queue.writeBuffer(maskBuf, aligned, tail);
    }
    const rectBuf = ensureBuf(dev, "patchSupport:rects", nRects * 16, usage);
    dev.queue.writeBuffer(rectBuf, 0, rects.buffer, rects.byteOffset, nRects * 16);
    const counts = ensureBuf(dev, "patchSupport:counts", nRects * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    await patchSupportResident({ mask: maskBuf, encoding, width, height, rects: rectBuf, nRects, counts }, dev);
    return new Uint32Array(await readBack(dev, "patchSupport:staging", counts, 0, nRects * 4));
  });
}
