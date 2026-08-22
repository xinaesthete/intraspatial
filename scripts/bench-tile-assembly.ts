// Bench: host tile assembly (what the playground does today) vs the device-side assembly pass
// (src/gpu/tiles/assemble.ts). Step 2 of docs/gpu-resident-loader.md — the measurement that says
// whether the CPU-decode fallback is worth building before any GPU codec exists.
//
// Reads REAL chunks straight off a local OME-Zarr store. It does not go through zarrita at all:
// this store's arrays declare a single codec (`experimental.openjph_htj2k`) and no sharding, so a
// chunk file IS the HTJ2K codestream. That is the "escape hatch" the design note describes — and
// exercising it here is also how we learn what breaks when a store *is* sharded.
//
//   pnpm bench:assembly [storeUrl] [arrayPath]
//
// Defaults to the 3DxN volumetric fixture. Decode is openjph-wasm (multi-component: a z-deep chunk
// is one component per z slice), matching playground/src/datasource/spatialDataVolume.ts.

import { readFile, stat } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { getDevice, releaseDevice } from "../src/gpu/device";
import { assembleTile, copyAssembledToTexture, uploadPlane } from "../src/gpu/tiles/assemble";

const STORE = process.argv[2] ?? "/Volumes/CrucialOx9/3DxN/8090_13_Punch1_fused_htj2k.zarr";
const ARRAY = process.argv[3] ?? "images/8090_13_Punch1_fused/0";
/** Chunk coordinates to sample, in the array's own (t, c, z, y, x) order. */
const CHUNKS: number[][] = [
  [0, 0, 0, 2, 3],
  [0, 0, 1, 2, 3],
  [0, 0, 2, 2, 2],
];
const REPEATS = 3;

const median = (xs: number[]): number => [...xs].sort((a, b) => a - b)[xs.length >> 1] ?? 0;
const ms = (n: number): string => `${n.toFixed(1)} ms`;
const mb = (n: number): string => `${(n / 1048576).toFixed(1)} MB`;

interface ArrayMeta {
  shape: number[];
  chunkShape: number[];
  dtype: string;
  codecs: string[];
}

async function readArrayMeta(store: string, arrayPath: string): Promise<ArrayMeta> {
  const root = JSON.parse(await readFile(`${store}/zarr.json`, "utf8")) as {
    consolidated_metadata?: { metadata: Record<string, Record<string, unknown>> };
  };
  const node = root.consolidated_metadata?.metadata?.[arrayPath];
  if (!node) throw new Error(`no consolidated metadata for '${arrayPath}' in ${store}/zarr.json`);
  const grid = node.chunk_grid as { configuration?: { chunk_shape?: number[] } } | undefined;
  return {
    shape: node.shape as number[],
    chunkShape: grid?.configuration?.chunk_shape ?? [],
    dtype: node.data_type as string,
    codecs: (node.codecs as { name: string }[]).map((c) => c.name),
  };
}

/** The host path the playground runs today, for a scalar volumetric brick:
 *  decoded uint16 → normalise into f32 (spatialDataVolume.getChunk) → quantise to R8
 *  (naiveVolumeRenderer.load). Both loops, both on the thread that is also rendering. */
function hostAssemble(src: Uint16Array, norm: number): { f32: Float32Array; r8: Uint8Array; tNorm: number; tQuant: number } {
  const t0 = performance.now();
  const f32 = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) f32[i] = (src[i] ?? 0) / norm;
  const t1 = performance.now();
  const r8 = new Uint8Array(f32.length);
  for (let i = 0; i < f32.length; i++) r8[i] = Math.round((f32[i] ?? 0) * 255);
  const t2 = performance.now();
  return { f32, r8, tNorm: t1 - t0, tQuant: t2 - t1 };
}

async function main(): Promise<void> {
  const meta = await readArrayMeta(STORE, ARRAY);
  const [, , cz, cy, cx] = meta.chunkShape;
  const w = cx ?? 1;
  const h = cy ?? 1;
  const d = cz ?? 1;
  const voxels = w * h * d;
  const norm = /16/.test(meta.dtype) ? 65535 : /32/.test(meta.dtype) ? 4294967295 : 255;
  const bits = /8/.test(meta.dtype) ? 8 : /16/.test(meta.dtype) ? 16 : 32;

  console.log(`store    ${STORE}`);
  console.log(`array    ${ARRAY}  shape ${meta.shape.join("×")}  ${meta.dtype}  codecs [${meta.codecs.join(", ")}]`);
  console.log(
    `chunk    ${meta.chunkShape.join("×")} → brick ${w}×${h}×${d} = ${voxels.toLocaleString()} voxels, ${mb(voxels * 2)} decoded\n`,
  );

  const { decode } = await import("openjph-wasm");
  const device = await getDevice();

  const rows: string[][] = [];
  for (const coords of CHUNKS) {
    const path = `${STORE}/${ARRAY}/c/${coords.join("/")}`;
    const bytes = new Uint8Array(await readFile(path));
    const compressed = (await stat(path)).size;

    const tDec: number[] = [];
    let decoded: Uint16Array | undefined;
    for (let r = 0; r < REPEATS; r++) {
      const t0 = performance.now();
      const img = await decode(bytes);
      tDec.push(performance.now() - t0);
      decoded = img.data as Uint16Array;
    }
    if (!decoded) throw new Error("decode produced nothing");
    if (decoded.length !== voxels) {
      throw new Error(`decode returned ${decoded.length} samples, expected ${voxels} — chunk/codestream mismatch`);
    }

    const tHostNorm: number[] = [];
    const tHostQuant: number[] = [];
    for (let r = 0; r < REPEATS; r++) {
      const { tNorm, tQuant } = hostAssemble(decoded, norm);
      tHostNorm.push(tNorm);
      tHostQuant.push(tQuant);
    }

    // Device path: upload the decoder's own u16 output verbatim, assemble + half-pack on the GPU,
    // copy into the texture the renderer would sample. No host pass over the samples at all.
    const tGpu: number[] = [];
    for (let r = 0; r < REPEATS; r++) {
      const t0 = performance.now();
      const plane = uploadPlane(device, decoded);
      const tile = await assembleTile({ device, planes: [plane], width: w, height: h, depth: d, bits, scale: 1 / norm, out: "f16" });
      const texture = device.createTexture({
        size: { width: w, height: h, depthOrArrayLayers: d },
        dimension: "3d",
        format: "r16float",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
      });
      copyAssembledToTexture(device, tile, texture, { width: w, height: h, depthOrArrayLayers: d });
      await device.queue.onSubmittedWorkDone();
      tGpu.push(performance.now() - t0);
    }

    const host = median(tHostNorm) + median(tHostQuant);
    rows.push([
      `c/${coords.join("/")}`,
      mb(compressed),
      ms(median(tDec)),
      ms(median(tHostNorm)),
      ms(median(tHostQuant)),
      ms(host),
      ms(median(tGpu)),
      `${(host / median(tGpu)).toFixed(1)}×`,
    ]);
  }

  const header = ["chunk", "compressed", "decode", "host:norm", "host:quant", "host total", "gpu total", "speedup"];
  const widths = header.map((hh, i) => Math.max(hh.length, ...rows.map((r) => (r[i] ?? "").length)));
  const line = (cells: string[]): string => cells.map((c, i) => c.padStart(widths[i] ?? 0)).join("  ");
  console.log(line(header));
  console.log(widths.map((n) => "-".repeat(n)).join("  "));
  for (const r of rows) console.log(line(r));

  console.log(`\nhost transient allocations per brick: ${mb(voxels * 4)} f32 + ${mb(voxels)} r8 = ${mb(voxels * 5)}`);
  console.log(`device path allocations per brick:   ${mb(voxels * 2)} upload + ${mb(voxels * 2)} f16 texture`);
  console.log(`host loop iterations per brick:      ${(voxels * 2).toLocaleString()}`);

  // Exactly once, last: a retained Dawn Instance holds a libuv handle open, so without this the
  // process never exits — and since stdout to a pipe is buffered, it looks like the bench produced
  // NO output at all rather than like a hang after the work finished.
  await releaseDevice();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
