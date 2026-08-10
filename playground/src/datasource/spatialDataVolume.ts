// SpatialDataVolume — the 3-D counterpart to `spatialDataLoader.ts`. Where that module flattens an
// OME-Zarr image to a single plane (`voxelDims0 = [x, y, 1]`, plane-only TileRenderer), this one
// builds a genuinely volumetric `Multiscale` + `Loader` from a store whose axes include `z`, so the
// existing raymarching volume renderer can consume it.
//
// ADDITIVE BY DESIGN: nothing here is imported by the 2-D image/alignment path, and
// `spatialDataLoader` (spatialdata.html + spatialscene.html) is untouched. Each page is its own
// module graph, so this page's codec registration cannot affect theirs.
//
// DECODER — `openjph-wasm`, not cornerstone. A z-deep chunk is written by the encoder as one
// JPEG2000 component PER Z SLICE (measured: a `[1,1,32,512,512]` chunk is a 97×75 codestream with
// `Csiz=32`). The `@cornerstonejs/codec-openjph` build that zarrextra@0.2.3 ships on npm cannot
// handle independent multi-component data — it keeps component 0 and replicates it — so volumetric
// chunks fail. zarrextra's SOURCE has already been switched to `openjph-wasm` (the from-source
// OpenJPH build with validated multi-component decode) but that isn't published yet, so here we
// register a custom `ImageCodecDecoder` backed by `openjph-wasm` directly. Once a zarrextra with
// that switch ships, this shim can simply be deleted.
// NB this also means NO `enableWorkerChunkDecode()` on this page: the worker bundle has cornerstone
// baked in, so it would bypass the shim. Decode is on the main thread here.
//
// CHUNK SEAM — native 3-D reads. We deliberately do NOT go through zarrextra's viv-compatible
// `PixelSource.getTile`: that seam is 2-D (an (x,y) tile plus a `selection` over the other axes), so
// a 32-deep brick would cost 32 getTile calls that each re-fetch AND re-decode the SAME underlying
// 3-D chunk (measured: 32 identical GETs per brick, no caching, no brick ever completing).
// Instead we reuse the zarrita that `zarrextra` already owns — `openExtraConsolidated` hands back a
// tree of `LazyZarrArray`s whose `.get()` yields a real `zarr.Array` — and call `Array.getChunk()`
// once per brick. One fetch, one decode, the store's native 3-D chunk. No new dependency, and the
// same zarrita instance that has the HTJ2K codec registered.
import {
  type Affine3,
  type ChunkId,
  chunkCounts,
  hostSamples,
  type Loader,
  type Multiscale,
  type Tile,
  type Vec3,
} from "../../../src/datasource";

let decodeReady: Promise<void> | null = null;

/** Register HTJ2K decode for this page against `openjph-wasm`, replacing zarrextra's default
 *  (cornerstone) decoder — see the multi-component note in the file header. Once, per page.
 *
 *  The decode itself runs in OUR OWN worker pool (`htj2kWorkerPool`), not on the main thread and not
 *  in zarrextra's worker (whose bundled cornerstone build cannot do multi-component). Measured, a
 *  level-3 brick is ~45 ms of synchronous wasm; the viewer streams 19–27 per camera move, so on the
 *  main thread that is about a second of blocking in 45 ms lumps. `?maindecode=1` puts it back on
 *  the main thread, so the two are comparable on the same store. */
function ensureVolumeDecode(zx: typeof import("zarrextra")): Promise<void> {
  if (!decodeReady) {
    decodeReady = (async () => {
      const mainThreadDecode = new URLSearchParams(location.search).get("maindecode") === "1";
      const pool = mainThreadDecode ? null : (await import("./htj2kWorkerPool")).decodePool();
      const decodeOnMain = mainThreadDecode ? (await import("openjph-wasm")).decode : null;
      const decode = async (bytes: Uint8Array) =>
        pool ? pool.decode(bytes) : await (decodeOnMain as NonNullable<typeof decodeOnMain>)(bytes);
      /** `openjph-wasm` returns planar, component-major samples — `data[(c*height + y)*width + x]`
       *  — which is already the C order zarr wants for a `[…, z, y, x]` chunk (component == z), so
       *  the buffer passes straight through. We only assert the codestream matches the chunk. */
      const decoder = async (encoded: Uint8Array, meta: { shape: number[] }): Promise<ArrayBufferView> => {
        const img = await decode(encoded);
        const s = meta.shape;
        const n = s.length;
        const x = s[n - 1] ?? 1;
        const y = s[n - 2] ?? 1;
        const c = n >= 3 ? (s[n - 3] ?? 1) : 1;
        // Everything above the trailing (c,y,x) must be singleton, or the flat buffer wouldn't line up.
        const lead = s.slice(0, Math.max(0, n - 3)).reduce((a, b) => a * b, 1);
        if (img.width !== x || img.height !== y || img.components !== c || lead !== 1)
          throw new Error(
            `openjph-wasm decode mismatch: codestream ${img.components}×${img.height}×${img.width} (comps×h×w) vs chunk shape [${s.join(",")}]`,
          );
        return img.data;
      };
      // biome-ignore lint/suspicious/noExplicitAny: the shipped ImageCodecDecoder type is wider than we need
      zx.registerExperimentalHtj2kCodec({ decoder: decoder as any });
    })();
  }
  return decodeReady;
}

/** A decoded zarr chunk (structural — zarrita is zarrextra's dep, not ours, so no value import). */
interface ZarrChunk {
  data: ArrayLike<number>;
  shape: number[];
  stride: number[];
}
/** The subset of `zarr.Array` we use. */
interface ZarrArrayLike {
  readonly shape: number[];
  readonly chunks: number[];
  readonly dtype: string;
  readonly dimensionNames?: readonly string[] | undefined;
  getChunk(coords: number[]): Promise<ZarrChunk>;
}
interface LazyZarrArrayLike {
  get(): Promise<ZarrArrayLike>;
}

export interface SpatialDataVolume {
  /** Cheap metadata only — a 3-D Multiscale (`voxelDims0[2] > 1`). */
  readonly ms: Multiscale;
  /** The impure seam the volume renderer calls; one native 3-D chunk per brick. */
  readonly loader: Loader;
  /** Human label for the HUD. */
  readonly label: string;
  /** Channels available on the `c` axis (the volume renders ONE scalar channel at a time). */
  readonly channelCount: number;
  /** Per-level voxel dimensions `[x, y, z]`, for the HUD / pyramid-depth diagnostics. */
  readonly levelDims: readonly (readonly [number, number, number])[];
  /** Sample a few coarse bricks and suggest a transfer-function window. Percentiles, not min/max:
   *  one hot voxel would otherwise wash the window out, and the default [0,1] shows nothing at all
   *  for data that only occupies the bottom few percent of its dtype range. */
  autoRange(opts?: AutoRangeOpts): Promise<AutoRange>;
}

export interface AutoRangeOpts {
  /** Pyramid level to sample. Default: the coarsest (cheapest, and covers the whole volume). */
  level?: number;
  /** Cap on bricks fetched (spread evenly across the level). Default 12. */
  maxBricks?: number;
  /** Low/high percentiles in [0,1]. Defaults 0.30 / 0.999. */
  loPct?: number;
  hiPct?: number;
}

export interface AutoRange {
  /** Suggested transfer-function window, in the loader's normalised [0,1] sample space. */
  lo: number;
  hi: number;
  /** Diagnostics: how much was actually sampled, and the extremes seen. */
  samples: number;
  min: number;
  max: number;
}

const axisIndex = (labels: readonly string[], axis: string): number => labels.findIndex((l) => l.toLowerCase() === axis);

/** Axis-aligned array→world placement for a volume: the largest PHYSICAL extent maps to `worldSpan`
 *  world units and the box is centred on the origin. `voxelSize` carries anisotropy (z step vs xy
 *  pitch); with the default `[1,1,1]` the volume is treated as isotropic. */
function volumePlacement(
  dims: readonly [number, number, number],
  voxelSize: readonly [number, number, number],
  worldSpan: number,
): Affine3 {
  const ext: Vec3 = [dims[0] * voxelSize[0], dims[1] * voxelSize[1], dims[2] * voxelSize[2]];
  const s = worldSpan / Math.max(ext[0], ext[1], ext[2], 1);
  return {
    origin: [(-ext[0] * s) / 2, (-ext[1] * s) / 2, (-ext[2] * s) / 2],
    axes: [
      [voxelSize[0] * s, 0, 0],
      [0, voxelSize[1] * s, 0],
      [0, 0, voxelSize[2] * s],
    ],
  };
}

export interface SpatialDataVolumeHandle {
  /** Names of the image elements in the store. */
  readonly imageNames: string[];
  /** Build a volumetric source for one element. Throws if it has no `z` axis. */
  volume(elementName: string, opts?: VolumeOpts): Promise<SpatialDataVolume>;
}

export interface VolumeOpts {
  /** Which channel of the `c` axis to render (the volume path is scalar). Default 0. */
  channel?: number;
  /** Assemble bricks ON THIS DEVICE (docs/gpu-resident-loader.md). Given one — it must be the
   *  renderer's own, via `adoptDevice`, or the texture is unusable by the thing that draws it —
   *  `getChunk` returns a texture-resident `Tile` and the two host passes over every sample
   *  (normalise → f32, then quantise → R8) never happen. Absent ⇒ the host path, unchanged. */
  device?: GPUDevice;
  /** World units the largest physical extent maps to. Default 256. */
  worldSpan?: number;
  /** Physical voxel size `[x, y, z]` for anisotropy. Default `[1,1,1]` (isotropic). */
  voxelSize?: readonly [number, number, number];
}

/** Open a SpatialData / OME-Zarr store and list its image elements (metadata only). */
export async function openSpatialDataVolume(url: string): Promise<SpatialDataVolumeHandle> {
  const core = await import("@spatialdata/core");
  const zx = await import("zarrextra");
  await ensureVolumeDecode(zx);
  const sdata = await core.readZarr(url);
  const images = (sdata.images ?? {}) as Record<string, { getStore(): unknown; attrs?: unknown }>;
  const imageNames = Object.keys(images);
  // Consolidated metadata lives ONLY at the store root — an element's own store view is rooted below
  // it and has no `consolidated_metadata`, so open the root once and navigate down to the element.
  const opened = await zx.openExtraConsolidated(url);
  const { tree } = zx.unwrap(opened);
  return {
    imageNames,
    volume: (elementName, opts = {}) => buildVolume(tree, images[elementName], elementName, imageNames, opts),
  };
}

/** The physical voxel size `[x, y, z]` from the OME multiscales level-0 `scale` transform, in the
 *  axes' own units (micrometer here). Without this a volume renders isotropic — wrong proportions
 *  whenever the z step differs from the xy pitch. Returns null when the store records no scale. */
function readVoxelSize(attrs: unknown, xi: number, yi: number, zi: number): [number, number, number] | null {
  const a = attrs as { ome?: { multiscales?: unknown }; multiscales?: unknown } | undefined;
  const multiscales = (a?.ome?.multiscales ?? a?.multiscales) as { datasets?: unknown[] }[] | undefined;
  const datasets = multiscales?.[0]?.datasets as { coordinateTransformations?: unknown[] }[] | undefined;
  // datasets are ordered finest-first by the spec, so [0] is level 0.
  const cts = datasets?.[0]?.coordinateTransformations;
  const scale = Array.isArray(cts)
    ? (cts.find((t) => (t as { type?: string })?.type === "scale") as { scale?: number[] } | undefined)?.scale
    : undefined;
  if (!Array.isArray(scale)) return null;
  const v: [number, number, number] = [Number(scale[xi]), Number(scale[yi]), Number(scale[zi])];
  return v.every((n) => Number.isFinite(n) && n > 0) ? v : null;
}

/** Walk a zarrextra ZarrTree, collecting the lazy array leaves (symbol-keyed metadata is skipped by
 *  `Object.entries`, and a leaf is identified by having a `get()`). */
function collectLazyArrays(node: unknown, out: LazyZarrArrayLike[]): void {
  if (!node || typeof node !== "object") return;
  for (const value of Object.values(node as Record<string, unknown>)) {
    if (!value || typeof value !== "object") continue;
    if (typeof (value as LazyZarrArrayLike).get === "function") out.push(value as LazyZarrArrayLike);
    else collectLazyArrays(value, out);
  }
}

async function buildVolume(
  tree: unknown,
  img: { getStore(): unknown; attrs?: unknown } | undefined,
  elementName: string,
  imageNames: string[],
  opts: VolumeOpts,
): Promise<SpatialDataVolume> {
  if (!img) throw new Error(`SpatialDataVolume: no image '${elementName}'; available: ${imageNames.join(", ") || "(none)"}`);

  // Reuse zarrextra's own zarrita (and its registered HTJ2K codec) rather than adding a dependency:
  // descend the consolidated tree to this element and take its per-level lazy arrays.
  const node = (tree as { images?: Record<string, unknown> } | undefined)?.images?.[elementName];
  if (!node) throw new Error(`SpatialDataVolume: '${elementName}' not found under images/ in the consolidated tree`);
  const lazies: LazyZarrArrayLike[] = [];
  collectLazyArrays(node, lazies);
  if (lazies.length === 0) throw new Error(`SpatialDataVolume: '${elementName}' exposes no zarr arrays`);

  const arrays = await Promise.all(lazies.map((l) => l.get()));
  const first = arrays[0];
  if (!first) throw new Error(`SpatialDataVolume: '${elementName}' has no levels`);

  const labels = first.dimensionNames;
  if (!labels || labels.length !== first.shape.length)
    throw new Error(
      `SpatialDataVolume: '${elementName}' has no dimension_names — cannot tell which axis is z. (OME-Zarr v0.5 / zarr v3 records these.)`,
    );

  const xi = axisIndex(labels, "x");
  const yi = axisIndex(labels, "y");
  const zi = axisIndex(labels, "z");
  const ci = axisIndex(labels, "c");
  if (xi === -1 || yi === -1) throw new Error(`SpatialDataVolume: axes ${JSON.stringify(labels)} lack x/y`);
  if (zi === -1)
    throw new Error(
      `SpatialDataVolume: '${elementName}' has no 'z' axis (axes ${JSON.stringify(labels)}) — it is a 2-D image, use the image viewer.`,
    );

  // Pyramid order: level 0 is the largest. (Tree key order isn't guaranteed to be level order.)
  const levels = [...arrays].sort((a, b) => (b.shape[xi] ?? 0) * (b.shape[yi] ?? 0) - (a.shape[xi] ?? 0) * (a.shape[yi] ?? 0));
  const base = levels[0];
  if (!base) throw new Error(`SpatialDataVolume: '${elementName}' has no levels`);

  const channelCount = ci === -1 ? 1 : (base.shape[ci] ?? 1);
  const channel = Math.max(0, Math.min(opts.channel ?? 0, channelCount - 1));

  const dimsOf = (a: ZarrArrayLike): [number, number, number] => [a.shape[xi] ?? 1, a.shape[yi] ?? 1, a.shape[zi] ?? 1];
  const voxelDims0 = dimsOf(base);
  const levelDims = levels.map(dimsOf);
  // Brick == the store's native chunk, so one getChunk fills one brick exactly.
  const chunkShape: [number, number, number] = [base.chunks[xi] ?? 1, base.chunks[yi] ?? 1, base.chunks[zi] ?? 1];
  const norm = /16/.test(base.dtype) ? 65535 : /32/.test(base.dtype) ? 4294967295 : 255;
  const bits = /16/.test(base.dtype) ? 16 : /32/.test(base.dtype) ? 32 : 8;

  // `Multiscale.chunkShape` is ONE shape for the whole pyramid, but a store may chunk each level
  // differently (commonly the coarse levels are a single chunk). That's fine only while every level's
  // brick grid still matches its real chunk grid — otherwise a brick would straddle store chunks and
  // `getChunk` would silently return the wrong voxels. Fail loudly instead.
  const grid = (extent: number, chunk: number): number => Math.ceil(extent / Math.max(1, chunk));
  levels.forEach((arr, level) => {
    const lvl = levelDims[level];
    if (!lvl) return;
    const mine = [grid(lvl[0], chunkShape[0]), grid(lvl[1], chunkShape[1]), grid(lvl[2], chunkShape[2])];
    const theirs = [grid(lvl[0], arr.chunks[xi] ?? 1), grid(lvl[1], arr.chunks[yi] ?? 1), grid(lvl[2], arr.chunks[zi] ?? 1)];
    if (mine.some((n, i) => n !== theirs[i]))
      throw new Error(
        `SpatialDataVolume: level ${level} chunk grid ${theirs.join("×")} doesn't match the brick grid ${mine.join("×")} ` +
          `(level-0 chunks ${chunkShape.join("×")} vs this level's ${[arr.chunks[xi], arr.chunks[yi], arr.chunks[zi]].join("×")}). ` +
          `Re-encode with a consistent chunk shape across levels.`,
      );
  });

  const voxelSize = opts.voxelSize ?? readVoxelSize(img.attrs, xi, yi, zi) ?? [1, 1, 1];

  const ms: Multiscale = {
    voxelDims0,
    chunkShape,
    levelCount: levels.length,
    levelDims,
    placements: [{ system: "global", worldFromArray: volumePlacement(voxelDims0, voxelSize, opts.worldSpan ?? 256) }],
    element: { kind: "scalar" },
    dtype: "f32",
  };

  /** One brick. `device` decides residency: given one, the samples are assembled into a texture on
   *  it and never touch the host; without one, the host loop below runs and the Tile carries `data`.
   *  Both callers are real — the renderer wants the device path, `autoRange` wants the host one
   *  (it is a percentile over samples, so a readback would be the whole point of the seam undone). */
  async function readChunk(id: ChunkId, device?: GPUDevice): Promise<Tile> {
    const arr = levels[id.level];
    const lvl = levelDims[id.level];
    if (!arr || !lvl) throw new Error(`SpatialDataVolume: no level ${id.level}`);

    // Chunk coordinates in the ARRAY's own dim order. `c` may itself be chunked, so split the
    // requested channel into a chunk coordinate plus an offset inside the chunk.
    const cChunk = ci === -1 ? 1 : (arr.chunks[ci] ?? 1);
    const cOff = ci === -1 ? 0 : channel % cChunk;
    const coords = arr.shape.map((_, d) =>
      d === xi ? id.x : d === yi ? id.y : d === zi ? id.z : d === ci ? Math.floor(channel / cChunk) : 0,
    );

    const chunk = await arr.getChunk(coords);
    const { data, stride } = chunk;
    // Stored chunks are full-size (edge chunks are fill-padded), so clip to the real extent —
    // `chunkArrayBox` places the brick by the same clipped box.
    const ex = Math.max(0, Math.min(chunkShape[0], lvl[0] - id.x * chunkShape[0]));
    const ey = Math.max(0, Math.min(chunkShape[1], lvl[1] - id.y * chunkShape[1]));
    const ez = Math.max(0, Math.min(chunkShape[2], lvl[2] - id.z * chunkShape[2]));
    if (ex === 0 || ey === 0 || ez === 0) throw new Error(`SpatialDataVolume: empty chunk at ${JSON.stringify(id)}`);

    const sx = stride[xi] ?? 1;
    const sy = stride[yi] ?? 1;
    const sz = stride[zi] ?? 1;
    const cBase = ci === -1 ? 0 : (stride[ci] ?? 0) * cOff;

    if (device) {
      // The device path: upload the decoder's own output verbatim and let the GPU do the
      // sub-box gather, the normalise and the half-conversion. The strides below are exactly
      // the ones the host loop used — a stored chunk is full-size even at the border, and a
      // chunked `c` axis offsets the channel — so nothing is repacked before the upload.
      const { assembleTileTexture, uploadPlane } = await import("../../../src/gpu/tiles/assemble");
      const plane = uploadPlane(device, data as unknown as ArrayBufferView);
      const texture = await assembleTileTexture({
        device,
        planes: [plane],
        width: ex,
        height: ey,
        depth: ez,
        bits,
        scale: 1 / norm,
        layout: { offset: cBase, x: sx, y: sy, z: sz },
      });
      // Free the 16 MB staging upload; only the texture survives. Safe HERE because this module
      // is browser-only — destroying a GPU resource mid-process is what segfaults Dawn-on-Node
      // (ADR-0002/0003), which is why the Tier-2 pools never destroy anything.
      plane.destroy();
      return { id, dims: [ex, ey, ez], element: { kind: "scalar" }, dtype: "f32", texture };
    }

    // Data3DTexture layout: x fastest, then y, then z.
    const out = new Float32Array(ex * ey * ez);
    for (let k = 0; k < ez; k++) {
      for (let j = 0; j < ey; j++) {
        const src = cBase + k * sz + j * sy;
        const dst = (k * ey + j) * ex;
        for (let i = 0; i < ex; i++) out[dst + i] = (data[src + i * sx] ?? 0) / norm;
      }
    }
    return { id, dims: [ex, ey, ez], element: { kind: "scalar" }, dtype: "f32", data: out };
  }

  const loader: Loader = { getChunk: (id) => readChunk(id, opts.device) };

  const iso = voxelSize.every((v) => v === 1);
  const vox = iso ? "voxel 1:1:1 (no scale metadata)" : `voxel ${voxelSize.map((v) => v.toPrecision(4)).join(" × ")}`;
  /** Fetch a spread of bricks from one (coarse) level, subsample them, and take percentiles. */
  async function autoRange(o: AutoRangeOpts = {}): Promise<AutoRange> {
    const L = Math.min(Math.max(0, o.level ?? levels.length - 1), levels.length - 1);
    const [nx, ny, nz] = chunkCounts(ms, L);
    const ids: ChunkId[] = [];
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) ids.push({ level: L, x, y, z });
    const maxBricks = Math.max(1, o.maxBricks ?? 12);
    const step = Math.max(1, Math.ceil(ids.length / maxBricks));
    const picked = ids.filter((_, i) => i % step === 0).slice(0, maxBricks);

    const vals: number[] = [];
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (const id of picked) {
      const t = await readChunk(id).catch(() => null); // host path: this reads samples on the CPU
      if (!t) continue;
      // Keep the sort cheap: at most ~40k samples per brick.
      const s = hostSamples(t);
      const stride = Math.max(1, Math.floor(s.length / 40000));
      for (let i = 0; i < s.length; i += stride) {
        const v = s[i] ?? 0;
        vals.push(v);
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (vals.length === 0) return { lo: 0, hi: 1, samples: 0, min: 0, max: 1 };

    vals.sort((a, b) => a - b);
    const at = (p: number): number => vals[Math.min(vals.length - 1, Math.max(0, Math.round(p * (vals.length - 1))))] ?? 0;
    let lo = at(o.loPct ?? 0.3);
    let hi = at(o.hiPct ?? 0.999);
    // Degenerate windows (flat brick, or hi <= lo) would divide by ~0 in the shader.
    if (!(hi > lo)) {
      lo = min;
      hi = max > min ? max : min + 1e-3;
    }
    return { lo, hi, samples: vals.length, min, max };
  }

  const label = `${elementName} · ${voxelDims0[0]}×${voxelDims0[1]}×${voxelDims0[2]} · ${levels.length} levels · c${channel} · brick ${chunkShape.join("×")} · ${vox}`;
  return { ms, loader, label, channelCount, levelDims, autoRange };
}
