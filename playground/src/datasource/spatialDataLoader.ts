// SpatialDataLoader (ADR-0010) — the Milestone-2 `Loader`, backed by *published*
// spatialdata.js. It adapts a SpatialData multiscale **image** element onto our
// Multiscale/Tile model:
//
//   @spatialdata/core  readZarr(url) → sdata.images[name] → the element (+ transforms)
//   zarrextra          loadOmeZarrMultiscalesFromStore(store) → one VivCompatiblePixelSource
//                      per Level, each with getTile({x,y,selection:{c}}) — the chunk seam,
//                      transparently OpenJPH-decoding HTJ2K codestreams.
//
// This is the real-bytes counterpart to the dep-free `SyntheticLoader`: two Loaders behind
// one interface is the proof the seam is real (ADR-0008). Heavy deps live here in the
// playground, never in the engine core (`src/datasource` stays dependency-free).
//
// The image is placed **axis-aligned** (the real sd.js `global` affine is deferred to 1b, per
// ADR-0010). Up to 4 channels are fetched per spatial (level,x,y) and interleaved as RAW planes
// into one `element: vec` Tile; the channel-composite material (tileChannelMaterial.ts) does
// colour/window/blend on the GPU. Channel defaults (colour + contrast + label) are seeded from
// OME `omero.channels`, with auto-contrast for >8-bit (fluorescence).

import type { Affine3, ChunkId, Loader, Multiscale, Tile, Vec3 } from "../../../src/datasource";
import { type ChannelSettings, defaultChannelColors, MAX_CHANNELS } from "./tileChannelMaterial";

// zarrextra's viv-compatible per-level pixel source (structural — avoids a value import here).
interface PixelSource {
  readonly labels: string[];
  readonly tileSize: number;
  readonly shape: number[];
  readonly dtype: string;
  getTile(props: { x: number; y: number; selection: Record<string, number> | number[]; signal?: AbortSignal }): Promise<{
    data: ArrayLike<number>;
    width: number;
    height: number;
  }>;
}

export interface SpatialDataImage {
  /** Cheap metadata only — the Multiscale the Datasource contributes. */
  readonly ms: Multiscale;
  /** The impure seam Resolve/TileRenderer calls. */
  readonly loader: Loader;
  /** Human label for the HUD (element name + level-0 pixel extent). */
  readonly label: string;
  /** Default per-channel render settings (colour + contrast window + visibility), seeded from
   *  OME `omero.channels` where present, else defaults + auto-contrast. Drives the material. */
  readonly channels: ChannelSettings[];
}

/** Parse an OME/omero channel colour (hex `"RRGGBB"` or `"#RRGGBB"`) to [r,g,b] in [0,1]. */
function parseHexColor(hex: unknown): [number, number, number] | null {
  if (typeof hex !== "string") return null;
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m?.[1]) return null;
  const n = Number.parseInt(m[1], 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

let workerReady: Promise<void> | null = null;
/** Enable zarrextra's off-main-thread codec worker pool once — MDV's `ensureChunkWorker` pattern
 *  (`enableWorkerChunkDecode` from `zarrextra/workers`, same `zarrextra@0.2.3`). The worker
 *  self-contains the OpenJPH/HTJ2K codec + wasm (zarrextra's optional `@cornerstonejs/codec-openjph`
 *  behind the `@fideus-labs/*` worker pool) and hooks zarrita's decode path, so `getTile` decodes
 *  HTJ2K in a worker: no main-thread codec registration, no `openjph-wasm` import, and no extra
 *  vite config (the worker URL uses the `import.meta.url` pattern Vite bundles natively). Decode
 *  stays CPU (ADR-0008 §9) but off the main thread, so the UI no longer stalls while tiles decode. */
function ensureWorkerDecode(): Promise<void> {
  if (!workerReady) {
    workerReady = (async () => {
      const zx = await import("zarrextra");
      // Register the codec id so the decode pipeline recognises `experimental.openjph_htj2k`; with
      // `@cornerstonejs/codec-openjph` now installed (a zarrextra optional dep, same as MDV) this
      // needs no custom decoder. Enabling the worker pool then routes the actual decode
      // off-main-thread — registration says *what* the codec is, the worker says *where* it runs.
      zx.registerExperimentalHtj2kCodec();
      const { enableWorkerChunkDecode } = await import("zarrextra/workers");
      enableWorkerChunkDecode();
    })();
  }
  return workerReady;
}

const axisIndex = (labels: string[], axis: string): number => {
  const i = labels.findIndex((l) => l.toLowerCase() === axis);
  if (i === -1) throw new Error(`SpatialDataLoader: image axes ${JSON.stringify(labels)} lack a '${axis}' axis`);
  return i;
};

/** Axis-aligned array→world placement, normalising the largest pixel extent to `worldSpan`
 *  world units and centring on the origin (1a; the real affine arrives in 1b). */
function axisAlignedPlacement(voxelDims0: readonly [number, number, number], worldSpan: number): Affine3 {
  const s = worldSpan / Math.max(voxelDims0[0], voxelDims0[1], 1);
  const origin: Vec3 = [(-voxelDims0[0] * s) / 2, (-voxelDims0[1] * s) / 2, 0];
  return {
    origin,
    axes: [
      [s, 0, 0],
      [0, s, 0],
      [0, 0, s],
    ],
  };
}

export interface SpatialDataHandle {
  /** Names of the image elements in the store — the demo's dropdown source. */
  readonly imageNames: string[];
  /** Build a renderable source for one image element, reusing the already-open store. */
  image(elementName: string, opts?: { worldSpan?: number; maxChannels?: number }): Promise<SpatialDataImage>;
}

/**
 * Open a SpatialData store (metadata only — `readZarr` parses zarr.json/attrs, no chunk bytes)
 * and list its image elements. Reuse the handle to build individual images without re-reading
 * the store. Throws if the URL is not a readable zarr / SpatialData object.
 */
export async function openSpatialData(url: string): Promise<SpatialDataHandle> {
  await ensureWorkerDecode();
  const core = await import("@spatialdata/core");
  const zx = await import("zarrextra");
  const sdata = await core.readZarr(url);
  const images = (sdata.images ?? {}) as Record<string, { getStore(): unknown; attrs?: unknown }>;
  const imageNames = Object.keys(images);
  return {
    imageNames,
    image: (elementName, opts = {}) => buildImage(zx, images[elementName], elementName, imageNames, opts),
  };
}

/** Convenience: open a store and build one image in a single call. */
export async function openSpatialDataImage(
  url: string,
  elementName: string,
  opts: { worldSpan?: number; maxChannels?: number } = {},
): Promise<SpatialDataImage> {
  return (await openSpatialData(url)).image(elementName, opts);
}

async function buildImage(
  zx: typeof import("zarrextra"),
  img: { getStore(): unknown; attrs?: unknown } | undefined,
  elementName: string,
  imageNames: string[],
  opts: { worldSpan?: number; maxChannels?: number },
): Promise<SpatialDataImage> {
  if (!img) {
    const have = imageNames.join(", ") || "(none)";
    throw new Error(`SpatialDataLoader: no image '${elementName}' in store; available: ${have}`);
  }

  const sources = (await zx.loadOmeZarrMultiscalesFromStore(img.getStore() as never)) as unknown as PixelSource[];
  const base = sources[0];
  if (!base) throw new Error(`SpatialDataLoader: image '${elementName}' has no multiscale levels`);
  const labels = base.labels;
  const xi = axisIndex(labels, "x");
  const yi = axisIndex(labels, "y");
  const ci = labels.findIndex((l) => l.toLowerCase() === "c");
  const nChannels = ci === -1 ? 1 : (base.shape[ci] ?? 1);
  // Composite up to MAX_CHANNELS (4, packed into one RGBA tile texture). Raw planes now — the
  // channel material does colour/window/composite on the GPU (ADR-0010 colormap/contrast).
  const lanes = Math.min(nChannels, opts.maxChannels ?? MAX_CHANNELS);
  const element = lanes >= 2 ? ({ kind: "vec", n: lanes as 2 | 3 | 4 } as const) : ({ kind: "scalar" } as const);
  const outLanes = element.kind === "vec" ? element.n : 1;

  const voxelDims0: [number, number, number] = [base.shape[xi] ?? 1, base.shape[yi] ?? 1, 1];
  // Real per-level dims: OME-Zarr pyramids floor-halve, so the downsample isn't exactly 2^L (here
  // L4 x = 11580/723 ≈ 16.017). Carrying the true shapes makes every level map to the same world
  // extent, so cross-level tiles align exactly instead of drifting toward the far edge.
  const levelDims: [number, number, number][] = sources.map((s) => [s.shape[xi] ?? 1, s.shape[yi] ?? 1, 1]);
  const tile = base.tileSize;
  // Normalise decoded integer samples into [0,1] by their dtype range; the contrast window then
  // lives in [0,1] space too.
  const norm = /16/.test(base.dtype) ? 65535 : /32/.test(base.dtype) ? 4294967295 : 255;
  const worldFromArray = axisAlignedPlacement(voxelDims0, opts.worldSpan ?? 256);

  const ms: Multiscale = {
    voxelDims0,
    chunkShape: [tile, tile, 1],
    levelCount: sources.length,
    levelDims,
    worldFromArray,
    element,
    dtype: "f32",
  };

  const readPlane = (src: PixelSource, c: number, x: number, y: number) => src.getTile({ x, y, selection: ci === -1 ? [0] : { c } });

  const loader: Loader = {
    async getChunk(id: ChunkId): Promise<Tile> {
      const src = sources[id.level];
      if (!src) throw new Error(`SpatialDataLoader: no level ${id.level}`);
      // One getTile per channel (each channel is a separate HTJ2K codestream: chunk shape
      // [1,tile,tile]). zarrextra decodes each transparently; we interleave RAW planes into one
      // Tile — the material tints/windows/composites them on the GPU.
      const planes = await Promise.all(Array.from({ length: outLanes }, (_, c) => readPlane(src, c, id.x, id.y)));
      const first = planes[0];
      if (!first) throw new Error("SpatialDataLoader: empty channel fetch");
      const w = first.width;
      const h = first.height;
      const data = new Float32Array(w * h * outLanes);
      for (let c = 0; c < outLanes; c++) {
        const plane = (planes[c] ?? first).data;
        for (let i = 0; i < w * h; i++) data[i * outLanes + c] = (plane[i] ?? 0) / norm;
      }
      return { id, dims: [w, h, 1], element, dtype: "f32", data };
    },
  };

  // Channel defaults: labels + colours + contrast window, seeded from omero where present.
  const omero = (
    img as { attrs?: { omero?: { channels?: Array<{ label?: unknown; color?: unknown; window?: { start?: number; end?: number } }> } } }
  ).attrs?.omero;
  const omeroCh = omero?.channels ?? [];
  const labelStrings = Array.from({ length: lanes }, (_, i) => {
    const l = omeroCh[i]?.label;
    return typeof l === "string" || typeof l === "number" ? String(l) : `ch${i}`;
  });
  const colors = defaultChannelColors(labelStrings);
  // Auto-contrast integer channels wider than uint8 (fluorescence) from the coarsest level, so
  // they aren't near-black at the default full-range window; uint8 (RGB) keeps [0,1].
  const autoLimits = norm > 255 ? await autoContrastFromCoarsest(sources, ms, readPlane, lanes, norm) : null;
  const channels: ChannelSettings[] = labelStrings.map((label, i) => {
    const win = omeroCh[i]?.window;
    const limits: [number, number] =
      win && typeof win.start === "number" && typeof win.end === "number"
        ? [win.start / norm, win.end / norm]
        : (autoLimits?.[i] ?? [0, 1]);
    return {
      label,
      color: parseHexColor(omeroCh[i]?.color) ?? colors[i] ?? [1, 1, 1],
      contrastLimits: limits,
      visible: true,
    };
  });

  return { ms, loader, label: `${elementName} · ${voxelDims0[0]}×${voxelDims0[1]} · ${sources.length} levels`, channels };
}

/** Per-channel [0, max] window computed from the coarsest level's tiles — a cheap viv-style
 *  auto-contrast so 16-bit fluorescence is visible without hand-tuning. */
async function autoContrastFromCoarsest(
  sources: PixelSource[],
  ms: Multiscale,
  readPlane: (src: PixelSource, c: number, x: number, y: number) => Promise<{ data: ArrayLike<number>; width: number; height: number }>,
  lanes: number,
  norm: number,
): Promise<[number, number][]> {
  const level = sources.length - 1;
  const src = sources[level];
  if (!src) return Array.from({ length: lanes }, () => [0, 1] as [number, number]);
  const { chunkCounts } = await import("../../../src/datasource");
  const [nx, ny] = chunkCounts(ms, level);
  const maxima = new Array<number>(lanes).fill(0);
  for (let c = 0; c < lanes; c++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const t = await readPlane(src, c, x, y).catch(() => null);
        if (!t) continue;
        for (let i = 0; i < t.data.length; i++) {
          const v = (t.data[i] ?? 0) / norm;
          if (v > (maxima[c] ?? 0)) maxima[c] = v;
        }
      }
    }
  }
  // A small headroom below the peak avoids one hot pixel washing the window out.
  return maxima.map((m) => [0, Math.max(1e-3, m * 0.9)] as [number, number]);
}
