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
// Slice 1a: `he_image` only, placed **axis-aligned** (the real sd.js `global` affine is
// deferred to 1b, per ADR-0010); RGB carried as an `element: vec3` — the loader fetches the
// three channels of each spatial (level,x,y) and interleaves them into one Tile.

import type { Affine3, ChunkId, Loader, Multiscale, Tile, Vec3 } from "../../../src/datasource";

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
}

let codecReady: Promise<void> | null = null;
/** Register the HTJ2K codec into zarrita's global registry, once. The *published* zarrextra
 *  dynamic-imports `@cornerstonejs/codec-openjph` for its default OpenJPH path; we instead
 *  inject our own decoder built on this project's `openjph-wasm` (ADR-0010: the local codec is
 *  the ecosystem's reference decoder). Decode runs CPU-side on the main thread (ADR-0008 §9);
 *  moving it to `enableWorkerChunkDecode()` is a later optimisation. */
function ensureCodecs(): Promise<void> {
  if (!codecReady) {
    codecReady = (async () => {
      const zx = await import("zarrextra");
      const ojph = await import("openjph-wasm");
      // zarrextra's ImageCodecDecoder: (encoded, meta, config) → decoded bytes. openjph-wasm's
      // decode returns planar component-major samples — exactly the chunk byte order zarrita
      // reshapes. One zarr chunk here is a single channel, so components === 1.
      const decoder = async (encoded: Uint8Array): Promise<ArrayBufferView> => {
        const img = await ojph.decode(encoded);
        return img.data;
      };
      zx.registerExperimentalHtj2kCodec({ decoder });
    })();
  }
  return codecReady;
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

/**
 * Open one image element of a SpatialData store and expose it as a `{ ms, loader }` source.
 * `url` is the store root (a FetchStore URL); `elementName` selects `sdata.images[elementName]`.
 */
export async function openSpatialDataImage(
  url: string,
  elementName: string,
  opts: { worldSpan?: number; maxChannels?: number } = {},
): Promise<SpatialDataImage> {
  await ensureCodecs();
  const core = await import("@spatialdata/core");
  const zx = await import("zarrextra");

  // Metadata only: readZarr parses zarr.json/attrs; no chunk bytes are fetched here.
  const sdata = await core.readZarr(url);
  const img = sdata.images?.[elementName];
  if (!img) {
    const have = Object.keys(sdata.images ?? {}).join(", ") || "(none)";
    throw new Error(`SpatialDataLoader: no image '${elementName}' in store; available: ${have}`);
  }

  const sources = (await zx.loadOmeZarrMultiscalesFromStore(img.getStore())) as unknown as PixelSource[];
  const base = sources[0];
  if (!base) throw new Error(`SpatialDataLoader: image '${elementName}' has no multiscale levels`);
  const labels = base.labels;
  const xi = axisIndex(labels, "x");
  const yi = axisIndex(labels, "y");
  const ci = labels.findIndex((l) => l.toLowerCase() === "c");
  const nChannels = ci === -1 ? 1 : (base.shape[ci] ?? 1);
  const lanes = Math.min(nChannels, opts.maxChannels ?? 3);
  const element = lanes >= 3 ? ({ kind: "vec", n: 3 } as const) : ({ kind: "scalar" } as const);

  const voxelDims0: [number, number, number] = [base.shape[xi] ?? 1, base.shape[yi] ?? 1, 1];
  const tile = base.tileSize;
  // Normalise decoded integer samples into [0,1] by their dtype range. uint8 (H&E) maps
  // cleanly; uint16 (fluorescence) is normalised too, but really wants a windowing/contrast
  // pass — that's the 1b channel-colormap work, not 1a.
  const norm = /16/.test(base.dtype) ? 65535 : /32/.test(base.dtype) ? 4294967295 : 255;
  const worldFromArray = axisAlignedPlacement(voxelDims0, opts.worldSpan ?? 256);

  const ms: Multiscale = {
    voxelDims0,
    chunkShape: [tile, tile, 1],
    levelCount: sources.length,
    worldFromArray,
    element,
    dtype: "f32",
  };

  const loader: Loader = {
    async getChunk(id: ChunkId): Promise<Tile> {
      const src = sources[id.level];
      if (!src) throw new Error(`SpatialDataLoader: no level ${id.level}`);
      // One getTile per channel (each channel is a separate HTJ2K codestream: chunk shape
      // [1,tile,tile]). zarrextra decodes each transparently; we interleave into one Tile.
      const planes = await Promise.all(
        Array.from({ length: lanes }, (_, c) => src.getTile({ x: id.x, y: id.y, selection: ci === -1 ? [0] : { c } })),
      );
      const first = planes[0];
      if (!first) throw new Error("SpatialDataLoader: empty channel fetch");
      const w = first.width;
      const h = first.height;
      const outLanes = element.kind === "vec" ? element.n : 1;
      const data = new Float32Array(w * h * outLanes);
      for (let c = 0; c < outLanes; c++) {
        const plane = (planes[Math.min(c, planes.length - 1)] ?? first).data;
        for (let i = 0; i < w * h; i++) data[i * outLanes + c] = (plane[i] ?? 0) / norm;
      }
      return { id, dims: [w, h, 1], element, dtype: "f32", data };
    },
  };

  return { ms, loader, label: `${elementName} · ${voxelDims0[0]}×${voxelDims0[1]} · ${sources.length} levels` };
}
