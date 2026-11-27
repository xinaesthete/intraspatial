// Zonal statistics over a pyramid level, tile by tile: every chunk of `level` that can hold a
// region is loaded (Loader / TileCache), labelled (an uploaded label tile, or polygons rasterised
// per overlap layer), and binned into ONE accumulator. Device memory is one chunk per channel plus
// the rows × bins counts, whatever the slide's size: a 520 Mpx resection at 512² chunks is ~2000
// tiles through the same few buffers.
//
// Passes over the tiles: a range pass when a channel has auto edges, the bin pass, and the exact
// pass when exact quantiles are asked for. A tile is fetched once per pass, so give a `cache`
// big enough for the level (or fixed ranges and no exact pass) to read the slide once.
//
// Every source must share the level's grid (voxel dims and chunk shape): registered rasters do.
// Polygons are in level-0 ARRAY space (x = column, y = row; pixel (i, j) covers [i, i+1) × [j, j+1)),
// and a level-L pixel is inside when its centre — mapped to level 0 the way `chunkArrayBox` maps
// chunks — is.
import { chunkCounts, chunkVoxelExtent, levelVoxelDims } from "#src/datasource/multiscale";
import type { TileCache } from "#src/datasource/tileCache";
import { chunkKey } from "#src/datasource/tileCache";
import type { ChunkId, Loader, Multiscale, Tile } from "#src/datasource/types";
import { elementLanes } from "#src/gpu/graph/handle";
import type { PolygonDomain } from "#src/spatial/domain";
import {
  type ZonalChannelOptions,
  type ZonalExactPlan,
  type ZonalHistogram,
  type ZonalTable,
  polygonLayers,
  polygonRingBuffers,
  zonalExactPlan,
  zonalExactQuantiles,
  zonalTable,
} from "#src/spatial/zonal";
import { perDevice, writeView } from "../device";
import type { ResidentBuffer } from "../graph/handle";
import { BufferPool } from "../graph/pool";
import { denseRasterPlane, type RasterMask } from "./histogram";
import { type LabelPlane, type ZonalChannelPlane, type ZonalTile, ZonalAccumulator, rasterisePolygonsResident } from "./zonal";

/** One pyramid, as the datasource layer serves it. */
export interface TiledRaster {
  readonly ms: Multiscale;
  readonly loader: Loader<Tile>;
  /** This source's own cache: keys are chunk ids, so two sources must never share one. */
  readonly cache?: TileCache<Tile>;
}

export interface ZonalTiledChannel extends ZonalChannelOptions {
  readonly source: TiledRaster;
  /** Supported where ANY of these is non-zero (numpy's `a != 0 | b != 0`; NaN counts). */
  readonly support?: TiledRaster | readonly TiledRaster[];
  /** Column prefix in the table. Default `c<index>`. */
  readonly name?: string;
}

export type ZonalRegions =
  /** Polygon k is region k (label k + 1), in level-0 array coordinates. May overlap. */
  | { readonly kind: "polygons"; readonly polygons: readonly PolygonDomain[] }
  /** A label raster on the same grid: label L is region L − 1. `regions` defaults to the
   *  largest label (found by a pre-scan of the label tiles). */
  | { readonly kind: "labels"; readonly source: TiledRaster; readonly regions?: number };

export interface ZonalTiledOptions {
  readonly level: number;
  readonly regions: ZonalRegions;
  readonly channels: readonly ZonalTiledChannel[];
  /** Bins per row. Default 1024. */
  readonly bins?: number;
  /** Table quantiles. Default `[0.1, 0.25, 0.5, 0.75, 0.9]`. */
  readonly quantiles?: readonly number[];
  /** Run the exact second pass for the table's quantiles (which must then include 0.25, 0.5, 0.75). */
  readonly exact?: boolean;
  /** Called after each tile of each pass. */
  readonly onTile?: (pass: "range" | "bin" | "collect", done: number, total: number) => void;
}

export interface ZonalTiledResult {
  readonly histogram: ZonalHistogram;
  readonly table: ZonalTable;
  readonly exact?: Float64Array;
  readonly plan?: ZonalExactPlan;
  /** Chunks that held a region (the ones each pass visited). */
  readonly tiles: number;
}

const scratchPool = perDevice((device) => new BufferPool(device));

function sameGrid(a: Multiscale, b: Multiscale, level: number): boolean {
  const da = levelVoxelDims(a, level);
  const db = levelVoxelDims(b, level);
  return da[0] === db[0] && da[1] === db[1] && a.chunkShape[0] === b.chunkShape[0] && a.chunkShape[1] === b.chunkShape[1];
}

async function fetch(src: TiledRaster, id: ChunkId): Promise<Tile> {
  const key = chunkKey(id);
  const hit = src.cache?.get(key);
  if (hit) return hit;
  const tile = await src.loader.getChunk(id);
  src.cache?.set(key, tile, tile.data?.byteLength ?? tile.buffer?.byteLength ?? 0);
  if (elementLanes(tile.element) !== 1 || tile.dims[2] !== 1) throw new Error(`zonal: tile ${key} must be a scalar plane`);
  return tile;
}

/** Labels → table over a pyramid level (see the header). */
export async function zonalStatsTiled(opts: ZonalTiledOptions, device?: GPUDevice): Promise<ZonalTiledResult> {
  const { level, channels } = opts;
  if (channels.length === 0) throw new Error("zonal: no channels");
  const base = channels[0]!.source.ms;
  if (!Number.isInteger(level) || level < 0 || level >= base.levelCount) throw new Error(`zonal: level ${level} is not in the pyramid`);
  const supports = channels.map((c) =>
    c.support === undefined ? [] : Array.isArray(c.support) ? (c.support as TiledRaster[]) : [c.support as TiledRaster],
  );
  const regionSrc = opts.regions.kind === "labels" ? opts.regions.source : undefined;
  for (const s of [...channels.map((c) => c.source), ...supports.flat(), ...(regionSrc ? [regionSrc] : [])]) {
    if (!sameGrid(base, s.ms, level)) throw new Error("zonal: every source must share level dims and chunk shape");
  }

  const [d0x, d0y] = base.voxelDims0;
  const [dLx, dLy] = levelVoxelDims(base, level);
  const sx = d0x / dLx;
  const sy = d0y / dLy;
  const [cw, ch] = base.chunkShape;
  const counts = chunkCounts(base, level);
  const all: ChunkId[] = [];
  for (let y = 0; y < counts[1]; y++) for (let x = 0; x < counts[0]; x++) all.push({ level, x, y, z: 0 });

  // Which chunks can hold a region, and how many regions there are.
  let regions: number;
  let layers: number[][] = [];
  let chunks: ChunkId[];
  const polygons = opts.regions.kind === "polygons" ? opts.regions.polygons : [];
  if (opts.regions.kind === "polygons") {
    regions = polygons.length;
    if (regions === 0) throw new Error("zonal: no polygons");
    layers = polygonLayers(polygons);
    chunks = all.filter((id) => {
      const [x0, y0, x1, y1] = [id.x * cw * sx, id.y * ch * sy, (id.x + 1) * cw * sx, (id.y + 1) * ch * sy];
      return polygons.some(({ bbox: b }) => b[0] < x1 && b[2] > x0 && b[1] < y1 && b[3] > y0);
    });
  } else {
    const src = opts.regions.source;
    let max = 0;
    const keep: ChunkId[] = [];
    for (const id of all) {
      const d = (await fetch(src, id)).data;
      if (!d) {
        keep.push(id); // device-resident labels: cannot pre-scan, so visit
        continue;
      }
      let m = 0;
      for (let i = 0; i < d.length; i++) if (d[i]! > m) m = d[i]!;
      if (m >= 1) keep.push(id);
      max = Math.max(max, Math.floor(m));
    }
    regions = opts.regions.regions ?? max;
    if (!(regions >= 1)) throw new Error("zonal: the label raster has no labels ≥ 1");
    chunks = keep;
  }

  const acc = await ZonalAccumulator.create(
    { regions, bins: opts.bins, channels: channels.map(({ range, valid }) => ({ range, valid })) },
    device,
  );
  const dev = acc.device;
  const pool = scratchPool(dev);
  const chunkBytes = cw * ch * 4;
  const leases: ResidentBuffer[] = [];
  const lease = () => {
    const l = pool.lease(chunkBytes);
    leases.push(l);
    return l.buffer;
  };
  const planeBufs = channels.map(() => lease());
  const supportBufs = channels.map((_, c) => (supports[c]!.length ? lease() : undefined));
  const labelBuf = lease();

  /** Load chunk `id` and run `fn` once per label layer, with every buffer filled. */
  const visit = async (id: ChunkId, fn: (tile: ZonalTile) => void) => {
    const [w, h] = chunkVoxelExtent(base, id);
    const n = w * h;
    const data = await Promise.all(channels.map((c) => fetch(c.source, id)));
    const sup = await Promise.all(supports.map((list) => Promise.all(list.map((s) => fetch(s, id)))));
    const planes: ZonalChannelPlane[] = data.map((tile, c) => {
      let plane;
      if (tile.data) {
        writeView(dev.queue, planeBufs[c]!, tile.data.subarray(0, n));
        plane = denseRasterPlane(planeBufs[c]!, w, h);
      } else if (tile.buffer) plane = denseRasterPlane(tile.buffer.buffer, w, h, tile.buffer.offset ?? 0);
      else throw new Error("zonal: a texture-resident tile cannot be reduced");
      const list = sup[c]!;
      let support: RasterMask | undefined;
      if (list.length === 1 && !list[0]!.data && list[0]!.buffer) {
        support = { buffer: list[0]!.buffer.buffer, byteOffset: list[0]!.buffer.offset ?? 0, dtype: "f32" };
      } else if (list.length) {
        const m = new Float32Array(n);
        for (const t of list) {
          if (!t.data) throw new Error("zonal: several support rasters must be host tiles");
          const d = t.data;
          for (let i = 0; i < n; i++) if (d[i] !== 0) m[i] = 1;
        }
        writeView(dev.queue, supportBufs[c]!, m);
        support = { buffer: supportBufs[c]!, dtype: "f32" };
      }
      return { plane, support };
    });

    if (regionSrc) {
      const t = await fetch(regionSrc, id);
      let labels: LabelPlane;
      if (t.data) {
        writeView(dev.queue, labelBuf, t.data.subarray(0, n));
        labels = { buffer: labelBuf, width: w, height: h, dtype: "f32" };
      } else if (t.buffer) labels = { buffer: t.buffer.buffer, byteOffset: t.buffer.offset ?? 0, width: w, height: h, dtype: "f32" };
      else throw new Error("zonal: a texture-resident label tile cannot be reduced");
      fn({ labels, channels: planes });
      return;
    }
    const frame = { width: w, height: h, originX: id.x * cw * sx, originY: id.y * ch * sy, scaleX: sx, scaleY: sy };
    for (const layer of layers) {
      const rings = polygonRingBuffers(
        layer.map((k) => polygons[k]!),
        frame,
        layer.map((k) => k + 1),
      );
      if (rings.count === 0) continue;
      await rasterisePolygonsResident(rings, { buffer: labelBuf, width: w, height: h }, dev);
      fn({ labels: { buffer: labelBuf, width: w, height: h, dtype: "u32" }, channels: planes });
    }
  };

  const pass = async (name: "range" | "bin" | "collect", fn: (tile: ZonalTile) => void) => {
    let done = 0;
    for (const id of chunks) {
      await visit(id, fn);
      opts.onTile?.(name, ++done, chunks.length);
    }
  };

  try {
    if (acc.needsRange) {
      await pass("range", (t) => acc.rangeTile(t));
      acc.finalizeEdges();
    }
    await pass("bin", (t) => acc.binTile(t));
    const histogram = await acc.read();
    const quantiles = opts.quantiles ?? [0.1, 0.25, 0.5, 0.75, 0.9];
    const names = channels.map((c, i) => c.name ?? `c${i}`);
    if (!opts.exact) return { histogram, table: zonalTable(histogram, { names, quantiles }), tiles: chunks.length };
    const plan = zonalExactPlan(histogram, quantiles);
    acc.beginCollect(plan);
    await pass("collect", (t) => acc.collectTile(t));
    const exact = zonalExactQuantiles(histogram, plan, await acc.readCollected());
    return { histogram, table: zonalTable(histogram, { names, quantiles, exact }), exact, plan, tiles: chunks.length };
  } finally {
    acc.release();
    for (const l of leases) pool.release(l);
  }
}
