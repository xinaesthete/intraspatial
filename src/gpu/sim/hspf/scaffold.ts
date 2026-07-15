// GeoTIFF ingestion for the HsPf scaffold (ADR-0011, decision 3): decode the real HbS + weights
// rasters into a `HspfScaffold`, converting nodata/ocean to the negative sentinel the kernel
// early-exits on. Deliberately BYO-data with `geotiff` (a step toward GIS / leaning on libraries)
// — NOT a reusable multiscale Loader (that belongs to the datasource context, ADR-0008).
//
// The core takes an `ArrayBuffer` so it is env-agnostic: the browser page fetches the `.tif` and
// passes `arrayBuffer()`; a Node test reads the file. The geo-transform is *captured* here but not
// yet used — barrier/count placement (phase 2) will consume it.
import { fromArrayBuffer } from "geotiff";
import type { HspfScaffold } from "./kernel";

/** The lat/long ↔ pixel transform captured from the HbS raster (unused in phase 1). */
export interface GeoTransform {
  /** Top-left corner in geographic coordinates [x, y] (e.g. [minLon, maxLat]). */
  origin: [number, number];
  /** Pixel size [dx, dy] in geographic units (dy is typically negative). */
  resolution: [number, number];
  /** [minX, minY, maxX, maxY]. */
  bbox: [number, number, number, number];
}

export interface ScaffoldWithGeo extends HspfScaffold {
  geo: GeoTransform;
}

interface DecodedRaster {
  data: ArrayLike<number>;
  width: number;
  height: number;
  nodata: number | null;
  origin: number[];
  resolution: number[];
  bbox: number[];
}

async function decode(buffer: ArrayBuffer): Promise<DecodedRaster> {
  const tiff = await fromArrayBuffer(buffer);
  const image = await tiff.getImage();
  const rasters = await image.readRasters({ interleave: false });
  const band = rasters[0];
  if (!band || typeof band === "number") throw new Error("hspf scaffold: GeoTIFF has no readable band");
  return {
    data: band as ArrayLike<number>,
    width: image.getWidth(),
    height: image.getHeight(),
    nodata: image.getGDALNoData(),
    origin: image.getOrigin(),
    resolution: image.getResolution(),
    bbox: image.getBoundingBox(),
  };
}

/** A finite land value, or NaN for nodata/ocean. */
function landValue(raw: ArrayLike<number>, i: number, nodata: number | null): number {
  const v = Number(raw[i]);
  if (!Number.isFinite(v)) return Number.NaN;
  if (nodata !== null && v === nodata) return Number.NaN;
  return v;
}

/** Decode the HbS + weights GeoTIFFs into a scaffold. Ocean/missing cells (nodata, non-finite,
 *  or negative HbS) become the `-2` sentinel; weights default to 0 where missing. The two rasters
 *  must share dimensions. */
export async function scaffoldFromGeoTIFFs(hbsBuffer: ArrayBuffer, weightsBuffer: ArrayBuffer): Promise<ScaffoldWithGeo> {
  const h = await decode(hbsBuffer);
  const w = await decode(weightsBuffer);
  if (h.width !== w.width || h.height !== w.height) {
    throw new Error(`hspf scaffold: HbS ${h.width}×${h.height} and weights ${w.width}×${w.height} differ in size`);
  }
  const n = h.width * h.height;
  const hbs = new Float32Array(n);
  const weights = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const hv = landValue(h.data, i, h.nodata);
    hbs[i] = Number.isFinite(hv) && hv >= 0 ? hv : -2; // ocean / missing / negative ⇒ sentinel
    const wv = landValue(w.data, i, w.nodata);
    weights[i] = Number.isFinite(wv) && wv > 0 ? wv : 0; // neutral where missing
  }
  const geo: GeoTransform = {
    origin: [h.origin[0] ?? 0, h.origin[1] ?? 0],
    resolution: [h.resolution[0] ?? 0, h.resolution[1] ?? 0],
    bbox: [h.bbox[0] ?? 0, h.bbox[1] ?? 0, h.bbox[2] ?? 0, h.bbox[3] ?? 0],
  };
  return { hbs, weights, width: h.width, height: h.height, geo };
}
