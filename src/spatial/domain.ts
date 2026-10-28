// The region a set of cells was observed in — the `W` that the cross-PCF's ρ_B, edge correction and
// membership all refer to. See docs/stats-on-arbitrary-domains.md.
//
// Two kinds. A **rect** is the analytic case `edgeCorrection.ts` already handles exactly. A **mask**
// is a raster of in/out pixels, which is how any other shape gets in: a tissue mask, a rasterised
// annotation polygon, a TMA core.
//
// ## The mask is clipped exactly — its only error is its staircase
//
// The clipped annulus area is `A_k(x) = Σ_{mask pixels p} |annulus_k(x) ∩ p|`, and every term is a
// `diskRectArea` call, so the integration is exact *for the union of mask pixels*. What remains is how
// well that union represents the true boundary. Measured (`pnpm bench:domain-mask`) at the covid
// binning, that costs ≤ 0.1% on g at 1–2 µm pixels and ≤ 0.34% at 4 µm. Counting pixel centres
// instead is indistinguishable in g but 2–6× noisier per anchor on oblique edges, and per-anchor
// values are what highlights and weights are built from.
//
// Most pixels are not cut by any bin edge and drop into a single bin as their full area, so only
// the pixels a bin-edge circle passes through pay for `diskRectArea`.
//
// ## Raster convention
//
// Same as `ScalarField` and the GPU splats: row-major `width × height` over `bbox`, row 0 at the TOP
// (worldY = maxY). Non-zero = inside. Pixels need not be square.

import { annulusAreasInto, diskRectArea } from "./edgeCorrection";

type Box = readonly [number, number, number, number];

export interface RectDomain {
  readonly kind: "rect";
  /** `[minX, minY, maxX, maxY]`, closed. */
  readonly bbox: Box;
}

export interface MaskDomain {
  readonly kind: "mask";
  /** World extent the raster spans, `[minX, minY, maxX, maxY]`. Outside it is outside the domain. */
  readonly bbox: Box;
  readonly width: number;
  readonly height: number;
  /** Row-major, row 0 = top (maxY). Non-zero = inside. */
  readonly data: Uint8Array;
  /** Type ids whose cells the mask was BUILT from — a density threshold, a wand similarity — when
   *  it came from the data rather than being drawn. A cross-PCF pair with both members in here is
   *  measured through a window its own cells chose, and `crossPCFMatrixBinned` flags it as
   *  `confounded`. Absent for a drawn or external mask. */
  readonly derivedFrom?: readonly number[];
}

export type Domain = RectDomain | MaskDomain;

export function rectDomain(bbox: Box): RectDomain {
  if (!(bbox[2] > bbox[0] && bbox[3] > bbox[1])) throw new Error(`rectDomain: bbox [${bbox.join(", ")}] has no positive extent`);
  return { kind: "rect", bbox };
}

/** A mask domain, checked: a raster whose length disagrees with its shape would be read as a
 *  different, silently sheared region. */
export function maskDomain(bbox: Box, width: number, height: number, data: Uint8Array, derivedFrom?: readonly number[]): MaskDomain {
  if (!(bbox[2] > bbox[0] && bbox[3] > bbox[1])) throw new Error(`maskDomain: bbox [${bbox.join(", ")}] has no positive extent`);
  if (!(Number.isInteger(width) && Number.isInteger(height) && width > 0 && height > 0)) {
    throw new Error(`maskDomain: ${width}×${height} is not a raster shape`);
  }
  if (data.length !== width * height) throw new Error(`maskDomain: ${data.length} pixels for a ${width}×${height} raster`);
  return derivedFrom ? { kind: "mask", bbox, width, height, data, derivedFrom } : { kind: "mask", bbox, width, height, data };
}

/**
 * A mask from any raster in the `ScalarField` convention — a density field, a mode projection, the
 * Gram wand's `similarityField` — inside wherever `keep(value)` holds.
 *
 * The raster's `bbox` becomes the domain's, so build the field over the OBSERVATION window: a field
 * padded past it (as `gaussianKdeField` pads by default) can mark unobserved ground as inside, which
 * is the loose-boundary bias of docs/stats-on-arbitrary-domains.md source 4.
 *
 * A mask built from the same cells the statistic is then measured on is a data-dependent window.
 * Measured (docs/stats-on-arbitrary-domains.md, "Masks from the data"): on independently clustered
 * types, a mask cut from both A's and B's densities reads g_AB 14–16% low, where the truth is 1. A
 * mask that never saw B's cells leaves g_AB unbiased. Pass `derivedFrom` — the type ids whose cells
 * built the field — so the PCF can flag the pairs this applies to.
 */
export function maskWhere(
  field: { readonly data: ArrayLike<number>; readonly width: number; readonly height: number; readonly bbox: Box },
  keep: (value: number) => boolean,
  opts: { readonly derivedFrom?: readonly number[] } = {},
): MaskDomain {
  const n = field.width * field.height;
  if (field.data.length !== n) throw new Error(`maskWhere: ${field.data.length} values for a ${field.width}×${field.height} raster`);
  const data = new Uint8Array(n);
  for (let i = 0; i < n; i++) data[i] = keep(field.data[i]!) ? 1 : 0;
  return maskDomain(field.bbox, field.width, field.height, data, opts.derivedFrom);
}

/** |W|. For a mask, the area of its non-zero pixels. */
export function domainArea(d: Domain): number {
  const [minX, minY, maxX, maxY] = d.bbox;
  if (d.kind === "rect") return (maxX - minX) * (maxY - minY);
  let n = 0;
  for (let i = 0; i < d.data.length; i++) if (d.data[i] !== 0) n++;
  return (n * (maxX - minX) * (maxY - minY)) / (d.width * d.height);
}

/** Whether `(x, y)` lies in the domain. A mask answers by the pixel containing the point, with its
 *  max edges closed so a cell exactly on `maxX`/`minY` is not lost to rounding. */
export function domainContains(d: Domain, x: number, y: number): boolean {
  const [minX, minY, maxX, maxY] = d.bbox;
  if (!(x >= minX && x <= maxX && y >= minY && y <= maxY)) return false;
  if (d.kind === "rect") return true;
  const i = Math.min(d.width - 1, Math.floor(((x - minX) / (maxX - minX)) * d.width));
  const j = Math.min(d.height - 1, Math.floor(((maxY - y) / (maxY - minY)) * d.height));
  return d.data[j * d.width + i] !== 0;
}

/**
 * Clipped areas of every annulus `[k·dr, (k+1)·dr)` around `(cx, cy)`, written into `out` (length
 * `nBins`). Exact for a rect, and exact for the pixel union of a mask.
 */
export function domainAnnulusAreasInto(out: Float64Array, d: Domain, cx: number, cy: number, dr: number, nBins: number): void {
  const [minX, minY, maxX, maxY] = d.bbox;
  if (d.kind === "rect") {
    annulusAreasInto(out, cx, cy, dr, nBins, minX, minY, maxX, maxY);
    return;
  }
  out.fill(0, 0, nBins);
  const rMax = nBins * dr;
  const pw = (maxX - minX) / d.width;
  const ph = (maxY - minY) / d.height;
  const full = pw * ph;
  const i0 = Math.max(0, Math.floor((cx - rMax - minX) / pw));
  const i1 = Math.min(d.width - 1, Math.floor((cx + rMax - minX) / pw));
  const j0 = Math.max(0, Math.floor((maxY - (cy + rMax)) / ph));
  const j1 = Math.min(d.height - 1, Math.floor((maxY - (cy - rMax)) / ph));
  for (let j = j0; j <= j1; j++) {
    const y1 = maxY - j * ph;
    const y0 = y1 - ph;
    const dyMin = Math.max(y0 - cy, 0, cy - y1);
    const dyMax = Math.max(Math.abs(cy - y0), Math.abs(cy - y1));
    for (let i = i0; i <= i1; i++) {
      if (d.data[j * d.width + i] === 0) continue;
      const x0 = minX + i * pw;
      const x1 = x0 + pw;
      const dxMin = Math.max(x0 - cx, 0, cx - x1);
      const dMin = Math.sqrt(dxMin * dxMin + dyMin * dyMin);
      if (dMin >= rMax) continue;
      const dxMax = Math.max(Math.abs(cx - x0), Math.abs(cx - x1));
      const dMax = Math.sqrt(dxMax * dxMax + dyMax * dyMax);
      // Walk the bin edges that pass through this pixel. `prev` is the area of the pixel inside the
      // disk of the last edge crossed; the part beyond the final edge, if the pixel ends inside
      // rMax, falls wholly in the bin that edge opens. An uncut pixel takes the first branch at once.
      let prev = 0;
      for (let e = Math.floor(dMin / dr) + 1; ; e++) {
        const r = e * dr;
        if (r >= dMax) {
          if (e <= nBins) out[e - 1]! += full - prev;
          break;
        }
        const a = diskRectArea(cx, cy, r, x0, y0, x1, y1);
        out[e - 1]! += a - prev;
        prev = a;
        if (e === nBins) break;
      }
    }
  }
}

/**
 * Per-anchor reciprocal annulus areas `1/A_k(x)` — the weight the cross-PCF puts on a pair in bin
 * `k`. An anchor whose disk of `rMax` the boundary cannot reach takes the precomputed full-annulus
 * weights; that is an exact answer, not an approximation, since nothing is clipped. Bins with no
 * area left get weight 0.
 *
 * The returned array is reused between boundary anchors: read it before the next call.
 */
export function inverseAnnulusAreas(d: Domain, rMax: number, nBins: number): (x: number, y: number) => Float64Array {
  const dr = rMax / nBins;
  const invFull = new Float64Array(nBins);
  for (let k = 0; k < nBins; k++) {
    const r0 = k * dr;
    const r1 = r0 + dr;
    invFull[k] = 1 / (Math.PI * (r1 * r1 - r0 * r0));
  }
  const scratch = new Float64Array(nBins);
  const inv = new Float64Array(nBins);
  const interior = interiorTest(d, rMax);
  return (x, y) => {
    if (interior(x, y)) return invFull;
    domainAnnulusAreasInto(scratch, d, x, y, dr, nBins);
    for (let k = 0; k < nBins; k++) inv[k] = scratch[k]! > 0 ? 1 / scratch[k]! : 0;
    return inv;
  };
}

/** Whether the disk of `rMax` around a point is wholly inside the domain. Conservative for a mask:
 *  it checks the disk's bounding square through a summed-area table of OUTSIDE pixels, so an anchor
 *  near a concave corner may be sent down the exact path needlessly, never the reverse. */
function interiorTest(d: Domain, rMax: number): (x: number, y: number) => boolean {
  const [minX, minY, maxX, maxY] = d.bbox;
  if (d.kind === "rect") return (x, y) => x - minX >= rMax && maxX - x >= rMax && y - minY >= rMax && maxY - y >= rMax;
  const w = d.width;
  const h = d.height;
  const pw = (maxX - minX) / w;
  const ph = (maxY - minY) / h;
  // sat[(j+1)(w+1) + (i+1)] = outside pixels in rows ≤ j, columns ≤ i.
  const sat = new Int32Array((w + 1) * (h + 1));
  for (let j = 0; j < h; j++) {
    let row = 0;
    for (let i = 0; i < w; i++) {
      if (d.data[j * w + i] === 0) row++;
      sat[(j + 1) * (w + 1) + i + 1] = sat[j * (w + 1) + i + 1]! + row;
    }
  }
  return (x, y) => {
    const i0 = Math.floor((x - rMax - minX) / pw);
    const i1 = Math.floor((x + rMax - minX) / pw);
    const j0 = Math.floor((maxY - (y + rMax)) / ph);
    const j1 = Math.floor((maxY - (y - rMax)) / ph);
    // A square reaching past the raster reaches outside the domain.
    if (i0 < 0 || j0 < 0 || i1 >= w || j1 >= h) return false;
    const W = w + 1;
    return sat[(j1 + 1) * W + i1 + 1]! - sat[j0 * W + i1 + 1]! - sat[(j1 + 1) * W + i0]! + sat[j0 * W + i0]! === 0;
  };
}
