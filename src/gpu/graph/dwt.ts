// dwt.ts — separable 2D Discrete Wavelet Transform for the op graph.
//
// A faithful Float32 port of the project's reference implementation
// (`docs-site/src/lib/dwt.ts`, itself ported from `viz/dwt.js`) so the op-graph
// wavelet ops stay numerically identical to the docs "Draw in the DWT domain" demo.
// Separable 2D Mallat pyramid with the two JPEG 2000 lifting filters:
//   - 5/3 REVERSIBLE   (LeGall, integer lifting, lossless on integer input)
//   - 9/7 IRREVERSIBLE (CDF 9/7, floating-point lifting, lossy)
// Boundary handling is whole-sample symmetric (mirror) extension (JPEG 2000).
//
// The forward transform packs coefficients in place in the standard Mallat layout
// (LL in the top-left corner shrinking each level; HL/LH/HH detail quadrants around
// it). The coefficient grid is the *wavelet-domain representation* of the image — an
// editable signal, not a picture (see ADR-0006). `dwtBands` describes that layout so
// band-selective ops can address subbands.
//
// f32 note: 5/3 uses `Math.floor` and is exact for integer inputs up to 2^24 (well
// beyond image bit depths); 9/7 is lossy by construction, with f32 round-trip error a
// few ×1e-3 (vs ~1e-6 for the f64 reference) — fine for analysis, flagged for codecs.

export type Kernel = "5/3" | "9/7";
export type BandType = "LL" | "HL" | "LH" | "HH";

/** A subband's location and size within the packed Mallat layout. */
export interface Band {
  level: number;
  type: BandType;
  x: number;
  y: number;
  w: number;
  h: number;
}

/* ---- CDF 9/7 lifting coefficients (standard JPEG 2000 values) ---- */
const A = -1.586134342059924; // alpha  (predict 1)
const B = -0.052980118572961; // beta   (update 1)
const G = 0.882911075530934; //  gamma  (predict 2)
const D = 0.443506852043971; //  delta  (update 2)
const K = 1.230174104914001; //  scaling K (low-pass), 1/K for high-pass

/** Whole-sample symmetric mirror of index `i` into `[0, n-1]` (no boundary repeat). */
export function mirror(i: number, n: number): number {
  if (n === 1) return 0;
  const period = 2 * (n - 1);
  let k = i % period;
  if (k < 0) k += period;
  return k < n ? k : period - k;
}

type Lift1D = (buf: Float32Array, base: number, stride: number, n: number, tmp: Float32Array) => void;

/* ---- 1D 5/3 reversible lifting (integer) ---- */
const fwd53_1d: Lift1D = (buf, base, stride, n, tmp) => {
  if (n < 2) return;
  for (let j = 0; j < n; j++) tmp[j] = buf[base + j * stride]!;
  const nLow = (n + 1) >> 1;
  for (let k = 1; k < n; k += 2) {
    const left = tmp[k - 1]!;
    const right = k + 1 < n ? tmp[k + 1]! : tmp[mirror(k + 1, n)]!;
    tmp[k] = tmp[k]! - Math.floor((left + right) / 2);
  }
  for (let k = 0; k < n; k += 2) {
    const left = k - 1 >= 0 ? tmp[k - 1]! : tmp[mirror(k - 1, n)]!;
    const right = k + 1 < n ? tmp[k + 1]! : tmp[mirror(k + 1, n)]!;
    tmp[k] = tmp[k]! + Math.floor((left + right + 2) / 4);
  }
  for (let k = 0; k < n; k++) {
    const dst = (k & 1) === 0 ? k >> 1 : nLow + (k >> 1);
    buf[base + dst * stride] = tmp[k]!;
  }
};

const inv53_1d: Lift1D = (buf, base, stride, n, tmp) => {
  if (n < 2) return;
  const nLow = (n + 1) >> 1;
  for (let k = 0; k < n; k++) {
    const src = (k & 1) === 0 ? k >> 1 : nLow + (k >> 1);
    tmp[k] = buf[base + src * stride]!;
  }
  for (let k = 0; k < n; k += 2) {
    const left = k - 1 >= 0 ? tmp[k - 1]! : tmp[mirror(k - 1, n)]!;
    const right = k + 1 < n ? tmp[k + 1]! : tmp[mirror(k + 1, n)]!;
    tmp[k] = tmp[k]! - Math.floor((left + right + 2) / 4);
  }
  for (let k = 1; k < n; k += 2) {
    const left = tmp[k - 1]!;
    const right = k + 1 < n ? tmp[k + 1]! : tmp[mirror(k + 1, n)]!;
    tmp[k] = tmp[k]! + Math.floor((left + right) / 2);
  }
  for (let j = 0; j < n; j++) buf[base + j * stride] = tmp[j]!;
};

/* ---- 1D CDF 9/7 irreversible lifting (floating point) ---- */
const fwd97_1d: Lift1D = (buf, base, stride, n, tmp) => {
  if (n < 2) return;
  for (let j = 0; j < n; j++) tmp[j] = buf[base + j * stride]!;
  const nLow = (n + 1) >> 1;
  for (let k = 1; k < n; k += 2) {
    const l = tmp[k - 1]!;
    const r = k + 1 < n ? tmp[k + 1]! : tmp[mirror(k + 1, n)]!;
    tmp[k] = tmp[k]! + A * (l + r);
  }
  for (let k = 0; k < n; k += 2) {
    const l = k - 1 >= 0 ? tmp[k - 1]! : tmp[mirror(k - 1, n)]!;
    const r = k + 1 < n ? tmp[k + 1]! : tmp[mirror(k + 1, n)]!;
    tmp[k] = tmp[k]! + B * (l + r);
  }
  for (let k = 1; k < n; k += 2) {
    const l = tmp[k - 1]!;
    const r = k + 1 < n ? tmp[k + 1]! : tmp[mirror(k + 1, n)]!;
    tmp[k] = tmp[k]! + G * (l + r);
  }
  for (let k = 0; k < n; k += 2) {
    const l = k - 1 >= 0 ? tmp[k - 1]! : tmp[mirror(k - 1, n)]!;
    const r = k + 1 < n ? tmp[k + 1]! : tmp[mirror(k + 1, n)]!;
    tmp[k] = tmp[k]! + D * (l + r);
  }
  for (let k = 0; k < n; k++) tmp[k] = tmp[k]! * ((k & 1) === 0 ? 1 / K : K);
  for (let k = 0; k < n; k++) {
    const dst = (k & 1) === 0 ? k >> 1 : nLow + (k >> 1);
    buf[base + dst * stride] = tmp[k]!;
  }
};

const inv97_1d: Lift1D = (buf, base, stride, n, tmp) => {
  if (n < 2) return;
  const nLow = (n + 1) >> 1;
  for (let k = 0; k < n; k++) {
    const src = (k & 1) === 0 ? k >> 1 : nLow + (k >> 1);
    tmp[k] = buf[base + src * stride]!;
  }
  for (let k = 0; k < n; k++) tmp[k] = tmp[k]! * ((k & 1) === 0 ? K : 1 / K);
  for (let k = 0; k < n; k += 2) {
    const l = k - 1 >= 0 ? tmp[k - 1]! : tmp[mirror(k - 1, n)]!;
    const r = k + 1 < n ? tmp[k + 1]! : tmp[mirror(k + 1, n)]!;
    tmp[k] = tmp[k]! - D * (l + r);
  }
  for (let k = 1; k < n; k += 2) {
    const l = tmp[k - 1]!;
    const r = k + 1 < n ? tmp[k + 1]! : tmp[mirror(k + 1, n)]!;
    tmp[k] = tmp[k]! - G * (l + r);
  }
  for (let k = 0; k < n; k += 2) {
    const l = k - 1 >= 0 ? tmp[k - 1]! : tmp[mirror(k - 1, n)]!;
    const r = k + 1 < n ? tmp[k + 1]! : tmp[mirror(k + 1, n)]!;
    tmp[k] = tmp[k]! - B * (l + r);
  }
  for (let k = 1; k < n; k += 2) {
    const l = tmp[k - 1]!;
    const r = k + 1 < n ? tmp[k + 1]! : tmp[mirror(k + 1, n)]!;
    tmp[k] = tmp[k]! - A * (l + r);
  }
  for (let j = 0; j < n; j++) buf[base + j * stride] = tmp[j]!;
};

interface KernelOps {
  fwd: Lift1D;
  inv: Lift1D;
  integer: boolean;
}

export const KERNELS: Record<Kernel, KernelOps> = {
  "5/3": { fwd: fwd53_1d, inv: inv53_1d, integer: true },
  "9/7": { fwd: fwd97_1d, inv: inv97_1d, integer: false },
};

/** Subband geometry of a `levels`-deep packed Mallat decomposition of a w×h grid. */
export function dwtBands(width: number, height: number, levels: number): Band[] {
  const bands: Band[] = [];
  let curW = width,
    curH = height;
  let lvl = 0;
  for (; lvl < levels && curW >= 2 && curH >= 2; lvl++) {
    const lowW = (curW + 1) >> 1;
    const lowH = (curH + 1) >> 1;
    bands.push({ level: lvl + 1, type: "HL", x: lowW, y: 0, w: curW - lowW, h: lowH });
    bands.push({ level: lvl + 1, type: "LH", x: 0, y: lowH, w: lowW, h: curH - lowH });
    bands.push({ level: lvl + 1, type: "HH", x: lowW, y: lowH, w: curW - lowW, h: curH - lowH });
    curW = lowW;
    curH = lowH;
  }
  bands.push({ level: lvl, type: "LL", x: 0, y: 0, w: curW, h: curH });
  return bands;
}

/** Forward 2D multi-level DWT → packed Mallat coefficient grid (new array). */
export function fdwt2d(src: ArrayLike<number>, width: number, height: number, kernel: Kernel, levels: number): Float32Array {
  const k = KERNELS[kernel];
  const data = Float32Array.from(src);
  const tmp = new Float32Array(Math.max(width, height));
  let curW = width,
    curH = height;
  for (let lvl = 0; lvl < levels && curW >= 2 && curH >= 2; lvl++) {
    for (let y = 0; y < curH; y++) k.fwd(data, y * width, 1, curW, tmp);
    for (let x = 0; x < curW; x++) k.fwd(data, x, width, curH, tmp);
    curW = (curW + 1) >> 1;
    curH = (curH + 1) >> 1;
  }
  return data;
}

/** Inverse 2D multi-level DWT of a packed Mallat coefficient grid → image (new array). */
export function idwt2d(src: ArrayLike<number>, width: number, height: number, kernel: Kernel, levels: number): Float32Array {
  const k = KERNELS[kernel];
  const data = Float32Array.from(src);
  const tmp = new Float32Array(Math.max(width, height));
  const sizes: Array<[number, number]> = [];
  let curW = width,
    curH = height;
  for (let lvl = 0; lvl < levels && curW >= 2 && curH >= 2; lvl++) {
    sizes.push([curW, curH]);
    curW = (curW + 1) >> 1;
    curH = (curH + 1) >> 1;
  }
  for (let i = sizes.length - 1; i >= 0; i--) {
    const [w, h] = sizes[i]!;
    for (let x = 0; x < w; x++) k.inv(data, x, width, h, tmp);
    for (let y = 0; y < h; y++) k.inv(data, y * width, 1, w, tmp);
  }
  return data;
}

/** True for subbands that carry detail (everything except the coarse LL). */
export const isDetailBand = (b: Band): boolean => b.type !== "LL";
