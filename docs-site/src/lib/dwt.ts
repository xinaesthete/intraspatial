/*
 * dwt.ts — Discrete Wavelet Transform, TypeScript port of `viz/dwt.js`.
 *
 * Separable 2D DWT (the Mallat pyramid) used by JPEG 2000 / HTJ2K, with the
 * two standard lifting filters:
 *   - 5/3 REVERSIBLE  (LeGall, integer lifting, lossless)
 *   - 9/7 IRREVERSIBLE (CDF 9/7, floating-point lifting, lossy)
 *
 * Boundary handling is whole-sample symmetric (mirror) extension, matching
 * the JPEG 2000 convention. Data is a flat Float64Array in row-major order.
 *
 * This is a faithful port of the vanilla reference so the React draw/erase
 * demo and the standalone primer stay numerically identical.
 */

/* ---- CDF 9/7 lifting coefficients (standard JPEG 2000 values) ---- */
const A = -1.586134342059924; // alpha  (predict 1)
const B = -0.052980118572961; // beta   (update 1)
const G = 0.882911075530934; //  gamma  (predict 2)
const D = 0.443506852043971; //  delta  (update 2)
const K = 1.230174104914001; //  scaling K (low-pass), 1/K for high-pass

export type Kernel = "5/3" | "9/7";

export interface ImagePlane {
  data: Float64Array;
  width: number;
  height: number;
}

export type BandType = "LL" | "HL" | "LH" | "HH";

export interface Band {
  level: number;
  type: BandType;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Decomposition {
  data: Float64Array;
  width: number;
  height: number;
  levels: number;
  kernel: Kernel;
  bands: Band[];
  llW: number;
  llH: number;
}

/* Symmetric index mirror for whole-sample symmetric extension. Maps any
 * integer index into [0, n-1] by reflecting about the endpoints without
 * repeating the boundary sample. */
export function mirror(i: number, n: number): number {
  if (n === 1) return 0;
  const period = 2 * (n - 1);
  let k = i % period;
  if (k < 0) k += period;
  return k < n ? k : period - k;
}

type Lift1D = (buf: Float64Array, base: number, stride: number, n: number, tmp: Float64Array) => void;

function tmpFor(n: number): Float64Array {
  return new Float64Array(n);
}

/* ---- 1D 5/3 reversible lifting (integer) ---- */
const fwd53_1d: Lift1D = (buf, base, stride, n, tmp) => {
  if (n < 2) return;
  for (let j = 0; j < n; j++) tmp[j] = buf[base + j * stride];
  const nLow = (n + 1) >> 1;
  // Predict: odd[k] -= floor((s[2k] + s[2k+2]) / 2)
  for (let k = 1; k < n; k += 2) {
    const left = tmp[k - 1];
    const right = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] -= Math.floor((left + right) / 2);
  }
  // Update: even[k] += floor((odd[k-1] + odd[k+1] + 2) / 4)
  for (let k = 0; k < n; k += 2) {
    const left = k - 1 >= 0 ? tmp[k - 1] : tmp[mirror(k - 1, n)];
    const right = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] += Math.floor((left + right + 2) / 4);
  }
  for (let k = 0; k < n; k++) {
    const dst = (k & 1) === 0 ? k >> 1 : nLow + (k >> 1);
    buf[base + dst * stride] = tmp[k];
  }
};

const inv53_1d: Lift1D = (buf, base, stride, n, tmp) => {
  if (n < 2) return;
  const nLow = (n + 1) >> 1;
  for (let k = 0; k < n; k++) {
    const src = (k & 1) === 0 ? k >> 1 : nLow + (k >> 1);
    tmp[k] = buf[base + src * stride];
  }
  for (let k = 0; k < n; k += 2) {
    const left = k - 1 >= 0 ? tmp[k - 1] : tmp[mirror(k - 1, n)];
    const right = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] -= Math.floor((left + right + 2) / 4);
  }
  for (let k = 1; k < n; k += 2) {
    const left = tmp[k - 1];
    const right = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] += Math.floor((left + right) / 2);
  }
  for (let j = 0; j < n; j++) buf[base + j * stride] = tmp[j];
};

/* ---- 1D CDF 9/7 irreversible lifting (floating point) ---- */
const fwd97_1d: Lift1D = (buf, base, stride, n, tmp) => {
  if (n < 2) return;
  for (let j = 0; j < n; j++) tmp[j] = buf[base + j * stride];
  const nLow = (n + 1) >> 1;
  for (let k = 1; k < n; k += 2) {
    const l = tmp[k - 1];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] += A * (l + r);
  }
  for (let k = 0; k < n; k += 2) {
    const l = k - 1 >= 0 ? tmp[k - 1] : tmp[mirror(k - 1, n)];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] += B * (l + r);
  }
  for (let k = 1; k < n; k += 2) {
    const l = tmp[k - 1];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] += G * (l + r);
  }
  for (let k = 0; k < n; k += 2) {
    const l = k - 1 >= 0 ? tmp[k - 1] : tmp[mirror(k - 1, n)];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] += D * (l + r);
  }
  for (let k = 0; k < n; k++) tmp[k] *= (k & 1) === 0 ? 1 / K : K;
  for (let k = 0; k < n; k++) {
    const dst = (k & 1) === 0 ? k >> 1 : nLow + (k >> 1);
    buf[base + dst * stride] = tmp[k];
  }
};

const inv97_1d: Lift1D = (buf, base, stride, n, tmp) => {
  if (n < 2) return;
  const nLow = (n + 1) >> 1;
  for (let k = 0; k < n; k++) {
    const src = (k & 1) === 0 ? k >> 1 : nLow + (k >> 1);
    tmp[k] = buf[base + src * stride];
  }
  for (let k = 0; k < n; k++) tmp[k] *= (k & 1) === 0 ? K : 1 / K;
  for (let k = 0; k < n; k += 2) {
    const l = k - 1 >= 0 ? tmp[k - 1] : tmp[mirror(k - 1, n)];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] -= D * (l + r);
  }
  for (let k = 1; k < n; k += 2) {
    const l = tmp[k - 1];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] -= G * (l + r);
  }
  for (let k = 0; k < n; k += 2) {
    const l = k - 1 >= 0 ? tmp[k - 1] : tmp[mirror(k - 1, n)];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] -= B * (l + r);
  }
  for (let k = 1; k < n; k += 2) {
    const l = tmp[k - 1];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] -= A * (l + r);
  }
  for (let j = 0; j < n; j++) buf[base + j * stride] = tmp[j];
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

/* 2D separable multi-level forward DWT (the Mallat pyramid). */
export function dwt2dForward(plane: ImagePlane, kernel: Kernel, levels: number): Decomposition {
  const { width, height } = plane;
  const k = KERNELS[kernel];
  const data = Float64Array.from(plane.data);
  const tmp = tmpFor(Math.max(width, height));

  let curW = width;
  let curH = height;
  const bands: Band[] = [];

  for (let lvl = 0; lvl < levels && curW >= 2 && curH >= 2; lvl++) {
    for (let y = 0; y < curH; y++) k.fwd(data, y * width, 1, curW, tmp);
    for (let x = 0; x < curW; x++) k.fwd(data, x, width, curH, tmp);
    const lowW = (curW + 1) >> 1;
    const lowH = (curH + 1) >> 1;
    bands.push({ level: lvl + 1, type: "HL", x: lowW, y: 0, w: curW - lowW, h: lowH });
    bands.push({ level: lvl + 1, type: "LH", x: 0, y: lowH, w: lowW, h: curH - lowH });
    bands.push({ level: lvl + 1, type: "HH", x: lowW, y: lowH, w: curW - lowW, h: curH - lowH });
    curW = lowW;
    curH = lowH;
  }
  bands.push({ level: levels, type: "LL", x: 0, y: 0, w: curW, h: curH });

  return { data, width, height, levels, kernel, bands, llW: curW, llH: curH };
}

/* 2D inverse DWT: undo the levels in reverse order. Accepts any object that
 * carries `data` + the transform geometry (e.g. an edited coefficient copy). */
export function dwt2dInverse(dec: { data: Float64Array; width: number; height: number; levels: number; kernel: Kernel }): ImagePlane {
  const { width, height, levels, kernel } = dec;
  const k = KERNELS[kernel];
  const data = Float64Array.from(dec.data);
  const tmp = tmpFor(Math.max(width, height));

  const sizes: Array<[number, number]> = [];
  let curW = width;
  let curH = height;
  for (let lvl = 0; lvl < levels && curW >= 2 && curH >= 2; lvl++) {
    sizes.push([curW, curH]);
    curW = (curW + 1) >> 1;
    curH = (curH + 1) >> 1;
  }
  for (let i = sizes.length - 1; i >= 0; i--) {
    const [w, h] = sizes[i];
    for (let x = 0; x < w; x++) k.inv(data, x, width, h, tmp);
    for (let y = 0; y < h; y++) k.inv(data, y * width, 1, w, tmp);
  }
  return { data, width, height };
}

/* Round-trip self-test; returns max abs reconstruction error per kernel. */
export function selfTest(width = 37, height = 53, levels = 4): Record<Kernel, number> {
  const results = {} as Record<Kernel, number>;
  for (const kernel of ["5/3", "9/7"] as Kernel[]) {
    const src = new Float64Array(width * height);
    for (let i = 0; i < src.length; i++) src[i] = Math.floor(Math.random() * 256) - 128;
    const dec = dwt2dForward({ data: src, width, height }, kernel, levels);
    const rec = dwt2dInverse(dec);
    let maxErr = 0;
    for (let i = 0; i < src.length; i++) maxErr = Math.max(maxErr, Math.abs(rec.data[i] - src[i]));
    results[kernel] = maxErr;
  }
  return results;
}

export const coeffs = { A, B, G, D, K };
