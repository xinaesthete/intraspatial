/*
 * dwt.js — Discrete Wavelet Transform reference implementation (vanilla JS).
 *
 * Implements the separable 2D DWT used by JPEG 2000 / HTJ2K, with both:
 *
 *   - 5/3 REVERSIBLE transform (integer lifting): perfectly reconstructs
 *     integer input. This is the lossless filter (LeGall 5/3).
 *
 *   - 9/7 IRREVERSIBLE transform (floating-point lifting): the Cohen-
 *     Daubechies-Feauveau 9/7 biorthogonal wavelet, used for lossy coding.
 *
 * Both are implemented with the LIFTING SCHEME, which factors the wavelet
 * filter bank into simple in-place predict/update steps. Lifting is:
 *   - fast (in-place, O(N)),
 *   - exactly invertible (just run the steps backwards with sign flips),
 *   - and, for 5/3, integer-exact.
 *
 * Boundary handling uses WHOLE-SAMPLE SYMMETRIC extension (mirror without
 * repeating the edge sample): ... x2 x1 | x0 x1 x2 ... | xN-2 xN-1 xN-2 ...
 * This matches the JPEG 2000 convention and keeps odd-length signals exact.
 *
 * Data layout: images are stored as a flat Float64Array of length w*h in
 * row-major order. The transform works on a rectangular region described by
 * an `ImagePlane { data, width, height }`.
 *
 * The transform is done IN PLACE on a copy. After a forward step on a
 * dimension of length n, the first ceil(n/2) entries are the low-pass
 * (approximation) coefficients and the remaining floor(n/2) are the
 * high-pass (detail) coefficients. We then recurse on the LL quadrant.
 */

'use strict';

/* ---- CDF 9/7 lifting coefficients (standard JPEG 2000 values) ---- */
const A = -1.586134342059924; // alpha  (predict 1)
const B = -0.052980118572961; // beta   (update 1)
const G = 0.882911075530934; // gamma  (predict 2)
const D = 0.443506852043971; // delta  (update 2)
const K = 1.230174104914001; // scaling K (low-pass), 1/K for high-pass

/* Prompt-supplied rounded constants, kept for reference / documentation:
 *   alpha=-1.586134342, beta=-0.052980118, gamma=0.882911075,
 *   delta=0.443506852,  K=1.230174105
 * The higher-precision values above give a cleaner round-trip. */

/* ------------------------------------------------------------------ *
 *  Symmetric index mirror for whole-sample symmetric extension.
 *  Maps any integer index into the valid range [0, n-1] by reflecting
 *  about the endpoints without repeating the boundary sample.
 * ------------------------------------------------------------------ */
function mirror(i, n) {
  if (n === 1) return 0;
  const period = 2 * (n - 1);
  let k = i % period;
  if (k < 0) k += period;
  return k < n ? k : period - k;
}

/* ================================================================== *
 *  1D 5/3 reversible lifting (integer).
 *
 *  Operates on a strided view of `buf`. Reads n samples at
 *  base + j*stride, writes the de-interleaved low/high result back to the
 *  same locations: lows first, then highs.
 *
 *  Forward lifting (in the de-interleaved even/odd picture):
 *     odd[k]  -= floor((even[k] + even[k+1]) / 2)            (predict)
 *     even[k] += floor((odd[k-1] + odd[k] + 2) / 4)          (update)
 *  where even = samples at positions 0,2,4,... and odd at 1,3,5,...
 * ================================================================== */

function tmpFor(n) {
  return new Float64Array(n);
}

function fwd53_1d(buf, base, stride, n, tmp) {
  if (n < 2) return;
  // Gather into a contiguous scratch buffer for clarity.
  for (let j = 0; j < n; j++) tmp[j] = buf[base + j * stride];

  const nLow = (n + 1) >> 1; // ceil(n/2)  -> even-indexed samples
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
  // De-interleave: lows (even) to front, highs (odd) after.
  for (let k = 0; k < n; k++) {
    const dst = (k & 1) === 0 ? (k >> 1) : nLow + (k >> 1);
    buf[base + dst * stride] = tmp[k];
  }
}

function inv53_1d(buf, base, stride, n, tmp) {
  if (n < 2) return;
  const nLow = (n + 1) >> 1;
  // Re-interleave lows/highs back to even/odd positions.
  for (let k = 0; k < n; k++) {
    const src = (k & 1) === 0 ? (k >> 1) : nLow + (k >> 1);
    tmp[k] = buf[base + src * stride];
  }
  // Undo update.
  for (let k = 0; k < n; k += 2) {
    const left = k - 1 >= 0 ? tmp[k - 1] : tmp[mirror(k - 1, n)];
    const right = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] -= Math.floor((left + right + 2) / 4);
  }
  // Undo predict.
  for (let k = 1; k < n; k += 2) {
    const left = tmp[k - 1];
    const right = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] += Math.floor((left + right) / 2);
  }
  for (let j = 0; j < n; j++) buf[base + j * stride] = tmp[j];
}

/* ================================================================== *
 *  1D CDF 9/7 irreversible lifting (floating point).
 *
 *  Four lifting steps (predict alpha, update beta, predict gamma,
 *  update delta) followed by scaling (lows *= 1/K, highs *= K). The
 *  even samples become low-pass, odd become high-pass.
 * ================================================================== */
function fwd97_1d(buf, base, stride, n, tmp) {
  if (n < 2) return;
  for (let j = 0; j < n; j++) tmp[j] = buf[base + j * stride];
  const nLow = (n + 1) >> 1;

  // Predict 1 (alpha) on odd samples.
  for (let k = 1; k < n; k += 2) {
    const l = tmp[k - 1];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] += A * (l + r);
  }
  // Update 1 (beta) on even samples.
  for (let k = 0; k < n; k += 2) {
    const l = k - 1 >= 0 ? tmp[k - 1] : tmp[mirror(k - 1, n)];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] += B * (l + r);
  }
  // Predict 2 (gamma) on odd samples.
  for (let k = 1; k < n; k += 2) {
    const l = tmp[k - 1];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] += G * (l + r);
  }
  // Update 2 (delta) on even samples.
  for (let k = 0; k < n; k += 2) {
    const l = k - 1 >= 0 ? tmp[k - 1] : tmp[mirror(k - 1, n)];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] += D * (l + r);
  }
  // Scaling.
  for (let k = 0; k < n; k++) {
    tmp[k] *= (k & 1) === 0 ? 1 / K : K;
  }
  // De-interleave.
  for (let k = 0; k < n; k++) {
    const dst = (k & 1) === 0 ? (k >> 1) : nLow + (k >> 1);
    buf[base + dst * stride] = tmp[k];
  }
}

function inv97_1d(buf, base, stride, n, tmp) {
  if (n < 2) return;
  const nLow = (n + 1) >> 1;
  for (let k = 0; k < n; k++) {
    const src = (k & 1) === 0 ? (k >> 1) : nLow + (k >> 1);
    tmp[k] = buf[base + src * stride];
  }
  // Undo scaling.
  for (let k = 0; k < n; k++) {
    tmp[k] *= (k & 1) === 0 ? K : 1 / K;
  }
  // Undo update 2 (delta).
  for (let k = 0; k < n; k += 2) {
    const l = k - 1 >= 0 ? tmp[k - 1] : tmp[mirror(k - 1, n)];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] -= D * (l + r);
  }
  // Undo predict 2 (gamma).
  for (let k = 1; k < n; k += 2) {
    const l = tmp[k - 1];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] -= G * (l + r);
  }
  // Undo update 1 (beta).
  for (let k = 0; k < n; k += 2) {
    const l = k - 1 >= 0 ? tmp[k - 1] : tmp[mirror(k - 1, n)];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] -= B * (l + r);
  }
  // Undo predict 1 (alpha).
  for (let k = 1; k < n; k += 2) {
    const l = tmp[k - 1];
    const r = k + 1 < n ? tmp[k + 1] : tmp[mirror(k + 1, n)];
    tmp[k] -= A * (l + r);
  }
  for (let j = 0; j < n; j++) buf[base + j * stride] = tmp[j];
}

/* ================================================================== *
 *  Kernel dispatch.
 * ================================================================== */
const KERNELS = {
  '5/3': { fwd: fwd53_1d, inv: inv53_1d, integer: true },
  '9/7': { fwd: fwd97_1d, inv: inv97_1d, integer: false },
};

/* ================================================================== *
 *  1D multi-level transform (used by the 1D lifting animator panel).
 *  Returns a record of each level for visualization.
 * ================================================================== */
function dwt1d(signal, kernel, levels) {
  const k = KERNELS[kernel];
  const data = Float64Array.from(signal);
  let n = data.length;
  const tmp = tmpFor(n);
  const stages = [];
  for (let lvl = 0; lvl < levels && n >= 2; lvl++) {
    const before = Float64Array.from(data.subarray(0, n));
    k.fwd(data, 0, 1, n, tmp);
    const nLow = (n + 1) >> 1;
    stages.push({
      level: lvl + 1,
      n,
      nLow,
      before,
      low: Float64Array.from(data.subarray(0, nLow)),
      high: Float64Array.from(data.subarray(nLow, n)),
    });
    n = nLow;
  }
  return { coeffs: data, stages };
}

/* ================================================================== *
 *  2D separable multi-level forward DWT (the Mallat pyramid).
 *
 *  At each level we transform every row then every column over the
 *  current LL region (top-left subregion of size curW x curH), then
 *  shrink the region to the new LL quadrant for the next level.
 *
 *  Returns { data, width, height, levels, kernel, bands } where `bands`
 *  describes the rectangle of every subband for the explorer panel.
 * ================================================================== */
function dwt2dForward(plane, kernel, levels) {
  const { width, height } = plane;
  const k = KERNELS[kernel];
  const data = Float64Array.from(plane.data);
  const tmp = tmpFor(Math.max(width, height));

  let curW = width;
  let curH = height;
  const bands = [];

  for (let lvl = 0; lvl < levels && curW >= 2 && curH >= 2; lvl++) {
    // Transform each row (horizontal pass).
    for (let y = 0; y < curH; y++) {
      k.fwd(data, y * width, 1, curW, tmp);
    }
    // Transform each column (vertical pass).
    for (let x = 0; x < curW; x++) {
      k.fwd(data, x, width, curH, tmp);
    }
    const lowW = (curW + 1) >> 1;
    const lowH = (curH + 1) >> 1;
    // Record the three detail subbands produced at this level.
    bands.push({ level: lvl + 1, type: 'HL', x: lowW, y: 0, w: curW - lowW, h: lowH });
    bands.push({ level: lvl + 1, type: 'LH', x: 0, y: lowH, w: lowW, h: curH - lowH });
    bands.push({ level: lvl + 1, type: 'HH', x: lowW, y: lowH, w: curW - lowW, h: curH - lowH });
    curW = lowW;
    curH = lowH;
  }
  // The final LL residual.
  bands.push({ level: levels, type: 'LL', x: 0, y: 0, w: curW, h: curH });

  return { data, width, height, levels, kernel, bands, llW: curW, llH: curH };
}

/* ================================================================== *
 *  2D inverse: undo the levels in reverse order.
 *  `dec` is the object returned by dwt2dForward (or a modified copy of
 *  its .data, e.g. after thresholding).
 * ================================================================== */
function dwt2dInverse(dec) {
  const { width, height, levels, kernel } = dec;
  const k = KERNELS[kernel];
  const data = Float64Array.from(dec.data);
  const tmp = tmpFor(Math.max(width, height));

  // Reconstruct the sequence of (curW, curH) used in the forward pass.
  const sizes = [];
  let curW = width;
  let curH = height;
  for (let lvl = 0; lvl < levels && curW >= 2 && curH >= 2; lvl++) {
    sizes.push([curW, curH]);
    curW = (curW + 1) >> 1;
    curH = (curH + 1) >> 1;
  }
  // Invert from the smallest region outward.
  for (let i = sizes.length - 1; i >= 0; i--) {
    const [w, h] = sizes[i];
    // Inverse columns first (mirror of forward order).
    for (let x = 0; x < w; x++) {
      k.inv(data, x, width, h, tmp);
    }
    // Inverse rows.
    for (let y = 0; y < h; y++) {
      k.inv(data, y * width, 1, w, tmp);
    }
  }
  return { data, width, height };
}

/* ================================================================== *
 *  Round-trip self-test. Returns the max absolute reconstruction error
 *  for both kernels on a random plane. Used by the console + a test page.
 * ================================================================== */
function selfTest(width = 37, height = 53, levels = 4) {
  const results = {};
  for (const kernel of ['5/3', '9/7']) {
    const src = new Float64Array(width * height);
    for (let i = 0; i < src.length; i++) {
      // Integer input so 5/3 can be checked for exactness.
      src[i] = Math.floor(Math.random() * 256) - 128;
    }
    const dec = dwt2dForward({ data: src, width, height }, kernel, levels);
    const rec = dwt2dInverse(dec);
    let maxErr = 0;
    for (let i = 0; i < src.length; i++) {
      maxErr = Math.max(maxErr, Math.abs(rec.data[i] - src[i]));
    }
    results[kernel] = maxErr;
  }
  return results;
}

/* Export for both browser (global) and Node (module.exports for tests). */
const DWT = {
  KERNELS,
  mirror,
  dwt1d,
  dwt2dForward,
  dwt2dInverse,
  selfTest,
  coeffs: { A, B, G, D, K },
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = DWT;
}
if (typeof window !== 'undefined') {
  window.DWT = DWT;
}
