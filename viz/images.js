/*
 * images.js — synthetic test-image generators and grayscale helpers.
 *
 * Every generator returns an ImagePlane: { data: Float64Array, width, height }
 * with pixel values in [0, 255]. These are deliberately chosen to expose
 * different wavelet behaviours:
 *   - radial gradient  -> almost all energy in LL, near-empty detail bands
 *   - circle / edges   -> detail energy concentrated along edges
 *   - grating          -> a single scale lights up one detail level
 *   - noise            -> energy spread across ALL bands (incompressible)
 *   - fractal          -> multi-scale structure across every level
 */
'use strict';

function makePlane(width, height) {
  return { data: new Float64Array(width * height), width, height };
}

function clamp255(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

function genRadial(width, height) {
  const p = makePlane(width, height);
  const cx = width / 2, cy = height / 2;
  const maxR = Math.hypot(cx, cy);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const r = Math.hypot(x - cx, y - cy) / maxR;
      p.data[y * width + x] = clamp255(255 * (1 - r * r));
    }
  return p;
}

function genCircles(width, height) {
  const p = makePlane(width, height);
  const cx = width / 2, cy = height / 2;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const r = Math.hypot(x - cx, y - cy);
      // Concentric hard rings + a filled disc => crisp edges.
      const ring = Math.floor(r / (Math.min(width, height) / 8)) % 2;
      const disc = r < Math.min(width, height) * 0.18 ? 1 : 0;
      p.data[y * width + x] = disc ? 235 : ring ? 60 : 200;
    }
  return p;
}

function genGrating(width, height) {
  const p = makePlane(width, height);
  const f = 8 * Math.PI / width; // diagonal sinusoid
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const v = Math.sin(f * x) * Math.cos(f * y * 0.7);
      p.data[y * width + x] = clamp255(128 + 110 * v);
    }
  return p;
}

function genNoise(width, height) {
  const p = makePlane(width, height);
  // A smooth base + white noise: base compresses, noise does not.
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const base = 128 + 60 * Math.sin(x * 0.03) * Math.sin(y * 0.04);
      p.data[y * width + x] = clamp255(base + (Math.random() - 0.5) * 90);
    }
  return p;
}

/* A Mandelbrot-slice fractal: rich multi-scale edges, a great stress test
 * for energy compaction across every decomposition level. */
function genFractal(width, height) {
  const p = makePlane(width, height);
  const maxIter = 80;
  const x0 = -2.1, x1 = 0.7, y0 = -1.2, y1 = 1.2;
  for (let py = 0; py < height; py++)
    for (let px = 0; px < width; px++) {
      const cr = x0 + (x1 - x0) * (px / width);
      const ci = y0 + (y1 - y0) * (py / height);
      let zr = 0, zi = 0, i = 0;
      while (i < maxIter && zr * zr + zi * zi < 4) {
        const t = zr * zr - zi * zi + cr;
        zi = 2 * zr * zi + ci;
        zr = t;
        i++;
      }
      // Smooth coloring.
      let v;
      if (i >= maxIter) v = 0;
      else {
        const mag = Math.sqrt(zr * zr + zi * zi);
        const sm = i + 1 - Math.log(Math.log(Math.max(mag, 1.0001))) / Math.LN2;
        v = 255 * (0.5 + 0.5 * Math.sin(0.3 * sm));
      }
      p.data[py * width + px] = clamp255(v);
    }
  return p;
}

const SYNTH_IMAGES = {
  radial: { label: 'Radial gradient', gen: genRadial },
  circles: { label: 'Rings & edges', gen: genCircles },
  grating: { label: 'Sinusoid grating', gen: genGrating },
  noise: { label: 'Smooth + noise', gen: genNoise },
  fractal: { label: 'Mandelbrot slice', gen: genFractal },
};

/* Decode an uploaded <img> (already loaded) to a grayscale ImagePlane,
 * resampling to `size` x `size`. */
function imageElementToPlane(img, size) {
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0, size, size);
  const px = ctx.getImageData(0, 0, size, size).data;
  const p = makePlane(size, size);
  for (let i = 0; i < size * size; i++) {
    // Rec. 601 luma.
    p.data[i] = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
  }
  return p;
}

const IMG = { SYNTH_IMAGES, makePlane, imageElementToPlane, clamp255 };
if (typeof window !== 'undefined') window.IMG = IMG;
