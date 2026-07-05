/*
 * dwtImages.ts — synthetic test-image generators, TS port of `viz/images.js`.
 * Each generator returns an ImagePlane with pixel values in [0, 255], chosen
 * to expose different wavelet behaviours.
 */
import type { ImagePlane } from "./dwt";

export function makePlane(width: number, height: number): ImagePlane {
  return { data: new Float64Array(width * height), width, height };
}

function clamp255(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

function genRadial(width: number, height: number): ImagePlane {
  const p = makePlane(width, height);
  const cx = width / 2,
    cy = height / 2;
  const maxR = Math.hypot(cx, cy);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const r = Math.hypot(x - cx, y - cy) / maxR;
      p.data[y * width + x] = clamp255(255 * (1 - r * r));
    }
  return p;
}

function genCircles(width: number, height: number): ImagePlane {
  const p = makePlane(width, height);
  const cx = width / 2,
    cy = height / 2;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const r = Math.hypot(x - cx, y - cy);
      const ring = Math.floor(r / (Math.min(width, height) / 8)) % 2;
      const disc = r < Math.min(width, height) * 0.18 ? 1 : 0;
      p.data[y * width + x] = disc ? 235 : ring ? 60 : 200;
    }
  return p;
}

function genGrating(width: number, height: number): ImagePlane {
  const p = makePlane(width, height);
  const f = (8 * Math.PI) / width;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const v = Math.sin(f * x) * Math.cos(f * y * 0.7);
      p.data[y * width + x] = clamp255(128 + 110 * v);
    }
  return p;
}

function genNoise(width: number, height: number): ImagePlane {
  const p = makePlane(width, height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const base = 128 + 60 * Math.sin(x * 0.03) * Math.sin(y * 0.04);
      p.data[y * width + x] = clamp255(base + (Math.random() - 0.5) * 90);
    }
  return p;
}

/* A Mandelbrot-slice fractal: rich multi-scale edges across every level. */
function genFractal(width: number, height: number): ImagePlane {
  const p = makePlane(width, height);
  const maxIter = 80;
  const x0 = -2.1,
    x1 = 0.7,
    y0 = -1.2,
    y1 = 1.2;
  for (let py = 0; py < height; py++)
    for (let px = 0; px < width; px++) {
      const cr = x0 + (x1 - x0) * (px / width);
      const ci = y0 + (y1 - y0) * (py / height);
      let zr = 0,
        zi = 0,
        i = 0;
      while (i < maxIter && zr * zr + zi * zi < 4) {
        const t = zr * zr - zi * zi + cr;
        zi = 2 * zr * zi + ci;
        zr = t;
        i++;
      }
      let v: number;
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

export interface SynthImage {
  label: string;
  gen: (w: number, h: number) => ImagePlane;
}

export const SYNTH_IMAGES: Record<string, SynthImage> = {
  fractal: { label: "Mandelbrot slice", gen: genFractal },
  circles: { label: "Rings & edges", gen: genCircles },
  radial: { label: "Radial gradient", gen: genRadial },
  grating: { label: "Sinusoid grating", gen: genGrating },
  noise: { label: "Smooth + noise", gen: genNoise },
};

/* Decode an uploaded <img> to a grayscale ImagePlane, resampled to size×size. */
export function imageElementToPlane(img: HTMLImageElement, size: number): ImagePlane {
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(img, 0, 0, size, size);
  const px = ctx.getImageData(0, 0, size, size).data;
  const p = makePlane(size, size);
  for (let i = 0; i < size * size; i++) {
    p.data[i] = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
  }
  return p;
}
