// PROTOTYPE: masks for the patch-grid page. A PNG (uploaded or fetched) is read as one value per
// pixel, the largest of its red, green and blue, so a greyscale raster reads as its grey level and
// a transparent pixel as 0. With no data to hand, a made-up tissue section stands in.

/** One 8-bit value per pixel, row-major, row 0 at the top. */
export interface MaskImage {
  readonly name: string;
  readonly width: number;
  readonly height: number;
  readonly values: Uint8Array;
}

/** Decode a PNG (or any image the browser reads) to per-pixel values. */
export async function decodeMask(blob: Blob, name: string): Promise<MaskImage> {
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const { width, height } = bitmap;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("No 2D context to read the image with.");
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const values = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < values.length; i++, j += 4) {
    values[i] = rgba[j + 3] === 0 ? 0 : Math.max(rgba[j]!, rgba[j + 1]!, rgba[j + 2]!);
  }
  return { name, width, height, values };
}

export async function fetchMask(url: string): Promise<MaskImage> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return decodeMask(await res.blob(), url.split("/").at(-1) ?? url);
}

/** A deterministic hash in [0, 1), for the made-up section's texture. */
function hash(x: number, y: number): number {
  let h = (x * 374761393 + y * 668265263) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

/**
 * A made-up section, 480 × 1600 px: a long core with ragged edges, a hole and a narrow neck, plus
 * two small separate fragments. The fragments get their own grids, which is the point of showing
 * them. Values are a smooth intensity (1–255) inside, 0 outside.
 */
export function syntheticMask(): MaskImage {
  const width = 480;
  const height = 1600;
  const values = new Uint8Array(width * height);
  const blobs = [
    { cx: 240, cy: 420, rx: 170, ry: 360 },
    { cx: 260, cy: 1080, rx: 150, ry: 300 },
    { cx: 80, cy: 1480, rx: 55, ry: 70 },
    { cx: 430, cy: 60, rx: 35, ry: 35 },
  ];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // Ragged edge: the radius wobbles with angle.
      let inside = 0;
      for (const b of blobs) {
        const dx = (x - b.cx) / b.rx;
        const dy = (y - b.cy) / b.ry;
        const a = Math.atan2(dy, dx);
        const r = 1 + 0.08 * Math.sin(5 * a + b.cx) + 0.05 * Math.sin(11 * a + b.cy);
        inside = Math.max(inside, r * r - (dx * dx + dy * dy));
      }
      // A hole, and the neck between the two main lobes.
      const hole = (x - 210) ** 2 / 50 ** 2 + (y - 360) ** 2 / 80 ** 2 < 1;
      const neck = y > 770 && y < 800 && Math.abs(x - 250) > 40;
      if (inside > 0 && !hole && !neck) values[y * width + x] = 1 + Math.round(254 * Math.min(1, inside * 3) * (0.6 + 0.4 * hash(y, x)));
    }
  }
  return { name: "made-up section", width, height, values };
}
