// Colours for an integer attribute: a palette is a list of deck colours (0–255), indexed by the
// attribute's value. Alpha 0 means "do not draw this value", which is how a host hides a category.

import { Buffer, type Device } from "@luma.gl/core";
import { categoricalHues, oklchToSrgbMapped, rgbBytes } from "intraspatial/color";
import type { Color } from "./DensityLayer";

export interface CategoricalOpts {
  /** OKLCh lightness and chroma shared by every colour, so no category reads as more important. */
  lightness?: number;
  chroma?: number;
}

/** `n` colours at equal steps of OKLCh hue: equal steps there look equal, unlike HSL. */
export function categoricalColors(n: number, opts: CategoricalOpts = {}): Color[] {
  const { lightness = 0.74, chroma = 0.13 } = opts;
  return categoricalHues(n).map((hue): Color => rgbBytes(oklchToSrgbMapped([lightness, chroma, hue])));
}

/** `n` colours along an arc of OKLCh hue, from `fromDeg` to `toDeg`: a ramp for position along a
 *  chain, which unlike a full circle does not end where it began. */
export function hueRamp(n: number, fromDeg = 20, toDeg = 320, opts: CategoricalOpts = {}): Color[] {
  const { lightness = 0.74, chroma = 0.13 } = opts;
  const rad = (d: number) => (d * Math.PI) / 180;
  return Array.from({ length: Math.max(2, n) }, (_, k): Color => {
    const t = k / (Math.max(2, n) - 1);
    return rgbBytes(oklchToSrgbMapped([lightness, chroma, rad(fromDeg + (toDeg - fromDeg) * t)]));
  });
}

/** A palette as the shaders read it: rgba in 0..1, four floats per stop, at least one stop. */
export function packPalette(colors: readonly Color[]): Float32Array {
  const stops = colors.length > 0 ? colors : ([[255, 255, 255, 255]] as const);
  const out = new Float32Array(stops.length * 4);
  stops.forEach((c, k) => out.set([c[0] / 255, c[1] / 255, c[2] / 255, (c[3] ?? 255) / 255], k * 4));
  return out;
}

/** Keep a layer's palette buffer holding `colors`: rewritten in place while it fits, else replaced. */
export function syncPalette(device: Device, id: string, current: Buffer | undefined, colors: readonly Color[]): Buffer {
  const data = packPalette(colors);
  if (current && current.byteLength >= data.byteLength) {
    current.write(data);
    return current;
  }
  current?.destroy();
  return device.createBuffer({ id, data, usage: Buffer.STORAGE | Buffer.COPY_DST });
}
