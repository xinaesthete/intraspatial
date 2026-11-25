// Keeps a 3D density field of the chain's beads up to date on the device, for the iso-surface
// layer to draw as metaballs: every frame the beads are splatted into a cube of voxels around the
// chain (`splatVolume`), straight from the simulation's position buffer.
//
// The box is fitted to the chain from the host's copy of the positions, which is a few frames
// old, so it is fitted loosely and only moved when the chain nears its faces or has shrunk well
// inside it. Between those moments the layer's box is constant and React is not involved.
import { useEffect, useState } from "react";
import { createSplatVolume, volumeBytes } from "../../../src/gpu/spatial/splatVolume";

export interface MetaballOptions {
  readonly device: GPUDevice | null;
  /** The simulation's positions, asked each frame; null when there is none. */
  readonly source: { positions(): GPUBuffer; n: number } | null;
  /** The host's recent copy of the positions, 4 floats per bead; for fitting the box only. */
  readonly hostPositions: () => Float32Array | null;
  readonly enabled: boolean;
  /** Voxels along each edge of the cube. */
  readonly resolution: number;
  /** How far a bead's influence reaches. */
  readonly radius: number;
}

export interface MetaballField {
  readonly field: GPUBuffer;
  readonly dims: readonly [number, number, number];
  readonly origin: readonly [number, number, number];
  readonly voxel: number;
}

type Vec3 = [number, number, number];
interface Box {
  lo: Vec3;
  hi: Vec3;
}

/** The box around the beads, grown by `pad` on every side. */
function fit(p: Float32Array, pad: number): Box {
  const lo: Vec3 = [Infinity, Infinity, Infinity];
  const hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < p.length; i += 4) {
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k]!, p[i + k]! - pad);
      hi[k] = Math.max(hi[k]!, p[i + k]! + pad);
    }
  }
  return { lo, hi };
}

const volume = (b: Box) => (b.hi[0] - b.lo[0]) * (b.hi[1] - b.lo[1]) * (b.hi[2] - b.lo[2]);
const holds = (outer: Box, inner: Box) => [0, 1, 2].every((k) => outer.lo[k]! <= inner.lo[k]! && outer.hi[k]! >= inner.hi[k]!);

export function useMetaballs(o: MetaballOptions): MetaballField | null {
  const { device, source, hostPositions, enabled, resolution, radius } = o;
  const [state, setState] = useState<{ key: object; value: MetaballField } | null>(null);
  const [key, setKey] = useState<object>({});
  const wanted = { device, source, hostPositions, enabled, resolution, radius };
  const same = Object.entries(wanted).every(([k, v]) => (key as Record<string, unknown>)[k] === v);
  if (!same) setKey(wanted);

  useEffect(() => {
    const { device, source, hostPositions, enabled, resolution, radius } = key as typeof wanted;
    if (!device || !source || !enabled) return;
    // The box follows the chain's shape, so a long thin chain is not given a cube mostly of nothing:
    // its longest side gets `resolution` voxels and the others as many as they need at that size.
    const field = device.createBuffer({
      label: "fold:metaballs",
      size: volumeBytes([resolution, resolution, resolution]),
      usage: GPUBufferUsage.STORAGE,
    });
    let stopped = false;
    let raf = 0;
    let box: Box | null = null;
    let current: Omit<MetaballField, "field"> | null = null;
    void createSplatVolume(device).then((splat) => {
      const tick = () => {
        if (stopped) return;
        const p = hostPositions();
        if (p && p.length === source.n * 4) {
          // A bead's reach, and half as much again for where the chain has got to since the copy.
          const need = fit(p, radius * 1.5);
          if (!box || !holds(box, need) || volume(box) > volume(need) * 2.5) {
            const slack = need.hi.map((h, k) => (h - need.lo[k]!) * 0.12);
            box = { lo: need.lo.map((l, k) => l - slack[k]!) as Vec3, hi: need.hi.map((h, k) => h + slack[k]!) as Vec3 };
            const size = box.hi.map((h, k) => h - box!.lo[k]!);
            const voxel = Math.max(...size) / resolution;
            const dims = size.map((e) => Math.min(resolution, Math.max(1, Math.ceil(e / voxel)))) as Vec3;
            current = { dims, origin: box.lo, voxel };
            setState({ key, value: { field, ...current } });
          }
          if (current) splat(source.positions(), source.n, field, { ...current, radius, stride: 4 });
        }
        raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    });
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      // Freed a little later: a frame already on its way may still be drawing it.
      setTimeout(() => field.destroy(), 1000);
    };
  }, [key]);

  return state?.key === key ? state.value : null;
}
