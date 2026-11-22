// Host arrays onto the device, and 3D density fields made from them there (`splatVolume`), for a
// page that draws the fields with `IsoSurfaceLayer`. Unlike `useMetaballs` nothing here runs every
// frame: a field is computed once when its inputs change and then only drawn.
//
// One position buffer serves every field. What differs between fields is a weight per point: 1 for
// a molecule of this gene and 0 for any other, or a cell's expression of this gene. That is the
// shape a density of cell expression takes too: the cells once, a weight vector per gene.
import { useEffect, useMemo, useState } from "react";
import { createSplatVolume, volumeBytes } from "../../../src/gpu/spatial/splatVolume";

/** `data` in a storage buffer on `device`; null until it is there. A new array is a new buffer. */
export function useGpuBuffer(device: GPUDevice | null, data: Float32Array | Uint32Array | null, label: string): GPUBuffer | null {
  const [state, setState] = useState<{ device: GPUDevice; data: object; buffer: GPUBuffer } | null>(null);
  useEffect(() => {
    if (!device || !data) return;
    const buffer = device.createBuffer({
      label,
      size: Math.max(16, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    setState({ device, data, buffer });
    return () => {
      // Freed a little later: a frame already on its way may still be drawing it.
      setTimeout(() => buffer.destroy(), 1000);
    };
  }, [device, data, label]);
  return state && state.device === device && state.data === data ? state.buffer : null;
}

export interface FieldBox {
  readonly dims: readonly [number, number, number];
  readonly origin: readonly [number, number, number];
  readonly voxel: number;
}

export interface DensityFieldsOptions {
  readonly device: GPUDevice | null;
  /** 4 f32 per point, x y z first. */
  readonly points: GPUBuffer | null;
  readonly n: number;
  /** One f32 per point for each field wanted; null for a slot that is not in use. */
  readonly weights: readonly [GPUBuffer | null, GPUBuffer | null, GPUBuffer | null];
  readonly box: FieldBox | null;
  /** How far a point's influence reaches. */
  readonly radius: number;
}

export interface DensityFields {
  /** One per weight slot, in the same order. */
  readonly fields: readonly (GPUBuffer | null)[];
  readonly box: FieldBox;
  /** Time from asking for the fields to the device saying they are done. */
  readonly ms: number;
}

export function useDensityFields(o: DensityFieldsOptions): { value: DensityFields | null; error: string | null } {
  const { device, points, n, box, radius } = o;
  const [w0, w1, w2] = o.weights;
  // One identity for this set of inputs: a result is shown only while it answers them.
  const key = useMemo(() => ({ device, points, n, w0, w1, w2, box, radius }), [device, points, n, w0, w1, w2, box, radius]);
  const [state, setState] = useState<{ key: object; value: DensityFields | null; error: string | null } | null>(null);

  useEffect(() => {
    const { device, points, n, box, radius } = key;
    if (!device || !points || !box || n === 0) return;
    const weights = [key.w0, key.w1, key.w2];
    let stale = false;
    const fields = weights.map((w, i) =>
      w ? device.createBuffer({ label: `density3d:field-${i}`, size: volumeBytes(box.dims), usage: GPUBufferUsage.STORAGE }) : null,
    );
    createSplatVolume(device)
      .then(async (splat) => {
        if (stale) return;
        const started = performance.now();
        fields.forEach((field, i) => {
          const w = weights[i];
          if (field && w) splat(points, n, field, { ...box, radius, stride: 4, weights: w });
        });
        await device.queue.onSubmittedWorkDone();
        if (!stale) setState({ key, value: { fields, box, ms: performance.now() - started }, error: null });
      })
      .catch((e: unknown) => {
        if (!stale) setState({ key, value: null, error: e instanceof Error ? e.message : String(e) });
      });
    return () => {
      stale = true;
      setTimeout(() => {
        for (const f of fields) f?.destroy();
      }, 1000);
    };
  }, [key]);

  return state?.key === key ? { value: state.value, error: state.error } : { value: null, error: null };
}
