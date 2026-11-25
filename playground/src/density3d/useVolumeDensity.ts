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

export interface DensityFieldsOptions<T> {
  readonly device: GPUDevice | null;
  /** What the points were read from. A field made from one source is never shown for another. */
  readonly source: object | null;
  /** 4 f32 per point, x y z first. */
  readonly points: GPUBuffer | null;
  readonly n: number;
  /** One f32 per point for each field wanted; null for a slot that is not in use. */
  readonly weights: readonly [GPUBuffer | null, GPUBuffer | null, GPUBuffer | null];
  readonly box: FieldBox | null;
  /** How far a point's influence reaches. */
  readonly radius: number;
  /** Anything the caller needs to read a field with, such as what each slot's level is measured
   *  against. Handed back with the fields it was made with, so a field still on screen while newer
   *  ones are worked out is read with its own inputs. A new tag asks for new fields. */
  readonly tag: T;
}

export interface DensityFields<T> {
  /** One per weight slot, in the same order. */
  readonly fields: readonly (GPUBuffer | null)[];
  readonly box: FieldBox;
  /** The `tag` these fields were made with. */
  readonly tag: T;
  /** Time from asking for the fields to the device saying they are done. */
  readonly ms: number;
}

/** The fields for the latest inputs, or while those are being worked out, the last ones done for
 *  the same device and points source: a surface stays on screen through a slider drag instead of
 *  blinking out. `pending` says newer fields are on their way. A result's buffers are freed once a
 *  newer one replaces it; a job that fails frees its own and leaves the shown result in place. */
export function useDensityFields<T>(o: DensityFieldsOptions<T>): {
  value: DensityFields<T> | null;
  error: string | null;
  pending: boolean;
} {
  const { device, source, points, n, box, radius, tag } = o;
  const [w0, w1, w2] = o.weights;
  // One identity for this set of inputs.
  const key = useMemo(
    () => ({ device, source, points, n, w0, w1, w2, box, radius, tag }),
    [device, source, points, n, w0, w1, w2, box, radius, tag],
  );
  const [shown, setShown] = useState<{ key: typeof key; value: DensityFields<T> } | null>(null);
  const [failed, setFailed] = useState<{ key: object; error: string } | null>(null);

  useEffect(() => {
    const { device, points, n, box, radius } = key;
    if (!device || !points || !box || n === 0) return;
    const weights = [key.w0, key.w1, key.w2];
    let stale = false;
    let adopted = false;
    let fields: (GPUBuffer | null)[] = [];
    let released = false;
    // Later, as work already submitted may still be using them.
    const release = () => {
      if (adopted || released) return;
      released = true;
      const doomed = fields;
      setTimeout(() => doomed.forEach((f) => f?.destroy()), 1000);
    };
    (async () => {
      const splat = await createSplatVolume(device);
      if (stale) return;
      const started = performance.now();
      // Running out of memory or a bad dispatch is reported through error scopes, not thrown.
      device.pushErrorScope("out-of-memory");
      device.pushErrorScope("validation");
      let thrown: unknown;
      try {
        fields = weights.map((w, i) =>
          w ? device.createBuffer({ label: `density3d:field-${i}`, size: volumeBytes(box.dims), usage: GPUBufferUsage.STORAGE }) : null,
        );
        fields.forEach((field, i) => {
          const w = weights[i];
          if (field && w) splat(points, n, field, { ...box, radius, stride: 4, weights: w });
        });
      } catch (e) {
        thrown = e;
      }
      const scoped = await Promise.all([device.popErrorScope(), device.popErrorScope()]);
      if (thrown !== undefined) throw thrown;
      const gpuError = scoped.find((e) => e !== null);
      if (gpuError) throw new Error(`density fields: ${gpuError.message}`);
      await device.queue.onSubmittedWorkDone();
      if (stale) return;
      adopted = true;
      setShown({ key, value: { fields, box, tag: key.tag, ms: performance.now() - started } });
    })().catch((e: unknown) => {
      release();
      if (!stale) setFailed({ key, error: e instanceof Error ? e.message : String(e) });
    });
    return () => {
      stale = true;
      release();
    };
  }, [key]);

  // A shown result's fields are freed when a newer one replaces it, or the page goes. Later, because a
  // frame already on its way may still be drawing them.
  useEffect(() => {
    const fields = shown?.value.fields;
    return () => {
      setTimeout(() => fields?.forEach((f) => f?.destroy()), 1000);
    };
  }, [shown]);

  // Old fields are shown only for the same device and points source, and a non-empty cloud.
  const usable = shown && shown.key.device === device && shown.key.source === source && n > 0 ? shown : null;
  const error = failed?.key === key ? failed.error : null;
  return { value: usable?.value ?? null, error, pending: usable !== null && usable.key !== key && error === null };
}
