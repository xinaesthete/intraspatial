// A Gram result's rasters, kept where a deck layer can draw them.
//
// `gramMatrixGpu` leaves its rasters in ONE pooled device buffer that the next call overwrites — and
// a null-envelope run makes a hundred such calls. A deck layer redraws every frame, whenever deck
// likes, including while the next compute is writing, so it cannot read that buffer. Each result
// therefore gets its own GPU-side copy, taken inside the GPU queue.
//
// Those copies are pooled, and a copy is reused only when no one can be drawing it: when its result
// is older than both the one on screen and the one before it (deck may still be drawing that for a
// frame). Results still in flight, or finished but not yet on screen, are never reused. Counting
// "the result before" only by order breaks: `useAsync` drops results whose inputs changed while they
// ran, but their GPU work has happened, so while a slider moves, superseded results can come and
// go in between the one on screen and its successor.

import { useCallback, useMemo, useRef } from "react";
import { getDevice } from "../../../src/gpu/device";
import type { GramMatrixGpuResult } from "../../../src/gpu/spatial/gramMatrix";

let queue: Promise<unknown> = Promise.resolve();
let generation = 0;

/** Run GPU work one task at a time, page-wide: every Gram, envelope and probe shares the pooled buffers. */
export function onGpu<T>(task: () => Promise<T>): Promise<T> {
  const run = queue.then(task, task);
  queue = run.catch(() => undefined);
  return run;
}

interface Snapshot {
  readonly buffer: GPUBuffer;
  /** The generation whose rasters it holds. */
  gen: number;
}

/**
 * Copy `res`'s rasters into a pooled buffer no one can be drawing — see the note at the top. `shown`
 * holds the generations on screen now and just before. Call inside `onGpu`.
 */
async function snapshot(pool: Snapshot[], shown: readonly number[], res: GramMatrixGpuResult, gen: number): Promise<GPUBuffer> {
  const device = await getDevice();
  const bytes = res.labels.length * res.height * res.resident.rowWords * 4;
  const newest = shown.length ? Math.max(...shown) : Number.NEGATIVE_INFINITY;
  const free = (s: Snapshot) => s.gen < newest && !shown.includes(s.gen);
  let slot = pool.find((s) => free(s) && s.buffer.size >= bytes);
  if (!slot) {
    // A free buffer too small for this result is replaced. Free means undrawn, so destroying is safe.
    const small = pool.findIndex(free);
    if (small >= 0) pool.splice(small, 1)[0]?.buffer.destroy();
    slot = {
      buffer: device.createBuffer({
        size: Math.max(bytes, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: "gram snapshot",
      }),
      gen,
    };
    pool.push(slot);
  }
  slot.gen = gen;
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(res.resident.buffer, 0, slot.buffer, 0, bytes);
  device.queue.submit([enc.finish()]);
  return slot.buffer;
}

export interface GramSnapshots {
  /** Snapshot `res` under a new generation. Call inside `onGpu`, straight after the compute. */
  readonly take: (res: GramMatrixGpuResult) => Promise<{ readonly generation: number; readonly rasters: GPUBuffer }>;
  /** Say which generation is on screen now. Call from an effect on the drawn result's generation. */
  readonly show: (generation: number | undefined) => void;
}

/** A snapshot pool for one page. Both functions are stable. */
export function useGramSnapshots(): GramSnapshots {
  const pool = useRef<Snapshot[]>([]);
  /** Generations on screen: the one before, and the current one. */
  const shown = useRef<number[]>([]);
  const take = useCallback(async (res: GramMatrixGpuResult) => {
    const gen = ++generation;
    return { generation: gen, rasters: await snapshot(pool.current, shown.current, res, gen) };
  }, []);
  const show = useCallback((gen: number | undefined) => {
    if (gen !== undefined && shown.current.at(-1) !== gen) shown.current = [...shown.current.slice(-1), gen];
  }, []);
  return useMemo(() => ({ take, show }), [take, show]);
}
