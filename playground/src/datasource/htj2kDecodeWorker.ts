// HTJ2K decode, off the main thread.
//
// The volume path could not use zarrextra's `enableWorkerChunkDecode()`: its worker bundle has
// `@cornerstonejs/codec-openjph` baked in, and that build cannot decode independent multi-component
// data — it keeps component 0 and replicates it — while a volumetric chunk is exactly that (one
// component per z slice). So `spatialDataVolume` registers its own `openjph-wasm` decoder, and that
// decoder ran on the main thread. Measured in the browser, a level-3 brick (388×297×32) is ~45 ms of
// synchronous wasm, and the viewer streams 19–27 of them per camera move — about a second of blocking
// in 45 ms lumps, each one three dropped frames at 60 Hz.
//
// This is the same decoder, in a worker. The decoded samples come back as a TRANSFERABLE
// ArrayBuffer, so the 7 MB result crosses threads without a copy, and the main thread's remaining
// share of a brick is an upload plus a compute dispatch (~5.6 ms, mostly async).
//
// Note what is NOT done here: the worker returns PIXELS, not wavelet coefficients. The eventual
// arrangement (docs/gpu-resident-loader.md §7) has the worker stop after the block decode and the
// GPU do the IDWT — less data across the boundary and less work in wasm — but that needs our own
// codec, whereas this needs only a `postMessage`.

import { decode } from "openjph-wasm";

export interface DecodeRequest {
  id: number;
  bytes: Uint8Array;
}

export type DecodeResponse =
  | { id: number; ok: true; data: ArrayBuffer; width: number; height: number; components: number; bytesPerSample: number }
  | { id: number; ok: false; error: string };

self.onmessage = async (e: MessageEvent<DecodeRequest>) => {
  const { id, bytes } = e.data;
  try {
    const img = await decode(bytes);
    const view = img.data;
    // `img.data` is a view onto the wasm heap copy openjph-wasm already made for us, so its buffer
    // is ours to transfer — no second copy on the way out.
    const buffer = view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength) as ArrayBuffer;
    const res: DecodeResponse = {
      id,
      ok: true,
      data: buffer,
      width: img.width,
      height: img.height,
      components: img.components,
      bytesPerSample: view.BYTES_PER_ELEMENT,
    };
    (self as unknown as Worker).postMessage(res, [buffer]);
  } catch (err) {
    const res: DecodeResponse = { id, ok: false, error: err instanceof Error ? err.message : String(err) };
    (self as unknown as Worker).postMessage(res);
  }
};
