// Scratch render targets that are reused between calls, and actually freed when replaced.
//
// Several paths keep a transient full-precision target around — the f32 accumulator behind an f16
// density field, the per-channel splat target in `gramMatrix`. They are worth reusing: a density
// field's size changes rarely, and reallocating per call would churn.
//
// What they must not do is what they used to do: drop the old texture on the floor when a
// different size is asked for. GPU memory is invisible to the JS heap — the GPUTexture object is a
// handful of bytes holding a 16 MB allocation — so the collector has no pressure signal and the
// timing of its release is undefined. In a long-lived app that resizes (the transcript page
// re-rasters as you pan), that is a leak that only resolves at the collector's convenience, and
// then as a spike. `destroy()` exists precisely so this is not left to the GC.
//
// Older comments in this repo said destroying mid-process segfaults Dawn-on-Node. It does not:
// `test/destroy-mid-process.gpu.test.ts` destroys hundreds of resources with work after each and
// exits clean. Those crashes were the Dawn **Instance** being collected out from under a live
// device, fixed in `device.ts` on 2026-07-29.
//
// Reuse rule: keep the existing texture when the format and usage match, it is big enough, and it
// is not more than twice the area asked for. Otherwise destroy it and allocate exactly what was
// asked. That bounds the waste at 2x and — unlike growing each axis to the largest ever seen —
// cannot turn a 2048x64 followed by a 64x2048 into a 2048x2048.

export interface ScratchDesc {
  readonly width: number;
  readonly height: number;
  readonly format: GPUTextureFormat;
  readonly usage: number;
}

interface Slot {
  tex: GPUTexture;
  width: number;
  height: number;
  format: GPUTextureFormat;
  usage: number;
}

const slots = new Map<string, Slot>();

const bytesPerTexel = (format: GPUTextureFormat): number => (format === "r16float" ? 2 : format === "rgba16float" ? 8 : 4);

function fits(s: Slot, d: ScratchDesc): boolean {
  if (s.format !== d.format || s.usage !== d.usage) return false;
  if (s.width < d.width || s.height < d.height) return false;
  return s.width * s.height <= 2 * d.width * d.height;
}

/** The scratch texture for `key`, reused where it can be and replaced (with the old one destroyed)
 *  where it cannot. The returned texture may be LARGER than requested, so read it by position —
 *  do not assume its extent is the region you asked for. */
export function scratchTexture(device: GPUDevice, key: string, d: ScratchDesc): GPUTexture {
  const got = slots.get(key);
  if (got && fits(got, d)) return got.tex;
  got?.tex.destroy();
  const tex = device.createTexture({
    size: { width: d.width, height: d.height },
    format: d.format,
    usage: d.usage,
  });
  slots.set(key, { tex, width: d.width, height: d.height, format: d.format, usage: d.usage });
  return tex;
}

/**
 * Free scratch targets, returning the bytes released.
 *
 * Nothing calls this on a timer: when to give the memory back is the application's decision, not
 * a library's — a page that re-rasters every pan wants them kept, one that has finished with a
 * view wants them gone. This is the lever; the policy is the caller's.
 *
 * `keyPrefix` frees one family (`"splatDensity"`, `"gramMatrix"`); omitted, it frees all.
 */
export function releaseScratchTextures(keyPrefix?: string): number {
  let bytes = 0;
  for (const [key, s] of [...slots]) {
    if (keyPrefix !== undefined && !key.startsWith(keyPrefix)) continue;
    bytes += s.width * s.height * bytesPerTexel(s.format);
    s.tex.destroy();
    slots.delete(key);
  }
  return bytes;
}

/** Bytes currently held in scratch targets — for a memory overlay, and for the tests that pin
 *  the replacement behaviour. */
export function scratchTextureBytes(): number {
  let bytes = 0;
  for (const s of slots.values()) bytes += s.width * s.height * bytesPerTexel(s.format);
  return bytes;
}
