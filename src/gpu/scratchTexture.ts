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
//
// Slots are kept PER DEVICE. A texture cannot cross devices, and a page can hold several live ones
// (deck.gl never destroys a finalised Deck's device — see `perDevice` in `device.ts`). With one slot
// per key, two devices asking for `splatDensity:accum` in turn would each destroy the other's and
// reallocate on every call.
//
// A caller must not await between taking a scratch texture and submitting the work that uses it.
// Another call for the same key can land in that await, find the slot does not fit, and destroy the
// texture — "Destroyed texture used in a submit", i.e. a blank field. That was measured with four
// concurrent f16 splats at alternating sizes on ONE device: three of the four came back empty. Per-
// device slots do not fix that case; only the ordering does (`halfNarrower` exists for it).

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

const slotsByDevice = new WeakMap<GPUDevice, Map<string, Slot>>();
/** Every device that has slots, held weakly. Release and accounting must reach them all, and a
 *  WeakMap cannot be iterated. A strong list would keep a finished device's textures alive until
 *  someone happened to call `releaseScratchTextures`, which nothing does on a timer. */
const devices = new Set<WeakRef<GPUDevice>>();

function slotsFor(device: GPUDevice): Map<string, Slot> {
  let slots = slotsByDevice.get(device);
  if (!slots) {
    slots = new Map();
    slotsByDevice.set(device, slots);
    devices.add(new WeakRef(device));
  }
  return slots;
}

/** Each live device's slots, dropping the registry entries of devices that have been collected. */
function* allSlots(): Generator<Map<string, Slot>> {
  for (const ref of devices) {
    const device = ref.deref();
    if (!device) {
      devices.delete(ref);
      continue;
    }
    const slots = slotsByDevice.get(device);
    if (slots) yield slots;
  }
}

const bytesPerTexel = (format: GPUTextureFormat): number => (format === "r16float" ? 2 : format === "rgba16float" ? 8 : 4);

function fits(s: Slot, d: ScratchDesc): boolean {
  if (s.format !== d.format || s.usage !== d.usage) return false;
  if (s.width < d.width || s.height < d.height) return false;
  return s.width * s.height <= 2 * d.width * d.height;
}

/** The scratch texture for `key` on `device`, reused where it can be and replaced (with the old one
 *  destroyed) where it cannot. The returned texture may be LARGER than requested, so read it by
 *  position — do not assume its extent is the region you asked for. Submit the work that uses it
 *  before any await (see the header). */
export function scratchTexture(device: GPUDevice, key: string, d: ScratchDesc): GPUTexture {
  const slots = slotsFor(device);
  const got = slots.get(key);
  if (got && fits(got, d)) return got.tex;
  got?.tex.destroy();
  const tex = device.createTexture({
    label: key,
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
 * `keyPrefix` frees one family (`"splatDensity"`, `"gramMatrix"`); omitted, it frees all. Either
 * way it frees them on every device.
 */
export function releaseScratchTextures(keyPrefix?: string): number {
  let bytes = 0;
  for (const slots of allSlots()) {
    for (const [key, s] of [...slots]) {
      if (keyPrefix !== undefined && !key.startsWith(keyPrefix)) continue;
      bytes += s.width * s.height * bytesPerTexel(s.format);
      s.tex.destroy();
      slots.delete(key);
    }
  }
  return bytes;
}

/** Bytes currently held in scratch targets — for a memory overlay, and for the tests that pin
 *  the replacement behaviour. */
export function scratchTextureBytes(): number {
  let bytes = 0;
  for (const slots of allSlots()) for (const s of slots.values()) bytes += s.width * s.height * bytesPerTexel(s.format);
  return bytes;
}
