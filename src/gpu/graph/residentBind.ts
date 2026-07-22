// Bind pooled raw `GPUBuffer`s into a bind group for a TypeGPU-declared layout.
//
// The kernels here are authored against TypeGPU bind-group layouts, but the Tier-2 pool
// (ADR-0017) hands out raw `GPUBuffer`s — it is backend-agnostic and knows nothing about
// TypeGPU. The obvious bridge is TypeGPU's two-argument `createBuffer(schema, existingBuffer)`
// wrapper form (as at sweptGpu.ts:86, splatDensity.ts:138, backend.node.ts:32).
//
// WE DELIBERATELY DO NOT USE THAT HERE. Wrapping makes the TypeGPU root a second owner of a
// buffer the pool already owns, and both free their buffers at process exit — a double free of
// the same Dawn handle, which is the atexit segfault class ADR-0002/0003 is about. A pool
// recycles buffers indefinitely, so it would mint that hazard on every lease.
//
// Instead we build the `GPUBindGroup` directly from the unwrapped layout and the raw buffers.
// Ownership stays entirely with the pool, and nothing is registered with the root.

import type { Root } from "./backend";

/** A binding: either a raw pooled buffer or a TypeGPU-owned one (params/uniform blocks, which
 *  are module-scoped and long-lived, so wrapping is not a concern for them). */
export type BindEntry = GPUBuffer | { readonly buffer: GPUBuffer };

// `Root["unwrap"]` is overloaded across buffers/layouts/pipelines, so it does not give a single
// usable parameter type here. These two helpers localise the casts rather than sprinkling them
// through each op.
type Unwrap = (x: unknown) => unknown;

function raw(root: Root, b: BindEntry): GPUBuffer {
  // A raw buffer passes straight through; a TypeGPU one is unwrapped to its device handle.
  if (typeof (b as GPUBuffer).size === "number") return b as GPUBuffer;
  return (root.unwrap as unknown as Unwrap)(b) as GPUBuffer;
}

/** Build a bind group for a `tgpu.bindGroupLayout` from raw buffers.
 *
 *  `entries` are positional and must be listed in the same order the layout declares them —
 *  TypeGPU assigns binding indices in declaration order, and this is where that correspondence
 *  is relied on. */
export function rawBindGroup(device: GPUDevice, root: Root, layout: unknown, entries: BindEntry[]): GPUBindGroup {
  return device.createBindGroup({
    layout: (root.unwrap as unknown as Unwrap)(layout) as GPUBindGroupLayout,
    entries: entries.map((e, binding) => ({ binding, resource: { buffer: raw(root, e) } })),
  });
}
