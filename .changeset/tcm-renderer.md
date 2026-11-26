---
"intraspatial": minor
---

TCM render on any device, with outputs the caller owns (`intraspatial/gpu/spatial/tcmRender`).

- `TcmRenderer(device, label)` owns everything it renders with: uniforms, uploads, the mark and Γ textures, and bind groups. So several TCMs coexist, on any number of devices. A render writes, encodes and submits in one synchronous call, so `destroy()` is safe at any time. Pipelines are shared per device. It throws on a device without `float32-blendable`.
- `renderResident({ a, b, rhoB }, options)` renders from points already on the device. A and B are each `{ points, weights, count }`, so both can be weightings of one point set: with 0/1 weights the result is exactly the TCM of the two subsets, and other non-negative weights act as masses. `options.grid` and `options.markGrid` are separate, so a view can render a window of Γ. `options.out` copies Γ (rows `tcmRowFloats(width)` apart) and max |Γ| (as f32 bits) into caller-owned buffers in the same submit.
- `render(a, b, params)`, `renderDensity(xs, ys, options)` and `read(texture)` are the host-array forms. `read` encodes its copy at the call, so later renders don't change what it returns.
- `renderTcm`, `computeTcmRender`, `renderKernelDensity` and `kernelDensityGpu` behave as before. They now take a `device` option, and use one renderer per device and key. `renderTcm` and `renderKernelDensity` return a `FieldTexture`, which has the same shape as before.
