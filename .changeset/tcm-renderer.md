---
"intraspatial": minor
---

`TcmRenderer` (`intraspatial/gpu/spatial/tcmRender`): the TCM render on any device, owning its outputs, so several TCMs coexist. Renders are synchronous, so `destroy()` is always safe.

- `renderResident` takes A and B as per-point weights over device buffers (one point set can serve both), separate Γ and mark grids, and optional caller-owned buffers for Γ and max |Γ|.
- `renderTcm`, `computeTcmRender`, `renderKernelDensity` and `kernelDensityGpu` are unchanged apart from a new `device` option.
