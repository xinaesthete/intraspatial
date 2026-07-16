# ADR-0014 — Procedural geometry render contract (depth · distance · G-buffer)

Status: **proposed / exploratory** (2026-07-16)

## Context

Procedural geometry needs to take part in more render passes than the lit-colour one: shadow **depth**
from a light camera (directional/spot), radial **distance** from a point light (omni shadows), a
**depth prepass**, and eventually a **G-buffer** for deferred lighting. [ADR-0013](0013-hybrid-strategy-dispatch-and-progen-parallelism.md)
established that shadow-casting is not a raymarch special case but a general contract, already shipped
for the terrain height-field in the psychogeo prior art. This ADR sketches that contract.

Two constraints shape it. First, **author the surface once** — the same discipline as CPU-golden ==
GPU-kernel ([ADR-0003](0003-use-gpu-tgsl-kernels.md)): if a depth pass re-derives the surface
independently of the lit pass, shadows detach from geometry. Second, the repo is **WebGPU-first**;
three.js is the presentation backend for standard PBR, but geometry and analysis live in our ops. So
the contract must be expressed in **our** terms, with three.js as *one* binding — not shaped by
three.js's material classes.

## Decision

### The contract: a surface, lowered to render roles

Each procedural geometry kind exposes a **surface program** — the one authored definition of where its
surface is and what it's made of, in our shading IR (WGSL-emittable):

```
surface(camera) -> { position, normal, valid }   // world-space, per fragment (or per vertex, meshes)
material         -> { albedo, roughness, metalness, emissive, id }
```

- **meshes (horn/swept, terrain):** `surface` is the **vertex transform** — the procedural
  displacement (`sweptShaderWgsl` / `applyPoint`, or the height-field lookup) that places and orients
  the vertex; `valid` is always true.
- **raymarch (implicit/hybrid):** `surface` is the **march** from the camera — `position` = the hit,
  `normal` = the field gradient, `valid` = hit/miss.

Every render pass is then a **generic function of the surface program + camera + lights**, not
something each kind re-implements:

| role | output | used by |
|------|--------|---------|
| `lit` | shade `material` under the lights at `surface` | colour pass |
| `depth` | clip-space depth of `surface.position` | directional/spot shadows, depth prepass, occlusion |
| `distance` | `length(surface.position − lightPos)` | point-light shadows |
| `gbuffer` | write `position` / `normal` / `material` / `id` to targets | deferred lighting, SSAO, picking |

`id` is the provenance address ([ADR-0012](0012-geometry-provenance-and-pick-to-feature-editing.md)) —
in a deferred pipeline it is simply one more G-buffer channel, so pick-to-feature falls out of the same
pass.

### three.js binding (the near-term, forward path)

The roles map onto three's per-object material slots — this is the concrete `depthMaterial` /
`distanceMaterial` the contract was asked for:

- `lit` → `Mesh*NodeMaterial` with `positionNode` / `normalNode` / `colorNode` from the surface program.
- `depth` → `object.customDepthMaterial` = `MeshDepthNodeMaterial` sharing the **same** `positionNode`
  (mesh) or a march-and-write-depth node (raymarch).
- `distance` → `object.customDistanceMaterial` for point lights.

With `castShadow = true` and those slots set, three runs the shadow passes automatically and the
surface — displaced or marched — is what casts. The only wrinkle is the two output shapes (planar depth
vs radial distance), which is the light type, not extra difficulty.

### Beyond shadows: the G-buffer, and life without three.js

The forward/three.js path bolts a depth (and distance) material onto each object. A **deferred**
pipeline is the same contract read differently: one geometry pass fills a G-buffer from every kind's
`gbuffer` role; one lighting pass applies PBR + all lights + shadows over it. This is where the
contract stops being three.js-shaped:

- **It unifies by construction.** Mesh and raymarch both *just fill the G-buffer*, so the material
  match (ADR-0013 §6) and shadow-casting cease to be separate problems — everything downstream sees
  position/normal/material, whatever produced it.
- **Custom lighting is ours.** Owning the lighting pass means many lights (clustered/tiled), our own
  BRDF and render-traits (the superegg/okLab trait mapping), analysis overlays, and per-fragment
  provenance — none of which three's forward PBR gives. Lighting cost decouples from geometry
  complexity and overdraw, which is what Progen-scale city rendering needs.
- **It is renderer-agnostic.** The G-buffer + lighting passes are plain WebGPU bound through our
  op-graph runtime ([ADR-0009](0009-rendering-as-ops.md) rendering-as-ops, resident buffers). three.js
  can composite the result or be absent entirely. three.js becomes optional presentation; the renderer
  is ours. The surface program is the seam that makes both bindings possible from one definition.

## Consequences

- The seam is `surface()` + `material`; `depth` / `distance` / `gbuffer` / `lit` are generic over it.
  A new kind implements the surface, not four passes; a new backend binds the roles, not each kind.
- **First step (cheap, unblocks shadows now):** implement the `depth` / `distance` roles for the
  existing kinds against three's `customDepthMaterial` / `customDistanceMaterial`. Horn shares its
  vertex transform; terrain ports the psychogeo height-field depth; the raymarch shares its march.
- **Larger step (renderer-owned):** the G-buffer + deferred lighting pass, which also serves the
  non-three.js target and the deep hybrid.
- **Deliberately open:** the exact `SurfaceProgram` type; whether to keep three's shadow-map machinery
  or own it; clustered lighting; when the G-buffer path is worth building.
- **Ties:** [ADR-0003](0003-use-gpu-tgsl-kernels.md) (author once, lower many), [ADR-0009](0009-rendering-as-ops.md)
  (rendering as ops), [ADR-0012](0012-geometry-provenance-and-pick-to-feature-editing.md) (id as a
  G-buffer channel), [ADR-0013](0013-hybrid-strategy-dispatch-and-progen-parallelism.md) (the hybrid it
  lights). Prior art: the psychogeo terrain depth pass (WebGL → WebGPU port).

This generalises from a shadow correction: three's `customDepthMaterial` is the pragmatic near-term, the
renderer-owned G-buffer is the target that also frees us from three.js.
