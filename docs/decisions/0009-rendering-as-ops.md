# ADR-0009 — Rendering as ops: demoting three.js/TSL to presentation

Status: **proposed / exploratory** (2026-07-05)

## Context

The datasource playground renderer (ADR-0008 §8) is built on **three.js + TSL** node
materials: `WebGPURenderer`, `MeshBasicNodeMaterial`, and TSL (`Fn`/`Loop`/`texture3D`/…) for
the volume raymarch. That was the pragmatic choice — three.js gives us a camera, orbit/transform
controls, a scene graph, transparent sorting, and a depth buffer for free, which let the
Decision-view and the depth-culling work land fast.

Two things have accumulated that make it worth writing down a direction:

1. **TSL is a friction surface, and three.js was never the target.** ADR-0008 explicitly treats
   three.js as *renderer-agnostic core, three.js as one backend* — not the destination. Building
   the volume renderer we hit real TSL wrinkles: the `Fn(([p]) => …)` arg-destructure fails to
   type (it resolves the zero-arg `NodeBuilder` overload), `struct().get()` returns an untyped
   `Node`, helper params typed via `ReturnType<typeof …>` are too narrow — five type errors that
   we cleared only by inlining and casting. None were real bugs, but the type story is poor and
   the node-graph indirection is a second, parallel compute abstraction sitting *beside* our own.

2. **We already have a compute op-graph.** `src/gpu/graph` is a demand-pulled graph of GPU ops
   with `"use gpu"` TGSL kernels (ADR-0003), an executor, per-node memoisation, a device/backend
   seam, and an expression-IR ⇄ graph duality (ADR-0007). The raymarch is *also* just a GPU
   kernel over buffers (the brick atlas + page table) parameterised by a camera. Expressing it in
   our own graph — rather than in three's node graph — would collapse two compute abstractions
   into one and put rendering on the same testable, memoised, device-managed footing as every
   other op.

ADR-0008 already anticipated this: *"render [may be] expressible in the pipeline … as long as
we don't shut off that path for future extension."* This ADR sketches that path.

## Decision (direction, not yet a migration)

Treat **rendering as a (family of) ops in the pull-graph**, with three.js demoted to a thin
**presentation + input** shell. Concretely, the target shape:

```
   … ◀─Tileset─ [Resolve] ◀─Selection─ [Select] ◀─Multiscale─ [Datasource]
        │
   resident GPU buffers (brick atlas + page table)
        │
        ▼
   [ Raymarch ] ◀── camera (pull-time input, per ADR-0008 §1)
        │
   output texture (+ optional depth)
        │
        ▼
   [ Present ]  ── thin three.js / raw-WebGPU blit to the canvas
```

- **`Raymarch` is a compute op.** Inputs: the resident atlas + page-table buffers (already GPU
  resident), the `worldToNorm` matrix, transfer-function params, and the **camera as a pull-time
  graph input** (the mechanism ADR-0008 already introduced). Output: a colour texture, and — where
  we need to compose with other geometry — a depth texture. It is a `"use gpu"` TGSL kernel like
  the DWT/sim ops, memoised by the executor, comparable in tests the way other op outputs are.
- **`Present` is the only three.js (or raw-canvas) surface.** It blits the op's output texture to
  the swapchain. Camera controls, window sizing, and gizmos stay on the three.js/DOM side and feed
  the camera *input*; they do not own the render.
- **The pull-graph stays pure.** `Select` and the transform math are already pure (ADR-0008 §2;
  this repo now does oriented select + a live affine). `Raymarch` is a pure function of (buffers,
  camera, params); only `Present` and `Resolve`'s `Loader` are effectful.

## Why

- **One compute abstraction, not two.** Our ops already handle device, memo, backends, and TGSL.
  Rendering joins them instead of living in three's parallel node system with its own type quirks.
- **Renderer-agnostic for real.** ADR-0008 wants the core free of three.js; a compute `Raymarch` +
  a swappable `Present` (three.js today, deck.gl layer or raw WebGPU later) delivers that.
- **Testability.** A compute raymarch's output texture can be pulled headless and compared, like
  the golden tests for the DWT/sim ops — hard to do through a three.js material.
- **It composes with the render-traits direction.** The trait DSL ⇄ graph ⇄ TSL/CPU IR work wants
  shaders that read sim/data buffers directly with an explicit vertex/transform stage; a raymarch
  op reading buffers is the same shape.

## Consequences / what three.js was doing for us that we must re-provide

Moving off three.js is not free — it was carrying real weight. A compute path must re-solve:

- **Camera + controls + gizmos.** Keep these DOM/three-side, emitting a plain `Camera` (we already
  have `cameraFromThree` and a pure `Camera`). Low risk — this seam exists.
- **Compositing with opaque geometry + depth.** Today the depth-culling leans on three's shared
  depth buffer (`viewportDepthTexture`) and transparent sort. A compute raymarch must read/write a
  shared depth texture explicitly and blend against whatever else is on screen. This is the
  hardest part and the main reason to keep three.js in `Present` initially (interop the depth
  texture) rather than go raw-WebGPU immediately.
- **Scene management for auxiliary geometry** (the probe rod, the image plane, wireframe overlays).
  These stay three.js in `Present`; only the *volume* becomes an op first.

## Migration (incremental, validated like the naive/brick A/B)

Do **not** rip out the three.js renderer. Mirror the pattern we just used for the pass-per-chunk
baseline: build the compute raymarch as a **parallel path** and validate it against the working
three.js TSL version before switching any default.

1. Keep the three.js `VolumeRenderer` (brick-page) as the interactive reference.
2. Add a `Raymarch` compute op that consumes the *same* atlas + page-table buffers and camera, and
   renders to an offscreen texture; `Present` blits it. Compare against the three.js output.
3. Move depth interop into `Present` (shared depth texture) so occlusion parity holds.
4. Once at parity, three.js is optional: `Present` can become a raw-WebGPU blit; deck.gl remains an
   additive layer option (ADR-0008).

## Status / open questions

Exploratory — this records the direction so we don't build *away* from it, not a commitment to
migrate now. Open:

- Does the executor/memo model want a first-class notion of an op whose output is a **texture**
  (vs a `FieldValue`)? Probably yes — a `RenderTarget`/`Texture` op kind.
- How does the **camera pull-time input** thread through the executor's memo keys (re-render on
  camera change, reuse resident buffers)? ADR-0008 §1 posits the mechanism; rendering is its first
  demanding consumer.
- Depth/occlusion interop with three.js during the transition — the concrete blocker to validate
  early.

### Horizon: raster passes as rendering ops, and blend-as-layers (noted 2026-07-06, not decided)

This ADR frames rendering-as-ops around the *compute* raymarch. A complementary direction, raised
while building the SpatialData channel compositor (ADR-0010): **raster passes are also a class of
rendering op** — not compute kernels, but owned GPU raster jobs (vertex-displaced grid + fragment
shading) that could equally live in the pull-graph. The motivating UX is a **Photoshop-style layers
panel** where the choice of per-layer **raster blend operations** (additive / max / alpha-over / …)
composes the output — which is exactly what a **TCM** (MuSpAn topographical correlation map) overlay
needs: a blended raster layer derived from spatial statistics. The first concrete substrate is the
tile channel-composite material (ADR-0010): it is authored now as a plain three.js TSL `NodeMaterial`
with the **blend as a parameter, not hardwired**, and channel settings shaped like viv's — so a
future "raster rendering op" *wraps* it rather than replacing it, and this door stays open at no
cost. **Deliberately not decided now:** whether raster passes become first-class graph ops, the op
signature, and the layers-panel model. Recorded so we don't build away from it.

## References

- ADR-0008 (view-driven datasource; render backends, pull-time camera input)
- ADR-0003 (`"use gpu"` TGSL kernels), ADR-0007 (expression-IR ⇄ graph duality)
- `src/gpu/graph` (executor, memo, ops, backend seam) — the op-graph rendering would join
- The five TSL type errors cleared in `volumeRenderer.ts`, and the brick-page vs pass-per-chunk
  A/B (`naiveVolumeRenderer.ts`) — the parallel-path validation pattern this migration reuses
