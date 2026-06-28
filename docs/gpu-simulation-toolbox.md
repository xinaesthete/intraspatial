# GPU simulation toolbox

Status: **building** — design + first primitive (2026-06-28)

The toolbox so far is *analytic*: every primitive maps inputs to one snapshot
(`splat`, `convolve`, `getisOrd`, `fuzzyAdjacency`, `kthNeighborDistance`). This note
opens a third front — **synthesis / simulation** — where state evolves over time:
mechanistic forces between points, chemotaxis, reaction–diffusion. The goal is not to
build all of it now but to fix *how it fits* the existing runtime so the pieces
compose with the analytic primitives rather than living beside them.

## The one new idea: iteration

Every existing op is single-pass. A simulation is a **state** advanced by a repeated
**step**:

```
state₀ → step → state₁ → step → state₂ → …          (download only when you look)
```

The [operation-graph runtime](gpu-resource-sync.md) already gives us exactly the
boundary a time-stepped solver needs:

- A step is an op whose inputs and outputs are the **same resource shapes** (a grid
  pair for fields, a points buffer for particles). Feeding a step's outputs into the
  next step is just another edge.
- The executor's **per-stage submit** *is* the per-step boundary — one submit per
  step, ping-ponging two physical buffers (resource-sync invariant 1: single writer
  per submit, never read-modify-write one buffer). The first primitive
  (`reactionDiffusionStep`) does exactly this internally for its `steps` batch.
- **Boundary-only transfer** (invariant 4): upload the seed once, run N steps
  resident on the GPU, read back only the frame you display. Readback is the slow,
  Dawn-on-Node-fragile op — a simulation is where keeping data resident pays the most.

So a simulation is a small **subgraph** that the host (or a future loop node) pulls
repeatedly, swapping the state handles each frame. No new execution model — just ops
whose output shape equals their input shape.

## Three sub-fronts and how they reuse what exists

### 1. Reaction–diffusion (fields) — ✅ first primitive

Two-species PDE on a grid: `∂U/∂t = Du∇²U + R_U`, `∂V/∂t = Dv∇²V + R_V`. The
Laplacian is a stencil — the same windowing machinery as `convolveSeparable`, though
the 5-point Laplacian is cheapest inlined. Implemented as
`src/gpu/sim/reactionDiffusion.ts` (Gray–Scott), exposed as the
`reactionDiffusionStep` graph op (two grids in, two grids out, `steps` per pull). This
is the validating slice for the iterative shape.

### 2. Mechanistic forces (particles) — → next

Per-point state (position, velocity); each step gathers pairwise/neighbour forces and
integrates. Reuse:

- **Neighbour gathering** — `nnDistance` / `kthNeighborDistance` already iterate every
  point against the cloud (the O(N²) brute force, and the planned uniform-grid index
  is the same structure a force solver wants for short-range cutoffs).
- **Integrator** — a pointwise op (`x += v·dt; v += a·dt`), the particle analogue of
  the field Euler step.

### 3. Chemotaxis / advection (coupling the two)

Agents move up a chemical gradient that they also secrete — the bridge between the two
fronts: `points → splat → (reaction-diffusion field) → ∇field → force on points`.
Every arrow there is already a primitive or a near-term one (`splatDensity` for
secretion, a `gradient` grid op, the force/integrator above). This is where the
simulation front and the analytic front close a loop.

## What stays on the CPU

Global reductions that gate a step (total mass, convergence checks) are cheap and
sequential — read back a scalar at a frame boundary, not every step. Adaptive
time-stepping control likewise lives on the host.

## Constraints (inherited)

- **No `Math.random` in kernels** — counter-based RNG (PCG/Philox) seeded per step for
  stochastic forces / noise; the CPU seed helpers stay reproducible.
- **No f32 atomics** — particle→grid scatter uses the additive-render splat path, not
  atomic accumulation (same decision as `splatDensity`).
- **Integer index arithmetic** — `i / w` transpiles to *float* division in TGSL; wrap
  grid-row math in `d.u32(...)` (learned building `reactionDiffusionStep`).
- **Dawn-on-Node readback ceiling (~512²)** — validate small in Node; large/animated
  runs are for the browser composer.

## Where to take it next

1. `gradient` grid op (central differences) — unlocks chemotaxis.
2. Particle integrator + a pairwise-force step (spring/Lennard-Jones/repulsion).
3. A composer demo: seed → reaction-diffusion subgraph pulled per frame, previewed as
   an animated grid.

See also [`gpu-resource-sync.md`](gpu-resource-sync.md) (the runtime + invariants),
[`gpu-spatial-analysis-toolbox.md`](gpu-spatial-analysis-toolbox.md) (points/grid
handles), and the concept page `docs-site/.../concepts/operation-graphs.md`.
