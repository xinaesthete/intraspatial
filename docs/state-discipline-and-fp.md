# State discipline in the page shells — and the Effect question

Status: **design note** (2026-07-30). Direction, not commitment — with one exception noted below.

Not an ADR, deliberately. Most of what follows is direction (Effect, the React conversion, the
package surface), and `docs/decisions/README.md` is explicit that a record earns the ADR form by
being built. The one part that *has* been built — the `Session` refactor of `playground/src/umapMain.ts`
— is a single-file refactor of a demo page, which is not an architectural decision so much as
paying off a debt. If the pattern spreads to the other page shells, that is when it becomes ADR-0022.

## The provocation

Four bugs of the same shape turned up in the UMAP page inside two sessions:

1. `graph build failed: process is not defined` — a debug hook in a browser-bundled module.
2. `cannot read properties of undefined (reading 'epoch')` — `frame()` awaits `driver.sync()`, a
   dataset swap lands during the await and sets `driver = undefined`, the continuation dereferences it.
3. `Error: This buffer has been destroyed` — the same window, landing a few milliseconds earlier, so
   the swap disposes the buffers the frame is *mid-map* of.
4. `loadStore` released the concurrency guard *before* calling `adopt`, which is what opened the
   window in the first place; `restartWith` never took the guard at all, and left `driver` pointing at
   a disposed object across an `await`.

Only (1) is a slip. (2)–(4) are one defect wearing three hats, and the observation that prompted this
note is fair: that is a lot of the same bug.

## What the bugs actually were

Not unmanaged effects. **A mutable resource freed while an async consumer still held it** — ownership
and cancellation, not purity.

This matters for choosing the remedy, and it is worth recording how it was established rather than
assumed. Hammering the UI could not reproduce (2) or (3) at all; the window is a real buffer map, a
few milliseconds wide. It reproduced deterministically only after `gpuDriver.sync()` was widened with
an artificial 300 ms delay and a single regenerate was fired inside the window. That experiment also
showed the first attempted fix was **incomplete**: a re-entrancy guard stops the *next* frame from
starting but cannot retract a frame already suspended inside a buffer map, so the destroyed-buffer
variant survived it. What fixed it was making the frame's async span an explicit value that a swap
could wait on.

That is the diagnosis the remedy has to answer: not "effects are untracked" but **"a lifetime was
implicit"**.

## The repo already has the discipline it stands accused of lacking

Worth stating plainly, because it narrows the problem enormously:

- **`src/gpu/graph/`** is an explicit, serialisable IR. Edges are *derived* from each node's `inputs`
  map rather than hand-declared; execution is lazy pull of the minimal required subgraph. Its header
  records choosing explicit over closure-based tracing specifically so a canvas could edit it. That is
  a stronger statement about state than most codebases which have adopted an effect system make.
- **`src/spatial/`** is pure functions over typed arrays. `pca.ts` was just split into four
  referentially-transparent steps precisely so a device implementation could substitute them one at a
  time and be diffed against them.
- **`src/gpu/spatial/`** kernels are stateless apart from deliberately pooled buffers.

Every one of the four bugs above is in `playground/src/*Main.ts`. The leak is not a missing paradigm
in the core; it is that the **page shells never got one**. They are imperative `main()` scripts that
grew ten module-level `let`s each, which is fine at 200 lines and a defect generator at 900.

## Why not Effect — for this

Effect has real answers here: `Scope`, `acquireRelease`, fiber interruption. The objection is not that
they are bad, it is that they are answers to a problem you must first have *named*, and naming it is
the whole of the fix. Effect would have given `framePending` a better name. It would not have told me
the frame had an async span.

Against adopting it for the render/page layer specifically:

- **It colours every function it touches.** The substance of this repo is `for` loops over
  `Float32Array` and WGSL template strings. There is no purchase there, and a boundary that runs
  through the middle of the hot path is worse than no boundary.
- **It is an all-or-nothing runtime.** Half a codebase in `Effect` is two idioms and two error models.
- **The measured cost of the alternative is one file.** See below — the fix is a value type and a
  generation counter, and it makes the invalid states unrepresentable rather than merely guarded.

### Where it would earn its place

**The datasource layer**, not the render loop. Zarr over HTTP with retries, partial failures, typed
error channels, cancellation on viewport change — that is Effect's home ground, and today it is
`try/catch` with stringly-typed messages. `playground/src/datasource/umapSource.ts` catches an
arbitrary error and reformats it into a status line; `openStore`/`tryOpenArray` in
`src/datasource/annDataIo.ts` swallow failures and return `undefined`. Those are exactly the places
where a typed error channel and a retry `Schedule` would replace real hand-rolled machinery rather
than decorate a loop.

It is also cleanly separable: one subtree, one boundary, no colouring of the kernels.

**Decision recorded so it is not re-derived:** not adopting Effect now, and not on the basis that FP
is unwarranted — on the basis that the bug class it is being proposed against is an ownership problem
with a much cheaper structural fix. Revisit it as a scoped experiment on the datasource layer, where
the failure modes are the ones it is actually good at.

## What landed instead: the session as a value

`umapMain.ts` held ten independent mutable bindings that had to be mutually consistent — `data`,
`graph`, `knn`, `driver`, `active`, `previous`, `speed`, `stress`, `canFade`, `paletteCache`. That is a
thousand representable states and a handful of valid ones, and **every bug above was an invalid
combination being briefly observable**.

Three changes, no framework:

1. **One immutable `Session`, replaced by assignment.** "A new `data.n` against an old embedding" stops
   being something to guard against and becomes something that cannot be written down. Making illegal
   states unrepresentable is the FP move that pays here, and it is the cheap one. The ten bindings are
   now one, plus a `channels` bag of derived per-point arrays that the session *owns* — mutation into
   a superseded session's channels is harmless because nothing can reach them again, so there is no
   "reset the derived state" step to forget.
2. **Object identity is the generation.** Async work captures the session it started from and compares
   by identity after each await. That replaces `rebuildToken`, which checked staleness *after*
   installing its driver — so a superseded rebuild could clobber a newer one, a latent bug the shape
   removes rather than fixes. On the stale path the work now disposes what it built, which the token
   version leaked.
3. **Swaps are serialised and coalescing** (`scheduleSwap`). A request superseded before it starts is
   dropped outright. This was the fix for something measured, not anticipated: a 60 ms click storm
   previously queued ~166 full k-NN builds, each running to completion before discovering it was
   stale, stalling the page for about 30 seconds. Re-measured after the change, the same 141-click
   storm leaves the page idle **immediately**, with the layout still running.

The re-entrancy guard and the awaited in-flight span stay — they are what makes disposal safe against
a frame suspended in a buffer map — but they now enforce an invariant the type is also expressing,
rather than being the only line of defence. Verified by re-running the deterministic race repro
(widened `sync()`, swaps timed into the window, all four paths) against the refactored disposal
ordering: no errors.

## React

React supplies (1) via `useReducer` and (3) via effect cleanup nearly for free, and (2) via the
standard stale-response ignore flag. Two things worth saying beyond that:

- **StrictMode is a bug detector for this exact class.** Double-invoked effects in development hunt
  precisely the acquire/release asymmetries that produced (3) and (4). That is an argument for the
  conversion independent of any UI benefit.
- **The layout driver must not be React state.** It is a device resource stepped 60×/s with its
  coordinates resident on the GPU; putting them in state re-renders every frame and defeats the entire
  point of the resident buffer (ADR-0017). The honest shape is React owning the session and the
  controls, a `useRef`'d imperative driver owning the animation, and one explicit boundary between
  them. The R3F spike established that WebGPU survives a React tree with a single scissor/viewport
  shim, so the precedent exists.

`umap.html` is the right pilot: the worst offender for state, and it now has verified behaviour to
diff a conversion against.

## Packaging for MDV: hooks before components

There is an attractive wrong turn here. `visx` is already a playground dependency and it would be easy
to start publishing chart components.

**The reusable artefact is the derivation, not the chart.** MDV has its own chart conventions and its
own state layer; shipping React components asks it to adopt our view layer — the highest-friction,
lowest-value part, and the part it can most easily write itself. What it cannot get elsewhere is
`pull(graph, sink)` over a shared `GPUDevice`: GPU-resident spatial statistics that never round-trip
through the host.

So: **one addition** to the three-package split in [`packaging-and-consumers.md`](packaging-and-consumers.md)
— a thin `@intraspatial/react` of *hooks* (`useDevice`, `useGraphPull`, `useResident`), not components.
Charts stay in the playground and the docs site until a second consumer wants the same one twice.

**One constraint to design around rather than discover.** The op graph is async and device-bound;
React hooks are synchronous. And the executor's readback invariant is *one download per pull,
regardless of chain depth* (`readbackBudget.gpu.test.ts`) — which means the cost scales with the number
of **pull sites**, not with graph complexity. A naive `useEffect(() => pull(...))` per component
therefore turns ten charts into ten readbacks per frame, which is the one thing the resident-buffer
design exists to avoid. The bridge has to be `useSyncExternalStore` over **a single scheduler per
device**, coalescing pulls across every subscriber. That is a requirement, not a refinement, and it
will decide whether the hooks package is pleasant or a footgun.

## What would change my mind

Recorded so the decision is falsifiable rather than merely held:

- **On Effect:** if the datasource experiment shows typed errors and `Schedule` genuinely collapsing
  the retry/partial-failure handling in the zarr layer, the argument for widening it gets much
  stronger — and the render loop is then a separate question, not a settled one.
- **On the session pattern:** if it needs materially more than a value type and a generation counter to
  hold up under the React conversion, that is evidence the problem was structural after all and a
  framework is worth its weight.
- **On hooks-before-components:** if MDV or psychogeo asks for the *same* chart twice, that is the
  signal to promote it, and the reluctance above stops applying.
