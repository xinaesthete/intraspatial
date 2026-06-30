# ADR-0005 — Columnar filters as a `support` facet (mask ⇄ index duality) and sparse columns

Status: **proposed** (2026-06-29)

## Decision

Model a **data filter / selection** as a **fourth, first-class facet** on a field,
orthogonal to the three [ADR-0004](./0004-field-type-model-and-volumetric-splat.md)
established. A field's facets become:

```ts
Field = { domain; axes?; element; support? }

domain  // which rows exist     — the table / point cloud (ADR-0004 Domain)
axes    // which columns/genes   — the OPEN tensor axis, runtime length (ADR-0004)
element // the algebra of a cell — scalar / complex / vec / quaternion (ADR-0004)
support // which rows are ACTIVE — the filter / selection            (NEW)
```

- **`support` is the filter.** `domain` says a row *exists*; `support` says a row is
  *currently selected*. Absent ⇒ all-active, so every existing op is untouched —
  exactly how ADR-0004 made `element` optional and defaulted to `scalar`.
- **`support` has two interchangeable encodings, chosen by scale.** This is the
  ADR-0004 scale-equivariance principle applied again: the *same* selection is one of
  two representations, picked by the backend, not two different concepts.

  1. **Soft dense mask** — one weight per row in `[0,1]`. A **hard** filter is the
     boxcar special case (weight ∈ {0,1}); **soft membership is the general thing.**
     This is the *windowing, not quadrats* philosophy the spatial front already
     adopted — a brush is a window with a falloff, not a crisp box. Masks compose
     **algebraically and elementwise**: `AND = a·b`, `OR = max(a,b)`, `NOT = 1−a`.
     Recomputed every frame as the user brushes; plugs straight into the
     weighted-permutation Monte-Carlo null already planned for the spatial front
     (a soft mask *is* the weight vector). This is the **interactive / small-N**
     representation.
  2. **Compact index list** — the selected row ids, built by **prefix-sum + stream
     compaction**. A filtered op then costs **O(selected), not O(N)**. This is the
     **cluster-scale / sparse** representation.

  These two are the same value under a `materializeSupport` op that converts
  mask ⇄ index. **This dense-mask ⇄ compact-index duality is the same pattern as
  scatter-vs-gather and dense-vs-lazy in [ADR-0004](./0004-field-type-model-and-volumetric-splat.md)**:
  one description, two backends, the executor picks. The compaction kernel
  (prefix-sum + counting-sort) is **the same primitive** as the planned 3D
  uniform-grid spatial index ([ADR-0004](./0004-field-type-model-and-volumetric-splat.md)
  builds the index first) — a second reason to land that primitive next.

### Sparse columns fold into the index encoding

A **sparse gene column** (CSR/CSC `(indices, values)`) is just the compact-index
encoding of a **tensor axis** — the identical machinery, pointed at columns instead of
rows. Row selection and sparse columns are one mechanism viewed along two facets.

Crucially, this stays scoped to the MDV working set: **we never densify the full
~20k-gene matrix.** Typical MDV use is ~10 genes at a time (the bound ADR-0004 already
set on G), so sparsity lives in **column selection (which genes) × row selection (the
filter)**, *not* in a giant resident sparse matrix. The dense small-G tensor from
ADR-0004 is the right resident form for the working set; sparsity is how we *choose*
that set, not how we store it.

### Filters are already expressible in the runtime

This is the elegant payoff: a filter is **just a subgraph that produces a `support`
field**, and the runtime already does everything needed.

- Brushing updates an op **param** → the mask node re-pulls.
- The **content-addressed memo** ([`memo.ts`](../../src/gpu/graph/memo.ts), key =
  `op | params | inputKeys`) invalidates **exactly** the downstream that depends on
  that mask — a param change re-keys that node and its dependents and nothing else,
  which is precisely what a live brush needs.
- The **React Flow composer already draws filter pipelines** — filters are not a new
  node type, just ops whose output port is a `support` field.

So **MDV's "one global filter, all linked views react"** maps directly onto **"one
`support` source fanning out through the DAG."** There is no new mechanism — filters
are ops emitting selection fields.

### MDV / Arrow interop is the payoff

If `support` is exposed as either a **GPU buffer (the mask)** or an **Arrow
boolean / dictionary column (the index)**, filters **round-trip with the viewer
zero-copy** — the same no-readback story as the existing splat-into-FBO render
overlay ([ADR-0003](./0003-use-gpu-tgsl-kernels.md) teardown notes). A selection
brushed in our graph and a selection set in MDV / SpatialData.js / deck.gl are the
same buffer, crossing the boundary without a host copy.

## Context & provenance

The user's framing, building directly on ADR-0004:

- A data filter should be a **first-class facet on fields, orthogonal to** ADR-0004's
  `domain` / `axes` / `element`. `support` = which rows are active.
- The filter has **two interchangeable encodings chosen by scale** — a soft dense mask
  (interactive, small-N, brushed every frame, composes algebraically) and a compact
  index list (cluster-scale, sparse, O(selected)). This is *the same* scale-equivariance
  move as ADR-0004, and the mask⇄index duality is *the same* duality as
  scatter-vs-gather / dense-vs-lazy there.
- **Hard filter = boxcar special case of soft membership** — "windowing, not quadrats",
  the philosophy already adopted on the spatial front; the soft mask is also the weight
  vector for the planned weighted-permutation Monte-Carlo null.
- **Sparse columns are the same index encoding pointed at a tensor axis.** And the
  scoping bound: in MDV we look at ~10 genes at a time, so **never densify the full
  matrix** — sparsity is column-selection × row-selection, not a resident sparse matrix.
- **Filters are already expressible**: a filter is a subgraph producing a `support`
  field; brushing → param change → mask re-pull → memo invalidates exactly the
  dependents. The composer already draws these. MDV's global-filter-fans-out is one
  `support` source through the DAG.

Grounding in the current runtime (mapped this session):

- [`GpuField` / `FieldValue`](../../src/gpu/graph/handle.ts) already carry the optional
  `element` facet ADR-0004 added (absent ⇒ `SCALAR`); `support` is the same shape of
  change — a new optional field property with an all-active default, **not** a
  type-system rewrite. `FieldValue.data` stays a flat typed array; a mask is one extra
  parallel weight array, an index list one extra `u32` array.
- The [`points`](../../src/gpu/graph/graph.ts) source packs `[x0,y0,x1,y1,...]`; a soft
  mask is one weight per point (`n` floats) and a compact index list is the `u32`
  subset of `[0..n)` — both align row-for-row with the existing domain.
- The [`memo`](../../src/gpu/graph/memo.ts) keys on `op | params | inputKeys`, so a mask
  node parameterised by the brush re-keys precisely when the brush moves; the
  fan-out-and-invalidate behaviour a live filter needs is **already the memo's
  behaviour**, not something to add.
- Element inference is opt-in via `OpType.inferElements` (parallel to `inferShapes`) in
  [`graph.ts`](../../src/gpu/graph/graph.ts); a support-propagation rule is the same
  opt-in shape — ops that don't declare one keep the all-active contract.

## Consequences

- A new **optional `support` facet** on [`GpuField` / `FieldValue`](../../src/gpu/graph/handle.ts)
  (absent ⇒ all-active, so existing ops are untouched — mirrors exactly how ADR-0004
  added optional `element`). The two carriers are a dense weight array (mask) or a
  `u32` index array (compact); strides/length derive from `domain` as before.
- Op definitions gain an **optional support-propagation rule** (parallel to
  `inferShapes` / `inferElements`): how an op carries / intersects the `support` of its
  inputs. Ops that omit it keep the all-active contract.
- **Reductions and statistics become weighted** — a sum/mean/variance over a soft mask
  is the weighted form, and over a compact index is the O(selected) form; both fall out
  of the two encodings.
- The **prefix-sum / stream-compaction kernel** (scan + counting-sort) is the shared new
  primitive — the *same* kernel the 3D uniform-grid spatial index needs
  ([ADR-0004](./0004-field-type-model-and-volumetric-splat.md)), so building it serves
  both fronts.
- A **`materializeSupport`** op converts mask ⇄ index (and is the seam where the backend
  chooses the scale-appropriate encoding).
- **MDV / Arrow bridge**: `support` exposed as a GPU mask buffer or an Arrow
  boolean/dictionary column round-trips zero-copy with the viewer.
- **Deferred**: the actual kernel implementations (mask compose, prefix-sum/compaction,
  weighted reductions) and the Arrow bridge itself. This ADR fixes the *facet model*;
  the kernels follow the spatial-index primitive.
- Revisit if a use case needs a genuinely **resident sparse matrix** (large G with most
  genes simultaneously active — then reconsider a stored CSR matrix instead of
  column-selection × row-selection), or if MDV's filter exchange turns out to need a
  representation neither a GPU mask nor an Arrow column covers cleanly.
