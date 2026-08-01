# MDV `Dimension` vs the `support` facet — and the case for a filter *graph*

**Design note, not an ADR** (2026-08-01). Nothing here is in flight, so per
`docs/decisions/README.md` it stays a note until someone starts building it. It exists because
[ADR-0005](decisions/0005-columnar-filters-and-sparse-support.md) asserts that *"MDV's 'one global
filter, all linked views react' maps directly onto 'one `support` source fanning out through the
DAG'"* — and never checked that against MDV's actual API. This checks it.

Sources read: `~/code/www/MDV` — `src/datastore/Dimension.js`, `CategoryDimension.js`,
`DataStore.js`, `filteredIndexWorker.ts`, `src/react/components/SelectionDialogComponent.tsx` and
`SelectionDialogReact.tsx`, and MDV's own
`docs/design/spatial-tables/05-dimension-async-filtering.md`.

**Headline: the claim holds for the representation and fails for the composition.** The two models
agree on more than expected — down to both having already built the mask⇄index duality — and differ
on exactly one thing that matters, which is the thing the filter-graph question is about.

**And the two projects already want the same answer.** MDV's filtering note intends the same DAG this
one argues for (§6), so the open question is not *whether* but *what the operators are* — which
neither document has yet written down.

## 1. What MDV actually does

Grounded in the source rather than in either project's docs.

**A `Dimension` is a byte-per-row mask.** `filterBuffer = new SharedArrayBuffer(parent.size)`,
`filterArray = new Uint8Array(filterBuffer)` (`Dimension.js:7-10`). One full-N byte array per active
filter, in shared memory so workers can read it without a copy.

**The byte is a 4-state enum, not a boolean:**

| value | meaning |
|---|---|
| 0 | passes |
| 1 | excluded by the **local** filter (the brush) |
| 2 | excluded by the **background** filter (a pinned category restriction) |
| 3 | both |

**The DataStore's `filterArray` is a per-row reference count of excluding dimensions**, not a mask.
`_applyStateTransition` does the bookkeeping: `if (++parentFilter[index] === 1) this.parent.filterSize--`.
A row is visible iff its count is 0. This is the crossfilter trick, and it is what makes removing one
chart's filter cheap.

**It also hard-wires conjunction.** A reference count can only accumulate exclusions, so the global
filter is `AND` over every registered dimension, by construction. There is no representation in which
an `OR` or a `NOT` could be written down.

**Evaluation is synchronous, main-thread, and O(N) per filter** — `filterPredicate`,
`filterCategories`, `filterRange/Square/Poly`, `filterValueset` all scan every row. `Dimension.js`
`console.warn`s when one exceeds 100 ms, which tells you this is felt. MDV's own note calls it
*"the main synchronous main-thread hotspot"*.

**The predicate is a JS closure.** `filterPredicate(args)` calls `args.predicate(i)` per row, so the
filter's *definition* is opaque code, not data — it cannot cross into a worker, WASM or the GPU.
MDV's note (§A4) independently identifies reifying filter args as serializable data as the
prerequisite for everything else.

**There is already a declared dependency set.** `filterColumns` is stored on the Dimension and
`reFilterOnDataChanged(columns)` re-runs the filter when any of them change. Primitive, but it is
dependency-tracked invalidation.

**`SelectionDialog` is one filter per column.** `SelectionDialogConfig.filters` is
`Record<string, SelectionDialogFilter | null>` **keyed by column field**
(`SelectionDialogReact.tsx:15-20`), and each column's component binds its own Dimension —
`dim.filter("filterCategories", [column.field], value, true)`. So the UI's whole expressive range is
*a conjunction of at most one predicate per column*. Two different ranges on one column is not
merely unsupported; there is nowhere to put the second one.

## 2. What ADR-0005 proposes

`support` as a fourth facet alongside `domain` / `axes` / `element`; absent ⇒ all-active. Two
encodings of one selection, chosen by scale:

- **soft dense mask**, one weight per row in `[0,1]`; a hard filter is the boxcar case. Composes
  elementwise: `AND = a·b`, `OR = max(a,b)`, `NOT = 1−a`.
- **compact index list** via prefix-sum + stream compaction, so a filtered op costs O(selected).

`materializeSupport` converts between them. A filter is *a subgraph producing a `support` field*, and
the content-addressed memo (`op | params | inputKeys`) invalidates exactly its dependents.

**Status: not landed.** Verified, not assumed — zero occurrences of `support` in
`src/gpu/graph/handle.ts`, and no `materializeSupport`, prefix-sum or stream-compaction anywhere in
`src/`. The ADR index has said so since the 2026-07-23 audit: *"the value lattice is half-built …
`support`, `extent`, `placement` are not [real]"*.

## 3. Side by side

| | MDV `Dimension` | ADR-0005 `support` |
|---|---|---|
| per-row carrier | `Uint8Array` over SAB, 4-state enum | weight array in `[0,1]`, or `u32` index list |
| soft membership | no — states, not weights | yes; hard filter is the boxcar case |
| composition | `AND` only, via parent ref-count | `AND = a·b`, `OR = max`, `NOT = 1−a` |
| composition is | implicit and global (register a Dimension) | explicit and local (a value on an edge) |
| compact index list | **yes** — `getFilteredIndices` → `filteredIndexWorker`, promise-cached | `materializeSupport` |
| filter definition | JS closure (opaque) | op subgraph (data) |
| invalidation | `filterColumns` + `reFilterOnDataChanged` | memo on `op │ params │ inputKeys` |
| evaluation | sync, main thread, O(N) | GPU kernel, O(N) or O(selected) |
| two-level filter | built in (background vs local) | absent |
| implemented? | **yes, in production, on 35M rows** | **no** |

**The last row is the one to keep in view.** One of these models is running on real data at scale and
the other is a paper design. Where they disagree, MDV's version has evidence behind it.

## 4. Where they already agree

More than ADR-0005 assumed, and this is the encouraging part.

- **The representation is the same shape.** A byte-per-row mask in a flat shared buffer *is* the dense
  encoding; ADR-0005's contribution there is only to widen the byte to a weight.
- **MDV already built `materializeSupport`.** `getFilteredIndices` runs a worker
  (`filteredIndexWorker.ts`) that turns the byte array into a compact `Uint32Array`, promise-cached
  and shared across charts — and it is load-bearing for the deck.gl path, which draws only passing
  points. The mask⇄index duality is not a proposal in MDV; it is production code. ADR-0005 should
  cite it as precedent rather than as future work.
- **Both dependency-track.** `filterColumns` and `inputKeys` are the same idea at different fidelities.
- **MDV independently arrived at our kernels.** Its §A1 proposes a per-row GPU predicate modelled on
  `nnDistance.ts`, and §B1 proposes `splatDensityGpu(xs, ys, {weights: passingMask})` for a live
  filtered density field. Those are ADR-0005's kernels, named in someone else's repo, pointed at our
  code. That is a strong signal the shared primitive is real.

## 5. Where they genuinely differ

Four differences; only the first is load-bearing.

1. **Composition algebra.** MDV can express a conjunction and nothing else. This is not a missing
   feature, it is the data structure: a reference count has no room for an `OR`. Everything below
   follows from it.
2. **Soft vs hard membership.** MDV's byte is an enum; a weight would give windowing-with-falloff and
   would double as the weight vector for the weighted-permutation null the spatial front wants. But
   note the enum is doing real work (see 4) that a bare weight would lose.
3. **Where the filter lives.** In MDV a Dimension is owned by a chart and registered on the store, so
   composition is a side effect of registration. Under `support` the selection is a value on an edge,
   so composition is written down.
4. **Predicate as code vs data.** MDV's closure can't leave the main thread. An op subgraph is data
   and can be hashed, memoised, serialised, and shipped to the GPU. MDV's note already wants this.

### The 4-state byte is an inlined two-node graph

Worth pulling out, because it is the best argument in this note.

MDV's `0/1/2/3` encoding is *background-filter* `AND` *local-filter*, flattened into one byte with
hand-written transition rules — `_applyStateTransition`, plus the promotion of `2 → 3` inside
`filterPredicate`, plus `setBackgroundFilter` re-running the local method afterwards, plus
`clearBackgroundFilter` subtracting 2. That is a two-input conjunction implemented as a state machine
because there was nowhere to put a second node.

So MDV *already needed* one level of filter composition, and paid for it in enum arithmetic. `noClear`
is a second symptom: a flag meaning "don't reset this node when the graph is cleared", which is
node-level lifecycle expressed as a boolean on a flat list.

**A filter graph is not a generalisation MDV doesn't need. It is the thing MDV built a special case
of.**

## 6. The filter graph

**Both notes intend the same thing here — a DAG whose nodes are selections and whose edges are set
operations.** Recording how that nearly got lost, because it is a drafting hazard rather than a
disagreement, and because it is evidence for the gap this section is about.

MDV's note opens by promising *"building filter-graph primitives"* (line 4) — the DAG reading. Every
subsequent use of the phrase is anchored to its §B and means **charts derived from a filter**:
"filter-histogram / filter-density graphs" (128), *"a unified 'filter graph' reduction"* collapsing
`binWorker`/`catWorker`/`boxPlotWorker` (138), *"making filtered graphs cheap"* (142), and decisively
*"so filter graphs **(B)** automatically reflect them"* (158), which binds the phrase to the charts
section by cross-reference. A reader who has not been told otherwise — this one — takes the six later
uses over the one earlier one.

The DAG intent *is* in the substance, just never under that name:

- **§A4, reify filter kinds as serializable data.** Filter definitions must be data before they can be
  nodes. This is the prerequisite for a DAG as much as for WASM/GPU.
- **§C1, a first-class "chart-local filter"** distinct from the global one — a second composition
  level, which is the same realisation as §5's "the 4-state byte is an inlined two-node graph".
- **§C4, reify chart-scope filters as Dimensions** — promoting an ad-hoc layer into a real node.

So the two notes agree on direction. What neither states is **the composition algebra** — no operator
set, no nesting, no set operations between selections. That is the actual gap, and it is why "filter
graph" could be read as charts for six paragraphs without anything contradicting it. It is also the
cheapest thing to fix: ADR-0005 already has the operators (`AND = a·b`, `OR = max(a,b)`, `NOT = 1−a`),
and they have never been written down next to MDV's model.

### What a DAG buys that a flat conjunction cannot

- `(A ∧ B) ∨ C` at all.
- Two filters on the same column — inexpressible today, since `filters` is keyed by field.
- **Named intermediate selections** reused by several views, instead of each chart re-deriving.
- **Per-chart scope as a node.** MDV's `useChartScopeFilterPredicate` is *"an entirely separate,
  in-memory, main-thread filter layered on top of the Dimension system"* (its note, §C item 3), and
  §C4 asks for chart-scope filters to be reified as Dimensions. In a DAG that is just a node with one
  consumer — no new concept.
- **The background filter stops being special** — it is an upstream node.
- **Set operations between selections**: "cells in gate A but not gate B" is a diff, not a re-brush.

### The operators, written down — and a defect in ADR-0005's

This is the gap identified above, and writing it out immediately found a problem, which is the
argument for writing it out.

ADR-0005 specifies **`AND = a·b`, `OR = max(a,b)`, `NOT = 1−a`**. Those three do not belong together.
In fuzzy-set terms a conjunction (t-norm) and a disjunction (t-conorm) are *De Morgan dual* under
`n(a) = 1−a` only when `S(a,b) = 1 − T(1−a, 1−b)`. The matched pairs are:

| conjunction | matched disjunction | idempotent? |
|---|---|---|
| `min(a,b)` (Gödel) | `max(a,b)` | **yes** |
| `a·b` (product) | `a + b − ab` (probabilistic sum) | no |
| `max(0, a+b−1)` (Łukasiewicz) | `min(1, a+b)` | no |

ADR-0005 took the conjunction from row 2 and the disjunction from row 1. Two consequences, both
verified numerically rather than argued:

1. **De Morgan fails.** For `a = 0.8, b = 0.6`: `1 − (1−a)(1−b) = 0.92`, but `max(a,b) = 0.8`. So
   `NOT(NOT a AND NOT b)` and `a OR b` are different selections. A user who builds one and a
   simplifier that rewrites it to the other get different answers.
2. **Product `AND` is not idempotent** — `a·a ≠ a` (0.8 → 0.64). **This is the one that matters for a
   DAG.** A diamond — two branches derived from the same upstream gate, recombined downstream — is
   the *normal* shape in a gating tree, and under product the shared ancestor gets multiplied in
   twice. The result then depends on the graph's topology rather than on the set being described,
   and it does so silently: the mask just gets dimmer.

MDV never had to face this because a flat conjunction cannot contain a diamond — every dimension
appears exactly once. **Idempotence is a requirement that only appears when you go from a list to a
graph**, which is a decent sign the DAG is a real change of model and not just nicer syntax.

Note also that **for hard 0/1 masks every pair above coincides** (verified: `a·b == min(a,b)` on
0/1). So this is invisible until masks are soft — and soft membership is the entire reason ADR-0005
widened the byte to a weight.

**Recommended: `min` / `max` / `1−a`.** Idempotent, De Morgan-consistent, associative, and the
cheapest of the three on the GPU. Product is the right choice only when the two masks are
*independent probabilities*; in a gating DAG they are usually nested or correlated, which is exactly
when product is wrong. Set difference is then `A ∧ ¬B = min(a, 1−b)`.

A caveat worth carrying rather than resolving here: `min`/`max` are not *strict* — they ignore the
non-extremal operand, so a soft brush ANDed with a soft window keeps only the tighter one and loses
the gradient of the other. If the weighted-permutation null wants that gradient, product may be
right *for that consumer* even though min is right for the graph. That would be an argument for the
operator being a property of the edge rather than a global constant, and it should be settled with a
real null in hand, not on paper.

### The domain already has this shape

The clinching argument is not architectural. **Flow and mass cytometry gating is a tree**, and IMC
data is cytometry — the COVID dataset in this repo is a 49-channel IMC panel. A gating hierarchy
(live → singlet → CD45⁺ → CD8⁺) is a chain of filters where every intermediate node is meaningful,
nameable, and reused. MDV's flat conjunction can represent the *leaf* of a gating tree and nothing
above it.

So a filter DAG is not abstraction for its own sake. It is the analysis model the users already have
in their heads, and the current model flattens it.

### What it costs

A DAG of soft masks gives up MDV's ref-count trick. Today, removing one dimension's filter is a
decrement per row; under a DAG, changing a node re-evaluates its dependents. That is fine on the GPU —
a full-N elementwise pass is trivial there and the memo bounds it to the affected subgraph — but it
is a real regression if evaluated on the host, and it is the reason the GPU kernel is a prerequisite
rather than an optimisation.

## 7. What this would mean for `cellTable`

The per-type split in `playground/src/datasource/cellTable.ts` is the concrete case that raised this.
It groups rows by `cell_type_id` into N separate clouds, `push()`ing row-by-row on the host, and it is
the eager-materialisation opposite of both models: not a mask, not a DAG, but N physical copies made
at load time. Re-grouping by another column redoes the whole split, and the selections cannot compose.

Under `support` it becomes **one domain plus a categorical selection producing N sibling supports** —
no copies, and the grouping becomes a node you can point at a different column.

One correction while here: the comment there says *"(ADR-0018 — never one merged cloud)"*, but
ADR-0018 says no such thing. What it says is that a points field is per-`(table, region)` because
*placement* is a property of the extracted cloud, so clouds with different affines must not be merged.
That is about coordinate systems, not cell types. **Splitting by `cell_type_id` is not required by
ADR-0018**, and the comment currently makes the design look more decided than it is.

## 8. What to build first, and what to answer first

The blocking primitive is the one ADR-0005 already named: **prefix-sum + stream compaction**. It is
also what ADR-0004's 3D uniform-grid index needs, so one kernel serves both fronts, and MDV's note
wants it too (its §A). Nothing else in this direction is worth starting before it exists.

Open questions, in the order they change the design:

1. **Does the soft mask survive contact with MDV's enum?** The `[0,1]` weight is cleaner and enables
   the weighted null, but the background/local distinction is carrying real product behaviour. Does
   it become two nodes, or a weight plus a flag? Two nodes is the honest answer and the more
   expensive one.
2. **Who owns the graph?** If MDV keeps its DataStore and we keep our op graph, a selection has to
   round-trip. ADR-0005 proposes a GPU buffer or an Arrow boolean/dictionary column; MDV's byte array
   is already a SAB, so the zero-copy story is plausible — but "plausible" is the current evidence
   level and it should be tested with one real selection before anything is designed around it.
3. **Is the DAG authored or inferred?** A gating tree is authored deliberately. The React Flow
   composer in `playground` already draws op graphs, so the UI substrate exists — but a gating UI for
   biologists is not the same artefact as a developer's node editor, and conflating them would be a
   mistake.
4. **What happens to `filterSize`?** Every MDV chart reads it. Under a DAG it is a reduction per node,
   which is cheap on the GPU and a readback on the host — and readback is the thing this repo has
   repeatedly measured as the expensive step.

## Relationship to other documents

- [ADR-0005](decisions/0005-columnar-filters-and-sparse-support.md) — the `support` facet. This note
  is its reality check; the ADR should absorb §4's finding that MDV already ships the index encoding.
- [ADR-0004](decisions/0004-field-type-model-and-volumetric-splat.md) — the facet model `support`
  extends, and the source of the shared scan/compaction requirement.
- MDV's `docs/design/spatial-tables/05-dimension-async-filtering.md` — the same problem from the other
  side, already proposing to reuse this repo's `getDevice()`, the `nnDistance.ts` pipeline idiom, and
  `splatDensityGpu`, and intending the same DAG (§6). Read both before starting either. The two notes
  between them cover the representation, the kernels, the async story and the cleanup — and neither
  writes down the composition algebra, which is the one thing a filter DAG cannot be built without.
- [`fuzzy-tda-and-windowing.md`](fuzzy-tda-and-windowing.md) — the windows-not-quadrats argument,
  which is the same argument as soft masks one front over.
