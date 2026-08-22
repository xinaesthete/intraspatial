# ADR-0023 — Composite field values, and borrowing a resident lease

Status: **proposed** (2026-08-22). Written the day the spike started; the implementation notes at
the bottom say what is built.
Implementation: `src/gpu/graph/handle.ts` (`bundle` shape + `parts`), `src/gpu/graph/executor.ts`
(borrow liveness), `src/gpu/graph/ops/bundleOps.ts` (extract/combine factories),
`src/gpu/graph/ops/gridIndex.ts` (first producer), `src/gpu/graph/ops/cellCounts.ts` (first consumer).

## Context

[ADR-0022](0022-gpu-uniform-grid-index.md)'s graph op emits the index as **three sibling ports** —
`start`, `items`, `lattice` — because the executor tracks resident ownership through `v.buffer`
alone, so buffers hidden inside an opaque `payload` would never be released. That works, and it is
wrong in a way the composer makes obvious:

- **A trio is not a value.** `start` and `items` are both `points`/u32, so nothing prevents wiring
  `start` from one index and `items` from another, or feeding `items` into a consumer's `start`
  input. Both mistakes typecheck, run, and return a plausible wrong neighbourhood.
- **It does not generalise.** kNN (indices + distances), a decimation (values + counts) and the
  `support` facet (a field + its mask — the next slice, note §4) all want the same thing: several
  buffers that are *one* value and must travel together.
- **It previews as nothing.** Three ports, two of them undownloadable u32; there is no port whose
  preview means "the index".

The value model already has a product type at the *sample* level — ADR-0004's element algebra
(complex, vec, quaternion). What is missing is a product at the *field* level.

## Decisions

### 1. A `bundle` shape kind, with named parts

```ts
{ kind: "bundle"; name: string; parts: Readonly<Record<string, Shape>> }   // Shape
parts?: Readonly<Record<string, FieldValue>>                               // FieldValue
```

`name` tags the concrete bundle type (`"gridIndex"`), exactly as `opaque` does, so port typing
stays nominal — a `gridIndex` bundle cannot be wired into a `knn` bundle input — while
`shapesEqual` also compares the parts structurally. Each part is a whole `FieldValue`, so a part
may be resident, host-side, or itself opaque, and the placement/element/role facets already work
per part with no new rules.

`numCells` of a bundle is **0**, like `opaque`: a bundle has no single sample count, and any
consumer that needs one asks a part.

Rejected: a fixed `index` shape kind (does not generalise, and every `switch` on `kind` grows a
case per domain type); buffers inside an `opaque` payload (the ownership hole above); a tuple
without names (the composer needs labels, and positional parts invite exactly the mix-ups this
ADR is removing).

### 2. Extraction **borrows**; the executor tracks the borrow

An extract op returns the part's `FieldValue` as-is — the same `ResidentBuffer`, no copy. It must
not `lease`, because copying `items` at millions of points to look at it defeats the point of the
bundle.

That breaks the current lifetime rule, which is one lease per `(node, port)`, released when the
port's last consumer runs: the bundle's lease would go back to the pool while the extracted value
still points at it. So:

> **A tick never owns a payload it can reach from an input.** When an output payload is identical
> (by object identity) to one reachable from any input, the executor records a **borrow edge**
> `out → in` instead of taking ownership. A port with live borrowers is not released; when a
> borrower dies, its lenders are re-checked.

This is refcounting, expressed as the dependency it actually is, and it reuses machinery that is
already there: `owned` is a list per port (a texture-bridged value legitimately owns two
payloads), and `transferOwnership` already moves a lease when `delay` passes its input through.

Rejected: copy-on-extract (throws away the reason for the bundle); a refcount integer on
`ResidentBuffer` (the pool would have to understand graph liveness, and a leaked increment is
invisible); making extraction a graph-rewrite that never runs (the executor would need to
special-case a whole op class, and a borrow edge is the general answer).

### 3. Bundles do not bridge to the host silently

The host bridge is f32-only (ADR-0017 stage 1). A bundle reaching a non-resident op is bridged
**part by part**, and a part that cannot bridge throws naming the part and its dtype — rather than
returning a bundle with holes in it. `pullResident` returns the bundle whole, which is how a test
and the composer preview see it.

### 4. Extract and combine are **factories**, not one dynamic op

`extractOp(bundle, part)` and `combineOp(bundle)` generate statically-typed ops per bundle type
(`gridIndex.start`, `gridIndex.items`, `gridIndex.lattice`, `gridIndex.bundle`). A single generic
`extract` with a runtime part name would need dynamic ports — the palette, `inferShapes` and edge
validation all read ports statically — and would lose per-part typing at build time. A new bundle
type costs a few lines of registration.

### 5. The composer calls it a **bucket grid**, and the port is `buckets`

`index` is ambiguous twice over in this graph: it reads as "an array index", and the bundle's own
`items` part holds exactly that. The registry keys stay `gridIndex` / `gridIndex.start` (they are
identifiers, and "uniform grid index" is the term in the literature — the ADRs keep it), but every
string a user reads uses the concrete name `src/spatial/bucketGrid.ts` already established:
`Bucket points into cells` produces `buckets`, `Points per cell` consumes them, and the part ops
read `Bucket grid → cell offsets` / `→ point ids` / `→ lattice`.

## Consequences

- `gridIndex` becomes single-output. ADR-0022's three-port shape is superseded before it had a
  consumer, so nothing migrates but its own tests.
- `pull()` of an index bundle still fails, now with a per-part message (`items` is u32). That is
  the same honest limit as before, said better.
- The memo stores bundles by content key like any value; a memoised bundle pins its parts, with
  the same "resident values accumulate until eviction" caveat ADR-0017 already carries.
- Feedback of a bundle is **not** supported in this slice: the ping-pong path adopts a single
  lease. It should be, and the borrow machinery is what makes it possible; out of scope here.
- The `support` facet (note §4) gets its natural carrier: a field bundled with its mask.

## Implementation notes

Built in the spike: the shape kind + `parts`, borrow liveness in the executor, the extract/combine
factories, `gridIndex` as a single `gridIndex` bundle, and `cellCounts` (bundle → grid of per-cell
counts) as the first real consumer — which is also what makes the composer example show something
rather than dangle.
