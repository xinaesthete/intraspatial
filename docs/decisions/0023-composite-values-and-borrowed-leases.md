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

`name` tags the concrete bundle type (`"bucketGrid"`), exactly as `opaque` does, so port typing
stays nominal — a `bucketGrid` bundle cannot be wired into a `knn` bundle input — while
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
not `lease`, because copying `pointIds` at millions of points to look at it defeats the point of the
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

### 3. Bundles bridge **part by part** — and the f32-only limit goes

A bundle reaching a non-resident op is bridged one part at a time. That immediately exposed
ADR-0017 stage 1's limit as a user-facing bug rather than a footnote: the index's parts are u32,
`readbackF32` decodes every word as a float, so the bridge refused — and the composer rendered
that refusal as an error the moment you selected the node.

So the limit goes for buffers. `readBackBytes` (`device.ts`) copies to a staging buffer and maps
it, and `downloadBuffer` (`executor.ts`) views the bytes as the value's own dtype. It is a
separate path rather than a second TypeGPU wrapper because every wrapper frees its buffer when
the root is torn down, and the pool recycles buffers — a u32-schema wrapper alongside the cached
f32 one would double-free at exit. Reinterpreting the decoded floats back to bits is not a fix
either: a u32 whose pattern is a signalling NaN can have its payload canonicalised passing through
a JS number.

Still f32-only: **textures**, whose readback goes through the render format. A texture-resident
bundle part throws, naming itself.

### 4. Extract and combine are **factories**, not one dynamic op

`extractOp(bundle, part)` and `combineOp(bundle)` generate statically-typed ops per bundle type
(`bucketGrid.cellOffsets`, `bucketGrid.pointIds`, `bucketGrid.lattice`, `bucketGrid.bundle`). A single generic
`extract` with a runtime part name would need dynamic ports — the palette, `inferShapes` and edge
validation all read ports statically — and would lose per-part typing at build time. A new bundle
type costs a few lines of registration.

### 5. A bundle port declares its members statically

`PortSpec` gains `bundle?: { name, parts }`. A port typed only `kind: "bundle"` tells a reader
nothing, and the part *sizes* are only known after a run — but the part *names* never change, so
they belong on the port. The composer's hover tooltip lists them before the graph has run and adds
each part's shape and dtype after it, and the palette can say what a port carries.

### 6. The composer calls it a **bucket grid**, and the port is `buckets`

`index` is ambiguous twice over in this graph: it reads as "an array index", and the bundle's own
`pointIds` part holds exactly that. The producing op keeps the key `gridIndex` — that is the algorithm's name and ADR-0022's — but
everything else uses the concrete vocabulary `src/spatial/bucketGrid.ts` already established:
the bundle's type tag is `bucketGrid` (it surfaces in errors and in the composer's value summary),
its parts are `cellOffsets` and `pointIds` (renamed from `start`/`items`, which is also what
`BucketGrid` now calls them), `Bucket points into cells` produces `buckets`, `Points per cell`
consumes them, and the part ops read `Bucket grid → cell offsets` / `→ point ids` / `→ lattice`.

## Consequences

- `gridIndex` becomes single-output. ADR-0022's three-port shape is superseded before it had a
  consumer, so nothing migrates but its own tests.
- `pull()` of an index bundle now works: the u32 parts come back as `Uint32Array`, so the composer
  can inspect the node instead of showing a bridge error.
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
