# Context — evo / Mutator

The ubiquitous language for the evolutionary layer (`src/evo`): Todd & Latham "Mutator"
aesthetic-selection primitives, generic over any op's or subgraph's `ParamSpec[]`. This glossary
covers **only** that layer; the other contexts have their own (see the root
[`CONTEXT-MAP.md`](../../CONTEXT-MAP.md)).

## Language

### The space

**Param spec**:
An op's declaration of one parameter ([`ParamSpec`](../gpu/graph/op.ts)): `name`, `type`,
`default`, `min`/`max`/`step`, `options`, `tags`, `units` (see ADR-0018 decision 5), and, for
evolution, `warp` and `delta`. The op
author controls the trait space by how they declare params: nothing in `src/evo` is per-op.
_Avoid_: gene (this repo also handles real genomic data).

**Trait**:
One evolvable axis derived from a param spec. A NUMBER trait (a bounded number/int, or an enum
over its option indices), an ENABLE trait (a bool), or FIXED (unbounded or single-option: passed
through, never bred).
_Avoid_: gene, dimension (fine informally).

**Trait space**:
The traits of a `ParamSpec[]`, from `traitSpaceFromParams`. `specimenToParams` reads a specimen
out as a graph node's `params`; `paramsToSpecimen` encodes params back.
_Avoid_: genome, genotype space.

**Coord**:
A NUMBER trait's position in [0, 1], what a specimen's `pos` holds. Every operator (mutate, marry,
steer, advance) works on coords, never on param values.
_Avoid_: normalised value (ambiguous about the warp).

**Warp**:
How a coord maps onto a param's [min, max], with SuperCollider's ControlSpec names: `"lin"`
(absent ⇒ this), `"exp"` (evenly in ratio; min and max non-zero and of the same sign), `"amp"`,
`"db"`, or a number (a curve). Read-out is `warpMap(coord, {min, max, warp})`, encoding is
`warpUnmap` ([`src/gpu/graph/warp.ts`](../gpu/graph/warp.ts)). A `freq` of 50–3000 Hz with
`warp: "exp"` reads 387 Hz (the geometric mean) at coord 0.5, so breeding, sliders and MIDI move
evenly in pitch. Declare frequencies in Hz with an exp warp, not as MIDI-valued params. An exp warp
over a range touching or crossing zero falls back to linear, with a console warning.
_Avoid_: scale, curve (a curve is one kind of warp).

**Delta**:
A param spec's `delta`: how far one mutation typically moves the param, in its own units
(organic's gene delta). The trait carries it as `deltaNorm`, in coord units: `delta / (max - min)`
when linear; when warped, the coord distance from the default to `default + delta` (from
`default - delta` if that would leave the range). So an exp param's step is the same *ratio*
everywhere in its range. Number/int params only: an enum's options have no distance between them,
so a delta there is ignored with a warning.
_Avoid_: step (that is `ParamSpec.step`, the quantisation), sigma.

### The individual and its operators

**Specimen**:
One bred individual: a coord per NUMBER trait (`pos`), a velocity through trait space (`vel`), a
bit per ENABLE trait, and its own RNG `seed`. A playhead in trait space.
_Avoid_: genome, individual (fine informally).

**Rate**:
The global mutation strength. `mutate` adds Gaussian noise with σ = `rate × deltaNorm` to each
unlocked NUMBER trait (`deltaNorm` 1 when the param declares no delta, so σ = `rate`, as before
deltas existed), and flips each ENABLE trait with probability `rate`.
_Avoid_: temperature.

**Mutate / marry / steer / advance**:
The operators ([`mutator.ts`](./mutator.ts)): noise on position; per-trait crossover of two
parents' position and velocity; an impulse to velocity; and integrating position by velocity.
All pure functions of `(specimen, …, rng)`, so a lineage replays.

**Locked**:
A trait excluded from mutation and marriage (`withLocked`): the artist's freeze. FIXED traits are
always locked.
