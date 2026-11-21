# Bond primitive

Status: **built** (2026-11-20) in `src/gpu/sim/springs/`; see [`spring-model.md`](spring-model.md)
for the whole model. This is the "mechanistic forces" front of
[`gpu-simulation-toolbox.md`](gpu-simulation-toolbox.md), starting from a port of the CSynth spring
table in `organic/TS/springs.ts` and `springsfs.ts`.

Three things changed between this note and the code:

- **One buffer, not two.** A slot is four words: partner and kind packed in word 0, then rest
  length, strength and a distance power. The power is CSynth's and was kept after all.
- **Rods are in.** One per particle, in their own array, applied after the step as in CSynth.
- **`bondForm` is brute force on the GPU.** The grid index is used on the CPU only, so far.

The table is not yet a graph bundle; the ops are plain functions and a `SpringsGpu` object.

**Everything below is the original design, kept for its reasoning.** Where it disagrees with
the three points above or with the code, the code is right. In particular: the table is one
buffer of four-word slots, not `link` and `spring`; the per-spring power is applied; the strain
rule breaks on `|strain|` unless the rule sets `tensionOnly`; and fix, pull, rods and torsion
springs are all built.

## Decision

- **A bond table is a fixed number of slots per particle.** Row `i` holds up to `K` bonds of
  particle `i`. Every bond is stored twice, once under each end. This is CSynth's `topologybuff`.
- **Forces are a gather.** A particle reads only its own row, so there are no atomics and no
  scatter. Storing each bond twice is what makes that possible.
- **A kernel only ever writes its own row.** Breaking and forming are rules that both ends evaluate
  independently and that are guaranteed to agree. CSynth never needed this, because the host edited
  its table.
- **The table is feedback state.** It moves through the graph as one bundle (ADR-0023) on a
  `feedback` edge, beside positions.
- **Fix and pull are per-particle buffers, not slots.** CSynth kept them in the table because it had
  one texture. Rods wait for a constraint pass.

## Layout

Two buffers of `n × K` entries, bundled as `bondTable`:

| part     | element | contents                                                       |
| -------- | ------- | -------------------------------------------------------------- |
| `link`   | `vec2u` | partner index (`0xFFFFFFFF` = empty), kind                     |
| `spring` | `vec2f` | rest length, strength                                          |

- **`K` is fixed per table**, default 8 as in CSynth. Hexagonal packing in 2D needs 6; dense 3D
  packing needs 12.
- **`kind` indexes a small uniform table of strength multipliers.** This is CSynth's `roleforces`.
  It lets one live control scale all matrix–matrix bonds, say, without rewriting the table.
- **Invariant:** `j` is in row `i` exactly when `i` is in row `j`, with identical `kind` and
  `spring`. Every op below preserves it and the tests assert it.

## Ops

| op          | in → out                           | what it does                                         |
| ----------- | ---------------------------------- | ---------------------------------------------------- |
| `bondForce` | points, table → force per point    | sum of spring forces over the row                    |
| `bondBreak` | points, table → table              | clears slots that fail the break rule                |
| `bondForm`  | points, grid index, table → table  | fills free slots from nearby particles               |

A host builder (`addBond`, `removeBond`) writes both ends, and refuses a bond when either row is
full. That alone is CSynth parity.

### `bondForce`

`F = k · m[kind] · (len − rest) · dir`, summed over the row.

- **Stability clamp from CSynth:** the per-step displacement along a bond is capped at half the
  bond's extension, since the other end is moving towards us too.
- CSynth's per-spring power term is left out until something needs it.

### `bondBreak`

Both ends must clear the slot in the same tick without talking to each other. So the rule has to
be a function of values both ends hold identically.

| rule   | test                                             | relaxation time            |
| ------ | ------------------------------------------------ | -------------------------- |
| strain | `len / rest > 1 + ε`                             | emergent from load         |
| rate   | `hash(min(i,j), max(i,j), tick) < dt / τ`        | set directly, `τ`          |

- **Strain** is the Elosegui-Artola model. `len` is the same on both sides because `a − b` and
  `b − a` differ only in sign.
- **Rate** gives a Maxwell material with a known relaxation time. Keying the counter-based RNG on
  the unordered pair is what keeps the two ends in agreement.
- Never breaking is the elastic limit.

### `bondForm`

Two passes, both gathers:

1. **Propose.** Each particle with a free slot walks the grid index and picks its nearest
   neighbour within `rForm` that it is not already bonded to. It writes `proposal[i] = j`.
2. **Accept.** Particle `i` takes the bond only if `proposal[proposal[i]] == i`. It writes `j`
   into its first free slot.

- **The handshake is symmetric by construction.** A particle only proposes when it has a free slot,
  so mutual proposals imply both rows have room.
- **At most one new bond per particle per tick.** That is a re-formation rate, which the model
  wants as a parameter anyway.
- **Rest length on forming is the current distance.** The new bond carries no stress, and that is
  how turnover relaxes the material. A fixed rest length is the alternative, for bonds that should
  pull.
- **Ties** (two neighbours at equal distance) break on the lower index, on both sides.

Explainer candidate: the propose/accept handshake, and a strained lattice relaxing as bonds turn
over.

## What comes from CSynth, and what does not

| CSynth                              | here                                                    |
| ----------------------------------- | ------------------------------------------------------- |
| `topologybuff`, `MAX_DEFS_PER_PARTICLE` | `bondTable`, `K`                                    |
| `addspring` / `removespring`        | host builder                                            |
| `spring()` in the shader            | `bondForce`                                             |
| `roleforces`                        | `kind` multipliers                                      |
| `pairforces` (pushapart, contact)   | separate pair-force op over the grid index, not this note |
| fix, pull                           | per-particle buffers, later                             |
| rod, torsion springs                | later                                                   |
| partner as a fractional float       | `u32` index                                             |
| `removeAllSprings` leaves the far end dangling | not allowed; the invariant holds             |

## Validation

- **CPU golden first**, in plain TS, as with `forces.ts`. The GPU ops are compared against it.
- **Invariants:** table symmetry after every op; `Σ F = 0`.
- **Stress relaxation.** Strain a bonded lattice, hold the strain, and record stress over time.
  With no breaking the stress is constant. With the rate rule it decays with half-time `τ ln 2`.
  This is the same half-time measurement the paper uses on its gels, so the simulated matrix is
  characterised the way the real one is.

## Slices

1. Table, host builder and `bondForce`, with the CPU golden. Static bonds only.
2. `bondBreak`, both rules, and the stress relaxation test.
3. `bondForm`.
4. Graph ops and a composer demo.

Slice 1 is useful on its own for force-directed layout.

## Open

- **Renumbering.** The grid index leaves particle indices alone, so bonds survive it. Compaction
  after division or death would renumber particles, and every partner index would need remapping.
  This belongs to the note on changing particle counts.
- **Per-particle `K`.** Cells and matrix particles may want different slot counts. One `K` is
  assumed until that costs too much memory.
- **Dawn's dispatch watchdog.** `bondForm`'s propose pass has the same per-point neighbour walk as
  indexed kNN, and inherits its limits.
