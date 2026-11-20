# Spring model

Status: **built** (2026-11-20), CPU reference and GPU kernels. Code: `src/gpu/sim/springs/`.
Playground: `playground/springs.html`. Bond table design: [`bond-primitive.md`](bond-primitive.md).

A port of CSynth's spring dynamics (`organic/TS/springs.ts`, `springsfs.ts`), plus bond turnover,
adhesion and growth. The additions are what a model of cells in a viscoelastic matrix needs
(Elosegui-Artola et al., _Nature Materials_ 22, 117–127, 2023).

## What is here

| file            | contents                                                                    |
| --------------- | --------------------------------------------------------------------------- |
| `params.ts`     | `SpringParams`: CSynth's names and defaults, then the new terms             |
| `topology.ts`   | bond slot table, rod, fix, pull, torsion; `addSpring`, `removeSpring`       |
| `torsion.ts`    | torsion spring forces and `dihedral`, from FoldSynth                        |
| `lines.ts`      | springs from lines drawn on the pair matrix, from FoldSynth                 |
| `step.ts`       | `stepSprings`: the CPU reference for one step                               |
| `turnover.ts`   | `breakBonds`, `formBonds`                                                   |
| `growth.ts`     | `divideCells`, `killParticle`                                               |
| `readouts.ts`   | outline area and circularity, bond stress, distance maps                    |
| `scenarios.ts`  | chromatin, spheroid in a matrix, stress relaxation test                     |
| `sweep.ts`      | `phaseDiagram`                                                              |
| `springsGpu.ts` | the step, break and form as compute kernels                                 |

In CSynth a force is a velocity increment per step (`vel = vel·use + force`, `pos += vel·springrate`).
That convention is kept, so CSynth parameter values carry over.

## Ported from CSynth

| CSynth                                             | here                                          |
| -------------------------------------------------- | --------------------------------------------- |
| `topologybuff`, `addspring`, `removespring`        | `SpringTopology`, `addSpring`, `removeSpring` |
| `spring()`, `roleforces`, per-spring power         | bonded term of `stepSprings`                  |
| rod, fix, pull, `pullfixdamp`                      | `addRod`, `setFix`, `setPull`                 |
| `pairforcesfull`: local and global pushapart       | pair term, same order, same names             |
| density pushapart, fractal force, backbone force   | same                                          |
| `xyzforce` against `distbuff`                      | same, against `SpringMatrices.dist`           |
| `contactforcesc` against `contactbuff`, `wrongfade`| same, against `SpringMatrices.contact`        |
| Lorentzian contact model (`m_force`)               | same                                          |
| boids (cohesion, alignment, separation)            | same, between all particles, not every eighth |
| viscosity, gravity, sphere container, centring     | same                                          |
| damping types 1 to 4, soft speed cap, half-step fix| same                                          |
| noise                                              | same, on a PCG hash rather than `fract`       |

Torsion springs come from FoldSynth instead (`TorsionSpring.java` in `gold/aafoldgrow`), because
CSynth's shader for them is not in the organic repo. `setTorsion(t, i, angle, stiffness)` holds the
dihedral of particles `i .. i+3`; `backbonetorsionspringforce` gives every backbone quad one. Two
deliberate differences from the Java: exact `atan2` rather than its fast approximation, and an
over-strong clamp that scales to the limit (the Java line has a precedence slip).

Not ported, with the reason:

| CSynth                                      | why not                                                  |
| ------------------------------------------- | -------------------------------------------------------- |
| `gradforce`, `noisefieldforce`              | sampled 3D fields; here those belong to graph fields     |
| `boost`, `pullspringmat`, `pullskelforce`   | interactive editing and skeleton tracking                |
| `minActive`, `maxActive`, `regionBoundary`  | partial chain activation; species cover the second       |
| `springminvel`                              | the shader line normalises to length 1, not the minimum  |
| sampled pairs (`PUSHSAMPLES`)               | `pairRange` and a neighbour grid replace it              |

One CSynth quirk is kept as found: a contact marked missing (≤ −9) zeroes the global terms for
that pair, and `patchval` is assigned there but only the Lorentzian path uses it.

## Matrix lines, from FoldSynth

The particle × particle matrix is an editing surface. A line across it ties every pair it crosses
with a spring (`lineCells`, `applyLines`). Direction is structure:

| line                                    | cells        | makes                          |
| --------------------------------------- | ------------ | ------------------------------ |
| alongside the diagonal (`helixLine`)    | `(i, i + d)` | a helix                        |
| across the diagonal (`sheetLines`)      | `(i, s − i)` | one rung of a sheet            |

- **A line is four numbers.** It stays editable, and could be bred or driven like any parameter.
- **`applyLines` is idempotent.** It takes out what its last call put in, then adds the new set,
  and never replaces a spring of another kind or a backbone neighbour.
- **Short-range lines add torsion springs**, centred on each tied pair, as FoldSynth does.
- Ported: `SSLine.getSpringLocations`, `snap`, `addHelix`, `addSheet`, the auto-torsion rule.
  Not ported: mask rectangles, freehand connect and erase, soft paint, `eliminateExcessSprings`.

The playground's "Draw a fold" setup is the editor: drag to draw, drag ends or body to move,
double-click to remove.

## Added

| term           | what it does                                                                        |
| -------------- | ----------------------------------------------------------------------------------- |
| species        | per-particle type; tables below are species × species                               |
| `repulsion`    | overlapping particles push apart in proportion to the overlap                       |
| `adhesion`     | attraction from touching out to `(1 + adhesionRange)` radii, strongest midway       |
| `mobility`     | per-species multiplier on force: an inverse viscosity                               |
| `motility`     | per-species random kick each step                                                   |
| `breakBonds`   | by strain past a yield, or at random with lifetime `tau`                            |
| `formBonds`    | mutual-nearest handshake; new bonds take their current length                       |
| `divideCells`  | cells divide into spare slots unless their local density is too high                |
| `pairRange`    | evaluate only nearby pairs                                                          |

Break and form decisions compare integer hashes of the unordered pair, so both ends of a bond
agree, and the CPU and GPU make identical choices.

## Changing settings on a running scenario

`spheroidScenario` and `relaxationScenario` return a scenario with `config` and
`configure(change)`. A change applies from the next step, to the particles and bonds already there.

| setting                                              | how it stays live                                   |
| ---------------------------------------------------- | --------------------------------------------------- |
| stiffness, cell–matrix bond strength                 | every bond has strength 1; `roleforces[kind]` scales it |
| repulsion, adhesion, mobility, motility              | fields of `scenario.params`                         |
| yield strain, bond lifetime, healing, division       | rules rebuilt from `config` on each `configure`     |
| cell radius, matrix radius, cell capacity, seed      | not live: they decide which particles exist         |

`scenario.params` can also be replaced directly for anything `SpringParams` holds.

## Checked

- **GPU against CPU.** Positions agree to 2×10⁻⁴ after six steps with every CSynth term on.
  Break and form produce the same table bit for bit.
- **Stress relaxation.** A pinned, stretched lattice holds its stress with no turnover. With
  lifetime τ it halves in τ·ln 2 steps, within 25%.
- **Chromatin.** A random tangle recovers a known curve from its distance matrix (correlation
  above 0.99) and from contacts alone (above 0.98).
- **Table invariant.** Symmetric after every op, including 60 rounds of random turnover.

## What the spheroid scenario does and does not show

It reproduces the paper's main split. With an elastic matrix the tissue grows a little and
stalls, round. With a matrix that yields, it grows about twice as large and is less round.

Five seeds each, 8000 steps, default geometry (61 cells at the start, area 52):

| matrix                                                     | cells      | area       | circularity  |
| ---------------------------------------------------------- | ---------- | ---------- | ------------ |
| elastic (never yields)                                     | 165 to 170 | 99 to 101  | 0.99         |
| soft, yields at 0.1, heals at once                         | 369 to 387 | 216 to 230 | 0.84 to 0.95 |
| stiff, yields at 0.3, heals slowly, motile cells (default) | 292 to 334 | 180 to 207 | 0.70 to 0.94 |

- **Growth is robust.** Every yielding run outgrows every elastic run.
- **Symmetry breaking is not.** It is strongest when damage stays where it happened (a stiff
  matrix that heals slowly, with motile cells), but two of five default runs stayed nearly
  round. One run tells you little; compare several seeds, or use the phase diagram.

Not shown: long branching fingers, or agreement with the paper's numbers. The force laws here
are this project's, not the authors' (their code is `anupamdata/ABM_VE_Matrix_Viscous_Tissue`
on GitHub and has not been read). The model is 2D.

## GPU limits

- **Pairs are brute force.** O(n²) per step, with `pairRange` as a cutoff. Walking the grid
  index instead would change no result.
- **Growth is on the host.** Download, divide, upload.
- **WGSL helpers must not return from inside a loop.** Doing so gave wrong answers on
  Dawn/Metal with no error; the helpers accumulate instead.
- **Slots are bound as `vec4u`.** An empty slot's bit pattern is a NaN, which `f32` need not
  preserve.

## Next

1. Grid-indexed pairs on the GPU.
2. Division on the GPU: match the k-th dividing cell to the k-th free slot with two prefix sums.
3. Graph ops, so the model composes with fields (chemotaxis: splat, gradient, force).
4. Read the authors' code and compare force laws.
5. 3D spheroid.

Explainer candidates: the propose and accept handshake, and a strained lattice relaxing as its
bonds turn over. The playground's stress relaxation setup is most of the second.
