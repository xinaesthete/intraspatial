---
"@intraspatial/deck": minor
"intraspatial": minor
---

Chain views on deck's WebGPU path.

- `TubeLayer`: an ordered chain of particles as an extruded tube, or with a `profile` as a protein-style cartoon (ribbons and arrows), eased between profiles. `cartoonProfile` builds the profile from a list of `ChainStructure` spans; `tubeProfile` exports the profile packing.
- `SurfaceLayer`: a lit mesh over particles from an index list.
- `TraceLayer`: the paths particles took, or the sheet a chain sweeps through time, from a `PositionHistory` ring of position snapshots filled by device copies (`traceSlot` finds a snapshot by age).
- `bondFaces` (spring model): the triangles closed by bonds, for `SurfaceLayer`.
