---
"intraspatial": minor
---

A spring model in `intraspatial/gpu/sim` (the `springs` namespace): a port of CSynth's spring dynamics with bond turnover, adhesion and growth.

- `stepSprings` is one step on the CPU. It has CSynth's bonded springs, rods, fixes and pulls, and its pair forces (pushapart, contact and distance matrices, the Lorentzian contact model, boids), under CSynth's parameter names.
- `setTorsion` and `backbonetorsionspringforce` add torsion springs, ported from FoldSynth: they hold the dihedral angle of four consecutive particles.
- `applyLines`, `lineCells`, `helixLine` and `sheetLines` turn lines drawn on the particle × particle matrix into springs, after FoldSynth's vector tools.
- `breakBonds` and `formBonds` break bonds by strain or at random and re-form them between mutual nearest neighbours. Turnover is what makes a bonded network viscoelastic.
- Per-species `repulsion`, `adhesion`, `mobility` and `motility`, and `divideCells` for growth.
- `createSpringsGpu` runs the step, break and form as compute kernels that match the CPU.
- `projectedShape`, `bondReadout` and `phaseDiagram` measure outline circularity, bond stress, and behaviour over a grid of two parameters.
- `chromatinScenario`, `chainScenario`, `spheroidScenario` and `relaxationScenario` are ready-made setups.
- Spheroid and relaxation scenarios have `config` and `configure(change)`: stiffness, yield, healing, adhesion, motility and division all change on a running scenario without rebuilding it.
