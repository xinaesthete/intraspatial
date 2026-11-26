---
"@intraspatial/deck": minor
---

`TcmLayer`: a topographical correlation map between two cell types, on deck's WebGPU device.

- Both types are weightings of one point set (`x`, `y`, `weightsA`, `weightsB`), so a layer per pair shares the coordinates. Props: `radius`, `sigma`, `alpha` (default 5), `kernel` (default `TOPHAT`), and `region`, the area B's expected density is averaged over (default: the points' bounds).
- The layer draws a diverging fill (`colorRange`, transparent near Γ = 0 by default) over ±`colorDomain` (default: max |Γ|), with iso-lines on both sides of 0. `scale: "relative"` takes levels as fractions of max |Γ|.
- Γ is rendered only for the view, plus a margin. The marks are rendered where A cells reach that view, at `markCellsPerRadius` cells per r (default 8).
- Each layer has its own `TcmRenderer`, so any number coexist. Renders are synchronous, so finalizing frees everything at once, and no submit is left using a destroyed buffer. GPU buffers are labelled `<layer id>:…`.
- The device needs `float32-blendable`; without it the layer reports an error and draws nothing.
- `packTcmPoints`, `densityOver`, `planMarks` and `fitSide` are the layer's helpers, exported.
