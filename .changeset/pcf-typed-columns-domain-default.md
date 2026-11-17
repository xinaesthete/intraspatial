---
"intraspatial": minor
---

PCF params: typed columns, optional `bbox`, and `edgeCorrected` on by default for a domain.

- `LabelledCells` takes `Coords` (so `Float32Array`/`Float64Array`) for `xs`/`ys` and `Uint8Array`/`Uint16Array`/`Int32Array` for `typeId`, read in place rather than copied.
- `bbox` is optional on `PcfParams` and `PcfMatrixBinnedParams` when `domain` is given (`PcfRegion`).
- **Behaviour change:** `edgeCorrected` now defaults to `true` when a `domain` is given, since a drawn gate's boundary is long relative to its area and the uncorrected estimate reads low. A bare `bbox` still defaults to `false`. Pass `edgeCorrected: false` to restore the old result for a domain.
