---
"@intraspatial/deck": minor
---

`TcmLayer`: a topographical correlation map between two cell types (`weightsA`, `weightsB` over one point set), drawn as a diverging field on deck's WebGPU device. Needs `float32-blendable`; without it the layer reports an error and draws nothing.
