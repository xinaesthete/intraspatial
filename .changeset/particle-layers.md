---
"@intraspatial/deck": minor
"intraspatial": minor
---

Particle views on deck's WebGPU path, drawing straight from a simulation's buffers.

- `ParticleLayer`: spheres (impostors that write depth) from a buffer of positions, coloured by category through a palette. Species and radius come from a per-particle attribute buffer or from separate `radii`; a buffer may be given as a function, asked for at every draw, so a simulation's ping-pong buffers can be bound.
- `BondLayer`: a line for each bond in a slot table (`K` `vec4u` slots per particle), coloured by bond kind.
- `categoricalColors`, `hueRamp`, `packPalette`, `syncPalette`: palettes as the layers read them. `CATEGORY_WGSL` and `BOND_WGSL` are the shader pieces they share.
- `SpringsGpu` exposes `attrs` and `slots` for a renderer to bind (`ATTR_FLOATS`, `ATTR_SPECIES`, `ATTR_RADIUS` give the layout), and `uploadTopology` sends a host edit of the bond table without touching positions. `createSpheroidGpuRunner` takes `hostCurrent: false` to skip the download after each advance, and returns its `gpu` and a `synced` count.
