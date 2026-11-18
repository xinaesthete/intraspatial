---
"intraspatial": patch
---

`polygonDomain` now throws on rings that cross themselves or each other (including collinear overlaps), naming the ring and edge indices. Previously `domainArea` (winding number) and `domainContains` (even–odd) disagreed on such rings, silently biasing ρ_B, edge correction and g. Touching at a vertex is still accepted. Callers passing hand-drawn lassos should repair self-crossing rings upstream before constructing the domain.
