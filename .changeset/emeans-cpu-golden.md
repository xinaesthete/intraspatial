---
"intraspatial": minor
---

k-means by e-means (Bessa et al. 2026, arXiv:2611.09501) in `intraspatial/spatial`. This is the CPU golden for a later GPU port.

- `emeans(data, n, d, k, opts)` runs mini-batch k-means with exponential-moving-average centroid updates. The batch size and step come from the published scaling laws, so the only inputs are Lloyd's. It also has a cosine step schedule, dead-cluster revival by splitting, and a finalisation epoch. It is deterministic per `seed`. It returns centroids, labels, per-cluster counts, the MSE and the `schedule` it ran.
- `lloyd(data, n, d, k, opts)` is the baseline. `emeansSchedule`, `emeansBatch` and `emeansAlpha0` expose the scaling laws. `assignNearest` is the nearest-centroid pass.
- Revival is off by default for d ≤ 3, where the authors found that it degrades results.
