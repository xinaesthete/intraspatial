---
"intraspatial": patch
---

Points tiles can now come from a multiscale element's sampled levels:

- `PointsTile` gains an optional `sampleFraction`, the share of the element's points a sampled tile was read from. Absent means every point, so existing tiles are unchanged.
- `pointsGridAtLevel(bounds, totalRows, level)` builds the grid at one quadtree level, so a loader can cut bigger tiles for a bigger window. `pointsGrid` now delegates to it.
