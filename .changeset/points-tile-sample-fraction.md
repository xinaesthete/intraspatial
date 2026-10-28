---
"intraspatial": patch
---

`PointsTile` gains an optional `sampleFraction`: the share of the element's points a tile was read from when it is a uniform random sample (a coarser level of a multiscale points element). Absent means every point, so existing tiles are unchanged.
