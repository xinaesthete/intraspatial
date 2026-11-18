---
"intraspatial": minor
---

Add `normaliseRings(shape, fillRule)` and a `fillRule` option on `polygonDomain` (#58). Crossing or overlapping rings, such as a self-intersecting lasso, are rewritten as simple rings covering the same region under `"evenodd"` (a figure-eight becomes two rings touching at the crossing; a loop drawn twice is outside) or `"nonzero"` (a point is inside unless the rings' windings round it sum to zero, so same-way overlaps merge, while one ring each way over the same area cancels to outside). Without `fillRule`, `polygonDomain` stays strict. Bad rings throw the same errors as before; a region that is empty under the rule throws too.

Also fixes the strict path: a hole whose every vertex touches its outer ring was sometimes read as an island; two nearly collinear edges lying end to end could be reported as crossing through rounding; and ring areas far from the origin lost precision (a unit square at 2²⁷ read as zero area).
