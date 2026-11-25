---
"@intraspatial/deck": minor
---

`PivotOrbitController`: deck's orbit controller, turning about a point the page picks instead of the view's target.

- `getPivot(pointer, viewport)` returns the world point a drag turns about (usually the thing under the pointer); null falls back to deck's own behaviour. `getAnchor` does the same for zoom and pan, so the point under the pointer stays under it at its own depth.
- `nearestPoint(positions, pointer, viewport, options)` finds the point nearest the pointer on screen in a packed position array, nearest the camera on ties.
