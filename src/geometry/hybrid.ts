// Hybrid decomposition (ADR-0013) — split an implicit model into the part the plane-BSP can mesh and
// the part it can't, so a scene can mesh the planar bulk and raymarch only the organic remainder,
// compositing the two by depth. `bsp.ts` still refuses a non-planar tree; here we pull the planar
// *skeleton* out of one and report where the rest lives.
//
// The load-bearing concept is **`nonPlanarRegions` — the region where the BSP won't work**, and its
// range is exactly: *nothing* (a fully planar model), *finite boxes* (local growths), or *infinite* (a
// noise-displaced half-space, or a smooth-union of two half-spaces whose blended seam runs forever).
// A finite box → a bounded raymarch proxy; an infinite region → the raymarch has no box to hide behind
// and must cover the view. `boundsSdf` (the AABB of the solid, honestly infinite for a half-space) is
// just the helper that computes those regions — a non-planar node declares its reach through it.

import { type Expr, evalExpr } from "./expr";
import type { Sdf } from "./implicit";
import type { Vec3 } from "./superellipsoid";

const val = (e: Expr): number => evalExpr(e, 0, 0);

/** An axis-aligned bounding box. Components may be ±∞ (an unbounded solid, e.g. a half-space). */
export interface AABB {
  min: Vec3;
  max: Vec3;
}

/** The whole of space — the extent of an unbounded solid, and the marker for an unbounded non-planar region. */
export const UNBOUNDED: AABB = { min: [-Infinity, -Infinity, -Infinity], max: [Infinity, Infinity, Infinity] };

const aUnion = (a: AABB, b: AABB): AABB => ({
  min: [Math.min(a.min[0], b.min[0]), Math.min(a.min[1], b.min[1]), Math.min(a.min[2], b.min[2])],
  max: [Math.max(a.max[0], b.max[0]), Math.max(a.max[1], b.max[1]), Math.max(a.max[2], b.max[2])],
});
const aInter = (a: AABB, b: AABB): AABB => ({
  min: [Math.max(a.min[0], b.min[0]), Math.max(a.min[1], b.min[1]), Math.max(a.min[2], b.min[2])],
  max: [Math.min(a.max[0], b.max[0]), Math.min(a.max[1], b.max[1]), Math.min(a.max[2], b.max[2])],
});
const aExpand = (a: AABB, r: number): AABB => ({
  min: [a.min[0] - r, a.min[1] - r, a.min[2] - r],
  max: [a.max[0] + r, a.max[1] + r, a.max[2] + r],
});
const aShift = (a: AABB, t: Vec3): AABB => ({
  min: [a.min[0] + t[0], a.min[1] + t[1], a.min[2] + t[2]],
  max: [a.max[0] + t[0], a.max[1] + t[1], a.max[2] + t[2]],
});
const aScale = (a: AABB, f: number): AABB => ({
  min: [a.min[0] * f, a.min[1] * f, a.min[2] * f],
  max: [a.max[0] * f, a.max[1] * f, a.max[2] * f],
});

/** A box with no positive extent (min > max on some axis) bounds an empty solid / empty region. */
export const aabbEmpty = (a: AABB): boolean => a.min[0] > a.max[0] || a.min[1] > a.max[1] || a.min[2] > a.max[2];
/** All six bounds finite — a box a raymarch proxy can actually be sized to. */
export const aabbFinite = (a: AABB): boolean => a.min.every(Number.isFinite) && a.max.every(Number.isFinite);
export const aabbOverlaps = (a: AABB, b: AABB): boolean =>
  a.min[0] <= b.max[0] &&
  b.min[0] <= a.max[0] &&
  a.min[1] <= b.max[1] &&
  b.min[1] <= a.max[1] &&
  a.min[2] <= b.max[2] &&
  b.min[2] <= a.max[2];

/** A conservative AABB of the solid `{ p : sd(p) ≤ 0 }`, in world space. Bounded primitives give tight
 *  boxes; a bare half-space is **unbounded** and reports `UNBOUNDED` (any intersection then clamps it
 *  back). Non-planar nodes widen by their reach: `displace` by its amplitude, `smoothUnion` by its
 *  blend `k`. This is the helper a non-planar node declares its extent through. */
export function boundsSdf(node: Sdf): AABB {
  switch (node.kind) {
    case "sphere": {
      const r = val(node.radius);
      return { min: [-r, -r, -r], max: [r, r, r] };
    }
    case "box": {
      const h: Vec3 = [val(node.half[0]), val(node.half[1]), val(node.half[2])];
      return { min: [-h[0], -h[1], -h[2]], max: [h[0], h[1], h[2]] };
    }
    case "plane":
      return UNBOUNDED; // a half-space is genuinely unbounded — not fudged to a scene box
    case "union":
      return aUnion(boundsSdf(node.a), boundsSdf(node.b));
    case "intersect":
      return aInter(boundsSdf(node.a), boundsSdf(node.b));
    case "subtract":
      return boundsSdf(node.a); // removing material only shrinks
    case "smoothUnion":
      return aExpand(aUnion(boundsSdf(node.a), boundsSdf(node.b)), val(node.k)); // blend bulges out by ~k
    case "translate":
      return aShift(boundsSdf(node.child), [val(node.t[0]), val(node.t[1]), val(node.t[2])]);
    case "scale":
      return aScale(boundsSdf(node.child), val(node.factor));
    case "displace":
      return aExpand(boundsSdf(node.child), Math.abs(val(node.amp))); // noise pushes the surface out by ≤ amp
  }
}

/** Whether `node` is entirely plane-representable — the plane BSP can mesh it exactly. False as soon as
 *  a `sphere`, `smoothUnion`, or `displace` appears anywhere in the tree. */
export function isPlanar(node: Sdf): boolean {
  switch (node.kind) {
    case "box":
    case "plane":
      return true;
    case "sphere":
    case "smoothUnion":
    case "displace":
      return false;
    case "union":
    case "intersect":
    case "subtract":
      return isPlanar(node.a) && isPlanar(node.b);
    case "translate":
    case "scale":
      return isPlanar(node.child);
  }
}

/** The planar skeleton of `node`: the plane-only sub-model the BSP can mesh, with non-planar nodes
 *  dropped (a `smoothUnion`/`union` keeps whichever sides are planar; `displace`/`scale`/`translate`
 *  keep their child's skeleton). Returns `undefined` if nothing planar remains. Outside the non-planar
 *  regions the true field equals this skeleton, so meshing it is exact there. */
export function planarSkeleton(node: Sdf): Sdf | undefined {
  switch (node.kind) {
    case "box":
    case "plane":
      return node;
    case "sphere":
      return undefined;
    case "union":
    case "smoothUnion": {
      // Both behave like a union away from the seam; keep the planar sides.
      const a = planarSkeleton(node.a);
      const b = planarSkeleton(node.b);
      if (a && b) return { kind: "union", a, b };
      return a ?? b;
    }
    case "intersect": {
      const a = planarSkeleton(node.a);
      const b = planarSkeleton(node.b);
      if (a && b) return { kind: "intersect", a, b };
      return a ?? b; // a lone planar operand over-covers the intersection — its region trims it
    }
    case "subtract": {
      const a = planarSkeleton(node.a);
      if (!a) return undefined;
      const b = planarSkeleton(node.b);
      return b ? { kind: "subtract", a, b } : a; // drop a non-planar tool; its region raymarches the cut
    }
    case "translate": {
      const c = planarSkeleton(node.child);
      return c ? { kind: "translate", t: node.t, child: c } : undefined;
    }
    case "scale": {
      const c = planarSkeleton(node.child);
      return c ? { kind: "scale", factor: node.factor, child: c } : undefined;
    }
    case "displace":
      return planarSkeleton(node.child); // the noise is a region; the base shape may still be planar
  }
}

/** **The region where the plane-BSP won't work** — curved primitives, noise, and the neighbourhood of a
 *  smooth-union seam. The result is one of three shapes: an **empty** array (fully planar — mesh it all,
 *  no raymarch); **finite** boxes (local features — a bounded raymarch proxy each); or a single
 *  **unbounded** box (`aabbFinite` is false — a displaced/blended half-space; the raymarch must cover
 *  the view). Overlapping boxes are merged, so a lone unbounded region collapses the whole list to one
 *  unbounded box. Everything outside is the meshed skeleton. */
export function nonPlanarRegions(node: Sdf): AABB[] {
  return mergeRegions(collectRegions(node));
}

function collectRegions(node: Sdf): AABB[] {
  switch (node.kind) {
    case "box":
    case "plane":
      return [];
    case "sphere":
      return [boundsSdf(node)];
    case "union":
    case "intersect":
    case "subtract":
      return [...collectRegions(node.a), ...collectRegions(node.b)];
    case "smoothUnion": {
      // The blend is active where the two operands are both within ~k — the overlap of their k-expanded
      // solids. Two bounded shapes → a box round the seam; two half-spaces → an unbounded seam.
      const k = val(node.k);
      const seam = aInter(aExpand(boundsSdf(node.a), k), aExpand(boundsSdf(node.b), k));
      const own = [...collectRegions(node.a), ...collectRegions(node.b)];
      return aabbEmpty(seam) ? own : [seam, ...own];
    }
    case "translate": {
      const t: Vec3 = [val(node.t[0]), val(node.t[1]), val(node.t[2])];
      return collectRegions(node.child).map((r) => aShift(r, t));
    }
    case "scale": {
      const f = val(node.factor);
      return collectRegions(node.child).map((r) => aScale(r, f));
    }
    case "displace":
      return [boundsSdf(node)];
  }
}

/** Merge overlapping boxes into their unions (to a fixed point) so nearby features become one region.
 *  An unbounded box overlaps everything, so it absorbs the list down to a single unbounded region. */
function mergeRegions(regions: AABB[]): AABB[] {
  const out = regions.filter((r) => !aabbEmpty(r));
  let merged = true;
  while (merged) {
    merged = false;
    for (let i = 0; i < out.length; i++)
      for (let j = i + 1; j < out.length; j++) {
        const a = out[i];
        const b = out[j];
        if (a && b && aabbOverlaps(a, b)) {
          out[i] = aUnion(a, b);
          out.splice(j, 1);
          merged = true;
          j--;
        }
      }
  }
  return out;
}
