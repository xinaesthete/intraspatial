// Leaf coordinate types shared across the two module towers.
//
// `Affine3` (and the `Vec3` it is built on) is needed in two places that must NOT
// depend on each other: the op-graph value model (`src/gpu/graph/handle.ts`, which
// deliberately imports nothing from `datasource`) carries a `ResolvedPlacement`
// whose matrix is an `Affine3`, and the datasource geometry (`src/datasource/math.ts`,
// which already depends on `handle`) is where the affine helpers live. Putting the
// bare types here — a module that imports nothing — lets both towers share the
// vocabulary without inverting the layer direction (`handle.ts` must never import
// `datasource`). `math.ts` re-exports these, so existing `datasource` imports are
// unchanged.

export type Vec3 = readonly [number, number, number];

/**
 * An affine placement of *array space* into world. Array coordinates are in
 * **level-0 voxel units**; one voxel step along array axis i moves `axes[i]` in
 * world. So `|axes[i]|` is the world size of a level-0 voxel along axis i, and a
 * general (rotated/anisotropic) `axes` lets a plane sit obliquely in 3-D.
 *
 *   world = origin + a0·axes[0] + a1·axes[1] + a2·axes[2]
 */
export interface Affine3 {
  readonly origin: Vec3;
  readonly axes: readonly [Vec3, Vec3, Vec3];
}
