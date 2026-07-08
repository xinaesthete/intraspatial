// The argument a transform-op accepts where a rotation angle is wanted. A typed `Angle` is
// canonical (radians); a bare number or `Expr` is interpreted in the catalogue's default Angle
// unit. Split into its own module so `swept` and `angle` share it without a cycle.

import type { Angle } from "./angle";
import type { Expr } from "./expr";

export type AngleLike = number | Expr | Angle;
