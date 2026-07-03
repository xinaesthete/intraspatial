// The Ceilidh caller's vocabulary — figures (target *states of motion*) and partner
// progression. This is the deepest bit of DANCERL reinterpreted: dancers don't snap to
// keyframe positions, they scramble toward a target *velocity field* at each call, in
// couples that advance through partners between figures. The caller op (ops/danceForces →
// caller) turns these into an acceleration, weighted so the target-change at a figure
// boundary reads as an urgent-but-smooth scramble (the integrator's jerk limit keeps it
// C²). Pure + deterministic; the app drives the frame clock (the conductor).
import { add, cross, normalize, scale, sub, vec3, type Vec3, type Vec3In } from "./vec3";

export type Figure = "swing" | "grandChain" | "gather" | "scatter" | "mill";

/** The called sequence, cycled (rotated by the caller's seed). */
export const FIGURE_SEQUENCE: readonly Figure[] = ["swing", "grandChain", "gather", "swing", "scatter", "mill"];

export interface FigureAt {
  figure: Figure;
  figureIndex: number;
  /** 0..1 through the current figure. */
  phase: number;
}

/** Which figure is called at `frame`, cycling every `period` frames. */
export function figureAt(frame: number, period: number, seed: number): FigureAt {
  const p = Math.max(1, period);
  const figureIndex = Math.floor(frame / p);
  const idx = (((figureIndex + seed) % FIGURE_SEQUENCE.length) + FIGURE_SEQUENCE.length) % FIGURE_SEQUENCE.length;
  const figure = FIGURE_SEQUENCE[idx] ?? "swing";
  return { figure, figureIndex, phase: (frame % p) / p };
}

/** Partner of agent `i` in figure `figureIndex` — a permutation that **advances each
 *  figure**, so couples form, swing, then break and re-pair at the next call (the Ceilidh
 *  progression / strip-the-willow weave). */
export function partnerIndex(i: number, figureIndex: number, n: number): number {
  if (n <= 1) return i;
  const step = 1 + (((figureIndex % (n - 1)) + (n - 1)) % (n - 1));
  return (i + step) % n;
}

const UP: Vec3 = vec3(0, 1, 0);

/** The target velocity for agent `i` under `figure` — the *state of motion* to scramble
 *  toward. `partner` is the position of `i`'s current partner (for `swing`). */
export function figureTargetVel(figure: Figure, p: Vec3In, i: number, partner: Vec3In, speed: number): Vec3 {
  switch (figure) {
    case "swing": {
      // orbit the couple midpoint — the pair swings around each other
      const mid = scale(add(p, partner), 0.5);
      const rel = sub(p, mid);
      return scale(normalize(cross(UP, rel)), speed);
    }
    case "gather":
      return scale(normalize(p), -speed);
    case "scatter":
      return scale(normalize(p), speed);
    case "grandChain": {
      // counter-rotating: even dancers one way about y, odd the other
      const sign = i % 2 === 0 ? 1 : -1;
      return scale(normalize(cross(UP, p)), speed * sign);
    }
    case "mill": {
      // low-coherence: a fixed pseudo-random heading per dancer, gentle
      const h = (Math.imul(i + 1, 2654435761) >>> 0) / 0xffffffff;
      const ang = h * Math.PI * 2;
      return scale([Math.cos(ang), 0, Math.sin(ang)], speed * 0.4);
    }
  }
}
