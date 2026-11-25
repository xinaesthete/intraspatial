// A surface from the bond table: wherever bonds close a small loop, fill it. Three particles all
// bonded to each other make a triangle; four bonded in a ring make a quad (two triangles). A
// β-sheet drawn with matrix lines is exactly such a ladder of quads (two backbone bonds and two
// rungs each), so it comes out as a sheet; a helix line gives a band winding between its turns.
import { NO_BOND, SLOT_WORDS, slotKind, slotPartner, type SpringTopology } from "./topology";

export interface FaceOptions {
  /** Bit mask of the bond kinds that may form a face's edge. Default: every kind. */
  readonly kinds?: number;
  /** Fill three-bond loops. Default true. */
  readonly triangles?: boolean;
  /** Fill four-bond loops. Default true. */
  readonly quads?: boolean;
}

/**
 * Triangles over the particles, as index triples: one per three-bond loop and two per four-bond
 * loop. Each loop is found once, from its lowest-numbered particle. A four-bond loop with a bond
 * across it is left to its triangles.
 */
export function bondFaces(t: SpringTopology, o: FaceOptions = {}): Uint32Array {
  const mask = o.kinds ?? 0xffffffff;
  const near: number[][] = Array.from({ length: t.n }, () => []);
  const bonded = new Set<number>();
  for (let i = 0; i < t.n; i++) {
    for (let k = 0; k < t.K; k++) {
      const bits = t.slotBits[(i * t.K + k) * SLOT_WORDS]!;
      if (bits === NO_BOND || ((mask >>> slotKind(bits)) & 1) === 0) continue;
      const j = slotPartner(bits);
      near[i]!.push(j);
      bonded.add(i * t.n + j);
    }
  }
  const has = (a: number, b: number) => bonded.has(a * t.n + b);
  const out: number[] = [];
  for (let a = 0; a < t.n; a++) {
    const above = near[a]!.filter((j) => j > a).sort((x, y) => x - y);
    for (let p = 0; p < above.length; p++) {
      for (let q = p + 1; q < above.length; q++) {
        const b = above[p]!;
        const c = above[q]!;
        if (has(b, c)) {
          if (o.triangles !== false) out.push(a, b, c);
          continue;
        }
        if (o.quads === false) continue;
        // The corner opposite `a`: bonded to both `b` and `c`, and not to `a` (that would be two triangles).
        for (const d of near[b]!) if (d > a && d !== c && has(c, d) && !has(a, d)) out.push(a, b, d, a, d, c);
      }
    }
  }
  return Uint32Array.from(out);
}
