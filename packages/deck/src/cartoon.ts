// A protein's classic "cartoon": the chain as a thin tube, with helices as broad ribbons facing
// out from their axis and sheet strands as flat arrows lying in the sheet. Makes a `TubeProfile`
// for `TubeLayer` from a list of structures. Host only.
//
// After FoldSynth's `FluidCartoonView` and `Fold.SecondaryStructure` (Peter Todd, Stephen Todd):
// narrow and broad radii, an arrowhead at the end of a strand, the helix direction from particles
// one turn apart, the strand direction from its partner strand, and "how far in" deciding which
// structure a particle shows when it is in several.

import type { Color } from "./DensityLayer";
import { NO_ORIENT, ORIENT_WORDS, roundProfile, SHAPE, SHAPE_FLOATS, type TubeProfile } from "./tubeProfile";

export type ChainStructure =
  /** Particles `start..end`, each held to the one `turn` further along. */
  | { readonly kind: "helix"; readonly start: number; readonly end: number; readonly turn: number }
  /** Two strands laid side by side: the pairs of particles facing each other. */
  | { readonly kind: "sheet"; readonly pairs: readonly (readonly [number, number])[] };

export interface CartoonStyle {
  /** Sizes are multiples of the tube's radius. Chain outside any structure: a round tube this size. */
  readonly coil: number;
  /** Thickness of a ribbon or strand. */
  readonly narrow: number;
  /** Width of a ribbon or strand. */
  readonly broad: number;
  /** Width of a strand's arrowhead, as a multiple of `broad`; 1 is no head. */
  readonly arrow: number;
  /** Colours that replace the tube's ramp inside a structure; null keeps the ramp. */
  readonly helixColor: Color | null;
  readonly sheetColor: Color | null;
  /** How square the section is, 0 an ellipse to 1 a rectangle. */
  readonly helixSquare: number;
  readonly sheetSquare: number;
}

export const DEFAULT_CARTOON: CartoonStyle = {
  coil: 0.6,
  narrow: 0.35,
  broad: 3,
  arrow: 1.7,
  helixColor: [236, 72, 153, 255],
  sheetColor: [250, 204, 21, 255],
  helixSquare: 0.2,
  sheetSquare: 0.9,
};

const COIL = 0;
const HELIX = 1;
const SHEET = 2;

/** The runs of consecutive values in a sorted list, as `[first, last]`. */
function runs(sorted: readonly number[]): [number, number][] {
  const out: [number, number][] = [];
  for (const v of sorted) {
    const last = out[out.length - 1];
    if (last && v === last[1] + 1) last[1] = v;
    else out.push([v, v]);
  }
  return out;
}

/** The cartoon of a chain of `count` particles holding `structures`. */
export function cartoonProfile(count: number, structures: readonly ChainStructure[], style: Partial<CartoonStyle> = {}): TubeProfile {
  const st = { ...DEFAULT_CARTOON, ...style };
  const profile = roundProfile(count);
  const { shapes, orients } = profile;
  const kind = new Uint8Array(count);
  /** How far inside its structure each particle is; the deepest structure is the one shown. */
  const depth = new Int32Array(count).fill(-1);
  const partners: number[][] = Array.from({ length: count }, () => []);
  const inside = (i: number) => i >= 0 && i < count;

  for (const s of structures) {
    if (s.kind === "helix") {
      const start = Math.max(0, s.start);
      const end = Math.min(count - 1, s.end);
      const turn = Math.max(1, Math.min(s.turn, end - start));
      for (let i = start; i <= end; i++) {
        const d = Math.min(end - i, i - start);
        if (d <= depth[i]!) continue;
        depth[i] = d;
        kind[i] = HELIX;
        // One turn back and one turn forward, each kept inside the helix: its axis hereabouts.
        const back = Math.max(start, i - turn);
        const fore = Math.min(end, i + turn);
        orients.set([back, back + turn, fore - turn, fore], i * ORIENT_WORDS);
      }
      continue;
    }
    const pairs = s.pairs.filter(([a, b]) => inside(a) && inside(b) && a !== b);
    for (const side of [0, 1] as const) {
      const strand = [...new Set(pairs.map((p) => p[side]))].sort((x, y) => x - y);
      for (const [first, last] of runs(strand)) {
        for (let i = first; i <= last; i++) {
          const d = Math.min(last - i, i - first);
          if (d > depth[i]!) {
            depth[i] = d;
            kind[i] = SHEET;
          }
        }
      }
    }
    for (const [a, b] of pairs) {
      if (!partners[a]!.includes(b)) partners[a]!.push(b);
      if (!partners[b]!.includes(a)) partners[b]!.push(a);
    }
  }

  // A strand faces its partner; a strand between two partners faces from one to the other.
  for (let i = 0; i < count; i++) {
    const [p, q] = partners[i]!;
    if (kind[i] !== SHEET || p === undefined) continue;
    orients.set(q === undefined ? [p, i, p, i] : [p, i, i, q], i * ORIENT_WORDS);
  }

  for (let i = 0; i < count; i++) {
    const at = i * SHAPE_FLOATS;
    // A stretch of structure starts and ends at the coil's size and is full size in between.
    const edge = kind[i] === COIL || kind[i - 1] !== kind[i] || kind[i + 1] !== kind[i];
    const width = edge ? st.coil : st.broad;
    const thick = edge ? st.coil : st.narrow;
    shapes.set([width, width, thick, thick], at);
    // The arrowhead: the span into a strand's last particle starts wide and comes to the coil.
    if (kind[i] === SHEET && !edge && kind[i + 2] !== SHEET && kind[i - 2] === SHEET) shapes[at + SHAPE.widthOut] = st.broad * st.arrow;
    const colour = kind[i] === HELIX ? st.helixColor : kind[i] === SHEET ? st.sheetColor : null;
    if (colour) shapes.set([colour[0] / 255, colour[1] / 255, colour[2] / 255, 1], at + SHAPE.colour);
    shapes[at + SHAPE.square] = kind[i] === HELIX ? st.helixSquare : kind[i] === SHEET ? st.sheetSquare : 0;
    if (kind[i] === COIL) orients.fill(NO_ORIENT, i * ORIENT_WORDS, (i + 1) * ORIENT_WORDS);
  }
  return profile;
}
