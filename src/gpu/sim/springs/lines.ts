// Springs from lines drawn on the particle × particle matrix: FoldSynth's vector tools
// (`SSLine`, `LineSet` and the builders in `FoldSpringControl`, by Peter Todd and Stephen Todd).
//
// The matrix has a cell for every pair of particles. A straight line across it names a run of
// pairs, and each pair gets a spring. The direction of the line is what makes it a structure:
//   - parallel to the diagonal, cells `(i, i + d)`: each particle tied to the one `d` further
//     along, which winds a helix;
//   - across the diagonal, cells `(i, s - i)`: two stretches of chain paired off against each
//     other running in opposite directions, which is one rung of a sheet.
// A line is four numbers, so it stays editable, and it can be bred or driven like any parameter.
import { addSpring, findSlot, removeSpring, setTorsion, slotKind, type SpringTopology } from "./topology";

/** A line on the matrix from cell `(a0, b0)` to cell `(a1, b1)`; `a` and `b` are particle
 *  indices and need not be whole numbers while a line is being dragged. */
export interface MatrixLine {
  readonly a0: number;
  readonly b0: number;
  readonly a1: number;
  readonly b1: number;
}

/**
 * The cells a line crosses, as `[a, b]` pairs: FoldSynth's `getSpringLocations`. A Bresenham
 * walk, with the two diagonal directions special-cased so that a helix or sheet line keeps a
 * constant offset or sum along its whole length.
 */
export function lineCells(line: MatrixLine): [number, number][] {
  let sx = line.a0;
  let sy = line.b0;
  let tx = line.a1;
  let ty = line.b1;
  const steep = Math.abs(ty - sy) > Math.abs(tx - sx);
  if (steep) {
    [sx, sy] = [sy, sx];
    [tx, ty] = [ty, tx];
  }
  if (sx > tx) {
    [sx, tx] = [tx, sx];
    [sy, ty] = [ty, sy];
  }
  const dErr = Math.abs(ty - sy) / (tx - sx);
  const yStep = sy < ty ? 1 : -1;
  const epsilon = 0.001;
  let x = Math.floor(sx + 0.5);
  let y: number;
  if (Math.abs(sx - sy - tx + ty) < epsilon) y = x + Math.round(sy - sx);
  else if (Math.abs(sx + sy - tx - ty) < epsilon) y = Math.round(sy + sx) - x;
  else y = Math.floor(sy + 0.5);

  const cells: [number, number][] = [];
  let error = 0;
  for (; x <= tx + 0.5; x++) {
    cells.push(steep ? [y, x] : [x, y]);
    error += dErr;
    if (Math.abs(error) >= 0.5) {
      y += yStep;
      error -= 1;
    }
  }
  return cells;
}

/** A helix over particles `s .. e`, each tied to the one `d` further along. `addHelix`. */
export function helixLine(s: number, e: number, d: number): MatrixLine {
  return { a0: s, b0: s + d, a1: e - d, b1: e };
}

/** A sheet of `strands` strands over particles `s .. e`, with `loop` particles in each turn
 *  between strands: one line per pair of neighbouring strands. `addSheet`. */
export function sheetLines(s: number, e: number, strands: number, loop: number): MatrixLine[] {
  const strandLen = Math.floor((e - s + 1 + loop) / strands) - loop;
  if (strandLen <= 0) return [];
  const lines: MatrixLine[] = [];
  let at = s;
  for (let k = 0; k < strands - 1; k++) {
    // Antiparallel neighbours: the first of this strand pairs with the last of the next.
    const sum = 2 * at + 2 * strandLen + loop - 1;
    lines.push({ a0: at, b0: sum - at, a1: at + strandLen - 1, b1: sum - (at + strandLen - 1) });
    at += strandLen + loop;
  }
  return lines;
}

/**
 * Snap a dragged end to the nearer diagonal direction from the line's other end, when it is
 * within `threshold` cells of it: FoldSynth's `snap`. Returns the adjusted `[a, b]`.
 */
export function snapEnd(otherA: number, otherB: number, a: number, b: number, threshold = 6): [number, number] {
  const dx = otherA - a;
  const dy = otherB - b;
  const parallel = dx - dy;
  const anti = dx + dy;
  if (Math.abs(parallel) < threshold && Math.abs(parallel) < Math.abs(anti)) return [a + parallel, b];
  if (Math.abs(anti) < threshold) return [a + anti, b];
  return [a, b];
}

export interface LineSpringOptions {
  /** Bond kind given to line springs, so they can be told from the backbone and re-applied. */
  readonly kind: number;
  /** Rest length of a line spring. FoldSynth's default is 1.5 particle diameters. */
  readonly rest: number;
  readonly strength: number;
  /** Springs between particles closer than `torsionMaxSeparation` along the chain also set a
   *  torsion spring centred between them, as FoldSynth does for helices. 0 strength is none. */
  readonly torsionAngle: number;
  readonly torsionStrength: number;
  readonly torsionMaxSeparation: number;
}

export const DEFAULT_LINE_SPRINGS: LineSpringOptions = {
  kind: 3,
  rest: 1.5,
  strength: 0.3,
  torsionAngle: 0.75,
  torsionStrength: 0.01,
  torsionMaxSeparation: 4,
};

/** What a call to `applyLines` put into the topology, so the next call can take it out again. */
export interface LineSprings {
  /** Bonded pairs, as `[a, b]` with `a < b`. */
  readonly pairs: readonly (readonly [number, number])[];
  /** First particle of each torsion spring set. */
  readonly torsions: readonly number[];
  /** Pairs wanted but not made because a particle's bond slots were full. */
  readonly skipped: number;
  /** The bond kind they were installed with; what the next call removes. */
  readonly kind: number;
}

/**
 * Make the topology's line springs match `lines`: take out what `previous` put in, then add a
 * spring for every cell every line crosses. Index neighbours (`|a - b| ≤ 1`) are left to the
 * backbone, and springs of other kinds are never replaced.
 */
export function applyLines(
  t: SpringTopology,
  lines: readonly MatrixLine[],
  o: LineSpringOptions = DEFAULT_LINE_SPRINGS,
  previous?: LineSprings,
): LineSprings {
  for (const [a, b] of previous?.pairs ?? []) {
    const slot = findSlot(t, a, b);
    if (slot >= 0 && slotKind(t.slotBits[slot]!) === (previous?.kind ?? o.kind)) removeSpring(t, a, b);
  }
  for (const i of previous?.torsions ?? []) t.torsion[i * 2 + 1] = 0;

  const pairs: [number, number][] = [];
  const torsions: number[] = [];
  const seen = new Set<number>();
  let skipped = 0;
  for (const line of lines) {
    for (const [p, q] of lineCells(line)) {
      const a = Math.min(p, q);
      const b = Math.max(p, q);
      if (a < 0 || b >= t.n || b - a <= 1 || seen.has(a * t.n + b)) continue;
      seen.add(a * t.n + b);
      const slot = findSlot(t, a, b);
      if (slot >= 0 && slotKind(t.slotBits[slot]!) !== o.kind) continue;
      if (!addSpring(t, a, b, o.rest, o.strength, 0, o.kind)) {
        skipped++;
        continue;
      }
      pairs.push([a, b]);
      if (o.torsionStrength !== 0 && b - a < o.torsionMaxSeparation) {
        // A torsion spring is indexed by the first of its four particles; centre it on the pair.
        const first = Math.trunc((a + b) / 2 - 1.5);
        if (first >= 0 && first + 3 < t.n) {
          setTorsion(t, first, o.torsionAngle, o.torsionStrength);
          torsions.push(first);
        }
      }
    }
  }
  return { pairs, torsions, skipped, kind: o.kind };
}
