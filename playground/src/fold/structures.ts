// Which stretches of the chain the matrix lines make into helices and sheets, for the cartoon view.
//
// This reads the lines, not the chain: it says what each line is trying to build, whether or not
// the chain has folded that way yet. FoldSynth could also find structures from which beads had
// actually come close (`Fold.computeSecondaryStructures`); that is not done here.
import type { ChainStructure } from "@intraspatial/deck";
import { lineCells, type MatrixLine } from "../../../src/gpu/sim/springs/index";

/**
 * The structure one line asks for, or null when it is neither along nor across the diagonal.
 * Along the diagonal, `turn` apart: a helix when the line is at least a turn long (each bead's tie
 * overlaps the next bead's), otherwise two strands running the same way. Across it: two strands
 * running opposite ways.
 */
function lineStructure(line: MatrixLine, n: number): ChainStructure | null {
  const pairs: [number, number][] = [];
  for (const [p, q] of lineCells(line)) {
    const a = Math.min(p, q);
    const b = Math.max(p, q);
    if (a >= 0 && b < n && b - a > 1) pairs.push([a, b]);
  }
  const first = pairs[0];
  if (!first || pairs.length < 3) return null;
  const turn = first[1] - first[0];
  if (pairs.every(([a, b]) => b - a === turn)) {
    if (pairs.length < turn) return { kind: "sheet", pairs };
    return { kind: "helix", start: Math.min(...pairs.map((p) => p[0])), end: Math.max(...pairs.map((p) => p[1])), turn };
  }
  const sum = first[0] + first[1];
  return pairs.every(([a, b]) => a + b === sum) ? { kind: "sheet", pairs } : null;
}

export function lineStructures(lines: readonly MatrixLine[], n: number): ChainStructure[] {
  return lines.map((l) => lineStructure(l, n)).filter((s) => s !== null);
}
