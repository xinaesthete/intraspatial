// The distance-matrix lens — the dance read as an N×N matrix (couples show as symmetric
// off-diagonal hot pairs; the caller's progression as migrating bands). It cross-links with
// the 3D stage: `highlight.row` tints a hovered dancer's row+column, and `highlight.pair`
// marks a hovered cell (i,j)+(j,i) and its rows/columns. `matrixCell` maps a pointer event
// back to (i,j). Redrawn every frame; capped at MAX agents.
export const MATRIX_MAX = 180;

export interface MatrixHighlight {
  row: number | null;
  pair: [number, number] | null;
}

export function matrixSize(n: number): number {
  return Math.min(n, MATRIX_MAX);
}

/** Map a pointer event over the matrix canvas to a cell [i, j] (i = row, j = col), or null
 *  if outside. */
export function matrixCell(canvas: HTMLCanvasElement, clientX: number, clientY: number, n: number): [number, number] | null {
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 1 || rect.height < 1) return null;
  const fx = (clientX - rect.left) / rect.width;
  const fy = (clientY - rect.top) / rect.height;
  if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return null;
  const m = matrixSize(n);
  const j = Math.min(m - 1, Math.max(0, Math.floor(fx * m)));
  const i = Math.min(m - 1, Math.max(0, Math.floor(fy * m)));
  return [i, j];
}

export function drawDistanceMatrix(canvas: HTMLCanvasElement, positions: Float32Array, n: number, highlight?: MatrixHighlight): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const m = matrixSize(n);
  if (m < 1) return;
  const img = ctx.createImageData(m, m);
  const data = img.data;

  let maxD = 1e-6;
  const d = new Float32Array(m * m);
  for (let i = 0; i < m; i++) {
    const xi = positions[i * 3] ?? 0,
      yi = positions[i * 3 + 1] ?? 0,
      zi = positions[i * 3 + 2] ?? 0;
    for (let j = i + 1; j < m; j++) {
      const dx = xi - (positions[j * 3] ?? 0);
      const dy = yi - (positions[j * 3 + 1] ?? 0);
      const dz = zi - (positions[j * 3 + 2] ?? 0);
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      d[i * m + j] = dist;
      d[j * m + i] = dist;
      if (dist > maxD) maxD = dist;
    }
  }

  const row = highlight?.row ?? null;
  const pair = highlight?.pair ?? null;
  const pi = pair ? pair[0] : -1;
  const pj = pair ? pair[1] : -1;

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      const t = 1 - Math.min(1, (d[i * m + j] ?? 0) / maxD); // near = hot
      let r = 40 + 200 * t * t;
      let g = 60 + 140 * t;
      let b = 110 + 120 * t;

      const onRow = row !== null && (i === row || j === row);
      const onPairLine = pair !== null && (i === pi || i === pj || j === pi || j === pj);
      if (onRow || onPairLine) {
        r += 45;
        g += 35;
        b += 20;
      }
      const isPairCell = pair !== null && ((i === pi && j === pj) || (i === pj && j === pi));
      if (isPairCell) {
        r = 255;
        g = 255;
        b = 210;
      }

      const o = (i * m + j) * 4;
      data[o] = Math.min(255, r);
      data[o + 1] = Math.min(255, g);
      data[o + 2] = Math.min(255, b);
      data[o + 3] = 255;
    }
  }

  if (canvas.width !== m || canvas.height !== m) {
    canvas.width = m;
    canvas.height = m;
  }
  ctx.putImageData(img, 0, 0);
}
