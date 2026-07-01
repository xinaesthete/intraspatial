// The distance-matrix lens — the dance read as an N×N matrix (the user's idea, and true
// to this repo's spatial-analysis identity). Couples show as symmetric off-diagonal hot
// pairs; the caller's progression as bands migrating across figures; figures as block
// patterns. Drawn to a small canvas; capped at MAX rows so it stays cheap.
const MAX = 96;

export function drawDistanceMatrix(canvas: HTMLCanvasElement, positions: Float32Array, n: number): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const m = Math.min(n, MAX);
  if (m < 1) return;
  const img = ctx.createImageData(m, m);
  const data = img.data;

  // one pass for the max distance (normalisation), then fill
  let maxD = 1e-6;
  const d = new Float32Array(m * m);
  for (let i = 0; i < m; i++) {
    const xi = positions[i * 3] ?? 0, yi = positions[i * 3 + 1] ?? 0, zi = positions[i * 3 + 2] ?? 0;
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

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      const t = 1 - Math.min(1, (d[i * m + j] ?? 0) / maxD); // near = 1 (hot)
      const o = (i * m + j) * 4;
      data[o] = 40 + 200 * t * t; // R
      data[o + 1] = 60 + 140 * t; // G
      data[o + 2] = 110 + 120 * t; // B
      data[o + 3] = 255;
    }
  }
  // draw at native m×m then let CSS scale the canvas up (crisp blocks)
  if (canvas.width !== m || canvas.height !== m) {
    canvas.width = m;
    canvas.height = m;
  }
  ctx.putImageData(img, 0, 0);
}
