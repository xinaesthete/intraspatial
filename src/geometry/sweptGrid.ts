// The `(slices + 1) × (stacks + 1)` vertex grid a Swept geometry is tessellated over — the pure
// index/parameter helpers shared by the CPU golden (`swept.ts`) and the GPU lowering
// (`sweptGpu.ts`). Kept apart from `swept.ts` so the GPU side needs no value import from it
// (ADR-0010; the CPU side lazily imports the GPU side, so that edge would be a cycle).

/** `(s, θ)` for grid corner `(col, row)` of a `slices × stacks` tessellation. */
export function gridSampleAngles(col: number, row: number, slices: number, stacks: number): { s: number; theta: number } {
  return { s: row / stacks, theta: (col / slices) * 2 * Math.PI };
}

/** Two triangles per quad over the `(slices + 1) × (stacks + 1)` vertex grid, wound so the
 *  face normal agrees with the analytic outward normal. */
export function gridIndices(slices: number, stacks: number): Uint32Array {
  const cols = slices + 1;
  const indices = new Uint32Array(slices * stacks * 6);
  let k = 0;
  for (let row = 0; row < stacks; row++) {
    for (let col = 0; col < slices; col++) {
      const a = row * cols + col;
      const b = a + 1;
      const c = a + cols;
      const dd = c + 1;
      indices[k++] = a;
      indices[k++] = c;
      indices[k++] = dd;
      indices[k++] = a;
      indices[k++] = dd;
      indices[k++] = b;
    }
  }
  return indices;
}
