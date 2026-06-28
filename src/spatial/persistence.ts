// Vietoris-Rips persistent homology (CPU) over a distance matrix — the reduction
// step that turns a (fuzzy / CkNN-rescaled) filtration into a persistence diagram.
//
// The GPU's job ends at a distance matrix: raw d_ij, the CkNN-rescaled
// d̃_ij = d_ij/√(ρ_iρ_j) (see `gpu/spatial/cknn.ts`), or any other. This module does
// the inherently-sequential part — building the VR complex up to triangles and
// reducing the GF(2) boundary matrix — which does not parallelise well and stays on
// the CPU (cf. Ripser/GUDHI; we implement the standard algorithm directly so the
// toolbox has a dependency-light reducer for small N).
//
// Computes H0 (connected components) and H1 (loops). H1 deaths require 2-simplices
// (triangles), so simplices of dimension 0,1,2 are built. Cost is O(N^3) triangles;
// intended for the small-N regime (tens to low hundreds of points).

export interface PersistencePair {
  /** Homology dimension: 0 = component, 1 = loop. */
  dim: number;
  /** Filtration value at which the feature is born. */
  birth: number;
  /** Filtration value at which it dies; `Infinity` for features still alive at
   *  `maxScale` (e.g. the essential components). */
  death: number;
}

export interface PersistenceResult {
  pairs: PersistencePair[];
}

export interface PersistenceOptions {
  /** Only include simplices whose filtration value is ≤ this. Bounds the cost and
   *  caps essential deaths. Default: the largest finite distance in the matrix. */
  maxScale?: number;
}

interface Simplex {
  verts: number[]; // sorted ascending
  filt: number;
  dim: number;
}

function symmetricDifference(a: number[], b: number[]): number[] {
  const out: number[] = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) { i++; j++; }
    else if (a[i]! < b[j]!) out.push(a[i++]!);
    else out.push(b[j++]!);
  }
  while (i < a.length) out.push(a[i++]!);
  while (j < b.length) out.push(b[j++]!);
  return out;
}

/** Vietoris-Rips persistence (H0 and H1) of the points described by a row-major
 *  `n*n` symmetric distance matrix. */
export function vietorisRipsPersistence(
  dist: ArrayLike<number>,
  n: number,
  opts: PersistenceOptions = {},
): PersistenceResult {
  const at = (i: number, j: number) => dist[i * n + j]!;

  let maxScale = opts.maxScale;
  if (maxScale === undefined) {
    let m = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) m = Math.max(m, at(i, j));
    maxScale = m;
  }

  // Build simplices (dim 0,1,2) within maxScale.
  const simplices: Simplex[] = [];
  for (let i = 0; i < n; i++) simplices.push({ verts: [i], filt: 0, dim: 0 });
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const dij = at(i, j);
      if (dij <= maxScale) simplices.push({ verts: [i, j], filt: dij, dim: 1 });
    }
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const dij = at(i, j);
      if (dij > maxScale) continue;
      for (let k = j + 1; k < n; k++) {
        const f = Math.max(dij, at(i, k), at(j, k));
        if (f <= maxScale) simplices.push({ verts: [i, j, k], filt: f, dim: 2 });
      }
    }

  // Filtration order: by (filt, dim, verts) so faces precede cofaces.
  simplices.sort((a, b) => a.filt - b.filt || a.dim - b.dim || cmpVerts(a.verts, b.verts));

  const index = new Map<string, number>();
  for (let idx = 0; idx < simplices.length; idx++) index.set(key(simplices[idx]!.verts), idx);

  // Boundary columns (face indices, sorted) over GF(2).
  const cols: number[][] = simplices.map((s) => {
    if (s.dim === 0) return [];
    const faces: number[] = [];
    for (let omit = 0; omit < s.verts.length; omit++) {
      const face = s.verts.filter((_, t) => t !== omit);
      faces.push(index.get(key(face))!);
    }
    faces.sort((a, b) => a - b);
    return faces;
  });

  // Standard reduction.
  const lowToCol = new Map<number, number>();
  const pairedCreator = new Set<number>();
  const pairs: PersistencePair[] = [];
  const low = (c: number[]) => (c.length ? c[c.length - 1]! : -1);

  for (let j = 0; j < cols.length; j++) {
    let col = cols[j]!;
    let l = low(col);
    while (l !== -1 && lowToCol.has(l)) {
      col = symmetricDifference(col, cols[lowToCol.get(l)!]!);
      l = low(col);
    }
    cols[j] = col;
    if (l !== -1) {
      lowToCol.set(l, j);
      const creator = simplices[l]!;
      const death = simplices[j]!.filt;
      if (death > creator.filt) pairs.push({ dim: creator.dim, birth: creator.filt, death });
      pairedCreator.add(l);
    }
  }

  // Essential features: creators (empty reduced columns) never paired.
  for (let j = 0; j < cols.length; j++) {
    if (cols[j]!.length === 0 && !pairedCreator.has(j)) {
      pairs.push({ dim: simplices[j]!.dim, birth: simplices[j]!.filt, death: Infinity });
    }
  }

  return { pairs };
}

/** Betti numbers (β0, β1, …) of the filtration at a given scale: count of pairs
 *  alive (`birth ≤ scale < death`) per dimension. */
export function bettiNumbers(result: PersistenceResult, scale: number): number[] {
  const b: number[] = [];
  for (const p of result.pairs) {
    if (p.birth <= scale && scale < p.death) b[p.dim] = (b[p.dim] ?? 0) + 1;
  }
  for (let i = 0; i < b.length; i++) if (b[i] === undefined) b[i] = 0;
  return b;
}

function key(verts: number[]): string {
  return verts.join(",");
}
function cmpVerts(a: number[], b: number[]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return a[i]! - b[i]!;
  return a.length - b.length;
}
