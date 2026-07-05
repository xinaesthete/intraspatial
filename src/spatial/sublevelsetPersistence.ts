// Sublevel-set persistence of a 2D scalar field — the "field" counterpart to the
// point-cloud Vietoris-Rips reducer in `persistence.ts`.
//
// Given a scalar field f sampled on a width*height grid, this builds the cubical
// complex (pixels = vertices, the 1-cells between 4-adjacent pixels = edges, the
// unit squares = 2-cells) under the *lower-star* filtration: a cell enters the
// filtration at the value of its largest vertex, so the sublevel set {f <= t}
// grows as t rises. Reducing the GF(2) boundary matrix (same algorithm as
// `persistence.ts`) yields H0 (connected components of the sublevel set) and H1
// (loops / holes).
//
// Why this lives alongside the VR reducer: it is what turns a KDE / distance field
// into a persistence diagram. The two fuzzy-vs-hard fields in `scalarField.ts`
// both reduce through here —
//   • sublevel-set of `distanceField`  = union of balls (hard, Cech-style);
//   • superlevel-set of `gaussianKdeField` (pass {superlevel:true}) = smooth
//     "fuzzy balls", whose C-infinity boundaries avoid the corner artefacts that
//     spawn spurious near-diagonal bars in the hard filtration.
//
// Cost: O(#cells) cells (~4*W*H) with the standard reduction; intended for the
// modest grids (tens of pixels a side) the demos and goldens use, computed once
// per field — not per animation frame.

export interface FieldPersistencePair {
  /** Homology dimension: 0 = component, 1 = loop. */
  dim: number;
  /** Field value at which the feature is born. For a sublevel filtration birth <=
   *  death; for a superlevel filtration (high values first) birth >= death. */
  birth: number;
  /** Field value at which it dies; `Infinity`/`-Infinity` for essential features
   *  still alive at the end of the filtration. */
  death: number;
  /** Row-major grid cell of the cell that *creates* the feature — for a component
   *  (dim 0) the local extremum (valley for sublevel, peak for superlevel) it is born
   *  at; for a loop (dim 1) a representative of the closing edge. Lets a caller link a
   *  spatial feature to its diagram point by identity rather than by value. */
  birthCell: number;
  /** Row-major grid cell of the cell that *kills* the feature — for a loop (dim 1)
   *  the square that fills it (a cell inside the hole); undefined for essential
   *  features that never die. */
  deathCell?: number;
}

export interface FieldPersistenceResult {
  pairs: FieldPersistencePair[];
}

export interface SublevelOptions {
  /** Filter on the *super*level sets {f >= t} (t descending) instead of the
   *  sublevel sets {f <= t}. Internally this just reduces -f; pairs are reported
   *  back in the original field's units (so birth >= death). Default false. */
  superlevel?: boolean;
}

interface Cell {
  /** Stable id (see the indexing scheme below). */
  id: number;
  /** Filtration value (lower-star: max over the cell's vertices), in -f units when
   *  superlevel. */
  filt: number;
  dim: number;
  /** Sorted face ids (the GF(2) boundary), or [] for a vertex. */
  faces: number[];
}

function symmetricDifference(a: number[], b: number[]): number[] {
  const out: number[] = [];
  let i = 0,
    j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      i++;
      j++;
    } else if (a[i]! < b[j]!) out.push(a[i++]!);
    else out.push(b[j++]!);
  }
  while (i < a.length) out.push(a[i++]!);
  while (j < b.length) out.push(b[j++]!);
  return out;
}

/**
 * Cubical sublevel-set persistence (H0 and H1) of a row-major `width*height`
 * scalar field.
 */
export function sublevelsetPersistence(
  field: ArrayLike<number>,
  width: number,
  height: number,
  opts: SublevelOptions = {},
): FieldPersistenceResult {
  const W = width,
    H = height;
  if (W <= 0 || H <= 0) throw new Error("sublevelsetPersistence: bad dimensions");
  if (field.length < W * H) throw new Error("sublevelsetPersistence: field too small");
  const sign = opts.superlevel ? -1 : 1;
  const f = (c: number, r: number) => sign * field[r * W + c]!;

  // ---- Id scheme: vertices | horizontal edges | vertical edges | squares ----
  const V = W * H;
  const HE = (W - 1) * H; // horizontal edge (c,r)-(c+1,r)
  const VE = W * (H - 1); // vertical edge   (c,r)-(c,r+1)
  const vId = (c: number, r: number) => r * W + c;
  const hId = (c: number, r: number) => V + r * (W - 1) + c;
  const vvId = (c: number, r: number) => V + HE + r * W + c;
  const sId = (c: number, r: number) => V + HE + VE + r * (W - 1) + c;

  const cells: Cell[] = [];
  // Vertices.
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) cells.push({ id: vId(c, r), filt: f(c, r), dim: 0, faces: [] });
  // Horizontal edges: lower-star value = max of the two endpoints.
  for (let r = 0; r < H; r++)
    for (let c = 0; c < W - 1; c++) {
      const a = vId(c, r),
        b = vId(c + 1, r);
      cells.push({ id: hId(c, r), filt: Math.max(f(c, r), f(c + 1, r)), dim: 1, faces: a < b ? [a, b] : [b, a] });
    }
  // Vertical edges.
  for (let r = 0; r < H - 1; r++)
    for (let c = 0; c < W; c++) {
      const a = vId(c, r),
        b = vId(c, r + 1);
      cells.push({ id: vvId(c, r), filt: Math.max(f(c, r), f(c, r + 1)), dim: 1, faces: a < b ? [a, b] : [b, a] });
    }
  // Squares: value = max over the 4 corners; boundary = its 4 edges.
  for (let r = 0; r < H - 1; r++)
    for (let c = 0; c < W - 1; c++) {
      const fl = Math.max(f(c, r), f(c + 1, r), f(c, r + 1), f(c + 1, r + 1));
      const faces = [hId(c, r), hId(c, r + 1), vvId(c, r), vvId(c + 1, r)].sort((p, q) => p - q);
      cells.push({ id: sId(c, r), filt: fl, dim: 2, faces });
    }

  // Filtration order: (filt, dim, id) so every face precedes its cofaces.
  cells.sort((a, b) => a.filt - b.filt || a.dim - b.dim || a.id - b.id);

  // Map original cell id -> position in the filtration, so boundary columns can
  // reference faces by filtration position (what the reduction operates on).
  const pos = new Map<number, number>();
  for (let i = 0; i < cells.length; i++) pos.set(cells[i]!.id, i);

  const cols: number[][] = cells.map((cell) => (cell.faces.length ? cell.faces.map((fid) => pos.get(fid)!).sort((a, b) => a - b) : []));

  // Map any cell id back to a representative row-major grid cell: a vertex is its own
  // cell; an edge / square reports its lowest-index corner. Lets callers link a
  // persistence pair to a spatial location (e.g. which component / hole it lives in).
  const repCell = (id: number): number => {
    if (id < V) return id;
    if (id < V + HE) {
      const k = id - V;
      return Math.floor(k / (W - 1)) * W + (k % (W - 1));
    }
    if (id < V + HE + VE) {
      const k = id - V - HE;
      return Math.floor(k / W) * W + (k % W);
    }
    const k = id - V - HE - VE;
    return Math.floor(k / (W - 1)) * W + (k % (W - 1));
  };

  // ---- Standard GF(2) reduction (as in persistence.ts) ----
  const lowToCol = new Map<number, number>();
  const pairedCreator = new Set<number>();
  const pairs: FieldPersistencePair[] = [];
  const low = (col: number[]) => (col.length ? col[col.length - 1]! : -1);
  const toField = (filt: number) => sign * filt; // undo the superlevel negation

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
      const creator = cells[l]!;
      const death = cells[j]!.filt;
      if (death > creator.filt)
        pairs.push({
          dim: creator.dim,
          birth: toField(creator.filt),
          death: toField(death),
          birthCell: repCell(creator.id),
          deathCell: repCell(cells[j]!.id),
        });
      pairedCreator.add(l);
    }
  }

  // Essential features (empty reduced columns never used as a pivot): alive to the
  // end of the filtration. In field units that is +Inf for sublevel, -Inf for
  // superlevel (toField flips the sign of +Inf accordingly).
  const essentialDeath = toField(Infinity);
  for (let j = 0; j < cols.length; j++)
    if (cols[j]!.length === 0 && !pairedCreator.has(j))
      pairs.push({
        dim: cells[j]!.dim,
        birth: toField(cells[j]!.filt),
        death: essentialDeath,
        birthCell: repCell(cells[j]!.id),
      });

  return { pairs };
}

/** Betti numbers (beta0, beta1, ...) of the sublevel/superlevel filtration at
 *  field value `t`: count of pairs alive at `t`. Direction is inferred from each
 *  pair (sublevel: birth <= t < death; superlevel: death < t <= birth). */
export function fieldBettiNumbers(result: FieldPersistenceResult, t: number): number[] {
  const b: number[] = [];
  for (const p of result.pairs) {
    const alive =
      p.birth <= p.death
        ? p.birth <= t && t < p.death // sublevel
        : p.death < t && t <= p.birth; // superlevel
    if (alive) b[p.dim] = (b[p.dim] ?? 0) + 1;
  }
  for (let i = 0; i < b.length; i++) if (b[i] === undefined) b[i] = 0;
  return b;
}
