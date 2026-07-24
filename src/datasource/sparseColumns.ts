// Extracting a few named columns out of an AnnData `X` — the pure arithmetic half, with no zarr
// in sight. The I/O that fetches the three constituent arrays lives on the playground side
// (ADR-0015's ownership boundary); this module receives them already read and does the indexing.
//
// AnnData writes `X` (and each `layers/<name>`) as one of three encodings, and which one you get
// decides how expensive "give me 20 genes" is:
//
//   dense  [n_obs, n_var]  — read the chunks covering the wanted columns; cost scales with the
//                            chunk grid, not the selection.
//   CSC    indptr[n_var+1] — the good case: gene `g` is the contiguous slice
//                            `indptr[g] .. indptr[g+1]` of `indices` (cell ids) and `data`. Reading
//                            20 genes touches only those slices.
//   CSR    indptr[n_obs+1] — the bad case: rows are cells, so a gene's entries are scattered
//                            across every row and there is no way to find them without walking
//                            the ENTIRE `indices` array. Selecting one gene costs the same as
//                            selecting all of them.
//
// That asymmetry is not a detail to hide behind an abstraction — CSR is what scanpy writes by
// default, so the common case is the expensive one, and the UI should say so rather than appear to
// hang. `sparseColumnCost` exists to be reported.
//
// **Output layout is column-major**: `values[g * nCells + i]` is gene `g` in cell `i`. That is
// exactly what `channelsFromExpression` in `src/spatial/gram.ts` consumes, and it keeps each gene's
// column contiguous so it can be handed on as a subarray view rather than copied.

/** The three encodings AnnData uses for `X` / `layers/*`. */
export type SparseEncoding = "csr" | "csc";

export interface SparseMatrix {
  readonly encoding: SparseEncoding;
  /** Compressed dimension offsets: length `n_obs+1` for CSR, `n_var+1` for CSC. */
  readonly indptr: ArrayLike<number>;
  /** Index within the *other* dimension: var index for CSR, cell index for CSC. */
  readonly indices: ArrayLike<number>;
  readonly data: ArrayLike<number>;
  readonly nCells: number;
  readonly nVars: number;
}

/**
 * How much of the matrix must be read to extract `nWanted` of `nVars` columns.
 *
 * Returned as a fraction of the stored non-zeros so a caller can warn before starting. CSC pays
 * roughly in proportion to the selection; CSR pays in full whatever you ask for.
 */
export function sparseColumnCost(encoding: SparseEncoding, nWanted: number, nVars: number): number {
  if (nVars <= 0) return 0;
  return encoding === "csc" ? Math.min(1, nWanted / nVars) : 1;
}

/**
 * Densify the selected columns of a sparse `X` into a column-major `Float64Array`.
 *
 * `wanted` is a list of var indices; duplicates are honoured (the same gene twice gives two
 * identical columns) because de-duplicating silently would misalign the caller's labels.
 *
 * Out-of-range entries in `indices` are skipped rather than throwing. A truncated or mis-typed
 * store is a real possibility, and dropping the offending entry loses one value where indexing
 * blindly would corrupt a neighbouring gene's column — a far worse failure, and a silent one.
 */
export function sparseToColumns(m: SparseMatrix, wanted: readonly number[]): Float64Array {
  const { nCells, nVars } = m;
  const out = new Float64Array(wanted.length * nCells);
  if (wanted.length === 0 || nCells === 0) return out;

  // Where does var `v` land in the output? -1 for "not selected". Built once so the CSR walk below
  // is a single pass with an O(1) test per non-zero rather than a scan of `wanted`.
  const slotOf = new Int32Array(nVars).fill(-1);
  const duplicates: Array<{ from: number; to: number }> = [];
  wanted.forEach((v, slot) => {
    if (v < 0 || v >= nVars) return;
    if (slotOf[v] === -1) slotOf[v] = slot;
    else duplicates.push({ from: slotOf[v]!, to: slot });
  });

  if (m.encoding === "csc") {
    for (let slot = 0; slot < wanted.length; slot++) {
      const v = wanted[slot]!;
      if (v < 0 || v >= nVars) continue;
      const lo = Number(m.indptr[v] ?? 0);
      const hi = Number(m.indptr[v + 1] ?? lo);
      const base = slot * nCells;
      for (let k = lo; k < hi; k++) {
        const cell = Number(m.indices[k] ?? -1);
        if (cell >= 0 && cell < nCells) out[base + cell] = Number(m.data[k] ?? 0);
      }
    }
  } else {
    // CSR: one pass over every stored non-zero, keeping only those whose var is selected.
    for (let cell = 0; cell < nCells; cell++) {
      const lo = Number(m.indptr[cell] ?? 0);
      const hi = Number(m.indptr[cell + 1] ?? lo);
      for (let k = lo; k < hi; k++) {
        const v = Number(m.indices[k] ?? -1);
        if (v < 0 || v >= nVars) continue;
        const slot = slotOf[v]!;
        if (slot >= 0) out[slot * nCells + cell] = Number(m.data[k] ?? 0);
      }
    }
  }

  // A var asked for twice was only filled once above; copy it into its other slots.
  for (const { from, to } of duplicates) out.copyWithin(to * nCells, from * nCells, (from + 1) * nCells);
  return out;
}

/**
 * The dense counterpart: pull selected columns out of a row-major `[nCells, nVars]` block.
 *
 * Separate from `sparseToColumns` rather than folded behind one union, because the caller's *I/O*
 * differs completely — dense reads chunk rectangles, sparse reads three 1-D arrays — and a shared
 * signature would only paper over that.
 */
export function denseToColumns(data: ArrayLike<number>, nCells: number, nVars: number, wanted: readonly number[]): Float64Array {
  const out = new Float64Array(wanted.length * nCells);
  for (let slot = 0; slot < wanted.length; slot++) {
    const v = wanted[slot]!;
    if (v < 0 || v >= nVars) continue;
    const base = slot * nCells;
    for (let i = 0; i < nCells; i++) out[base + i] = Number(data[i * nVars + v] ?? 0);
  }
  return out;
}

export interface ColumnStats {
  readonly label: string;
  /** Σ w over cells — the channel's total mark mass, which is what the Gram normalisation divides by. */
  readonly total: number;
  readonly max: number;
  /** How many cells carry a non-zero value. */
  readonly nonZero: number;
}

/**
 * Per-column summaries, for the UI to show what was actually loaded.
 *
 * Worth surfacing rather than assuming: a gene selected out of curiosity may be expressed in three
 * cells, and a co-location statistic computed over three points is noise with a confident-looking
 * number attached. The `nonZero` count is the cheapest available warning.
 */
export function columnStats(values: ArrayLike<number>, nCells: number, labels: readonly string[]): ColumnStats[] {
  return labels.map((label, g) => {
    let total = 0;
    let max = 0;
    let nonZero = 0;
    const base = g * nCells;
    for (let i = 0; i < nCells; i++) {
      const v = Number(values[base + i] ?? 0);
      if (v !== 0) nonZero++;
      total += v;
      if (v > max) max = v;
    }
    return { label, total, max, nonZero };
  });
}
