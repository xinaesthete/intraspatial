import { describe, expect, it } from "vitest";
import { columnStats, denseToColumns, type SparseMatrix, sparseColumnCost, sparseToColumns } from "./sparseColumns";

// One small matrix, written out densely, then encoded both ways. Every test compares against the
// dense truth, so a mistake in either encoding shows up as a disagreement with something obvious
// rather than as two wrong answers that happen to match.
//
//        var0  var1  var2  var3
// cell0    0    2.5    0     7
// cell1    1     0     0     0
// cell2    0     0    3.5    0
// cell3    4    -1     0     9
const N_CELLS = 4;
const N_VARS = 4;
const DENSE = [
  [0, 2.5, 0, 7],
  [1, 0, 0, 0],
  [0, 0, 3.5, 0],
  [4, -1, 0, 9],
].flat();

/** Row-compressed: one slice per CELL, `indices` are var ids. */
const CSR: SparseMatrix = {
  encoding: "csr",
  indptr: [0, 2, 3, 4, 7],
  indices: [1, 3, 0, 2, 0, 1, 3],
  data: [2.5, 7, 1, 3.5, 4, -1, 9],
  nCells: N_CELLS,
  nVars: N_VARS,
};

/** Column-compressed: one slice per VAR, `indices` are cell ids. */
const CSC: SparseMatrix = {
  encoding: "csc",
  indptr: [0, 2, 4, 5, 7],
  indices: [1, 3, 0, 3, 2, 0, 3],
  data: [1, 4, 2.5, -1, 3.5, 7, 9],
  nCells: N_CELLS,
  nVars: N_VARS,
};

/** The expected column-major output for a selection, straight off the dense table. */
function expected(wanted: number[]): number[] {
  const out: number[] = [];
  for (const v of wanted) for (let i = 0; i < N_CELLS; i++) out.push(DENSE[i * N_VARS + v]!);
  return out;
}

describe("sparseToColumns", () => {
  it("agrees with the dense matrix for both encodings, on every single column", () => {
    for (let v = 0; v < N_VARS; v++) {
      expect([...sparseToColumns(CSR, [v])], `csr var${v}`).toEqual(expected([v]));
      expect([...sparseToColumns(CSC, [v])], `csc var${v}`).toEqual(expected([v]));
    }
  });

  it("emits column-major output, so each gene's column is contiguous", () => {
    const wanted = [3, 1];
    const got = sparseToColumns(CSC, wanted);
    expect(got.length).toBe(wanted.length * N_CELLS);
    expect([...got]).toEqual(expected(wanted));
    // The layout claim, made concrete: slot 0 is var 3's whole column.
    expect([...got.subarray(0, N_CELLS)]).toEqual([7, 0, 0, 9]);
  });

  it("preserves the caller's order and does not sort the selection", () => {
    expect([...sparseToColumns(CSR, [2, 0, 3])]).toEqual(expected([2, 0, 3]));
    expect([...sparseToColumns(CSC, [2, 0, 3])]).toEqual(expected([2, 0, 3]));
  });

  it("honours a repeated var rather than silently de-duplicating it", () => {
    // De-duplicating would shift every later column against the caller's label list — a
    // misalignment that produces plausible numbers under the wrong names.
    for (const m of [CSR, CSC]) {
      const got = sparseToColumns(m, [1, 1, 0]);
      expect([...got], m.encoding).toEqual(expected([1, 1, 0]));
      expect([...got.subarray(0, N_CELLS)]).toEqual([...got.subarray(N_CELLS, 2 * N_CELLS)]);
    }
  });

  it("keeps negative values — expression columns are not always counts", () => {
    // var1 holds a −1 (a scaled/centred layer). Zero-filling is the default, so a real negative
    // must survive rather than be treated as absent.
    expect([...sparseToColumns(CSR, [1])]).toEqual([2.5, 0, 0, -1]);
    expect([...sparseToColumns(CSC, [1])]).toEqual([2.5, 0, 0, -1]);
  });

  it("skips out-of-range indices instead of corrupting a neighbouring column", () => {
    const broken: SparseMatrix = { ...CSR, indices: [1, 99, 0, 2, 0, 1, 3] };
    const got = sparseToColumns(broken, [3, 1]);
    // The bad entry was var 99 in cell 0; var 3 loses that cell, var 1 is untouched.
    expect([...got.subarray(0, N_CELLS)]).toEqual([0, 0, 0, 9]);
    expect([...got.subarray(N_CELLS)]).toEqual([2.5, 0, 0, -1]);
  });

  it("tolerates BigInt64Array indptr/indices, as an int64 zarr array produces", () => {
    const big: SparseMatrix = {
      ...CSC,
      indptr: BigInt64Array.from([0n, 2n, 4n, 5n, 7n]) as unknown as ArrayLike<number>,
      indices: BigInt64Array.from([1n, 3n, 0n, 3n, 2n, 0n, 3n]) as unknown as ArrayLike<number>,
    };
    expect([...sparseToColumns(big, [0, 3])]).toEqual(expected([0, 3]));
  });

  it("returns an empty result for an empty selection", () => {
    expect(sparseToColumns(CSR, []).length).toBe(0);
  });
});

describe("denseToColumns", () => {
  it("matches the sparse paths", () => {
    const wanted = [3, 0, 2];
    expect([...denseToColumns(DENSE, N_CELLS, N_VARS, wanted)]).toEqual(expected(wanted));
    expect([...denseToColumns(DENSE, N_CELLS, N_VARS, wanted)]).toEqual([...sparseToColumns(CSC, wanted)]);
  });
});

describe("sparseColumnCost", () => {
  it("reports CSC as proportional to the selection and CSR as a full scan", () => {
    // The asymmetry the UI has to warn about: picking 10 of 20000 genes reads 0.05% of a CSC and
    // 100% of a CSR, and scanpy writes CSR by default.
    expect(sparseColumnCost("csc", 10, 20_000)).toBeCloseTo(0.0005, 10);
    expect(sparseColumnCost("csr", 10, 20_000)).toBe(1);
    expect(sparseColumnCost("csc", 30_000, 20_000)).toBe(1); // clamped
    expect(sparseColumnCost("csr", 0, 0)).toBe(0);
  });
});

describe("columnStats", () => {
  it("summarises each column, counting non-zeros", () => {
    const values = sparseToColumns(CSC, [0, 1]);
    const stats = columnStats(values, N_CELLS, ["gA", "gB"]);
    expect(stats[0]).toEqual({ label: "gA", total: 5, max: 4, nonZero: 2 });
    // gB is [2.5, 0, 0, -1]: total 1.5, max 2.5 (not |−1|), two non-zero cells.
    expect(stats[1]).toEqual({ label: "gB", total: 1.5, max: 2.5, nonZero: 2 });
  });

  it("surfaces a near-empty gene, which is the case worth warning about", () => {
    const values = sparseToColumns(CSC, [2]);
    expect(columnStats(values, N_CELLS, ["gC"])[0]!.nonZero).toBe(1);
  });
});
