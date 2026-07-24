// CSV parsing and column inspection for cell tables — the dependency-free half.
//
// Most data that actually reaches us is not a well-formed SpatialData store; it is an export
// somebody made: one row per cell, an x, a y, and a column of type labels. Requiring a zarr store
// before any of the statistics can be looked at is the wrong bar, so this reads that export.
//
// Kept here, with no imports, for two reasons: the parsing is generic library code rather than
// playground glue (ADR-0015 boundary), and it is where the tests can actually run — the CPU suite
// covers `src/**`, not `playground/**`.
//
// The parser is small but not naive: it handles RFC-4180 quoting (commas and newlines inside
// quotes, "" escapes), because those appear the moment a label contains a comma — and a
// split(",") parser corrupts such a file SILENTLY, shifting every later column by one.

/** Split CSV text into rows of fields, honouring RFC-4180 quoting. */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  let i = 0;
  const push = () => {
    row.push(field);
    field = "";
  };
  const endRow = () => {
    push();
    // Drop the trailing blank line a file usually ends with, but keep genuinely empty middle rows
    // out too — neither is a cell.
    if (row.length > 1 || row[0] !== "") rows.push(row);
    row = [];
  };
  while (i < text.length) {
    const c = text[i]!;
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"'; // "" is an escaped quote
          i += 2;
          continue;
        }
        quoted = false;
        i++;
        continue;
      }
      field += c;
      i++;
      continue;
    }
    if (c === '"') {
      quoted = true;
      i++;
    } else if (c === ",") {
      push();
      i++;
    } else if (c === "\r") {
      i++; // CRLF — the \n does the work
    } else if (c === "\n") {
      endRow();
      i++;
    } else {
      field += c;
      i++;
    }
  }
  if (field !== "" || row.length > 0) endRow();
  return rows;
}

export interface CsvColumnInfo {
  readonly name: string;
  readonly index: number;
  /** Fraction of sampled values that parse as finite numbers. */
  readonly numericFraction: number;
  /** Distinct values seen in the sample — small counts suggest a type/label column. */
  readonly distinct: number;
}

export interface CsvSchema {
  readonly headers: string[];
  readonly columns: CsvColumnInfo[];
  readonly nRows: number;
  /** Best guesses, by name and then by shape of the data. */
  readonly suggestedX?: string;
  readonly suggestedY?: string;
  readonly suggestedType?: string;
}

const SAMPLE = 2000;

/** Inspect the parsed rows: which columns look like x, y, and the cell type?
 *
 *  Name matching first (it is nearly always right and is what a user expects), then a fallback on
 *  the shape of the data: coordinates are numeric with many distinct values, a type column has few. */
export function inspectCsv(rows: string[][]): CsvSchema {
  const headers = rows[0] ?? [];
  const body = rows.slice(1);
  const columns: CsvColumnInfo[] = headers.map((name, index) => {
    let numeric = 0;
    let seen = 0;
    const distinct = new Set<string>();
    for (let r = 0; r < Math.min(body.length, SAMPLE); r++) {
      const v = body[r]?.[index] ?? "";
      if (v === "") continue;
      seen++;
      if (distinct.size < 5000) distinct.add(v);
      if (Number.isFinite(Number(v))) numeric++;
    }
    return { name, index, numericFraction: seen ? numeric / seen : 0, distinct: distinct.size };
  });

  const byName = (re: RegExp) => columns.find((c) => re.test(c.name))?.name;
  const numericCols = columns.filter((c) => c.numericFraction > 0.95);
  const suggestedX = byName(/^(x|x_?(centroid|coord|position|um|px)?|centroid_?x|col)$/i) ?? numericCols[0]?.name;
  const suggestedY =
    byName(/^(y|y_?(centroid|coord|position|um|px)?|centroid_?y|row)$/i) ?? numericCols.find((c) => c.name !== suggestedX)?.name;
  // A type column: named like one, or else the non-coordinate column with the fewest distinct
  // values (and more than one — a constant column types nothing).
  const suggestedType =
    byName(/cell.?type|celltype|phenotype|cluster|annot|label|class|population|lineage/i) ??
    columns
      .filter((c) => c.name !== suggestedX && c.name !== suggestedY && c.distinct > 1 && c.distinct <= 200)
      .sort((a, b) => a.distinct - b.distinct)[0]?.name;

  return { headers, columns, nRows: body.length, suggestedX, suggestedY, suggestedType };
}

export interface CsvGroupOptions {
  readonly xColumn: string;
  readonly yColumn: string;
  readonly typeColumn: string;
}

export interface CsvGrouping {
  /** Distinct type labels in FIRST-APPEARANCE order — a file whose types are already in a
   *  meaningful order keeps it, which sorting would throw away. */
  readonly order: string[];
  /** Centroids per type label. Never merged into one cloud (ADR-0018). */
  readonly clouds: Map<string, { xs: number[]; ys: number[] }>;
  /** Rows dropped for an unparseable coordinate or a blank type — surfaced rather than hidden,
   *  because a file that silently loses half its cells should say so. */
  readonly skipped: number;
}

/** Group parsed CSV rows into per-type centroid clouds.
 *
 *  Type values stay STRINGS: a CSV of names ("T cell", "Tumour") works exactly as well as one of
 *  integers, and the names survive into the UI — the difference between a readable matrix and a
 *  grid of anonymous numbers. */
export function groupCsvCells(rows: readonly string[][], opts: CsvGroupOptions): CsvGrouping {
  const headers = rows[0] ?? [];
  const ix = headers.indexOf(opts.xColumn);
  const iy = headers.indexOf(opts.yColumn);
  const it = headers.indexOf(opts.typeColumn);
  if (ix < 0 || iy < 0 || it < 0) {
    throw new Error(
      `cellCsv: missing column(s) — x='${opts.xColumn}' y='${opts.yColumn}' type='${opts.typeColumn}'; have: ${headers.join(", ")}`,
    );
  }
  const order: string[] = [];
  const clouds = new Map<string, { xs: number[]; ys: number[] }>();
  let skipped = 0;
  for (let r = 1; r < rows.length; r++) {
    const row = rows[r]!;
    const x = Number(row[ix]);
    const y = Number(row[iy]);
    const t = row[it] ?? "";
    if (!Number.isFinite(x) || !Number.isFinite(y) || t === "") {
      skipped++;
      continue;
    }
    let bucket = clouds.get(t);
    if (!bucket) {
      bucket = { xs: [], ys: [] };
      clouds.set(t, bucket);
      order.push(t);
    }
    bucket.xs.push(x);
    bucket.ys.push(y);
  }
  return { order, clouds, skipped };
}
