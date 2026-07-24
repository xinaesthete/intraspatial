import { describe, expect, it } from "vitest";
import { groupCsvCells, inspectCsv, parseCsv } from "./cellCsv";

describe("parseCsv", () => {
  it("handles RFC-4180 quoting — the thing a split(',') parser corrupts silently", () => {
    const rows = parseCsv('a,b,c\n1,"hello, world",3\n4,"say ""hi""",6\n');
    expect(rows).toEqual([
      ["a", "b", "c"],
      ["1", "hello, world", "3"],
      ["4", 'say "hi"', "6"],
    ]);
  });

  it("keeps newlines inside quotes, and tolerates CRLF and a missing final newline", () => {
    expect(parseCsv('a,b\r\n1,"two\nlines"')).toEqual([
      ["a", "b"],
      ["1", "two\nlines"],
    ]);
  });

  it("drops only the trailing blank line", () => {
    expect(parseCsv("a\n1\n2\n")).toEqual([["a"], ["1"], ["2"]]);
  });
});

describe("inspectCsv", () => {
  const csv = ["x,y,cell_type,area", "1,2,T cell,10.5", "3,4,Tumour,11.5", "5,6,T cell,9.5"].join("\n");

  it("finds x, y and the type column by name", () => {
    const s = inspectCsv(parseCsv(csv));
    expect(s.suggestedX).toBe("x");
    expect(s.suggestedY).toBe("y");
    expect(s.suggestedType).toBe("cell_type");
    expect(s.nRows).toBe(3);
  });

  it("falls back to the SHAPE of the data when the headers are unhelpful", () => {
    // No recognisable names: coordinates are numeric with many distinct values, the type column has
    // few — that is the only signal left, and it is enough to make a sensible default.
    const odd = ["c0,c1,c2", "1,2,A", "3,4,B", "5,6,A", "7,8,B", "9,10,A"].join("\n");
    const s = inspectCsv(parseCsv(odd));
    expect(s.suggestedX).toBe("c0");
    expect(s.suggestedY).toBe("c1");
    expect(s.suggestedType).toBe("c2");
  });

  it("reports numeric fraction and distinct counts per column", () => {
    const s = inspectCsv(parseCsv(csv));
    expect(s.columns.find((c) => c.name === "x")!.numericFraction).toBe(1);
    expect(s.columns.find((c) => c.name === "cell_type")!.numericFraction).toBe(0);
    expect(s.columns.find((c) => c.name === "cell_type")!.distinct).toBe(2);
  });
});

describe("groupCsvCells", () => {
  const rows = parseCsv(["x,y,cell_type", "1,2,T cell", "3,4,Tumour", "5,6,T cell", "bad,7,Tumour", "8,9,"].join("\n"));
  const opts = { xColumn: "x", yColumn: "y", typeColumn: "cell_type" };

  it("groups by type in first-appearance order, keeping the labels", () => {
    const g = groupCsvCells(rows, opts);
    expect(g.order).toEqual(["T cell", "Tumour"]);
    expect(g.clouds.get("T cell")).toEqual({ xs: [1, 5], ys: [2, 6] });
    expect(g.clouds.get("Tumour")!.xs).toEqual([3]);
  });

  it("skips unparseable rows rather than poisoning a cloud with NaN, and counts them", () => {
    const g = groupCsvCells(rows, opts);
    expect(g.skipped).toBe(2); // the "bad" x, and the blank type
    for (const c of g.clouds.values()) expect(c.xs.every(Number.isFinite)).toBe(true);
  });

  it("rejects a column that is not there, naming what is", () => {
    expect(() => groupCsvCells(rows, { ...opts, typeColumn: "nope" })).toThrow(/nope.*x, y, cell_type/s);
  });
});
