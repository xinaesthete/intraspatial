import { describe, expect, it } from "vitest";
import { type Affine2, compose2, IDENTITY2, micrometresPer, type NgffAxis, resolveNgffXY } from "./ngffTransform";

const XY: NgffAxis[] = [
  { name: "x", type: "space", unit: "micrometer" },
  { name: "y", type: "space", unit: "micrometer" },
];
const CYX: NgffAxis[] = [
  { name: "c", type: "channel" },
  { name: "y", type: "space", unit: "unit" },
  { name: "x", type: "space", unit: "unit" },
];
/** SpatialData's shape: one transform tagged with its input and output coordinate systems. */
const tagged = (t: Record<string, unknown>, outName = "Leap034", axes: NgffAxis[] = XY) => ({
  ...t,
  input: { name: "xy", axes },
  output: { name: outName, axes },
});
const apply = (m: Affine2, x: number, y: number): [number, number] => [m.a * x + m.c * y + m.tx, m.b * x + m.d * y + m.ty];

describe("micrometresPer", () => {
  it("converts the length units", () => {
    expect(micrometresPer("micrometer")).toBe(1);
    expect(micrometresPer("µm")).toBe(1);
    expect(micrometresPer("nanometer")).toBe(1e-3);
    expect(micrometresPer("millimeter")).toBe(1e3);
    expect(micrometresPer("meter")).toBe(1e6);
  });

  it("returns undefined — NOT 1 — for anything that is not a stated length", () => {
    // The distinction the whole module exists for: "unknown scale" must not collapse into
    // "one micrometre". SpatialData writes the literal "unit" when nothing was specified, and
    // "pixel" is a real NGFF unit that is nonetheless not a physical length.
    expect(micrometresPer("unit")).toBeUndefined();
    expect(micrometresPer("pixel")).toBeUndefined();
    expect(micrometresPer(undefined)).toBeUndefined();
    expect(micrometresPer("")).toBeUndefined();
    expect(micrometresPer("furlong")).toBeUndefined();
  });
});

describe("compose2", () => {
  it("applies the first transform, then the second", () => {
    const scale: Affine2 = { a: 2, b: 0, c: 0, d: 3, tx: 0, ty: 0 };
    const shift: Affine2 = { ...IDENTITY2, tx: 10, ty: 20 };
    expect(apply(compose2(scale, shift), 1, 1)).toEqual([12, 23]); // scale then shift
    expect(apply(compose2(shift, scale), 1, 1)).toEqual([22, 63]); // shift then scale
  });
});

describe("resolveNgffXY", () => {
  it("reads a real SpatialData shapes element (identity, placeholder unit)", () => {
    // Verbatim from Leap034's shapes/Leap034_imc_cell_shapes.
    const cts = [
      tagged({ type: "identity" }, "Leap034", [
        { name: "x", type: "space", unit: "unit" },
        { name: "y", type: "space", unit: "unit" },
      ]),
    ];
    const r = resolveNgffXY(cts, { target: "Leap034" })!;
    expect(r.system).toBe("Leap034");
    expect(r.unit).toBe("unit");
    expect(r.micrometres).toBeUndefined(); // the store did NOT state a physical scale
    expect(apply(r.affine, 7, 9)).toEqual([7, 9]);
    expect(r.unsupported).toEqual([]);
  });

  it("reads scale and translation, honouring axis NAMES not positions", () => {
    // c,y,x ordering: a naive [0]/[1] read would take the channel scale as x.
    const cts = [tagged({ type: "scale", scale: [1, 0.5, 0.25] }, "Leap034", CYX)];
    const r = resolveNgffXY(cts, { target: "Leap034" })!;
    expect(apply(r.affine, 4, 4)).toEqual([1, 2]); // x·0.25, y·0.5
  });

  it("composes a sequence in order", () => {
    const cts = [
      tagged({
        type: "sequence",
        transformations: [
          { type: "scale", scale: [2, 2] },
          { type: "translation", translation: [100, 200] },
        ],
      }),
    ];
    const r = resolveNgffXY(cts, { target: "Leap034" })!;
    expect(apply(r.affine, 1, 1)).toEqual([102, 202]);
    expect(r.micrometres).toBe(1); // axes say micrometer
  });

  it("keeps rotation from an affine rather than flattening it to a scale", () => {
    // 90° rotation: (x,y) → (−y, x).
    const cts = [
      tagged({
        type: "affine",
        affine: [
          [0, -1, 0],
          [1, 0, 0],
        ],
      }),
    ];
    const r = resolveNgffXY(cts, { target: "Leap034" })!;
    const [x, y] = apply(r.affine, 1, 0);
    expect(x).toBeCloseTo(0, 10);
    expect(y).toBeCloseTo(1, 10);
  });

  it("picks the transform whose OUTPUT matches the requested system", () => {
    const cts = [tagged({ type: "scale", scale: [10, 10] }, "other"), tagged({ type: "scale", scale: [2, 2] }, "Leap034")];
    expect(apply(resolveNgffXY(cts, { target: "Leap034" })!.affine, 1, 1)).toEqual([2, 2]);
    expect(apply(resolveNgffXY(cts, { target: "other" })!.affine, 1, 1)).toEqual([10, 10]);
    expect(apply(resolveNgffXY(cts)!.affine, 1, 1)).toEqual([10, 10]); // no target ⇒ first
  });

  it("treats a BARE ngff list as a sequence, using the supplied axes", () => {
    // Dataset-level transforms carry no input/output; they compose rather than compete.
    const cts = [
      { type: "scale", scale: [1, 0.5, 0.5] },
      { type: "translation", translation: [0, 10, 20] },
    ];
    const r = resolveNgffXY(cts, { axes: CYX })!;
    expect(apply(r.affine, 4, 4)).toEqual([22, 12]); // x: 4·0.5+20, y: 4·0.5+10
    expect(r.unit).toBe("unit");
  });

  it("reports transform types it cannot honour instead of silently approximating", () => {
    const cts = [tagged({ type: "byDimension", transformations: [] })];
    const r = resolveNgffXY(cts, { target: "Leap034" })!;
    expect(r.unsupported).toEqual(["byDimension"]);
    expect(r.affine).toEqual(IDENTITY2); // fell back, and said so
  });

  it("returns undefined when there is nothing to resolve", () => {
    expect(resolveNgffXY(undefined)).toBeUndefined();
    expect(resolveNgffXY([])).toBeUndefined();
  });
});
