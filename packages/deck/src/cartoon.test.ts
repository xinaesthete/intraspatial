import { describe, expect, it, vi } from "vite-plus/test";

vi.mock("intraspatial/color", () => import("../../../src/color/index"));

import { cartoonProfile } from "./cartoon";
import { ease, mixProfiles, NO_ORIENT, ORIENT_WORDS, packTransition, roundProfile, SHAPE, SHAPE_FLOATS } from "./tubeProfile";

const shape = (p: { shapes: Float32Array }, i: number) => Array.from(p.shapes.subarray(i * SHAPE_FLOATS, i * SHAPE_FLOATS + 4));
const orient = (p: { orients: Uint32Array }, i: number) => Array.from(p.orients.subarray(i * ORIENT_WORDS, (i + 1) * ORIENT_WORDS));
const STYLE = { coil: 1, narrow: 0.5, broad: 4, arrow: 2 };

describe("cartoonProfile", () => {
  it("with no structures is a round tube", () => {
    const p = cartoonProfile(5, [], STYLE);
    expect(Array.from(p.shapes)).toEqual(Array.from(roundProfile(5).shapes));
    expect(Array.from(p.orients)).toEqual(new Array(20).fill(NO_ORIENT));
  });

  it("makes a helix a ribbon, coil-sized at its ends, facing along particles one turn apart", () => {
    const p = cartoonProfile(12, [{ kind: "helix", start: 2, end: 9, turn: 3 }], STYLE);
    expect(shape(p, 1)).toEqual([1, 1, 1, 1]);
    expect(shape(p, 2)).toEqual([1, 1, 1, 1]);
    expect(shape(p, 5)).toEqual([4, 4, 0.5, 0.5]);
    expect(shape(p, 9)).toEqual([1, 1, 1, 1]);
    expect(p.shapes[5 * SHAPE_FLOATS + SHAPE.colour + 3]).toBe(1);
    expect(p.shapes[1 * SHAPE_FLOATS + SHAPE.colour + 3]).toBe(0);
    // Mid-helix: a turn back and a turn forward. Near the start the backward turn is the first one.
    expect(orient(p, 5)).toEqual([2, 5, 5, 8]);
    expect(orient(p, 3)).toEqual([2, 5, 3, 6]);
    expect(orient(p, 1)).toEqual([NO_ORIENT, NO_ORIENT, NO_ORIENT, NO_ORIENT]);
  });

  it("makes a strand an arrow that faces its partner strand", () => {
    // Strands 0..4 and 10..6, facing each other.
    const pairs = [0, 1, 2, 3, 4].map((a) => [a, 10 - a] as const);
    const p = cartoonProfile(12, [{ kind: "sheet", pairs }], STYLE);
    expect(shape(p, 0)).toEqual([1, 1, 1, 1]);
    expect(shape(p, 2)).toEqual([4, 4, 0.5, 0.5]);
    // The last full-size particle of each strand leaves at the arrowhead's width.
    expect(shape(p, 3)).toEqual([4, 8, 0.5, 0.5]);
    expect(shape(p, 9)).toEqual([4, 8, 0.5, 0.5]);
    expect(shape(p, 4)).toEqual([1, 1, 1, 1]);
    expect(orient(p, 2)).toEqual([8, 2, 8, 2]);
    expect(orient(p, 8)).toEqual([2, 8, 2, 8]);
  });

  it("faces a strand between two partners from one to the other", () => {
    const first = [0, 1, 2, 3].map((a) => [a, 9 - a] as const);
    const second = [6, 7, 8, 9].map((a) => [a, 21 - a] as const);
    const p = cartoonProfile(16, [
      { kind: "sheet", pairs: first },
      { kind: "sheet", pairs: second },
    ]);
    expect(orient(p, 7)).toEqual([2, 7, 7, 14]);
  });

  it("shows the structure a particle is deeper inside", () => {
    const p = cartoonProfile(
      20,
      [
        { kind: "helix", start: 0, end: 10, turn: 3 },
        { kind: "sheet", pairs: [8, 9, 10, 11, 12].map((a) => [a, 27 - a] as const) },
      ],
      { ...STYLE, helixSquare: 0, sheetSquare: 1 },
    );
    expect(p.shapes[5 * SHAPE_FLOATS + SHAPE.square]).toBe(0);
    expect(p.shapes[10 * SHAPE_FLOATS + SHAPE.square]).toBe(1);
  });
});

describe("profile changes", () => {
  const round = roundProfile(3);
  const ribbon = cartoonProfile(3, [{ kind: "helix", start: 0, end: 2, turn: 1 }], { ...STYLE, coil: 4 });

  it("eases from 0 to 1 and stays there", () => {
    expect([ease(-1), ease(0), ease(0.5), ease(1), ease(2)]).toEqual([0, 0, 0.5, 1, 1]);
  });

  it("mixes sizes and keeps whichever direction exists", () => {
    const half = mixProfiles(round, ribbon, 0.5);
    expect(shape(half, 1)).toEqual([2.5, 2.5, 0.75, 0.75]);
    expect(orient(mixProfiles(round, ribbon, 0.1), 1)).toEqual(orient(ribbon, 1));
    expect(orient(mixProfiles(ribbon, round, 0.9), 1)).toEqual(orient(ribbon, 1));
  });

  it("packs both ends per particle, sharing a direction only one end names", () => {
    const packed = packTransition(round, ribbon);
    expect(packed.shapes.length).toBe(3 * SHAPE_FLOATS * 2);
    expect(Array.from(packed.shapes.subarray(SHAPE_FLOATS * 2, SHAPE_FLOATS * 2 + 4))).toEqual([1, 1, 1, 1]);
    expect(Array.from(packed.shapes.subarray(SHAPE_FLOATS * 3, SHAPE_FLOATS * 3 + 4))).toEqual(shape(ribbon, 1));
    expect(Array.from(packed.orients.subarray(ORIENT_WORDS * 2, ORIENT_WORDS * 3))).toEqual(orient(ribbon, 1));
    expect(Array.from(packed.orients.subarray(ORIENT_WORDS * 3, ORIENT_WORDS * 4))).toEqual(orient(ribbon, 1));
  });
});
