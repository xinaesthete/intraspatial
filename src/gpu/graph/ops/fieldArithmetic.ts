// Basic arithmetic and linear-algebra operators over fields, element-aware (ADR-0004).
//
// add/sub/scale are lane-wise linear, so they work for *any* element (scalar, complex,
// vec, quaternion). `mul` is the element's algebra product (ordinary / complex / Hamilton
// — vec is rejected). dot/cross/normalize are the vector operators. All are CPU Tier-1
// ops; element preconditions are checked at graph-build time in `inferElements`.

import { addFields, crossFields, dotFields, mulFields, normalize, scaleField, subFields } from "../elementMath";
import type { ElementType, FieldValue, ResolvedPlacement, Shape } from "../handle";
import { elementLabel, elementsEqual, shapesEqual, systemsAgree } from "../handle";
import type { OpType, Params } from "../op";

/** Placement agreement for a pointwise binary op (ADR-0018 decision 3): reject combining two
 *  fields in different coordinate systems (or a placed + an array-space one — `systemsAgree`
 *  throws on that), then pass the first input's placement through (a pointwise op does not move
 *  the field). Shared by add/sub/mul/dot/cross — their first uses of the agreement check. */
function agreePlacement(op: string, inputs: (ResolvedPlacement | undefined)[]): (ResolvedPlacement | undefined)[] {
  if (!systemsAgree(inputs[0], inputs[1])) {
    throw new Error(`${op}: inputs are in different coordinate systems (cannot combine across systems)`);
  }
  return [inputs[0]];
}

const SCALAR: ElementType = { kind: "scalar" };

function requireSameShape(a: Shape, b: Shape, op: string): Shape {
  if (!shapesEqual(a, b)) throw new Error(`${op}: input shapes differ`);
  return a;
}

function requireSameElement(a: ElementType, b: ElementType, op: string): ElementType {
  if (!elementsEqual(a, b)) throw new Error(`${op}: element mismatch (${elementLabel(a)} vs ${elementLabel(b)})`);
  return a;
}

/** A pointwise binary op whose output element equals its (matching) input element.
 *  `validate` runs at graph-build time to reject elements the op doesn't support. */
function binaryOp(
  name: string,
  label: string,
  describe: string,
  combine: (el: ElementType, a: ArrayLike<number>, b: ArrayLike<number>) => Float32Array,
  validate?: (el: ElementType) => void,
): OpType {
  const body = (inputs: FieldValue[]): FieldValue[] => {
    const el = inputs[0]!.element ?? SCALAR;
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: el, data: combine(el, inputs[0]!.data!, inputs[1]!.data!) }];
  };
  return {
    name,
    label,
    describe,
    inputs: [
      { name: "a", kind: "any" },
      { name: "b", kind: "any" },
    ],
    outputs: [{ name: "out", kind: "any", dtype: "f32" }],
    params: [],
    inferShapes: (inputs) => [requireSameShape(inputs[0]!, inputs[1]!, name)],
    inferElements: (inputs) => {
      const el = requireSameElement(inputs[0]!, inputs[1]!, name);
      validate?.(el);
      return [el];
    },
    inferPlacement: (inputs) => agreePlacement(name, inputs),
    execute: async (_ctx, inputs) => body(inputs),
    cpuGolden: (inputs) => body(inputs),
  };
}

export const addFieldsOp = binaryOp("addFields", "Add", "Pointwise sum a + b (any matching element).", (_el, a, b) => addFields(a, b));
export const subFieldsOp = binaryOp("subFields", "Subtract", "Pointwise difference a − b (any matching element).", (_el, a, b) =>
  subFields(a, b),
);

// Algebra product a · b. CPU Tier-1 like the rest of the element ops (and `addGrids`).
// A GPU kernel exists (`complexMulGpu.ts`, validated by `element.gpu.test.ts`) but is
// deliberately NOT wired in here: importing that second `"use gpu"` module into the op
// registry pulled it into the module graph of *every* fork that loads the registry and
// destabilised Dawn-on-Node teardown in unrelated graph tests (ADR-0002/0003). Wiring
// it back in is a follow-up once that interaction is understood; correctness is
// unaffected (the math is identical to the kernel it mirrors).
export const mulFieldsOp = binaryOp(
  "mulFields",
  "Multiply",
  "Pointwise algebra product a · b (scalar ×, complex multiply, Hamilton product).",
  (el, a, b) => mulFields(el, a, b),
  (el) => {
    if (el.kind === "vec") throw new Error("mulFields: vec has no algebra product — use dotFields or crossFields");
  },
);

// --- the fuzzy-set operators (AND / OR / NOT) -------------------------------------------
//
// Lane-wise min, max and 1−a. Pointwise arithmetic like everything above, and they live
// here for that reason — but their *purpose* is composing the soft selection masks in
// `filterOps.ts`, and the choice of these three rather than any other triple is load
// bearing.
//
// ADR-0005's body specifies `AND = a·b` with `OR = max(a,b)`, which mixes a product t-norm
// with a Gödel t-conorm. They are not De Morgan dual (`1−(1−a)(1−b) = 0.92` against
// `max = 0.8` at a=0.8, b=0.6), and product AND is not idempotent (`a·a = 0.64` against
// `a = 0.8`). The second breaks a **diamond** — two branches off one upstream gate,
// recombined — which is the ordinary shape of a gating tree: the shared ancestor gets
// multiplied in twice and the answer starts depending on the graph's topology rather than
// on the set being described. Silently, because the mask merely gets dimmer.
//
// min/max/1−a are idempotent, De Morgan-consistent, associative and the cheapest of the
// three. Set difference is `min(a, 1−b)`. Every pair coincides on hard 0/1 masks, which is
// why the defect survived — soft membership is the whole point of the encoding. Full
// derivation in `docs/mdv-dimension-vs-support-facet.md` §6; ADR-0005 carries the
// correction in its status block.
//
// A caveat worth keeping visible rather than resolving: min/max are not *strict* — they
// ignore the non-extremal operand, so a soft brush ANDed with a soft window keeps the
// tighter one and loses the other's gradient. If a weighted-permutation null turns out to
// want that gradient, the product may be right *for that consumer*, which would be an
// argument for the operator being a property of the edge. That should be settled with a
// real null in hand, not on paper.

function lanewise(f: (x: number, y: number) => number) {
  return (a: ArrayLike<number>, b: ArrayLike<number>): Float32Array => {
    if (a.length !== b.length) throw new Error(`lane count mismatch: ${a.length} vs ${b.length}`);
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = f(a[i]!, b[i]!);
    return out;
  };
}

const minLanes = lanewise(Math.min);
const maxLanes = lanewise(Math.max);

export const minFieldsOp = binaryOp(
  "minFields",
  "Min (AND)",
  "Lane-wise min — fuzzy AND, and the intersection of two selection masks.",
  (_el, a, b) => minLanes(a, b),
);

export const maxFieldsOp = binaryOp(
  "maxFields",
  "Max (OR)",
  "Lane-wise max — fuzzy OR, and the union of two selection masks.",
  (_el, a, b) => maxLanes(a, b),
);

/** invert(a) = 1 − a: fuzzy NOT, and the complement of a selection mask. Involutive, so
 *  `invert(invert(a))` is `a` exactly — which `1/(1+a)` and friends are not. */
export const invertFieldOp: OpType = {
  name: "invertField",
  label: "Invert (NOT)",
  describe: "1 − a — fuzzy NOT, and the complement of a selection mask.",
  inputs: [{ name: "in", kind: "any" }],
  outputs: [{ name: "out", kind: "any", dtype: "f32" }],
  params: [],
  inferShapes: (inputs) => [inputs[0]!],
  inferElements: (inputs) => [inputs[0]!],
  async execute(_ctx, inputs) {
    return [invertField(inputs[0]!)];
  },
  cpuGolden(inputs) {
    return [invertField(inputs[0]!)];
  },
};

function invertField(v: FieldValue): FieldValue {
  const src = v.data!;
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = 1 - src[i]!;
  return { shape: v.shape, dtype: "f32", element: v.element ?? SCALAR, data: out };
}

/** scale(a): multiply every lane by a real scalar param. Element-preserving. */
export const scaleFieldOp: OpType = {
  name: "scaleField",
  label: "Scale",
  describe: "Multiply a field by a real scalar (element-preserving).",
  inputs: [{ name: "in", kind: "any" }],
  outputs: [{ name: "out", kind: "any", dtype: "f32" }],
  params: [{ name: "s", type: "number", default: 1, min: -8, max: 8, step: 0.1, describe: "scalar multiplier" }],
  inferShapes: (inputs) => [inputs[0]!],
  inferElements: (inputs) => [inputs[0]!],
  async execute(_ctx, inputs, params) {
    const el = inputs[0]!.element ?? SCALAR;
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: el, data: scaleField(inputs[0]!.data!, params.s as number) }];
  },
  cpuGolden(inputs, params: Params) {
    const el = inputs[0]!.element ?? SCALAR;
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: el, data: scaleField(inputs[0]!.data!, params.s as number) }];
  },
};

/** dot(a, b): pointwise vector dot product → a real (scalar) field. */
export const dotFieldsOp: OpType = {
  name: "dotFields",
  label: "Dot",
  describe: "Pointwise vector dot product a · b → a real field.",
  inputs: [
    { name: "a", kind: "any" },
    { name: "b", kind: "any" },
  ],
  outputs: [{ name: "out", kind: "any", dtype: "f32" }],
  params: [],
  inferShapes: (inputs) => [requireSameShape(inputs[0]!, inputs[1]!, "dotFields")],
  inferElements(inputs) {
    const el = requireSameElement(inputs[0]!, inputs[1]!, "dotFields");
    if (el.kind !== "vec") throw new Error(`dotFields: requires vec, got ${elementLabel(el)}`);
    return [SCALAR];
  },
  inferPlacement: (inputs) => agreePlacement("dotFields", inputs),
  async execute(_ctx, inputs) {
    const el = inputs[0]!.element ?? SCALAR;
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: SCALAR, data: dotFields(el, inputs[0]!.data!, inputs[1]!.data!) }];
  },
  cpuGolden(inputs) {
    const el = inputs[0]!.element ?? SCALAR;
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: SCALAR, data: dotFields(el, inputs[0]!.data!, inputs[1]!.data!) }];
  },
};

/** cross(a, b): pointwise vec3 cross product → a vec3 field. */
export const crossFieldsOp: OpType = {
  name: "crossFields",
  label: "Cross",
  describe: "Pointwise vec3 cross product a × b → a vec3 field.",
  inputs: [
    { name: "a", kind: "any" },
    { name: "b", kind: "any" },
  ],
  outputs: [{ name: "out", kind: "any", dtype: "f32" }],
  params: [],
  inferShapes: (inputs) => [requireSameShape(inputs[0]!, inputs[1]!, "crossFields")],
  inferElements(inputs) {
    const el = requireSameElement(inputs[0]!, inputs[1]!, "crossFields");
    if (el.kind !== "vec" || el.n !== 3) throw new Error(`crossFields: requires vec3, got ${elementLabel(el)}`);
    return [el];
  },
  inferPlacement: (inputs) => agreePlacement("crossFields", inputs),
  async execute(_ctx, inputs) {
    const el = inputs[0]!.element ?? SCALAR;
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: el, data: crossFields(el, inputs[0]!.data!, inputs[1]!.data!) }];
  },
  cpuGolden(inputs) {
    const el = inputs[0]!.element ?? SCALAR;
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: el, data: crossFields(el, inputs[0]!.data!, inputs[1]!.data!) }];
  },
};

/** normalize(a): scale each sample to unit magnitude (vec / quaternion). */
export const normalizeFieldOp: OpType = {
  name: "normalizeField",
  label: "Normalize",
  describe: "Scale each sample to unit magnitude (vec / quaternion).",
  inputs: [{ name: "in", kind: "any" }],
  outputs: [{ name: "out", kind: "any", dtype: "f32" }],
  params: [],
  inferShapes: (inputs) => [inputs[0]!],
  inferElements(inputs) {
    const el = inputs[0]!;
    if (el.kind !== "vec" && el.kind !== "quaternion") {
      throw new Error(`normalizeField: requires vec or quaternion, got ${elementLabel(el)}`);
    }
    return [el];
  },
  async execute(_ctx, inputs) {
    const el = inputs[0]!.element ?? SCALAR;
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: el, data: normalize(el, inputs[0]!.data!) }];
  },
  cpuGolden(inputs) {
    const el = inputs[0]!.element ?? SCALAR;
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: el, data: normalize(el, inputs[0]!.data!) }];
  },
};
