// Basic arithmetic and linear-algebra operators over fields, element-aware (ADR-0004).
//
// add/sub/scale are lane-wise linear, so they work for *any* element (scalar, complex,
// vec, quaternion). `mul` is the element's algebra product (ordinary / complex / Hamilton
// — vec is rejected). dot/cross/normalize are the vector operators. All are CPU Tier-1
// ops; element preconditions are checked at graph-build time in `inferElements`.
import type { ElementType, FieldValue, Shape } from "../handle";
import { elementLabel, elementsEqual, shapesEqual } from "../handle";
import type { OpType, Params } from "../op";
import { addFields, crossFields, dotFields, mulFields, normalize, scaleField, subFields } from "../elementMath";

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
    execute: async (_ctx, inputs) => body(inputs),
    cpuGolden: (inputs) => body(inputs),
  };
}

export const addFieldsOp = binaryOp("addFields", "Add", "Pointwise sum a + b (any matching element).", (_el, a, b) => addFields(a, b));
export const subFieldsOp = binaryOp("subFields", "Subtract", "Pointwise difference a − b (any matching element).", (_el, a, b) => subFields(a, b));

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
