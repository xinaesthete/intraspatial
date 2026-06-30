// The complex element type made usable: construct a complex field from two real
// grids, take it apart again, conjugate it, and read its modulus (ADR-0004). These
// are CPU Tier-1 ops (like `addGrids`) — the work is lane-wise and memory-bound; a
// GPU pass is a later optimisation, not a correctness concern. Element preconditions
// are checked in `inferElements`, i.e. at graph-build time.
import type { ElementType, Shape } from "../handle";
import { elementLabel, shapesEqual } from "../handle";
import type { OpType } from "../op";
import { conjugate, extractLane, magnitude, packComplex } from "../elementMath";

const COMPLEX: ElementType = { kind: "complex" };
const SCALAR: ElementType = { kind: "scalar" };

function requireSameShape(a: Shape, b: Shape, op: string): Shape {
  if (!shapesEqual(a, b)) throw new Error(`${op}: input shapes differ`);
  return a;
}

function requireElement(el: ElementType, ok: ElementType["kind"][], op: string): void {
  if (!ok.includes(el.kind)) throw new Error(`${op}: requires ${ok.join("/")}, got ${elementLabel(el)}`);
}

/** complex(re, im): pack two real grids into one complex field — the constructor that
 *  collapses the reaction–diffusion two-feedback workaround into a single signal. */
export const complexOp: OpType = {
  name: "complex",
  label: "Complex",
  describe: "Combine two real fields into a single complex field (re + i·im).",
  inputs: [
    { name: "re", kind: "any" },
    { name: "im", kind: "any" },
  ],
  outputs: [{ name: "out", kind: "any", dtype: "f32" }],
  params: [],
  inferShapes(inputs) {
    return [requireSameShape(inputs[0]!, inputs[1]!, "complex")];
  },
  inferElements(inputs) {
    requireElement(inputs[0]!, ["scalar"], "complex");
    requireElement(inputs[1]!, ["scalar"], "complex");
    return [COMPLEX];
  },
  async execute(_ctx, inputs) {
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: COMPLEX, data: packComplex(inputs[0]!.data!, inputs[1]!.data!) }];
  },
  cpuGolden(inputs) {
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: COMPLEX, data: packComplex(inputs[0]!.data!, inputs[1]!.data!) }];
  },
};

function laneOp(name: string, label: string, describe: string, lane: number): OpType {
  const body = (inputs: Parameters<OpType["execute"]>[1]) => [
    { shape: inputs[0]!.shape, dtype: "f32" as const, element: SCALAR, data: extractLane(COMPLEX, inputs[0]!.data!, lane) },
  ];
  return {
    name,
    label,
    describe,
    inputs: [{ name: "in", kind: "any" }],
    outputs: [{ name: "out", kind: "any", dtype: "f32" }],
    params: [],
    inferShapes: (inputs) => [inputs[0]!],
    inferElements(inputs) {
      requireElement(inputs[0]!, ["complex"], name);
      return [SCALAR];
    },
    execute: async (_ctx, inputs) => body(inputs),
    cpuGolden: (inputs) => body(inputs),
  };
}

export const realPartOp = laneOp("realPart", "Real part", "Extract the real component of a complex field.", 0);
export const imagPartOp = laneOp("imagPart", "Imag part", "Extract the imaginary component of a complex field.", 1);

/** conjugate(a): complex/quaternion conjugate; element-preserving. */
export const conjugateOp: OpType = {
  name: "conjugate",
  label: "Conjugate",
  describe: "Complex / quaternion conjugate (negate the imaginary / vector part).",
  inputs: [{ name: "in", kind: "any" }],
  outputs: [{ name: "out", kind: "any", dtype: "f32" }],
  params: [],
  inferShapes: (inputs) => [inputs[0]!],
  inferElements(inputs) {
    requireElement(inputs[0]!, ["complex", "quaternion"], "conjugate");
    return [inputs[0]!];
  },
  async execute(_ctx, inputs) {
    const el = inputs[0]!.element ?? SCALAR;
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: el, data: conjugate(el, inputs[0]!.data!) }];
  },
  cpuGolden(inputs) {
    const el = inputs[0]!.element ?? SCALAR;
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: el, data: conjugate(el, inputs[0]!.data!) }];
  },
};

/** magnitude(a): per-sample |a| of any element → a real (scalar) field. */
export const magnitudeOp: OpType = {
  name: "magnitude",
  label: "Magnitude",
  describe: "Per-sample magnitude |a| of any element → a real field.",
  inputs: [{ name: "in", kind: "any" }],
  outputs: [{ name: "out", kind: "any", dtype: "f32" }],
  params: [],
  inferShapes: (inputs) => [inputs[0]!],
  inferElements() {
    return [SCALAR];
  },
  async execute(_ctx, inputs) {
    const el = inputs[0]!.element ?? SCALAR;
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: SCALAR, data: magnitude(el, inputs[0]!.data!) }];
  },
  cpuGolden(inputs) {
    const el = inputs[0]!.element ?? SCALAR;
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: SCALAR, data: magnitude(el, inputs[0]!.data!) }];
  },
};
