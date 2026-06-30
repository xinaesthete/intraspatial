// Vector calculus on grid fields: gradient, divergence, Laplacian, and the
// structure-tensor orientation (a closed-form 2×2 symmetric eigen-decomposition — the
// "eigenvectors" of a scalar field's local geometry). These exercise the element model
// directly: the gradient of a scalar field is a `vec` field, divergence takes one back
// to scalar. CPU Tier-1; opt-in registered (registerElementOps) like the rest.
import type { ElementType, FieldValue, Shape } from "../handle";
import { elementLabel } from "../handle";
import type { OpType, Params } from "../op";

const SCALAR: ElementType = { kind: "scalar" };
const VEC2: ElementType = { kind: "vec", n: 2 };

function grid(s: Shape): { w: number; h: number } {
  if (s.kind !== "grid") throw new Error("fieldCalculus: input must be a grid");
  return { w: s.width, h: s.height };
}

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);
/** Sample a scalar grid with clamped (replicate) boundaries. */
const sample = (d: ArrayLike<number>, w: number, h: number, x: number, y: number) =>
  d[clamp(y, 0, h - 1) * w + clamp(x, 0, w - 1)]!;

/** Central-difference gradient of a scalar field → an interleaved vec2 field [fx, fy]. */
function gradient(d: ArrayLike<number>, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h * 2);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const fx = (sample(d, w, h, x + 1, y) - sample(d, w, h, x - 1, y)) * 0.5;
      const fy = (sample(d, w, h, x, y + 1) - sample(d, w, h, x, y - 1)) * 0.5;
      const i = (y * w + x) * 2;
      out[i] = fx;
      out[i + 1] = fy;
    }
  }
  return out;
}

/** Divergence of a vec2 field [vx, vy] → a scalar field ∂vx/∂x + ∂vy/∂y. */
function divergence(d: ArrayLike<number>, w: number, h: number): Float32Array {
  const vx = (x: number, y: number) => d[(clamp(y, 0, h - 1) * w + clamp(x, 0, w - 1)) * 2]!;
  const vy = (x: number, y: number) => d[(clamp(y, 0, h - 1) * w + clamp(x, 0, w - 1)) * 2 + 1]!;
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      out[y * w + x] = (vx(x + 1, y) - vx(x - 1, y)) * 0.5 + (vy(x, y + 1) - vy(x, y - 1)) * 0.5;
    }
  }
  return out;
}

/** 5-point Laplacian of a scalar field. */
function laplacian(d: ArrayLike<number>, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      out[y * w + x] =
        sample(d, w, h, x + 1, y) + sample(d, w, h, x - 1, y) +
        sample(d, w, h, x, y + 1) + sample(d, w, h, x, y - 1) -
        4 * sample(d, w, h, x, y);
    }
  }
  return out;
}

/** Per-sample gradient magnitude |∇f| → a scalar field. */
function gradMag(d: ArrayLike<number>, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const fx = (sample(d, w, h, x + 1, y) - sample(d, w, h, x - 1, y)) * 0.5;
      const fy = (sample(d, w, h, x, y + 1) - sample(d, w, h, x, y - 1)) * 0.5;
      out[y * w + x] = Math.hypot(fx, fy);
    }
  }
  return out;
}

/** Structure-tensor orientation. Smooth the outer product of the gradient over a box
 *  window, then take the dominant eigenvector of the 2×2 symmetric tensor
 *  J = [[Jxx, Jxy], [Jxy, Jyy]] in closed form. Output is a vec2 = the principal
 *  eigenvector (dominant gradient direction) scaled by the local coherence
 *  (λ₊−λ₋)/(λ₊+λ₋) ∈ [0,1], so |output| is how oriented the neighbourhood is. */
function structureOrientation(d: ArrayLike<number>, w: number, h: number, radius: number): Float32Array {
  // Per-pixel gradient outer-product components.
  const gxx = new Float32Array(w * h), gyy = new Float32Array(w * h), gxy = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const fx = (sample(d, w, h, x + 1, y) - sample(d, w, h, x - 1, y)) * 0.5;
      const fy = (sample(d, w, h, x, y + 1) - sample(d, w, h, x, y - 1)) * 0.5;
      const i = y * w + x;
      gxx[i] = fx * fx; gyy[i] = fy * fy; gxy[i] = fx * fy;
    }
  }
  const r = Math.max(0, radius | 0);
  const boxAt = (g: Float32Array, x: number, y: number) => {
    let s = 0, n = 0;
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
      s += g[clamp(y + dy, 0, h - 1) * w + clamp(x + dx, 0, w - 1)]!; n++;
    }
    return s / n;
  };
  const out = new Float32Array(w * h * 2);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const Jxx = boxAt(gxx, x, y), Jyy = boxAt(gyy, x, y), Jxy = boxAt(gxy, x, y);
      // Closed-form symmetric 2×2 eigensystem.
      const tr = Jxx + Jyy;
      const diff = Jxx - Jyy;
      const disc = Math.sqrt(diff * diff + 4 * Jxy * Jxy);
      const lam1 = (tr + disc) * 0.5; // larger eigenvalue
      const lam2 = (tr - disc) * 0.5;
      const coherence = tr > 1e-12 ? (lam1 - lam2) / tr : 0;
      // Dominant eigenvector direction: θ = ½·atan2(2Jxy, Jxx−Jyy).
      const theta = 0.5 * Math.atan2(2 * Jxy, diff);
      const i = (y * w + x) * 2;
      out[i] = Math.cos(theta) * coherence;
      out[i + 1] = Math.sin(theta) * coherence;
    }
  }
  return out;
}

function requireElement(el: ElementType, ok: ElementType["kind"][], op: string): void {
  if (!ok.includes(el.kind)) throw new Error(`${op}: requires ${ok.join("/")}, got ${elementLabel(el)}`);
}

/** A unary grid op: validate input element, produce an output of a fixed element. */
function unary(
  name: string,
  label: string,
  describe: string,
  category: string,
  inEl: ElementType["kind"][],
  outEl: ElementType,
  compute: (d: ArrayLike<number>, w: number, h: number, params: Params) => Float32Array,
  params: OpType["params"] = [],
): OpType {
  const body = (inputs: FieldValue[], p: Params): FieldValue[] => {
    const { w, h } = grid(inputs[0]!.shape);
    return [{ shape: inputs[0]!.shape, dtype: "f32", element: outEl, data: compute(inputs[0]!.data!, w, h, p) }];
  };
  return {
    name, label, describe, category,
    inputs: [{ name: "in", kind: "grid" }],
    outputs: [{ name: "out", kind: "grid", dtype: "f32" }],
    params,
    inferShapes: (inputs) => {
      grid(inputs[0]!);
      return [inputs[0]!];
    },
    inferElements: (inputs) => {
      requireElement(inputs[0]!, inEl, name);
      return [outEl];
    },
    execute: async (_ctx, inputs, p) => body(inputs, p),
    cpuGolden: (inputs, p) => body(inputs, p),
  };
}

export const gradientOp = unary("gradient", "Gradient", "Central-difference gradient of a scalar field → a vec2 field.", "Linear algebra", ["scalar"], VEC2, (d, w, h) => gradient(d, w, h));
export const gradientMagnitudeOp = unary("gradientMagnitude", "Gradient magnitude", "Per-sample |∇f| of a scalar field.", "Linear algebra", ["scalar"], SCALAR, (d, w, h) => gradMag(d, w, h));
export const laplacianOp = unary("laplacian", "Laplacian", "5-point Laplacian ∇²f of a scalar field.", "Linear algebra", ["scalar"], SCALAR, (d, w, h) => laplacian(d, w, h));
export const divergenceOp = unary("divergence", "Divergence", "Divergence ∇·v of a vec2 field → a scalar field.", "Linear algebra", ["vec"], SCALAR, (d, w, h) => divergence(d, w, h));
export const structureOrientationOp = unary(
  "structureOrientation",
  "Structure orientation",
  "Dominant local orientation (structure-tensor eigenvector) × coherence → a vec2 field.",
  "Linear algebra",
  ["scalar"],
  VEC2,
  (d, w, h, p) => structureOrientation(d, w, h, (p.radius as number) ?? 2),
  [{ name: "radius", type: "int", default: 2, min: 0, max: 16, describe: "structure-tensor smoothing radius" }],
);
