// Vector-calculus ops on known fields (ADR-0004 element model in use). CPU, fast.
import { describe, it, expect } from "vitest";
import type { FieldValue, Shape } from "./index";
import { gradientOp, gradientMagnitudeOp, laplacianOp, divergenceOp, structureOrientationOp } from "./ops/fieldCalculus";

const W = 16, H = 16;
const gridShape: Shape = { kind: "grid", width: W, height: H };

function scalar(f: (x: number, y: number) => number): FieldValue {
  const data = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) data[y * W + x] = f(x, y);
  return { shape: gridShape, dtype: "f32", element: { kind: "scalar" }, data };
}
function vec2(fx: (x: number, y: number) => number, fy: (x: number, y: number) => number): FieldValue {
  const data = new Float32Array(W * H * 2);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const i = (y * W + x) * 2; data[i] = fx(x, y); data[i + 1] = fy(x, y); }
  return { shape: gridShape, dtype: "f32", element: { kind: "vec", n: 2 }, data };
}
const interior = (x: number, y: number) => x > 0 && x < W - 1 && y > 0 && y < H - 1;
const near = (a: number, b: number, e = 1e-4) => Math.abs(a - b) < e;

describe("field calculus", () => {
  it("gradient of a linear ramp is the constant slope (and is a vec2 field)", () => {
    const [out] = gradientOp.cpuGolden!([scalar((x, y) => 2 * x + 3 * y)], {});
    expect(out!.element).toEqual({ kind: "vec", n: 2 });
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (!interior(x, y)) continue;
      const i = (y * W + x) * 2;
      expect(near(out!.data![i]!, 2) && near(out!.data![i + 1]!, 3)).toBe(true);
    }
  });

  it("Laplacian of x² is 2 in the interior; of a ramp is 0", () => {
    const [sq] = laplacianOp.cpuGolden!([scalar((x) => x * x)], {});
    const [ramp] = laplacianOp.cpuGolden!([scalar((x, y) => x + y)], {});
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (!interior(x, y)) continue;
      expect(near(sq!.data![y * W + x]!, 2)).toBe(true);
      expect(near(ramp!.data![y * W + x]!, 0)).toBe(true);
    }
  });

  it("divergence of v=(x,y) is 2; gradient magnitude of a ramp is √13", () => {
    const [div] = divergenceOp.cpuGolden!([vec2((x) => x, (_x, y) => y)], {});
    const [mag] = gradientMagnitudeOp.cpuGolden!([scalar((x, y) => 2 * x + 3 * y)], {});
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (!interior(x, y)) continue;
      expect(near(div!.data![y * W + x]!, 2)).toBe(true);
      expect(near(mag!.data![y * W + x]!, Math.sqrt(13), 1e-3)).toBe(true);
    }
    expect(div!.element).toEqual({ kind: "scalar" });
  });

  it("structure orientation of vertical stripes points along x with high coherence", () => {
    // f varies only in x → gradient is along x → dominant eigenvector ≈ (±1, 0).
    const [o] = structureOrientationOp.cpuGolden!([scalar((x) => Math.cos((2 * Math.PI * 3 * x) / W))], { radius: 2 });
    expect(o!.element).toEqual({ kind: "vec", n: 2 });
    const cx = 8, cy = 8, i = (cy * W + cx) * 2;
    const ox = o!.data![i]!, oy = o!.data![i + 1]!;
    expect(Math.abs(oy)).toBeLessThan(0.1); // ~no y component
    expect(Math.hypot(ox, oy)).toBeGreaterThan(0.5); // coherent (oriented)
  });

  it("rejects wrong element at build-ish time (divergence needs vec, gradient needs scalar)", () => {
    expect(() => divergenceOp.inferElements!([{ kind: "scalar" }], {})).toThrow(/vec/);
    expect(() => gradientOp.inferElements!([{ kind: "vec", n: 2 }], {})).toThrow(/scalar/);
  });
});
