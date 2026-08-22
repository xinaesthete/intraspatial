import { beforeAll, describe, expect, it } from "vitest";
import { getDevice } from "../../device";
import { Graph, pullData } from "../index";

// The graph node must agree with its `cpuGolden` exactly (min/max involves no arithmetic), and
// a resident chain — threshold → open → convolve — must run without any host round-trip breaking
// the bit-exactness of the middle op. Mirrors the LIDAR foliage chain: dz ≥ t → r=1 opening.

const W = 24;
const H = 20;

function firstMismatch(a: Float32Array, b: Float32Array): string | undefined {
  if (a.length !== b.length) return `length ${a.length} != ${b.length}`;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return `index ${i}: got ${a[i]}, want ${b[i]}`;
  return undefined;
}

describe("morphology graph op", () => {
  beforeAll(async () => {
    await getDevice();
  });

  it("every mode matches cpuGolden bit-exactly on a grey-scale field", async () => {
    const field = new Float32Array(W * H);
    for (let i = 0; i < field.length; i++) field[i] = ((i * 2654435761) % 89) / 89;
    for (const op of ["erode", "dilate", "open", "close"]) {
      const g = new Graph();
      const out = g.op1("morphology", { grid: g.grid(field, W, H) }, { op, radius: 2 });
      const gpu = await pullData(g, out);
      const cpu = await pullData(g, out, { mode: "cpu" });
      expect(firstMismatch(gpu, cpu), op).toBeUndefined();
      expect(new Set(Array.from(gpu)).size).toBeGreaterThan(1);
    }
  });

  it("resident chain threshold -> opening (r=1) drops a thin outline, keeps a blob", async () => {
    // dz-like field: 0 everywhere, a 1-px hollow square at 3 m, a 5x5 block at 4 m.
    const dz = new Float32Array(W * H);
    for (let i = 3; i <= 10; i++) {
      dz[3 * W + i] = 3;
      dz[10 * W + i] = 3;
      dz[i * W + 3] = 3;
      dz[i * W + 10] = 3;
    }
    for (let y = 13; y <= 17; y++) for (let x = 13; x <= 17; x++) dz[y * W + x] = 4;
    const g = new Graph();
    const mask = g.op1("threshold", { in: g.grid(dz, W, H) }, { thresh: 1, soft: false });
    const canopy = g.op1("morphology", { grid: mask }, { op: "open", radius: 1 });
    const gpu = await pullData(g, canopy);
    const cpu = await pullData(g, canopy, { mode: "cpu" });
    expect(firstMismatch(gpu, cpu)).toBeUndefined();
    let total = 0;
    for (const v of gpu) total += v;
    expect(total).toBe(25); // only the block survives
    expect(gpu[3 * W + 5]).toBe(0);
  });
});
