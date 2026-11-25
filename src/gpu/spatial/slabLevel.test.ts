import { describe, expect, it } from "vite-plus/test";
import { slabLevel } from "./splatVolume";

// The kernel summed over an even slab, by brute quadrature: points on a fine grid through the slab,
// each weighing its share of `perArea`, and the kernel read at the slab's middle.
function quadrature(perArea: number, radius: number, thickness: number): number {
  const steps = 400;
  const layers = thickness === 0 ? 1 : 200;
  const h = (2 * radius) / steps;
  let sum = 0;
  for (let k = 0; k < layers; k++) {
    const z = thickness === 0 ? 0 : -thickness / 2 + ((k + 0.5) * thickness) / layers;
    for (let i = 0; i < steps; i++) {
      const x = -radius + (i + 0.5) * h;
      for (let j = 0; j < steps; j++) {
        const y = -radius + (j + 0.5) * h;
        const q = Math.max(1 - (x * x + y * y + z * z) / radius ** 2, 0);
        sum += q * q;
      }
    }
  }
  return (sum * h * h * perArea) / layers;
}

describe("slabLevel", () => {
  it("matches quadrature for a plane, thin and thick slabs", () => {
    const cases = [0, 3, 12, 23.9, 24, 40, 100].map((t) => [slabLevel(2.5, 12, t), quadrature(2.5, 12, t)] as const);
    const worst = Math.max(...cases.map(([got, want]) => Math.abs(got / want - 1)));
    expect(worst).toBeLessThan(1e-3);
  });

  it("is continuous where the slab reaches the sphere, and whole-sphere beyond", () => {
    const below = slabLevel(1, 5, 10 - 1e-9);
    const at = slabLevel(1, 5, 10);
    expect(Math.abs(below / at - 1)).toBeLessThan(1e-9);
    // Per unit volume, a deep slab is the kernel's volume integral, 32πR³/105.
    expect(slabLevel(1000, 5, 1000)).toBeCloseTo((32 * Math.PI * 125) / 105, 9);
  });
});
