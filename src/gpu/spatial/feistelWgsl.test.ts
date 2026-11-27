import { describe, expect, it } from "vite-plus/test";
import { feistelDomain, feistelKeyCpu, feistelPermuteCpu } from "./feistelWgsl";

describe("feistelPermuteCpu", () => {
  it("is a bijection on [0, n) for awkward n", () => {
    for (const n of [1, 2, 3, 5, 33, 64, 65, 1000, 4097]) {
      const { halfBits, halfMask } = feistelDomain(n);
      for (const s of [0, 1, 7]) {
        const key = feistelKeyCpu(12345, s);
        const seen = new Uint8Array(n);
        for (let i = 0; i < n; i++) seen[feistelPermuteCpu(i, key, n, halfBits, halfMask)]!++;
        expect(seen.every((c) => c === 1)).toBe(true);
      }
    }
  });

  it("gives different shuffles different permutations", () => {
    const n = 50;
    const { halfBits, halfMask } = feistelDomain(n);
    const perm = (s: number) => Array.from({ length: n }, (_, i) => feistelPermuteCpu(i, feistelKeyCpu(1, s), n, halfBits, halfMask));
    expect(perm(0)).not.toEqual(perm(1));
  });
});
