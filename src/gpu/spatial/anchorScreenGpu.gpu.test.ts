import { describe, expect, it } from "vite-plus/test";
import { mulberry32 } from "#src/evo/rng";
import { anchorNullExceedances, anchorScreen, prepareAnchorBlock } from "#src/spatial/anchorScreen";
import { randomPermutation } from "#src/spatial/permute";
import { anchorNullExceedancesGpu, anchorScreenGpu, FEISTEL_MIN_REGIONS } from "./anchorScreenGpu";
import { feistelDomain, feistelKeyCpu, feistelPermuteCpu } from "./feistelWgsl";

// ρ comes from the CPU on both paths, so it must be identical. With the default shuffle the GPU
// draws the CPU's own permutations, so p must be identical too. With the Feistel shuffle p agrees
// statistically, and driving the CPU null with the Feistel twin's permutations must reproduce the
// GPU counts exactly (ties included).

function table(seed: number, n: number, m: number, opts: { ties?: number; nanRate?: number } = {}) {
  const rnd = mulberry32(seed);
  const q = (v: number) => (opts.ties ? Math.round(v * opts.ties) / opts.ties : v);
  const anchor = Float64Array.from({ length: n }, () => q(rnd()));
  const cols: Float64Array[] = [];
  for (let j = 0; j < m; j++) {
    const w = (rnd() * 2 - 1) * (j % 3 === 0 ? 2 : 0.2);
    cols.push(Float64Array.from(anchor, (a) => (rnd() < (opts.nanRate ?? 0) ? Number.NaN : q(w * a + rnd()))));
  }
  return { anchor, cols };
}

function feistelStream(n: number, seed: number) {
  const { halfBits, halfMask } = feistelDomain(n);
  const buf = new Uint32Array(n);
  let s = 0;
  return () => {
    const key = feistelKeyCpu(seed, s++);
    for (let t = 0; t < n; t++) buf[t] = feistelPermuteCpu(t, key, n, halfBits, halfMask);
    return buf;
  };
}

describe("anchorNullExceedancesGpu", () => {
  it("default shuffle: the CPU screen's p exactly, with ties and partial masks", async () => {
    // Heavy ties and small n make many null |ρ| equal the observed one exactly; the NaNs give the
    // CPU-side partial-mask groups their share of the same permutations.
    const { anchor, cols } = table(3, 12, 60, { ties: 3, nanRate: 0.05 });
    const features = new Map(cols.map((c, j) => [`f${j}`, c]));
    const block = prepareAnchorBlock(anchor, cols);
    expect(block.groups.some((g) => g.full) && block.groups.some((g) => !g.full)).toBe(true);
    const cpu = anchorScreen({ anchor, features }, { seed: 2 });
    const gpu = await anchorScreenGpu({ anchor, features }, { seed: 2 });
    expect(Array.from(gpu.p)).toEqual(Array.from(cpu.p));
  });

  it("feistel shuffle: counts exactly what the CPU counts on the same permutations, ties included", async () => {
    const { anchor, cols } = table(3, 24, 40, { ties: 3 });
    const block = prepareAnchorBlock(anchor, cols);
    expect(block.groups.length).toBe(1);
    const gpu = await anchorNullExceedancesGpu(block, 999, { seed: 77, shuffle: "feistel" });
    const cpu = anchorNullExceedances(block, 999, feistelStream(block.nAnchor, 77));
    expect(Array.from(gpu)).toEqual(Array.from(cpu));
  });

  it("stays exact across several submitted batches, both shuffles", async () => {
    // 40 regions x 2,000 features: the per-batch work cap splits 999 shuffles into two submits.
    const { anchor, cols } = table(4, 40, 2000, { ties: 10 });
    const block = prepareAnchorBlock(anchor, cols);
    const feistel = await anchorNullExceedancesGpu(block, 999, { seed: 5, shuffle: "feistel" });
    expect(Array.from(feistel)).toEqual(Array.from(anchorNullExceedances(block, 999, feistelStream(block.nAnchor, 5))));
    const fy = await anchorNullExceedancesGpu(block, 999, { seed: 5 });
    const rnd = mulberry32(5);
    const buf = new Uint32Array(block.nAnchor);
    expect(Array.from(fy)).toEqual(Array.from(anchorNullExceedances(block, 999, () => randomPermutation(block.nAnchor, rnd, buf))));
  });

  it("falls back from feistel below FEISTEL_MIN_REGIONS", async () => {
    const { anchor, cols } = table(6, FEISTEL_MIN_REGIONS - 1, 30);
    const block = prepareAnchorBlock(anchor, cols);
    const a = await anchorNullExceedancesGpu(block, 299, { seed: 1, shuffle: "feistel" });
    const b = await anchorNullExceedancesGpu(block, 299, { seed: 1 });
    expect(Array.from(a)).toEqual(Array.from(b));
  });
});

describe("anchorScreenGpu", () => {
  it("feistel shuffle: ρ exactly, p within Monte Carlo error of the CPU", async () => {
    const { anchor, cols } = table(8, 120, 300);
    const features = new Map(cols.map((c, j) => [`f${j}`, c]));
    const samples = Array.from({ length: 120 }, (_, i) => (i < 70 ? "A" : "B"));
    const cpu = anchorScreen({ anchor, features, samples });
    const gpu = await anchorScreenGpu({ anchor, features, samples }, { shuffle: "feistel" });
    expect(gpu.sample).toEqual(cpu.sample);
    expect(Array.from(gpu.rho)).toEqual(Array.from(cpu.rho));
    expect(Array.from(gpu.nFinite)).toEqual(Array.from(cpu.nFinite));
    // Two independent 999-shuffle estimates of the same p: z = Δp / sqrt(2 p(1-p) / 1000).
    let sumZ2 = 0;
    let big = 0;
    for (let j = 0; j < cpu.p.length; j++) {
      const pm = (cpu.p[j]! + gpu.p[j]!) / 2;
      const sd = Math.sqrt((2 * pm * (1 - pm)) / 1000);
      const z = sd > 0 ? (cpu.p[j]! - gpu.p[j]!) / sd : 0;
      sumZ2 += z * z;
      if (Math.abs(z) > 4) big++;
    }
    expect(big).toBe(0);
    expect(sumZ2 / cpu.p.length).toBeLessThan(1.5);
  });
});
