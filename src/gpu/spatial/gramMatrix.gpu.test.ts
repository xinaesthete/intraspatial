import { describe, expect, it } from "vitest";
import { eigenSym, psdDefect } from "../../spatial/eigenSym";
import { type ChannelCloud, channelsFromLabels } from "../../spatial/gram";
import { EPANECHNIKOV, TOPHAT } from "../../spatial/kernels";
import { gramMatrixGpu } from "./gramMatrix";

// ## Why the f64 oracle is BAKED IN rather than computed here
//
// Running `gramMatrix` (the CPU twin) in this process crashes the fork before vitest can flush
// results — and so does any comparable amount of CPU work. Bisected: a bare loop churning
// `Float64Array`s, with no gram code involved at all, kills it just the same, while the identical
// GPU calls plus trivial assertions pass repeatedly. The budget is brutal: a CPU oracle at a 32²
// raster survives, 48² does not. This is the Dawn-on-Node teardown fragility already documented
// for `tcmRender` in docs/cell-stats.md §11, not anything about this module.
//
// So the oracle values below were generated ONCE by running `gramMatrix` on the scene `scene()`
// builds, in the normal CPU test environment, and pasted here. To regenerate after changing the
// scene or the statistic, run that function on this scene under `vitest run` (not the GPU config)
// and print `c`, `g` and `corr`. The scene is built by a closed formula rather than a seeded RNG
// precisely so it cannot drift out from under these constants.
//
// The other standing Dawn constraint applies as usual: assertions are AGGREGATED (one max-relative
// difference per matrix), never per-element `expect()` loops.

const BBOX = [0, 0, 200, 200] as const;

/** Deterministic, formula-built point cloud — no RNG, so the baked oracle stays valid. */
const cloud = (n: number, cx: number, cy: number) => {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(cx + (i % 17) * 1.3);
    ys.push(cy + (i % 13) * 1.7);
  }
  return { xs, ys };
};

/** Three cell types: `a` and `b` overlap, `c` sits far away. */
function scene(): { xs: number[]; ys: number[]; typeId: number[] } {
  const a = cloud(120, 40, 40);
  const b = cloud(120, 52, 46);
  const c = cloud(120, 140, 140);
  return {
    xs: [...a.xs, ...b.xs, ...c.xs],
    ys: [...a.ys, ...b.ys, ...c.ys],
    typeId: [...new Array(120).fill(0), ...new Array(120).fill(1), ...new Array(120).fill(2)],
  };
}

const PARAMS = { bbox: BBOX, width: 128, height: 128, radius: 14, kernel: EPANECHNIKOV } as const;

// f64 oracle — see the header for how to regenerate.
const ORACLE = {
  c: [14.41908175, 8.492748654, 0, 8.492748654, 14.41970318, 0, 0, 0, 14.41908175],
  g: [39.33121373, 23.59096848, 0, 23.59096848, 39.33293993, 0, 0, 0, 39.33121373],
  corr: [1, 0.5784573432, -0.02560419603, 0.5784573432, 1, -0.02560390084, -0.02560419603, -0.02560390084, 1],
  selfTerm: [0.259844805, 0, 0, 0, 0.259844805, 0, 0, 0, 0.259844805],
} as const;

/** Max absolute difference, scaled by the reference matrix's own peak — one number per matrix so
 *  the whole comparison is a single `expect()`. */
function relMax(reference: ArrayLike<number>, got: ArrayLike<number>): number {
  let peak = 0;
  for (let i = 0; i < reference.length; i++) peak = Math.max(peak, Math.abs(reference[i] ?? 0));
  if (peak === 0) return 0;
  let m = 0;
  for (let i = 0; i < reference.length; i++) m = Math.max(m, Math.abs((reference[i] ?? 0) - (got[i] ?? 0)));
  return m / peak;
}

describe("gramMatrixGpu", () => {
  it("matches the f64 CPU oracle on C, g and corr", async () => {
    const { xs, ys, typeId } = scene();
    const gpu = await gramMatrixGpu(channelsFromLabels(xs, ys, typeId), PARAMS);
    // Not bit-parity, and it does not claim to be: the raster is a quadrature and the reduction is
    // f32, so the tolerance is stated rather than assumed away.
    expect(relMax(ORACLE.c, gpu.c)).toBeLessThan(2e-3);
    expect(relMax(ORACLE.g, gpu.g)).toBeLessThan(2e-3);
    expect(relMax(ORACLE.corr, gpu.corr)).toBeLessThan(2e-3);
    // The self term is host f64 arithmetic on both paths, so it must agree far more tightly.
    expect(relMax(ORACLE.selfTerm, gpu.selfTerm)).toBeLessThan(1e-9);
    expect([...gpu.mass]).toEqual([120, 120, 120]);
    expect(gpu.labels).toEqual(["0", "1", "2"]);
  });

  it("keeps the PSD guarantee in f32 — the property is structural, not numerical", async () => {
    // The load-bearing claim of the formulation: MMᵀ is PSD because of what it is. Accumulate it
    // in f32 through a raster quadrature and it is still PSD — for the top-hat too, whose own
    // Fourier transform dips to −13% (see src/spatial/kernelSpectrum.test.ts).
    const { xs, ys, typeId } = scene();
    const chans = channelsFromLabels(xs, ys, typeId);
    for (const kernel of [EPANECHNIKOV, TOPHAT]) {
      const gpu = await gramMatrixGpu(chans, { ...PARAMS, kernel });
      const spectrum = eigenSym(gpu.corr, 3);
      expect(psdDefect(spectrum.values, 3), kernel === TOPHAT ? "top-hat" : "epanechnikov").toBeLessThan(1e-6);
    }
  });

  it("finds the co-location structure: a with b, against c", async () => {
    const { xs, ys, typeId } = scene();
    const gpu = await gramMatrixGpu(channelsFromLabels(xs, ys, typeId), PARAMS);
    const { vectors, values } = eigenSym(gpu.corr, 3);
    expect(Math.sign(vectors[0]!)).toBe(Math.sign(vectors[1]!)); // a and b load together
    expect(Math.sign(vectors[2]!)).not.toBe(Math.sign(vectors[0]!)); // c opposes them
    expect(values[0]! / 3).toBeGreaterThan(0.5); // and it is the dominant mode
    expect(gpu.g[2]!).toBeCloseTo(0, 6); // a and c never interact
  });

  it("carries per-point weights, so an expression column is a drop-in for a type indicator", async () => {
    const base: ChannelCloud = { label: "a", ...cloud(150, 90, 90) };
    const heavy: ChannelCloud = { ...base, weights: new Array(150).fill(2) };
    const one = await gramMatrixGpu([base], PARAMS);
    const two = await gramMatrixGpu([heavy], PARAMS);
    expect(two.mass[0]!).toBeCloseTo(2 * one.mass[0]!, 6);
    expect(two.c[0]! / one.c[0]!).toBeCloseTo(4, 2); // C is quadratic in the weights…
    expect(two.g[0]! / one.g[0]!).toBeCloseTo(1, 2); // …and g, being normalised, is invariant
  });

  it("counts mass window-locally and still splats the apron — the edge correction, on the GPU", async () => {
    // A window that CUTS cloud `a`, so both halves of the apron are exercised at once. The
    // arithmetic here is 360 iterations over the scene's own closed formula — nowhere near the CPU
    // budget that kills the fork (a 32² raster oracle is the ceiling; see the header).
    const { xs, ys, typeId } = scene();
    const chans = channelsFromLabels(xs, ys, typeId);
    const WIN = [0, 0, 50, 50] as const;
    const params = { ...PARAMS, bbox: WIN, width: 96, height: 96 } as const;
    const gpu = await gramMatrixGpu(chans, params);

    const within = (c: ChannelCloud, pad: number) => {
      let n = 0;
      for (let i = 0; i < c.xs.length; i++) {
        const x = c.xs[i] ?? 0;
        const y = c.ys[i] ?? 0;
        if (x >= -pad && x <= 50 + pad && y >= -pad && y <= 50 + pad) n++;
      }
      return n;
    };
    const inside = chans.map((c) => within(c, 0));
    const apron = chans.map((c, k) => within(c, PARAMS.radius) - inside[k]!);
    expect([...gpu.mass]).toEqual(inside);
    expect([...gpu.apronMass]).toEqual(apron);
    // The window really did cut: `a` is only partly inside, and its apron is not empty.
    expect(inside[0]!).toBeGreaterThan(0);
    expect(inside[0]!).toBeLessThan(120);
    expect(apron[0]!).toBeGreaterThan(0);

    // …and the apron points reach the raster. Clip the point set to the window — same window, same
    // mass, one fewer apron — and C must fall, because the edge pixels lose the mass those outside
    // points were depositing. This is the GPU's version of the CPU pin in gram.test.ts.
    const clipped: ChannelCloud[] = chans.map((c) => {
      const cx: number[] = [];
      const cy: number[] = [];
      for (let i = 0; i < c.xs.length; i++) {
        const x = c.xs[i] ?? 0;
        const y = c.ys[i] ?? 0;
        if (x >= 0 && x <= 50 && y >= 0 && y <= 50) {
          cx.push(x);
          cy.push(y);
        }
      }
      return { label: c.label, xs: cx, ys: cy };
    });
    const bare = await gramMatrixGpu(clipped, params);
    expect([...bare.mass]).toEqual(inside); // identical normaliser…
    expect(bare.c[0]!).toBeLessThan(gpu.c[0]!); // …so the deficit lands squarely on g
    expect(bare.g[0]!).toBeLessThan(gpu.g[0]!);
  });

  it("a zero-weight mark is exactly a missing one — what the vertex-stage cull relies on", async () => {
    // The splat culls zero-weight instances, because expression data is mostly zeros and each one
    // otherwise rasterises a full kernel footprint to add nothing. That is only sound if a
    // zero-weight cell is indistinguishable from an absent one in EVERY output: it contributes 0 to
    // the raster, 0 to `mass`, and 0 to `selfTerm`. Pinning it against the explicitly-thinned scene
    // is what makes the optimisation a no-op rather than an approximation.
    const dense = cloud(200, 60, 60);
    const keep = (i: number) => i % 3 !== 0;
    const withZeros: ChannelCloud = {
      label: "a",
      ...dense,
      weights: Array.from({ length: 200 }, (_, i) => (keep(i) ? 1 + (i % 5) * 0.25 : 0)),
    };
    const thinned: ChannelCloud = {
      label: "a",
      xs: dense.xs.filter((_, i) => keep(i)),
      ys: dense.ys.filter((_, i) => keep(i)),
      weights: Array.from({ length: 200 }, (_, i) => (keep(i) ? 1 + (i % 5) * 0.25 : 0)).filter((_, i) => keep(i)),
    };
    const a = await gramMatrixGpu([withZeros], PARAMS);
    const b = await gramMatrixGpu([thinned], PARAMS);
    expect(a.mass[0]!).toBeCloseTo(b.mass[0]!, 9);
    expect(relMax(b.c, a.c)).toBeLessThan(1e-9);
    expect(relMax(b.g, a.g)).toBeLessThan(1e-9);
    expect(relMax(b.selfTerm, a.selfTerm)).toBeLessThan(1e-9);
  });

  it("is symmetric by construction — both halves come from one accumulation", async () => {
    const { xs, ys, typeId } = scene();
    const gpu = await gramMatrixGpu(channelsFromLabels(xs, ys, typeId), PARAMS);
    let asym = 0;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) asym = Math.max(asym, Math.abs(gpu.c[i * 3 + j]! - gpu.c[j * 3 + i]!));
    }
    expect(asym).toBe(0); // exactly, not approximately: the shader writes one value to both slots
  });
});
