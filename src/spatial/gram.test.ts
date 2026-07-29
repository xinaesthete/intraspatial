import { describe, expect, it } from "vitest";
import { eigenSym, psdDefect } from "./eigenSym";
import {
  apronCoverage,
  type ChannelCloud,
  channelsFromExpression,
  channelsFromLabels,
  coLocationModes,
  effectiveRadius,
  type GramParams,
  gramMatrix,
  projectMode,
  splatChannel,
} from "./gram";
import { EPANECHNIKOV, GAUSSIAN, KERNELS, kernelLabel, QUARTIC, roughness, TOPHAT, TRIWEIGHT } from "./kernels";
import { crossPCFMatrix } from "./pcf";

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

const BBOX = [0, 0, 200, 200] as const;

/** A `ChannelCloud` whose coordinates are concrete arrays, so tests can concatenate them into the
 *  flat `LabelledCells` shape `crossPCFMatrix` takes. */
type Cloud = ChannelCloud & { readonly xs: number[]; readonly ys: number[] };

function uniformCloud(label: string, n: number, seed: number): Cloud {
  const r = rng(seed);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(r() * 200);
    ys.push(r() * 200);
  }
  return { label, xs, ys };
}

/** Points in a disk of radius `rad` about (cx, cy) — a planted cluster. */
function blob(label: string, n: number, cx: number, cy: number, rad: number, seed: number): Cloud {
  const r = rng(seed);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = r() * 2 * Math.PI;
    const d = rad * Math.sqrt(r());
    xs.push(cx + d * Math.cos(t));
    ys.push(cy + d * Math.sin(t));
  }
  return { label, xs, ys };
}

describe("splatChannel", () => {
  it("conserves mass — Σ M·A_pix = total weight", () => {
    // Points kept a kernel radius clear of the boundary, so none of the mass falls off the raster.
    const p: GramParams = { bbox: BBOX, width: 200, height: 200, radius: 12 };
    const c = blob("a", 300, 100, 100, 50, 1);
    const pixelArea = (200 * 200) / (200 * 200);
    for (const kernel of KERNELS) {
      const m = splatChannel(c, { ...p, kernel });
      let mass = 0;
      for (const v of m) mass += v * pixelArea;
      expect(mass, kernelLabel(kernel)).toBeCloseTo(300, 0);
      expect(Math.abs(mass - 300) / 300, kernelLabel(kernel)).toBeLessThan(2e-3);
    }
  });

  it("honours per-point weights linearly", () => {
    const p: GramParams = { bbox: BBOX, width: 128, height: 128, radius: 15 };
    const base = blob("a", 50, 100, 100, 40, 2);
    const doubled: ChannelCloud = { ...base, weights: new Array(50).fill(2) };
    const m1 = splatChannel(base, p);
    const m2 = splatChannel(doubled, p);
    for (let i = 0; i < m1.length; i += 97) expect(m2[i]!).toBeCloseTo(2 * m1[i]!, 10);
  });

  it("puts row 0 at the TOP of the bbox, matching ScalarField and splatDensity", () => {
    const p: GramParams = { bbox: BBOX, width: 64, height: 64, radius: 10 };
    const high = splatChannel({ label: "h", xs: [100], ys: [190] }, p); // near maxY
    let bestRow = -1;
    let best = -Infinity;
    for (let row = 0; row < 64; row++) {
      for (let col = 0; col < 64; col++) {
        if (high[row * 64 + col]! > best) {
          best = high[row * 64 + col]!;
          bestRow = row;
        }
      }
    }
    expect(bestRow).toBeLessThan(8); // high worldY ⇒ low row index
  });
});

describe("the Gram matrix is a pair sum with the effective kernel J ⊛ J", () => {
  const p: GramParams = { bbox: BBOX, width: 400, height: 400, radius: 20, kernel: EPANECHNIKOV };

  it("C_aa for a single point equals (J⊛J)(0) = ∫J² = roughness(J, r)", () => {
    // A closed form from kernels.ts, reached by an entirely different route (a raster inner
    // product). If the splat normalisation or the pixel area were wrong, this would not hold.
    for (const kernel of [TOPHAT, EPANECHNIKOV, QUARTIC, TRIWEIGHT, GAUSSIAN]) {
      const g = gramMatrix([{ label: "one", xs: [100], ys: [100] }], { ...p, kernel });
      expect(g.c[0]!, kernelLabel(kernel)).toBeCloseTo(roughness(kernel, p.radius), 6);
      // …and that is exactly the self-term the association matrix subtracts.
      expect(g.selfTerm[0]!, kernelLabel(kernel)).toBeCloseTo(roughness(kernel, p.radius), 12);
    }
  });

  it("two single points interact out to 2·radius and no further", () => {
    const at = (d: number) =>
      gramMatrix(
        [
          { label: "a", xs: [100 - d / 2], ys: [100] },
          { label: "b", xs: [100 + d / 2], ys: [100] },
        ],
        p,
      ).c[1]!;
    const r = p.radius;
    expect(at(0)).toBeCloseTo(roughness(EPANECHNIKOV, r), 6);
    expect(at(0.5 * r)).toBeGreaterThan(0);
    expect(at(1.5 * r)).toBeGreaterThan(0);
    expect(at(2 * r + 1)).toBeCloseTo(0, 12); // beyond the effective support
    expect(at(3 * r)).toBe(0);
    // Monotone decreasing in separation, as a radial autocorrelation must be.
    const ds = [0, 0.4, 0.8, 1.2, 1.6].map((f) => at(f * r));
    for (let i = 1; i < ds.length; i++) expect(ds[i]!).toBeLessThan(ds[i - 1]!);
    expect(effectiveRadius(p)).toBe(2 * r);
  });
});

describe("normalisation", () => {
  it("g = 1 under complete spatial randomness, for every kernel", () => {
    // The kernel-agnostic property: E[C_ab] = W_a·W_b/|ROI| for any unit-mass kernel, so the same
    // normalisation serves the whole family with no πr² in sight. The radius is kept small
    // relative to the ROI so the edge deficit measured in the next test stays ~1%.
    const chans = [uniformCloud("a", 6000, 11), uniformCloud("b", 6000, 22)];
    for (const kernel of KERNELS) {
      const g = gramMatrix(chans, { bbox: BBOX, width: 300, height: 300, radius: 3, kernel });
      expect(g.g[1]!, kernelLabel(kernel)).toBeGreaterThan(0.98);
      expect(g.g[1]!, kernelLabel(kernel)).toBeLessThan(1.02);
    }
  });

  it("the residual shortfall under CSR is the Mode-1 ROI edge effect, and it grows with radius", () => {
    // NOT a defect in the normalisation — it is the missing-neighbour bias the plan's Mode 2 is
    // designed to remove. A cell near the ROI boundary has no neighbours beyond it, so E[M(x)]
    // there is ρ·κ(x) with κ < 1 the in-ROI kernel mass, and E[C] = ρ_aρ_b∫κ² < ρ_aρ_b|ROI|.
    // Quantified here as the baseline the viewport apron below is measured against.
    const chans = [uniformCloud("a", 6000, 11), uniformCloud("b", 6000, 22)];
    const at = (radius: number, kernel = EPANECHNIKOV) => gramMatrix(chans, { bbox: BBOX, width: 300, height: 300, radius, kernel }).g[1]!;
    const ladder = [5, 8, 14, 25].map((r) => at(r));
    for (const v of ladder) expect(v).toBeLessThan(1);
    for (let i = 1; i < ladder.length; i++) expect(ladder[i]!).toBeLessThan(ladder[i - 1]!);
    expect(ladder[3]!).toBeLessThan(0.9); // ~15% low at r = 25 on a 200-unit ROI

    // The smoother kernels lose less, because at matched support radius their mass sits further
    // from the boundary — the same μ₂ ordering kernels.ts measures, showing up as edge bias.
    expect(at(14, TOPHAT)).toBeLessThan(at(14, EPANECHNIKOV));
    expect(at(14, EPANECHNIKOV)).toBeLessThan(at(14, TRIWEIGHT));
  });

  it("mass equals cell count for one-hot channels and total weight otherwise", () => {
    const p: GramParams = { bbox: BBOX, width: 64, height: 64, radius: 10 };
    const c = uniformCloud("a", 137, 5);
    expect(gramMatrix([c], p).mass[0]!).toBe(137);
    const w = new Array(137).fill(0.25);
    expect(gramMatrix([{ ...c, weights: w }], p).mass[0]!).toBeCloseTo(137 * 0.25, 10);
  });

  it("mass is WINDOW-local — points outside the bbox are splatted but not counted", () => {
    // The two halves of the apron, isolated. Same points either way; the bbox is half the width, so
    // roughly half the cloud falls outside it. `mass` must track the window, or the CSR expectation
    // would be built from an intensity the window does not have.
    const c = uniformCloud("a", 4000, 77); // uniform on [0,200]²
    const half: GramParams = { bbox: [0, 0, 100, 200], width: 128, height: 256, radius: 10 };
    const res = gramMatrix([c], half);
    const inside = c.xs.filter((x) => x <= 100).length;
    expect(res.mass[0]!).toBe(inside);
    expect(res.mass[0]!).toBeLessThan(c.xs.length);
    // The apron tally counts only what is within `radius` of the window — the points that actually
    // deposited mass onto window pixels, not everything beyond the edge.
    const inApron = c.xs.filter((x, i) => x > 100 && x <= 110 && (c.ys[i] ?? 0) <= 210).length;
    expect(res.apronMass[0]!).toBe(inApron);
  });

  it("converges as the raster refines", () => {
    const chans = [blob("a", 600, 70, 70, 45, 31), blob("b", 600, 90, 90, 45, 32)];
    const base = { bbox: BBOX, radius: 16, kernel: EPANECHNIKOV } as const;
    const coarse = gramMatrix(chans, { ...base, width: 128, height: 128 });
    const fine = gramMatrix(chans, { ...base, width: 512, height: 512 });
    expect(Math.abs(coarse.g[1]! - fine.g[1]!) / fine.g[1]!).toBeLessThan(0.01);
  });
});

describe("the viewport apron", () => {
  /** An interior window: points fill [0,200]², the statistic is measured on the middle 150². */
  const WIN = [25, 25, 175, 175] as const;
  const chans = [uniformCloud("a", 6000, 11), uniformCloud("b", 6000, 22)];
  const params = (radius: number) => ({ bbox: WIN, width: 225, height: 225, radius, kernel: EPANECHNIKOV }) as const;

  /** The same clouds with every point outside the window thrown away — i.e. no apron supplied. */
  const clipTo =
    (b: readonly [number, number, number, number]) =>
    (c: Cloud): Cloud => {
      const xs: number[] = [];
      const ys: number[] = [];
      for (let i = 0; i < c.xs.length; i++) {
        const x = c.xs[i]!;
        const y = c.ys[i]!;
        if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) {
          xs.push(x);
          ys.push(y);
        }
      }
      return { label: c.label, xs, ys };
    };

  it("removes the edge deficit outright — g = 1 under CSR at every radius", () => {
    // Contrast with the ladder above, which on the same points measured over their own extent gives
    // 0.966 / 0.940 / 0.897 / 0.822. Here there is no ladder at all: the deficit is gone, not
    // reduced, because the window's M really is complete.
    for (const radius of [5, 8, 14, 25]) {
      const g = gramMatrix(chans, params(radius)).g[1]!;
      expect(g, `r=${radius}`).toBeGreaterThan(0.97);
      expect(g, `r=${radius}`).toBeLessThan(1.03);
    }
  });

  it("and the correction lives in the POINT SET, not in extra raster pixels", () => {
    // The load-bearing measurement. Clip the cloud to the window — identical raster, identical
    // window, one fewer apron — and the full deficit ladder returns. Which is why there is no
    // padded-raster machinery here: a point outside the bbox deposits onto the bbox's own edge
    // pixels regardless, because the splat's footprint is clipped to the raster, not the point set.
    const clipped = chans.map(clipTo(WIN));
    const ladder = [5, 14, 25].map((r) => gramMatrix(clipped, params(r)).g[1]!);
    for (let i = 1; i < ladder.length; i++) expect(ladder[i]!).toBeLessThan(ladder[i - 1]!);
    expect(ladder[0]!).toBeLessThan(0.98);
    expect(ladder[2]!).toBeLessThan(0.85);
    // ... and every one of them is worse than the same radius WITH the apron.
    for (const [i, r] of [5, 14, 25].entries()) expect(ladder[i]!).toBeLessThan(gramMatrix(chans, params(r)).g[1]!);
  });

  it("is stable under a shrinking window — which is what makes an interactive camera honest", () => {
    // g must not move when the viewport does. Both C and the expectation are window-local, so a
    // zoom changes the area measured but not the statistic measured over it.
    for (const inset of [25, 50, 70]) {
      const bbox = [inset, inset, 200 - inset, 200 - inset] as const;
      const side = 200 - 2 * inset;
      const g = gramMatrix(chans, { bbox, width: 1.5 * side, height: 1.5 * side, radius: 10, kernel: EPANECHNIKOV }).g[1]!;
      expect(g, `inset=${inset}`).toBeGreaterThan(0.94);
      expect(g, `inset=${inset}`).toBeLessThan(1.06);
    }
  });

  it("apronCoverage says whether the window was interior at all", () => {
    // The honest diagnostic: there is no way to correct a window at the edge of the data, so the
    // code reports that rather than quietly returning a biased number.
    const interior = gramMatrix(chans, params(14));
    for (const v of apronCoverage(interior, 14)) expect(v).toBeGreaterThan(0.9);
    for (const v of apronCoverage(interior, 14)) expect(v).toBeLessThan(1.1);

    // Window = the data's own extent: nothing outside, coverage 0, and g carries the full deficit.
    const edge = gramMatrix(chans, { bbox: BBOX, width: 300, height: 300, radius: 14, kernel: EPANECHNIKOV });
    for (const v of apronCoverage(edge, 14)) expect(v).toBe(0);
    expect(edge.g[1]!).toBeLessThan(0.95);
  });
});

describe("positive semi-definiteness — the reason for the Gram form", () => {
  const scene = () => [
    blob("a", 500, 60, 60, 35, 41),
    blob("b", 500, 70, 70, 35, 42), // overlaps a
    blob("c", 500, 150, 150, 35, 43), // far from both
    uniformCloud("d", 500, 44),
  ];
  const p: GramParams = { bbox: BBOX, width: 256, height: 256, radius: 18, kernel: EPANECHNIKOV };

  it("C = MMᵀ is PSD to machine precision, for every kernel including the non-PD ones", () => {
    // The structural point: PSD-ness of MMᵀ does not depend on the kernel being positive-definite.
    // The top-hat's Fourier transform dips to −13% (kernelSpectrum.test.ts) and this still holds.
    for (const kernel of KERNELS) {
      const g = gramMatrix(scene(), { ...p, kernel });
      const m = coLocationModes(g);
      expect(psdDefect(m.values, 4), kernelLabel(kernel)).toBeLessThan(1e-12);
    }
  });

  it("corr has an exact unit diagonal and trace K, so `explained` is a real variance share", () => {
    const g = gramMatrix(scene(), p);
    const K = 4;
    for (let a = 0; a < K; a++) expect(g.corr[a * K + a]!).toBe(1);
    const m = coLocationModes(g);
    let trace = 0;
    for (const v of m.values) trace += v;
    expect(trace).toBeCloseTo(K, 10);
    let sum = 0;
    for (const e of m.explained) sum += e;
    expect(sum).toBeCloseTo(1, 12);
  });

  it("g stays PSD on self-clustering populations — by accident, not by guarantee", () => {
    // Worth pinning as the honest counterweight to the next test: on the common case (each type
    // clusters with itself) the inflated diagonal carries the matrix, and decomposing `g` happens
    // to be harmless. Nothing about the construction promises this.
    const g = gramMatrix(scene(), p);
    expect(coLocationModes(g, { matrix: "g" }).psdDefect).toBe(0);
  });

  it("interdigitated populations make the PUBLISHED matrix maximally indefinite", () => {
    // The case that decides the design. Two types alternating on a lattice: each repels itself at
    // the pitch while sitting right next to the other, so the diagonal empties out. This is
    // `crossPCFMatrix` — the published statistic, no Gram form involved — and at the pitch radius
    // it returns [[0, ~2.1], [~2.1, 0]], whose eigenvalues are ±2.1.
    const xs: number[] = [];
    const ys: number[] = [];
    const typeId: number[] = [];
    const pitch = 8;
    for (let i = 0; i < 24; i++) {
      for (let j = 0; j < 24; j++) {
        xs.push(10 + i * pitch);
        ys.push(10 + j * pitch);
        typeId.push((i + j) % 2);
      }
    }
    const published = crossPCFMatrix({ xs, ys, typeId }, { bbox: BBOX, radius: 9 });
    expect(published.g[0]!).toBeCloseTo(0, 10); // no same-type pair within the pitch
    expect(published.g[1]!).toBeGreaterThan(2);
    const spectrum = eigenSym(published.g, 2);
    expect(psdDefect(spectrum.values, 2)).toBeCloseTo(1, 6); // as indefinite as a matrix can be
    expect(spectrum.values[1]!).toBeLessThan(-2);

    // The Gram form's own normalised `g` inherits the same problem — it is the normalisation, not
    // the pair-counting method, that costs definiteness…
    const gram = gramMatrix(channelsFromLabels(xs, ys, typeId), {
      bbox: BBOX,
      width: 400,
      height: 400,
      radius: 4.5,
      kernel: EPANECHNIKOV,
    });
    expect(coLocationModes(gram, { matrix: "g" }).psdDefect).toBeGreaterThan(0.5);

    // …while `corr` and the raw Gram `C` are PSD on exactly the same data, exactly.
    expect(coLocationModes(gram).psdDefect).toBe(0);
    expect(psdDefect(eigenSym(gram.c, 2).values, 2)).toBe(0);
  });
});

describe("co-location modes", () => {
  const scene = [
    blob("a", 700, 60, 60, 30, 51),
    blob("b", 700, 62, 58, 30, 52), // tracks a
    blob("c", 700, 150, 150, 30, 53), // anti-correlated with a and b
  ];
  const p: GramParams = { bbox: BBOX, width: 256, height: 256, radius: 16, kernel: EPANECHNIKOV };

  it("the variance of a projected mode field equals its eigenvalue", () => {
    // An exact identity — Var(Σ_a v_a z_a) = vᵀ·Corr·v = λ — and the sharpest single check that
    // the projection, the standardisation and the decomposition all agree.
    const g = gramMatrix(scene, p);
    const m = coLocationModes(g);
    for (let k = 0; k < 3; k++) {
      const y = projectMode(g, m, k);
      let mean = 0;
      for (const v of y) mean += v;
      mean /= y.length;
      let varr = 0;
      for (const v of y) varr += (v - mean) ** 2;
      varr /= y.length;
      expect(varr, `mode ${k}`).toBeCloseTo(m.values[k]!, 8);
      expect(mean, `mode ${k} is centred`).toBeCloseTo(0, 8);
    }
  });

  it("mode 1 separates the co-located pair from the distant channel by loading sign", () => {
    const g = gramMatrix(scene, p);
    const m = coLocationModes(g);
    const [va, vb, vc] = [m.vectors[0]!, m.vectors[1]!, m.vectors[2]!];
    expect(Math.sign(va)).toBe(Math.sign(vb)); // a and b travel together
    expect(Math.sign(vc)).not.toBe(Math.sign(va)); // c opposes them
    expect(m.explained[0]!).toBeGreaterThan(0.5); // and it is the dominant structure
  });

  it("projected mode fields are mutually orthogonal across pixels", () => {
    const g = gramMatrix(scene, p);
    const m = coLocationModes(g);
    const y0 = projectMode(g, m, 0);
    const y1 = projectMode(g, m, 1);
    let dot = 0;
    for (let i = 0; i < y0.length; i++) dot += y0[i]! * y1[i]!;
    expect(Math.abs(dot / y0.length)).toBeLessThan(1e-8);
  });
});

describe("agreement with the existing cross-PCF oracle", () => {
  it("reproduces crossPCFMatrix's structure: co-located > 1, segregated < 1", () => {
    // Not a numeric parity test — the effective kernel is J⊛J, not a top-hat of the same radius,
    // so the two probe different scales by construction. What must agree is the finding.
    const a = blob("a", 500, 60, 60, 30, 61);
    const b = blob("b", 500, 63, 57, 30, 62);
    const c = blob("c", 500, 150, 150, 30, 63);
    const xs = [...a.xs, ...b.xs, ...c.xs];
    const ys = [...a.ys, ...b.ys, ...c.ys];
    const typeId = [...new Array(500).fill(0), ...new Array(500).fill(1), ...new Array(500).fill(2)];

    const oracle = crossPCFMatrix({ xs, ys, typeId }, { bbox: BBOX, radius: 20 });
    const mine = gramMatrix(channelsFromLabels(xs, ys, typeId), {
      bbox: BBOX,
      width: 384,
      height: 384,
      radius: 10, // effective support 2r = 20, matching the oracle's disk radius
      kernel: EPANECHNIKOV,
    });

    for (const [i, j] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ] as [number, number][]) {
      const bothClustered = oracle.g[i * 3 + j]! > 1;
      expect(mine.g[i * 3 + j]! > 1, `pair ${i},${j}`).toBe(bothClustered);
    }
    // The co-located pair is strongly positive and the segregated pairs are near-empty in both.
    expect(oracle.g[1]!).toBeGreaterThan(2);
    expect(mine.g[1]!).toBeGreaterThan(2);
    expect(oracle.g[2]!).toBeLessThan(0.1);
    expect(mine.g[2]!).toBeLessThan(0.1);
  });

  it("channelsFromLabels partitions the cells exactly once", () => {
    const xs = [1, 2, 3, 4, 5];
    const ys = [1, 1, 1, 1, 1];
    const chans = channelsFromLabels(xs, ys, [7, 3, 7, 3, 3], (id) => `t${id}`);
    expect(chans.map((c) => c.label)).toEqual(["t3", "t7"]);
    expect(chans[0]!.xs).toEqual([2, 4, 5]);
    expect(chans[1]!.xs).toEqual([1, 3]);
  });
});

describe("weighted marks — the AnnData X case", () => {
  const xs = [30, 60, 90, 120, 150, 60, 90];
  const ys = [40, 40, 40, 40, 40, 120, 120];
  const p: GramParams = { bbox: BBOX, width: 200, height: 200, radius: 25, kernel: EPANECHNIKOV };

  it("one-hot expression columns reproduce the cell-type channels exactly", () => {
    // The claim that types are the one-hot case of a general mark, tested rather than asserted.
    const typeId = [0, 1, 0, 1, 0, 1, 0];
    const byLabel = gramMatrix(channelsFromLabels(xs, ys, typeId), p);
    const n = xs.length;
    const x = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) x[typeId[i]! * n + i] = 1;
    const byExpr = gramMatrix(channelsFromExpression(xs, ys, x, ["0", "1"]), p);
    for (let i = 0; i < 4; i++) expect(byExpr.c[i]!).toBeCloseTo(byLabel.c[i]!, 10);
    expect([...byExpr.mass]).toEqual([...byLabel.mass]);
  });

  it("the self term is diagonal-only for types but fills the matrix for shared-point genes", () => {
    const typeId = [0, 1, 0, 1, 0, 1, 0];
    const byLabel = gramMatrix(channelsFromLabels(xs, ys, typeId), p);
    expect(byLabel.selfTerm[1]!).toBe(0); // disjoint point sets share no cell
    expect(byLabel.selfTerm[0]!).toBeGreaterThan(0);

    // Two genes expressed in the SAME cells: the within-cell product leaks into the off-diagonal.
    const n = xs.length;
    const x = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      x[i] = 1;
      x[n + i] = 2;
    }
    const genes = gramMatrix(channelsFromExpression(xs, ys, x, ["g1", "g2"]), p);
    expect(genes.selfTerm[1]!).toBeCloseTo(roughness(EPANECHNIKOV, p.radius) * 2 * n, 8);
    expect(genes.selfTerm[1]!).toBeGreaterThan(0);
  });

  it("subtracting the self term is what makes g comparable to a pair count", () => {
    // Two genes that are perfectly co-expressed but spatially SPREAD OUT (every cell far from
    // every other, relative to the kernel). Without the correction they look co-located purely
    // because they live in the same cells; with it, the association collapses toward CSR.
    const far = { xs: [20, 180, 20, 180], ys: [20, 20, 180, 180] };
    const n = 4;
    const x = new Float64Array(2 * n).fill(1);
    const q: GramParams = { bbox: BBOX, width: 256, height: 256, radius: 12, kernel: EPANECHNIKOV };
    const g = gramMatrix(channelsFromExpression(far.xs, far.ys, x, ["g1", "g2"]), q);
    expect(g.selfTerm[1]!).toBeGreaterThan(0);
    // ALL of C is the self term here. The agreement is RELATIVE, not exact: `selfTerm` uses the
    // continuum ∫J² from kernels.ts while `c` is a Riemann sum over pixels, so they differ by the
    // raster's O(pixel²) discretisation — ~1e-5 here, and it shrinks as the raster refines.
    expect(Math.abs(g.c[1]! - g.selfTerm[1]!) / g.selfTerm[1]!).toBeLessThan(1e-4);
    expect(Math.abs(g.g[1]!)).toBeLessThan(1e-3); // …so the corrected association is ~0
  });

  it("zero-weight points contribute nothing", () => {
    const n = xs.length;
    const w = new Array(n).fill(1);
    w[2] = 0;
    const withZero = gramMatrix([{ label: "a", xs, ys, weights: w }], p);
    const dropped = gramMatrix([{ label: "a", xs: xs.filter((_, i) => i !== 2), ys: ys.filter((_, i) => i !== 2) }], p);
    expect(withZero.c[0]!).toBeCloseTo(dropped.c[0]!, 10);
  });
});
