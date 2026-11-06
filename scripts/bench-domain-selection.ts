// What does a mask built FROM the data do to the cross-PCF measured inside it?
// `docs/stats-on-arbitrary-domains.md` ("Masks from the data") quotes these tables.
//
// A drawn ROI is fixed before anyone looks at the cells. A mask thresholded from a density field, or
// from the Gram wand's similarity to a sampled spot, is chosen BY the cells, so the cells inside it
// are not a fair sample of the process — the window has been placed where the pattern happened to
// look a certain way. This measures that on patterns whose truth is known: three mutually
// independent types A, B, C, so every cross-g is exactly 1 for the process.
//
// Every mask covers about the same area, and the statistic is the domain-aware estimator
// (`crossPCFMatrixBinned` with `domain`), so the mask's own geometry is corrected exactly and
// anything left is selection.
//
// "thinned": the cells are split by a fair coin; the mask is built from one half and g measured on
// the other. For a Poisson process the halves are independent (thinning), so the mask cannot know
// anything about the cells it is applied to.
//
//   pnpm bench:domain-selection          # ~1 min

import { type Domain, maskWhere, rectDomain } from "../src/spatial/domain";
import { coLocationModes, gramMatrix, channelsFromLabels, pixelZ, similarityField, whiteningMatrix } from "../src/spatial/gram";
import { mulberry32 } from "../src/spatial/kernelAnalysis";
import { crossPCFMatrixBinned, type LabelledCells } from "../src/spatial/pcf";
import { csr, independentClustered, type PointPattern } from "../src/spatial/pointPatterns";
import { gaussianKdeField } from "../src/spatial/scalarField";

const WINDOW = [0, 0, 1000, 1000] as const;
const PX = 250; // 4 µm pixels: source 2/3 of the mask note puts that at ≤ 0.34% on g
const SIGMA = 40; // density bandwidth, and the Gram splat radius
const KEEP = 0.4; // every mask keeps ~40% of the window
const SEEDS = 20;
const A = 0;
const B = 1;
const C = 2;

function quantile(v: ArrayLike<number>, q: number): number {
  const s = Float64Array.from(v).sort();
  return s[Math.min(s.length - 1, Math.floor(q * s.length))]!;
}

function without(c: LabelledCells, t: number): LabelledCells {
  const keep = c.typeId.flatMap((id, i) => (id === t ? [] : [i]));
  return { xs: keep.map((i) => c.xs[i]!), ys: keep.map((i) => c.ys[i]!), typeId: keep.map((i) => c.typeId[i]!) };
}

function ofType(c: LabelledCells, t: number | null) {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < c.xs.length; i++) {
    if (t === null || c.typeId[i] === t) {
      xs.push(c.xs[i]!);
      ys.push(c.ys[i]!);
    }
  }
  return { xs, ys };
}

/** Densest KEEP of the window, by a KDE of the given type (null: all cells). */
function densityMask(c: LabelledCells, t: number | null, exclude?: number): Domain {
  const s = ofType(exclude === undefined ? c : without(c, exclude), t);
  const f = gaussianKdeField(s.xs, s.ys, { width: PX, height: PX, bbox: [...WINDOW], sigma: SIGMA, radiusSigma: 3 });
  const cut = quantile(f.data, 1 - KEEP);
  return maskWhere(f, (v) => v > cut);
}

/** The Gram wand: the KEEP of the window most like a reference spot. `ref` picks the spot (on all
 *  channels); `exclude` drops one type's channel from the similarity the mask is cut from. */
function wandMask(c: LabelledCells, ref: "co-rich" | "random", rnd: () => number, exclude?: number): Domain {
  const params = { bbox: WINDOW, width: PX, height: PX, radius: SIGMA } as const;
  const gram = gramMatrix(channelsFromLabels(c.xs, c.ys, c.typeId), params);
  let q = Math.floor(rnd() * PX * PX);
  if (ref === "co-rich") {
    // Where A and B are jointly densest: the spot someone looking for A–B contact would click.
    const P = PX * PX;
    let best = -Infinity;
    for (let p = 0; p < P; p++) {
      const s = (gram.rasters[A * P + p]! - gram.mean[A]!) / gram.sd[A]! + (gram.rasters[B * P + p]! - gram.mean[B]!) / gram.sd[B]!;
      if (s > best) {
        best = s;
        q = p;
      }
    }
  }
  const rest = exclude === undefined ? null : without(c, exclude);
  const metric = rest ? gramMatrix(channelsFromLabels(rest.xs, rest.ys, rest.typeId), params) : gram;
  const modes = coLocationModes(metric);
  const K = metric.labels.length;
  const W = whiteningMatrix(modes.vectors, modes.values, K, K);
  const d = similarityField(metric, W, K, pixelZ(metric, q % PX, Math.floor(q / PX)));
  const cut = quantile(d, KEEP);
  return maskWhere({ data: d, width: PX, height: PX, bbox: WINDOW }, (v) => v <= cut);
}

function split(c: LabelledCells, rnd: () => number): [LabelledCells, LabelledCells] {
  const h: LabelledCells[] = [0, 1].map(() => ({ xs: [] as number[], ys: [] as number[], typeId: [] as number[] }));
  for (let i = 0; i < c.xs.length; i++) {
    const t = h[rnd() < 0.5 ? 0 : 1]! as { xs: number[]; ys: number[]; typeId: number[] };
    t.xs.push(c.xs[i]!);
    t.ys.push(c.ys[i]!);
    t.typeId.push(c.typeId[i]!);
  }
  return [h[0]!, h[1]!];
}

/** g_AB, g_AC and g_AA per bin. */
function measure(c: LabelledCells, domain: Domain): number[][] {
  const res = crossPCFMatrixBinned(c, { bbox: WINDOW, domain, rMax: 30, nBins: 3, nTypes: 3, edgeCorrected: true });
  const at = (a: number, b: number) => [0, 1, 2].map((k) => res.g[a * 9 + b * 3 + k]!);
  return [at(A, B), at(A, C), at(A, A)];
}

const MASKS = [
  "window (no mask)",
  "density of all cells",
  "density of A",
  "density of C (independent)",
  "wand, co-rich A+B spot",
  "wand, random spot",
  "density of all but B",
  "wand co-rich, B left out",
  "thinned: window",
  "thinned: density of all",
  "thinned: density of A",
  "thinned: wand co-rich",
] as const;

function run(name: string, make: (seed: number) => PointPattern) {
  const sums = MASKS.map(() => [0, 1, 2].map(() => [0, 0, 0]));
  const sq = MASKS.map(() => [0, 1, 2].map(() => [0, 0, 0]));
  for (let s = 0; s < SEEDS; s++) {
    const pat = make(1000 + s);
    const rnd = mulberry32(77 + s);
    const [h1, h2] = split(pat, rnd);
    const rows = [
      measure(pat, rectDomain(WINDOW)),
      measure(pat, densityMask(pat, null)),
      measure(pat, densityMask(pat, A)),
      measure(pat, densityMask(pat, C)),
      measure(pat, wandMask(pat, "co-rich", rnd)),
      measure(pat, wandMask(pat, "random", rnd)),
      measure(pat, densityMask(pat, null, B)),
      measure(pat, wandMask(pat, "co-rich", rnd, B)),
      measure(h2, rectDomain(WINDOW)),
      measure(h2, densityMask(h1, null)),
      measure(h2, densityMask(h1, A)),
      measure(h2, wandMask(h1, "co-rich", rnd)),
    ];
    rows.forEach((r, m) =>
      r.forEach((bins, st) =>
        bins.forEach((g, k) => {
          sums[m]![st]![k]! += g;
          sq[m]![st]![k]! += g * g;
        }),
      ),
    );
  }
  console.log(`\n## ${name} — mean g over ${SEEDS} realisations (± 2 SE), bins [0,10) [10,20) [20,30)`);
  const cell = (m: number, st: number) =>
    [0, 1, 2]
      .map((k) => {
        const mean = sums[m]![st]![k]! / SEEDS;
        const se = Math.sqrt(Math.max(0, sq[m]![st]![k]! / SEEDS - mean * mean) / SEEDS);
        return `${mean.toFixed(3)}±${(2 * se).toFixed(3)}`;
      })
      .join(" ");
  console.log(`${"mask".padEnd(28)} | g_AB (truth 1)                      | g_AC (truth 1)                      | g_AA`);
  MASKS.forEach((label, m) => console.log(`${label.padEnd(28)} | ${cell(m, 0)} | ${cell(m, 1)} | ${cell(m, 2)}`));
}

run("CSR, three independent types, 1500 each", (seed) => csr({ n: 1500, types: 3, seed, bbox: WINDOW }));
run("Independently clustered (σ 30, 120 parents per type), 1500 each", (seed) =>
  independentClustered({ n: 1500, types: 3, seed, bbox: WINDOW, sigma: 30, parents: 120 }),
);
