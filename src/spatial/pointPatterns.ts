// Standard 2-D spatial point processes, each carrying the analytic answer a spatial statistic
// should recover from it.
//
// This is the point-pattern counterpart of `syntheticManifolds.ts` (high-dimensional geometry, for
// UMAP) and `datasource/syntheticLoader.ts` (analytic rasters behind the real `Loader`). Those two
// exist; labelled 2-D clouds did not, and were hand-rolled — with slightly different definitions —
// in `syntheticCellTable`, `pcfBootstrap.test.ts`, `gram.test.ts`, `quadratCorrelation.test.ts`,
// `pcf.test.ts` and several GPU tests.
//
// **The reason to share them is not that the code was duplicated. It is that a statistic needs a
// fixture whose true answer is known.** An ad-hoc "clustered blob" can only support a weak
// assertion — g came out bigger than 1, the band got wider — and a weak assertion passes for an
// estimator that is wrong by a constant factor. The processes below were chosen because their pair
// correlation function is known in closed form (or is exactly zero over a known range), so a test
// can compare against `truth.crossG(a, b, r)` instead of against another implementation's opinion.
//
// **Where no closed form exists, `crossG` returns `undefined` rather than an approximation.** Half
// an oracle is genuinely useful — a hard-core process pins g ≡ 0 below the core diameter and says
// nothing above it, and the part it does pin is exact. An approximated "expected" value would be
// indistinguishable from a wrong one at the tolerances these tests run at.
//
// ## Two generation details that the formulae depend on
//
// **Cluster parents are simulated on a window expanded by `PARENT_MARGIN`·σ.** A cluster whose
// parent lies just outside the ROI still drops offspring inside it. Simulating parents only within
// the ROI depletes the border, which is not a small effect — it biases g downward at exactly the
// radii the clustering is meant to show, and the closed forms below would then be wrong.
//
// **Parent counts and cluster sizes are Poisson, not fixed.** With a deterministic `m` offspring per
// parent the Neyman-Scott pcf picks up a factor `(m−1)/m` (5% at m = 20), and a deterministic parent
// count deflates it again by ~1/N_parents. Both are avoidable by drawing Poisson, so `n` here is an
// EXPECTED count rather than an exact one. For a fixture that trades on its analytic truth that is
// the right way round.
//
// ## The self-pair caveat
//
// `crossPCF` counts a cell against itself at distance 0 when the same cloud is passed twice, so an
// auto-correlation check must skip bin 0 or split the type. `truth.crossG(a, a, r)` gives the true
// auto-pcf of the underlying process, which does NOT include that artefact.

import { mulberry32 } from "./kernelAnalysis";
import type { LabelledCells } from "./pcf";

/** How many σ beyond the ROI to place cluster parents. At 4σ a Gaussian kernel has ~3e-5 of its
 *  mass left, which is far below the sampling noise of any test here. */
const PARENT_MARGIN = 4;

export type Bbox = readonly [number, number, number, number];

/**
 * What the generating process says the answer is.
 *
 * `crossG` is the true pair correlation between types `a` and `b` at separation `r` — the quantity
 * `crossPCF` estimates. `a === b` asks for the auto-pcf. `undefined` means the process has no closed
 * form there, which is a statement about the process, not a gap to be filled in later.
 */
export interface PatternTruth {
  crossG(a: number, b: number, r: number): number | undefined;
  /** What this pattern is FOR: the property a test using it should be pinning down. */
  readonly note: string;
}

export interface PointPattern extends LabelledCells {
  readonly xs: number[];
  readonly ys: number[];
  readonly typeId: number[];
  /** Human names, indexed by type id. Type ids are dense and 0-based. */
  readonly typeNames: string[];
  /** The window the process was observed through. Statistics must be given this, not the point
   *  cloud's own extent — the extent shrinks as `n` falls and would make ρ depend on sample size. */
  readonly bbox: Bbox;
  readonly truth: PatternTruth;
}

export interface PatternOptions {
  /** Expected points PER TYPE. Stochastic patterns land near it, not on it — see the header. */
  readonly n?: number;
  readonly seed?: number;
  /** Observation window. Defaults to a 1000-unit square, which at the default σ leaves room for the
   *  clustered patterns to show several cluster diameters. */
  readonly bbox?: Bbox;
  /** Number of cell types. Every pattern here takes it, because a pair is a special case and not the
   *  interesting one: the Gram-matrix / eigen-mode work on `cellmodes.html` reads a K×K matrix, and
   *  the questions it answers — which types form a guild, how many independent modes there are —
   *  have no two-type version at all. Defaults to 2 where a pair is the natural reading of the
   *  process, more where it is not. */
  readonly types?: number;
}

// --- primitives ----------------------------------------------------------------------

/** One standard normal from a uniform stream (Box-Muller, one of the two values kept).
 *
 *  Not the sum-of-uniforms approximation the old fixtures used: the closed forms below are Gaussian
 *  kernel results, and a kernel that is only approximately Gaussian makes them approximately true —
 *  which defeats the purpose of having them. */
function gauss(rnd: () => number): number {
  const u1 = Math.max(1e-12, rnd());
  const u2 = rnd();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Poisson deviate.
 *
 *  Knuth's product method below ~30, a normal approximation above it. The switch is not tidiness:
 *  Knuth compares against `exp(-mean)`, which UNDERFLOWS TO ZERO past mean ≈ 745, after which the
 *  loop runs until a product of uniforms reaches denormal zero and returns a number with no relation
 *  to the mean. Asking for 1728 points returned about 500, silently — and because the cross-pcf of a
 *  parent-offspring process goes as 1/λ_A, that showed up as a 22% error in a closed form that was
 *  in fact correct. A `csr` fixture would never have caught it: g ≡ 1 whatever the intensity. */
function poisson(mean: number, rnd: () => number): number {
  if (mean <= 0) return 0;
  if (mean > 30) return Math.max(0, Math.round(mean + Math.sqrt(mean) * gauss(rnd)));
  const limit = Math.exp(-mean);
  let k = 0;
  let p = 1;
  do {
    k++;
    p *= rnd();
  } while (p > limit);
  return k - 1;
}

const areaOf = (b: Bbox) => Math.max(1e-12, (b[2] - b[0]) * (b[3] - b[1]));

/** Homogeneous Poisson points in `b`, expected count `n`. */
function poissonPoints(n: number, b: Bbox, rnd: () => number): { xs: number[]; ys: number[] } {
  const count = poisson(n, rnd);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < count; i++) {
    xs.push(b[0] + rnd() * (b[2] - b[0]));
    ys.push(b[1] + rnd() * (b[3] - b[1]));
  }
  return { xs, ys };
}

const expand = (b: Bbox, m: number): Bbox => [b[0] - m, b[1] - m, b[2] + m, b[3] + m];
const inside = (b: Bbox, x: number, y: number) => x >= b[0] && x < b[2] && y >= b[1] && y < b[3];

/** Offspring per parent needed for `n` points to land INSIDE the ROI.
 *
 *  Parents live on the expanded window, so offspring are spread over that larger area and only
 *  `|ROI|/|outer|` of them land in view — a third of them lost at the default σ. The pcf does not
 *  depend on the cluster size (it is a function of κ and σ alone), so this only makes `n` mean what
 *  it says; it is not a correctness fix. */
function offspringPerParent(n: number, nParents: number, bbox: Bbox, outer: Bbox): number {
  return (n / nParents) * (areaOf(outer) / areaOf(bbox));
}

/** The pcf of a Thomas process: `1 + exp(−r²/4σ²) / (4πκσ²)`, κ = parent intensity. */
const thomasG = (r: number, sigma: number, kappa: number) =>
  1 + Math.exp((-r * r) / (4 * sigma * sigma)) / (4 * Math.PI * sigma * sigma * kappa);

/** The CROSS pcf between a parent process of intensity λ and its own offspring: `1 + k(r)/λ`, where
 *  `k` is the dispersal density itself — one displacement, not the difference of two, which is why
 *  this is `2σ²` where `thomasG` is `4σ²`. */
const parentOffspringG = (r: number, sigma: number, lambda: number) =>
  1 + Math.exp((-r * r) / (2 * sigma * sigma)) / (2 * Math.PI * sigma * sigma * lambda);

const CONST_ONE: PatternTruth["crossG"] = () => 1;

// --- the patterns --------------------------------------------------------------------

const DEFAULT_BBOX: Bbox = [0, 0, 1000, 1000];

function opts(o: PatternOptions, defaultTypes: number): { n: number; bbox: Bbox; rnd: () => number; K: number } {
  return {
    n: o.n ?? 1500,
    bbox: o.bbox ?? DEFAULT_BBOX,
    rnd: mulberry32((o.seed ?? 1) >>> 0),
    K: Math.max(1, Math.floor(o.types ?? defaultTypes)),
  };
}

/** `A, B, C …` past the alphabet, then `T27`. Short enough for a matrix axis label. */
function names(K: number, stem: string): string[] {
  return Array.from({ length: K }, (_, i) => (K <= 26 ? `${stem} ${String.fromCharCode(65 + i)}` : `${stem} ${i + 1}`));
}

/** Complete spatial randomness: K mutually independent homogeneous Poisson processes. */
export function csr(o: PatternOptions = {}): PointPattern {
  const { n, bbox, rnd, K } = opts(o, 2);
  const xs: number[] = [];
  const ys: number[] = [];
  const typeId: number[] = [];
  for (let t = 0; t < K; t++) {
    const p = poissonPoints(n, bbox, rnd);
    for (let i = 0; i < p.xs.length; i++) {
      xs.push(p.xs[i]!);
      ys.push(p.ys[i]!);
      typeId.push(t);
    }
  }
  return {
    xs,
    ys,
    typeId,
    typeNames: names(K, "csr"),
    bbox,
    truth: {
      crossG: CONST_ONE,
      note: `g ≡ 1 everywhere, for all ${(K * (K + 1)) / 2} pairs. The null yardstick: anything an estimator reports here that is not 1 is its own error, and the K×K matrix should have exactly one mode — none.`,
    },
  };
}

/**
 * A Thomas cluster process, randomly split into two labels — the canonical clustered pattern.
 *
 * The split is not decoration. A single-type pattern forces any cross-type consumer to pass the same
 * cloud twice, and `crossPCF` then counts every cell against itself at distance 0: the page read
 * g(r→0) = 17.4 for what is a ~4 process, all of it self-pairs. Splitting removes the artefact
 * without touching the answer, because **independent thinning preserves the pcf** — each label is a
 * thinned copy of the same process, so the cross-pcf between them equals the auto-pcf of the whole,
 * exactly.
 *
 * It also earns its keep pedagogically. Both types are strongly cross-clustered here and there is no
 * association whatever between them — they are one population with a coin flip on top. Beside
 * `independentClustered` (separately clumped, cross ≡ 1) it makes the pair of cases that random
 * labelling and random shift respectively answer.
 */
export function thomas(o: PatternOptions & { sigma?: number; parents?: number } = {}): PointPattern {
  const { n, bbox, rnd, K } = opts(o, 2);
  const sigma = o.sigma ?? 30;
  const nParents = o.parents ?? 40;
  const outer = expand(bbox, PARENT_MARGIN * sigma);
  // κ is defined over the EXPANDED window, because that is the window the parents actually live in
  // and the pcf formula wants the true parent intensity.
  const kappa = nParents / areaOf(outer);
  // K·n, because `n` is per type and the label draw splits the population K ways.
  const perParent = offspringPerParent(K * n, nParents, bbox, outer);
  const parents = poissonPoints(nParents, outer, rnd);
  const xs: number[] = [];
  const ys: number[] = [];
  const typeId: number[] = [];
  for (let p = 0; p < parents.xs.length; p++) {
    const m = poisson(perParent, rnd);
    for (let k = 0; k < m; k++) {
      const x = parents.xs[p]! + gauss(rnd) * sigma;
      const y = parents.ys[p]! + gauss(rnd) * sigma;
      if (inside(bbox, x, y)) {
        xs.push(x);
        ys.push(y);
        typeId.push(Math.min(K - 1, Math.floor(rnd() * K)));
      }
    }
  }
  return {
    xs,
    ys,
    typeId,
    typeNames: names(K, "clump"),
    bbox,
    truth: {
      // Auto and cross alike: thinning does not change the pcf, so both labels and the pair between
      // them all follow the same closed form.
      crossG: (_a, _b, r) => thomasG(r, sigma, kappa),
      note: `One clustered population split ${K} ways at random, so g(r) = 1 + exp(−r²/4σ²)/(4πκσ²) for EVERY pair (σ=${sigma}, κ=${kappa.toExponential(3)}); g(0⁺) ≈ ${thomasG(0, sigma, kappa).toFixed(2)}. Every type is strongly cross-clustered with every other and yet none is associated with any — it is one population wearing ${K} labels, so the K×K matrix should show ONE mode.`,
    },
  };
}

/** Two clustered types with SEPARATE parents — each self-clustered, mutually independent.
 *
 *  The pattern that separates the two envelope nulls, and the reason it is worth naming: random
 *  labelling rejects here almost every time and is RIGHT to (the types are not exchangeable — each
 *  clumps on its own), while the truth about their ASSOCIATION is that there is none. Any test or
 *  demo about which null answers which question needs this pattern specifically.
 *
 *  **The estimator does not return the truth here, and by a knowable amount.** The cross-pcf is
 *  ≡ 1 for the process, but `crossPCF` reads about 0.92 at short range with the default 40 clusters
 *  per type, because it normalises by the OBSERVED global ρ̂_B and a realisation with few clusters is
 *  a lumpy intensity surface. Measured over 20 realisations, the shortfall falls as the clusters get
 *  more numerous — 0.923 at 40 parents, 0.979 at 120, 1.002 at 400 — so it is O(1/n_clusters), not a
 *  fixed bias. Anything calibrating a false-positive rate on this pattern is calibrating against
 *  0.92, not against 1. */
export function independentClustered(o: PatternOptions & { sigma?: number; parents?: number } = {}): PointPattern {
  const { n, bbox, rnd, K } = opts(o, 2);
  const sigma = o.sigma ?? 30;
  const nParents = o.parents ?? 40;
  const outer = expand(bbox, PARENT_MARGIN * sigma);
  const kappa = nParents / areaOf(outer);
  const xs: number[] = [];
  const ys: number[] = [];
  const typeId: number[] = [];
  const perParent = offspringPerParent(n, nParents, bbox, outer);
  for (let t = 0; t < K; t++) {
    const parents = poissonPoints(nParents, outer, rnd);
    for (let p = 0; p < parents.xs.length; p++) {
      const m = poisson(perParent, rnd);
      for (let k = 0; k < m; k++) {
        const x = parents.xs[p]! + gauss(rnd) * sigma;
        const y = parents.ys[p]! + gauss(rnd) * sigma;
        if (inside(bbox, x, y)) {
          xs.push(x);
          ys.push(y);
          typeId.push(t);
        }
      }
    }
  }
  return {
    xs,
    ys,
    typeId,
    typeNames: names(K, "clustered"),
    bbox,
    truth: {
      // Each type clusters on its own parents; the two parent sets are independent, so the CROSS
      // pcf is flat 1 while both auto-pcfs are well above it.
      crossG: (a, b, r) => (a === b ? thomasG(r, sigma, kappa) : 1),
      note: `Each of the ${K} types clustered on its own centres, all mutually independent: auto-g ≫ 1 but every cross-g ≡ 1. Association is absent; exchangeability is too. The K×K matrix should be diagonal — ${K} modes, not one.`,
    },
  };
}

/** Types 1…K−1 scattered around the cells of type 0 — genuine co-location, on a shared anchor.
 *
 *  With more than two types this says something a pair cannot: the recruited types are co-located
 *  with the anchor AND with each other, without any of them influencing another. Their mutual
 *  association is entirely inherited from the anchor they share — which is the multi-type form of
 *  the confound the partial correlation exists to remove, and it has a closed form on both legs. */
export function colocalised(o: PatternOptions & { sigma?: number; perAnchor?: number } = {}): PointPattern {
  const { n, bbox, rnd, K } = opts(o, 2);
  const sigma = o.sigma ?? 25;
  // Mean B recruited per A. λ_B = λ_A · perAnchor, so 1 gives the two types the same intensity and
  // `n` means the same thing for both — which is what every other pattern here promises.
  const perParent = o.perAnchor ?? 1;
  // A is simulated on the expanded window so that B offspring whose parent falls just outside are
  // present inside it; only the A points inside the ROI are reported (a Poisson process restricted
  // to a sub-window is still Poisson at the same intensity, so λ_A is unchanged).
  const outer = expand(bbox, PARENT_MARGIN * sigma);
  const lambdaA = n / areaOf(bbox);
  const allA = poissonPoints(lambdaA * areaOf(outer), outer, rnd);
  const xs: number[] = [];
  const ys: number[] = [];
  const typeId: number[] = [];
  for (let i = 0; i < allA.xs.length; i++) {
    if (inside(bbox, allA.xs[i]!, allA.ys[i]!)) {
      xs.push(allA.xs[i]!);
      ys.push(allA.ys[i]!);
      typeId.push(0);
    }
  }
  for (let t = 1; t < K; t++) {
    for (let i = 0; i < allA.xs.length; i++) {
      const m = poisson(perParent, rnd);
      for (let k = 0; k < m; k++) {
        const x = allA.xs[i]! + gauss(rnd) * sigma;
        const y = allA.ys[i]! + gauss(rnd) * sigma;
        if (inside(bbox, x, y)) {
          xs.push(x);
          ys.push(y);
          typeId.push(t);
        }
      }
    }
  }
  return {
    xs,
    ys,
    typeId,
    typeNames: ["anchor", ...names(K - 1, "recruited")],
    bbox,
    truth: {
      crossG: (a, b, r) => {
        if (a === b) return undefined; // an offspring type against itself has no closed form here
        // Anchor against a recruited type: ONE displacement, so 2σ². Two recruited types: the
        // difference of two independent displacements, so 4σ² — the same distinction that separates
        // `parentOffspringG` from `thomasG`, and getting it the wrong way round would inflate the
        // claimed correlation length by √2.
        if (a === 0 || b === 0) return parentOffspringG(r, sigma, lambdaA);
        return thomasG(r, sigma, lambdaA);
      },
      note: `${K - 1} recruited type(s) sit around one anchor: anchor↔recruited g(r) = 1 + exp(−r²/2σ²)/(2πσ²λ_A), and recruited↔recruited g(r) = 1 + exp(−r²/4σ²)/(4πσ²λ_A) — co-located purely by sharing the anchor, with no influence between them (σ=${sigma}).`,
    },
  };
}

/** K types in disjoint vertical strips separated by `gap` — mutual exclusion, ordered.
 *
 *  Beyond two types this gains a property a pair has no room for: the exclusion is GRADED by how far
 *  apart the strips are, so the true K×K matrix has a band structure and the type ordering is
 *  recoverable from it. That is a much sharper test of an ordination than a two-block split. */
export function segregated(o: PatternOptions & { gap?: number } = {}): PointPattern {
  const { n, bbox, rnd, K } = opts(o, 2);
  const gap = o.gap ?? 120;
  const span = bbox[2] - bbox[0];
  // K strips of equal width with K−1 gaps between them.
  const width = Math.max(1e-6, (span - (K - 1) * gap) / K);
  const xs: number[] = [];
  const ys: number[] = [];
  const typeId: number[] = [];
  for (let t = 0; t < K; t++) {
    const x0 = bbox[0] + t * (width + gap);
    const p = poissonPoints(n, [x0, bbox[1], x0 + width, bbox[3]], rnd);
    for (let i = 0; i < p.xs.length; i++) {
      xs.push(p.xs[i]!);
      ys.push(p.ys[i]!);
      typeId.push(t);
    }
  }
  /** Closest any point of strip `a` can be to any point of strip `b`. Exact, from the geometry. */
  const minSep = (a: number, b: number) => {
    const d = Math.abs(a - b);
    return d === 0 ? 0 : (d - 1) * width + d * gap;
  };
  return {
    xs,
    ys,
    typeId,
    typeNames: names(K, "strip"),
    bbox,
    truth: {
      // Exact and edge-effect-free: no pair can exist below the strips' separation, whatever the
      // estimator does about boundaries. Above it the geometry of two rectangles decides the value,
      // and that has no useful closed form.
      crossG: (a, b, r) => (a !== b && r < minSep(a, b) ? 0 : undefined),
      note: `${K} disjoint strips ${width.toFixed(0)} wide, ${gap} apart: cross-g ≡ 0 below ${minSep(0, 1).toFixed(0)} for neighbours and ${K > 2 ? `${minSep(0, K - 1).toFixed(0)} for the two ends` : "further for any wider separation"}, exactly. Non-zero there is wrong, not noisy.`,
    },
  };
}

/** Simple sequential inhibition — a hard core of `core`, no two points closer than that.
 *
 *  Split into two labels for the same reason `thomas` is: the exclusion holds between any two points
 *  regardless of label, so the cross-pcf is 0 below the core exactly as the auto-pcf is, and the page
 *  gets a real A ≠ B pair instead of a bin 0 full of self-pairs. */
export function hardcore(o: PatternOptions & { core?: number } = {}): PointPattern {
  const { n, bbox, rnd, K } = opts(o, 2);
  const core = o.core ?? 25;
  const core2 = core * core;
  const xs: number[] = [];
  const ys: number[] = [];
  // Rejection with a hard attempt budget: SSI has no guarantee of reaching `n`, and at high
  // intensity it jams. Stopping on the budget is honest — the pattern is thinner than asked for and
  // still exactly hard-core, which is the property the truth claims.
  const target = K * n; // `n` is per type, and the label draw below splits it K ways
  const maxAttempts = target * 60;
  for (let a = 0; a < maxAttempts && xs.length < target; a++) {
    const x = bbox[0] + rnd() * (bbox[2] - bbox[0]);
    const y = bbox[1] + rnd() * (bbox[3] - bbox[1]);
    let ok = true;
    for (let i = 0; i < xs.length; i++) {
      const dx = xs[i]! - x;
      const dy = ys[i]! - y;
      if (dx * dx + dy * dy < core2) {
        ok = false;
        break;
      }
    }
    if (ok) {
      xs.push(x);
      ys.push(y);
    }
  }
  return {
    xs,
    ys,
    typeId: xs.map(() => Math.min(K - 1, Math.floor(rnd() * K))),
    typeNames: names(K, "inhibited"),
    bbox,
    truth: {
      crossG: (_a, _b, r) => (r < core ? 0 : undefined),
      note: `No two points within ${core}, whatever their labels: g ≡ 0 below it for every pair, exactly. Above it SSI has no closed form, so nothing is claimed there.`,
    },
  };
}

/** Two independent types sharing a left-to-right intensity ramp.
 *
 *  The first-order / second-order confound, as a fixture. The types are independent — the true
 *  cross-g is 1 at every r — but both are commoner on the right, so an estimator that assumes a
 *  constant ρ_B (Mode 1, which is what this repo computes) reports g > 1 and it looks like
 *  association. Any claim that a statistic separates "clustered together" from "both common in the
 *  same place" has to be demonstrated on this. */
export function gradient(o: PatternOptions & { contrast?: number } = {}): PointPattern {
  const { n, bbox, rnd, K } = opts(o, 2);
  const contrast = o.contrast ?? 8;
  const xs: number[] = [];
  const ys: number[] = [];
  const typeId: number[] = [];
  const w = bbox[2] - bbox[0];
  // Thinning a homogeneous process by λ(x)/λmax, so the two types are independent by construction
  // rather than by a shared random draw.
  //
  // The acceptance probability averages `(contrast + 1) / (2·contrast)` over a uniform x, so the
  // pre-thinning count has to be divided by that for `n` to survive it. Getting this wrong is
  // invisible in g — thinning does not change the pcf — and shows up only as the wrong number of
  // cells: at contrast 8 an earlier version returned 3,097 for 1,200 asked.
  const preThin = (n * 2 * contrast) / (contrast + 1);
  for (let t = 0; t < K; t++) {
    const target = poisson(preThin, rnd);
    for (let i = 0; i < target; i++) {
      const x = bbox[0] + rnd() * w;
      const y = bbox[1] + rnd() * (bbox[3] - bbox[1]);
      const accept = (1 + (contrast - 1) * ((x - bbox[0]) / w)) / contrast;
      if (rnd() < accept) {
        xs.push(x);
        ys.push(y);
        typeId.push(t);
      }
    }
  }
  return {
    xs,
    ys,
    typeId,
    typeNames: names(K, "ramped"),
    bbox,
    truth: {
      crossG: CONST_ONE,
      note: `All ${K} types ${contrast}× denser on the right, and mutually independent: the TRUE g ≡ 1 for every pair. A homogeneous-ρ estimator will not say so — and on the K×K matrix it manufactures a single strong mode out of nothing, which is the multi-type form of the trap.`,
    },
  };
}

/**
 * K types in G guilds: co-located within a guild, independent between guilds.
 *
 * The pattern that has no two-type version, and the one the Gram-matrix / eigen-mode page exists to
 * find. Each guild gets its own set of cluster centres and every type in that guild is an
 * independent thinning of that guild's process. So the true K×K pcf matrix is BLOCK DIAGONAL —
 * `thomasG` inside a block, exactly 1 outside it — and a correct ordination should recover G modes
 * and the membership, not K and not 1.
 *
 * Both halves are exact. Within a guild the types are thinnings of one process, and thinning
 * preserves the pcf; between guilds the parent sets are independent. Nothing is approximated, which
 * makes this usable as a scoring target and not just a picture: the recovered blocks either match
 * `guildOf` or they do not.
 */
export function guilds(o: PatternOptions & { sigma?: number; parents?: number; guilds?: number } = {}): PointPattern {
  const { n, bbox, rnd, K } = opts(o, 6);
  const sigma = o.sigma ?? 40;
  const nParents = o.parents ?? 25;
  const G = Math.max(1, Math.min(K, Math.floor(o.guilds ?? 2)));
  const outer = expand(bbox, PARENT_MARGIN * sigma);
  const kappa = nParents / areaOf(outer);
  /** Contiguous blocks, remainder spread over the first guilds so none is empty. */
  const guildOf = (t: number) => Math.min(G - 1, Math.floor((t * G) / K));
  const sizeOf = (g: number) => {
    let c = 0;
    for (let t = 0; t < K; t++) if (guildOf(t) === g) c++;
    return c;
  };
  const xs: number[] = [];
  const ys: number[] = [];
  const typeId: number[] = [];
  for (let g = 0; g < G; g++) {
    const members: number[] = [];
    for (let t = 0; t < K; t++) if (guildOf(t) === g) members.push(t);
    // The guild's whole population is `members.length · n`, then split among its members — so `n` is
    // per TYPE regardless of how the guilds are sized.
    const perParent = offspringPerParent(members.length * n, nParents, bbox, outer);
    const parents = poissonPoints(nParents, outer, rnd);
    for (let p = 0; p < parents.xs.length; p++) {
      const m = poisson(perParent, rnd);
      for (let k = 0; k < m; k++) {
        const x = parents.xs[p]! + gauss(rnd) * sigma;
        const y = parents.ys[p]! + gauss(rnd) * sigma;
        if (inside(bbox, x, y)) {
          xs.push(x);
          ys.push(y);
          typeId.push(members[Math.min(members.length - 1, Math.floor(rnd() * members.length))]!);
        }
      }
    }
  }
  const blocks = Array.from({ length: G }, (_, g) => sizeOf(g)).join("+");
  return {
    xs,
    ys,
    typeId,
    typeNames: Array.from({ length: K }, (_, t) => `G${guildOf(t) + 1}·${String.fromCharCode(97 + t)}`),
    bbox,
    truth: {
      crossG: (a, b, r) => (guildOf(a) === guildOf(b) ? thomasG(r, sigma, kappa) : 1),
      note: `${K} types in ${G} guilds (${blocks}), co-located within a guild and independent between: the true K×K matrix is block diagonal, g(0⁺) ≈ ${thomasG(0, sigma, kappa).toFixed(2)} inside a block and exactly 1 outside. An ordination should find ${G} modes and this membership.`,
    },
  };
}

// --- registry ------------------------------------------------------------------------

export interface PointPatternSpec {
  readonly key: string;
  readonly label: string;
  readonly describe: string;
  readonly make: (o: PatternOptions) => PointPattern;
}

/** Ordered as an argument: the null first, then the two ways a pattern departs from it, then the
 *  cases that are easy to confuse with those departures. */
export const POINT_PATTERNS: PointPatternSpec[] = [
  { key: "csr", label: "Complete spatial randomness", describe: "Two independent Poisson types. g ≡ 1 — the yardstick.", make: csr },
  { key: "thomas", label: "Thomas clusters", describe: "One clustered type, with a closed-form g(r).", make: thomas },
  {
    key: "colocalised",
    label: "Co-located pair",
    describe: "B recruited around A — real association, known cross-g.",
    make: colocalised,
  },
  {
    key: "segregated",
    label: "Segregated pair",
    describe: "Disjoint territories: cross-g is exactly 0 below the gap.",
    make: segregated,
  },
  {
    key: "independentClustered",
    label: "Independently clustered",
    describe: "Both types clumped, neither aware of the other. Cross-g ≡ 1.",
    make: independentClustered,
  },
  {
    key: "guilds",
    label: "Guilds (block structure)",
    describe: "K types in G co-locating groups — the true matrix is block diagonal.",
    make: guilds,
  },
  { key: "hardcore", label: "Hard-core inhibition", describe: "A minimum spacing: g ≡ 0 below the core.", make: hardcore },
  {
    key: "gradient",
    label: "Shared density gradient",
    describe: "Independent, but both denser to the right — the classic confound.",
    make: gradient,
  },
];

export function makePointPattern(key: string, o: PatternOptions = {}): PointPattern {
  const spec = POINT_PATTERNS.find((p) => p.key === key);
  if (!spec) throw new Error(`pointPatterns: no generator '${key}' (have ${POINT_PATTERNS.map((p) => p.key).join(", ")})`);
  return spec.make(o);
}

/** Split a pattern into one cloud per type — the shape the per-type consumers want. */
export function patternClouds(p: PointPattern): { id: number; label: string; xs: number[]; ys: number[] }[] {
  return p.typeNames.map((label, id) => {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < p.xs.length; i++) {
      if (p.typeId[i] === id) {
        xs.push(p.xs[i]!);
        ys.push(p.ys[i]!);
      }
    }
    return { id, label, xs, ys };
  });
}
