// Synthetic datasets whose *geometry* is the point — the data the UMAP page and the
// scaling benchmark run on when there is no real store to hand.
//
// The first version of the playground generated isotropic Gaussian blobs, and blobs are
// a bad demonstration of UMAP: separated clusters are the one case where almost any
// projection works, so the embedding stabilises into coloured dots and there is nothing
// to see. Everything interesting about the algorithm — whether it unrolls a curved
// sheet, where it has to tear a surface it cannot flatten, what a continuum looks like
// next to discrete types, how much apparent structure it invents from noise — needs data
// with real structure to reveal it.
//
// **Every generator is built the same way**, and that uniformity is what makes the
// gene-programme controls work across all of them:
//
//   1. A generator emits a low-dimensional **latent** `[n, L]` — the true manifold —
//      plus ground truth: a discrete `label` and, where the manifold is a continuum, a
//      continuous `truth` (arc length, pseudotime, angle).
//   2. `expressManifold` maps each latent axis onto a **block of genes** with random
//      positive loadings, and adds noise. A "gene programme" is then exactly one latent
//      axis, so switching a programme off deletes one dimension of the true manifold and
//      the embedding has to relax into what is left.
//
// **Scale normalisation.** Latents are scaled so the RMS pairwise distance is 1, so a
// generator's parameterisation cannot leak into anything downstream. The scaling is global
// (one factor for the whole latent block) rather than per-axis: standardising axes
// independently would stretch a swiss roll's height against its radius and destroy the
// very geometry the generator exists to produce.
//
// **Noise is a fraction of the RMS pairwise distance**, so the figure means the same thing
// for every generator, and each generator carries its own default because the right amount
// differs enormously between them.
//
// **A thin manifold is the wrong thing to generate, and this is not obvious.** A trajectory
// drawn as a near-exact curve — points scattered by a fraction of their own spacing — comes
// out of UMAP as a few hundred disconnected beads, each internally perfect (trustworthiness
// 1.000) and globally scattered. That is not an implementation fault: umap-learn was run on
// the identical matrix and produced the same picture, to within a few per cent on every
// summary (median edge length 0.199 vs our 0.192, extent 33 vs 33, largest connected piece
// 129 points in both), with spectral init and with random. UMAP shatters clean 1-D
// manifolds, full stop.
//
// The generators therefore give their continua the width real data has — cells scatter
// around a trajectory, they do not sit on it. `branching` at `noise` 0.45 is a clean
// three-armed tree at every n tried; at 0.10 it is beads. The transverse spread is what
// makes the data locally full-dimensional, and that is what UMAP needs to keep a continuum
// connected. Where a generator's default looks oddly large, that is why.

import { mulberry32 } from "./umapLayout";

/** A generated manifold: the truth, before it is turned into gene expression. */
export interface Manifold {
  readonly name: string;
  readonly n: number;
  /** Row-major `[n, nLatent]`, scaled to unit RMS pairwise distance. */
  readonly latent: Float32Array;
  readonly nLatent: number;
  /** One name per latent axis — these become the gene-programme labels. */
  readonly latentNames: string[];
  /** Discrete ground truth (branch, ring, cell type). */
  readonly label: Uint8Array;
  readonly labelNames: string[];
  /** Continuous ground truth where the manifold has one: pseudotime, arc length, angle. */
  readonly truth?: Float32Array;
  readonly truthName?: string;
  /** True when `truth` is an angle, so a colour ramp should wrap rather than run end to end. */
  readonly truthCyclic?: boolean;
  /** The manifold's own flat parameterisation `[n, intrinsicDim]`, in the same units as
   *  `latent`, where one exists — the answer an embedding is trying to recover.
   *
   *  Only the generators whose manifold really is a flat sheet bent into the ambient space
   *  carry this. A sphere and a pair of linked rings deliberately do not: neither has a
   *  flat parameterisation, which is the whole reason they are in the list, and inventing
   *  one would turn a genuine obstruction into a score to be beaten. */
  readonly intrinsic?: Float32Array;
  readonly intrinsicDim?: number;
  /** Expression noise this generator wants by default, as a fraction of the RMS pairwise
   *  distance. Tuned per shape by looking at the result — see the module header on why the
   *  continuum generators want so much of it. */
  readonly noise: number;
  /** What a faithful embedding should look like — shown in the page, and the thing to
   *  check a result against. */
  readonly expect: string;
}

// --- shared helpers ------------------------------------------------------------------

function gaussian(rnd: () => number): number {
  const u = Math.max(rnd(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
}

/**
 * RMS distance between random pairs.
 *
 * Sampled rather than exhaustive: this runs at n = 100k in the benchmark, where all pairs
 * would be 10^10 distances to normalise a constant. A few thousand pairs pin the RMS to
 * well within the precision anything downstream cares about.
 */
function rmsPairwise(x: ArrayLike<number>, n: number, dim: number, rnd: () => number, samples = 4000): number {
  let acc = 0;
  let count = 0;
  for (let s = 0; s < samples; s++) {
    const i = Math.floor(rnd() * n);
    const j = Math.floor(rnd() * n);
    if (i === j) continue;
    let d = 0;
    for (let c = 0; c < dim; c++) {
      const t = x[i * dim + c]! - x[j * dim + c]!;
      d += t * t;
    }
    acc += d;
    count++;
  }
  return Math.sqrt(acc / Math.max(count, 1)) || 1;
}

interface Draft {
  latent: Float32Array;
  nLatent: number;
  latentNames: string[];
  label: Uint8Array;
  labelNames: string[];
  truth?: Float32Array;
  truthName?: string;
  truthCyclic?: boolean;
  intrinsic?: Float32Array;
  intrinsicDim?: number;
}

/** Normalise the scale and stamp on the descriptive fields.
 *
 *  `intrinsic` is scaled by the same factor, not renormalised on its own: it shares the
 *  latent's units by construction, and rescaling it independently would change the sheet's
 *  aspect ratio relative to the ambient shape it is meant to be compared against. */
function finish(name: string, n: number, draft: Draft, noise: number, expect: string, rnd: () => number): Manifold {
  const scale = 1 / rmsPairwise(draft.latent, n, draft.nLatent, rnd);
  for (let t = 0; t < draft.latent.length; t++) draft.latent[t] = draft.latent[t]! * scale;
  if (draft.intrinsic) for (let t = 0; t < draft.intrinsic.length; t++) draft.intrinsic[t] = draft.intrinsic[t]! * scale;
  return { name, n, ...draft, noise, expect };
}

// --- generators ----------------------------------------------------------------------

/** Well-separated blobs: the baseline, and deliberately the least interesting one.
 *
 *  Kept because it is the case people picture when they say "UMAP", and having it in the
 *  list is what makes the others read as different rather than as broken. Latent axes are
 *  one-hot type indicators, so switching two types' programmes off genuinely merges
 *  them — they stop being distinguishable rather than merely blurring. */
function blobs(n: number, seed: number, clusters = 5): Manifold {
  const rnd = mulberry32(seed);
  const L = clusters;
  const latent = new Float32Array(n * L);
  const label = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const c = i % clusters;
    label[i] = c;
    latent[i * L + c] = 1;
  }
  return finish(
    "blobs",
    n,
    {
      latent,
      nLatent: L,
      latentNames: Array.from({ length: clusters }, (_, c) => `type ${c + 1}`),
      label,
      labelNames: Array.from({ length: clusters }, (_, c) => `type ${c + 1}`),
    },
    0.55,
    "Separated islands, one per type. Switch two type programmes off and those two islands merge.",
    rnd,
  );
}

/** A bifurcating trajectory — the shape single-cell data actually tends to have.
 *
 *  A root branch that splits into `branches` arms. This is the generator to judge an
 *  embedding by: the correct answer is a connected tree with the branch point intact, and
 *  the common failure is arms detaching into separate blobs, which reads as discrete cell
 *  types that are not there. */
function branching(n: number, seed: number, branches = 3): Manifold {
  const rnd = mulberry32(seed);
  const L = 3;
  const latent = new Float32Array(n * L);
  const label = new Uint8Array(n);
  const truth = new Float32Array(n);
  // Arms leave the branch point spread evenly around a cone, so no two are collinear and
  // the embedding cannot fake the tree by folding one arm onto another.
  const dirs = Array.from({ length: branches }, (_, b) => {
    const a = (b / branches) * Math.PI * 2;
    return [0.7, Math.cos(a) * 0.9, Math.sin(a) * 0.9] as const;
  });
  const split = 0.45;
  for (let i = 0; i < n; i++) {
    const t = rnd();
    truth[i] = t;
    if (t < split) {
      label[i] = 0;
      latent[i * L] = t;
    } else {
      const b = Math.floor(rnd() * branches);
      label[i] = b + 1;
      const s = t - split;
      const d = dirs[b]!;
      latent[i * L] = split + d[0] * s;
      latent[i * L + 1] = d[1] * s;
      latent[i * L + 2] = d[2] * s;
    }
  }
  return finish(
    "branching",
    n,
    {
      latent,
      nLatent: L,
      latentNames: ["progression", "fate axis 1", "fate axis 2"],
      label,
      labelNames: ["root", ...Array.from({ length: branches }, (_, b) => `branch ${b + 1}`)],
      truth,
      truthName: "pseudotime",
    },
    0.45,
    "A connected tree: one trunk splitting into arms. Arms that detach into islands are an artefact, not cell types.",
    rnd,
  );
}

/** The swiss roll: a flat 2-D sheet rolled up in 3-D.
 *
 *  The classic manifold-learning test, and the sharpest demonstration of what a
 *  neighbourhood graph buys you — points on adjacent turns are close in ambient distance
 *  and far along the sheet, so anything that trusts raw distance (PCA, MDS) glues the
 *  turns together. A correct embedding is a rectangle with the bands in order.
 *
 *  **The pitch matters more than the shape.** The textbook parameterisation
 *  (`r = t`, `t` from 1.5π over two turns) leaves adjacent turns about eight times further
 *  apart than a point's nearest neighbour, which makes the k-NN graph trivially correct and
 *  the demonstration vacuous — the roll only looks hard. Here the spiral has an explicit
 *  `pitch` that a k-neighbourhood can plausibly jump: measured at n = 3000, the radius
 *  reached at `n_neighbors` 12 is 0.65 of the gap and at 40 it is 1.20, so the roll really
 *  does collapse if you push the slider.
 *
 *  **The sheet is square, and that is not cosmetic.** An earlier version derived the height
 *  from `n` to hold the sampling density fixed, which produced a 20:1 ribbon — and a ribbon
 *  is globally a 1-D object, so it shattered exactly the way the header describes. Square
 *  keeps it honestly 2-D. The cost is that the sampling density now rises with `n`: below
 *  about 2500 points the roll is too coarsely sampled and the graph short-circuits on its
 *  own, which is a limitation of this generator rather than a lesson about UMAP.
 *
 *  Expect a lacework, not a solid rectangle. The bands come out in order and adjacent turns
 *  stay apart — the sheet is genuinely unrolled — but a clean 2-D manifold comes out of UMAP
 *  full of holes, for the reason in the module header. */
function swissRoll(n: number, seed: number, turns = 1.5): Manifold {
  const rnd = mulberry32(seed);
  const L = 3;
  const latent = new Float32Array(n * L);
  const intrinsic = new Float32Array(n * 2);
  const label = new Uint8Array(n);
  const truth = new Float32Array(n);
  const R0 = 1.2;
  const PITCH = 1;
  const thetaMax = turns * Math.PI * 2;
  // Arc length along an Archimedean spiral, ignoring the dr/dθ term: it contributes
  // (pitch/2π)² ≈ 0.025 against r² ≥ 1.4, i.e. under 1%, and the intrinsic coordinate is
  // only ever used up to a monotone rescaling.
  const arc = (th: number) => R0 * th + (PITCH * th * th) / (4 * Math.PI);
  const arcMax = arc(thetaMax);
  // Square: as wide as it is long, so the sheet is 2-D at every scale rather than a ribbon
  // that is globally 1-D. n points then sit `arcMax / sqrt(n)` apart.
  const HEIGHT = arcMax;
  // Sample uniformly in ARC LENGTH, not in the angle. Uniform in θ looks equivalent and is
  // not: arc length grows as θ², so the outer turns come out roughly three times sparser
  // than the inner ones, their k-neighbourhoods reach correspondingly further, and the
  // graph short-circuits across the gap at the outside of the roll while the inside is
  // fine. The embedding then tangles in a way that looks like an optimiser problem and is
  // really a sampling one. Inverting `arc` is one quadratic.
  const invArc = (s: number) => {
    const q = PITCH / (4 * Math.PI);
    return (-R0 + Math.sqrt(R0 * R0 + 4 * q * s)) / (2 * q);
  };
  for (let i = 0; i < n; i++) {
    const h = rnd();
    const th = invArc(rnd() * arcMax);
    const r = R0 + (PITCH * th) / (2 * Math.PI);
    latent[i * L] = r * Math.cos(th);
    latent[i * L + 1] = h * HEIGHT;
    latent[i * L + 2] = r * Math.sin(th);
    intrinsic[i * 2] = arc(th);
    intrinsic[i * 2 + 1] = h * HEIGHT;
    truth[i] = arc(th) / arcMax;
    // Bands along the sheet, so the discrete colouring shows whether turns stayed in order.
    label[i] = Math.min(5, Math.floor(truth[i]! * 6));
  }
  return finish(
    "swissRoll",
    n,
    {
      latent,
      nLatent: L,
      latentNames: ["roll x", "sheet height", "roll z"],
      label,
      labelNames: Array.from({ length: 6 }, (_, b) => `band ${b + 1}`),
      truth,
      truthName: "position along roll",
      intrinsic,
      intrinsicDim: 2,
    },
    0.12,
    "A lacework sheet — holey, but with the bands running in order across it. Bands out of order means turns of the roll were glued together; raise n_neighbors until they are.",
    rnd,
  );
}

/** Two interlocked rings — a manifold with a topological obstruction.
 *
 *  There is no way to lay two linked circles flat in 2-D without cutting one, so this is
 *  the case where the *right* answer contains a tear. Worth having in the list precisely
 *  because it makes visible that a break in a UMAP is sometimes forced by the projection
 *  rather than present in the data. */
function linkedRings(n: number, seed: number): Manifold {
  const rnd = mulberry32(seed);
  const L = 3;
  const latent = new Float32Array(n * L);
  const label = new Uint8Array(n);
  const truth = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const ring = i % 2;
    const a = rnd() * Math.PI * 2;
    label[i] = ring;
    truth[i] = a / (Math.PI * 2);
    if (ring === 0) {
      latent[i * L] = Math.cos(a);
      latent[i * L + 1] = Math.sin(a);
      latent[i * L + 2] = 0;
    } else {
      // Offset along x and standing in the x–z plane, so it threads the first ring.
      latent[i * L] = 1 + Math.cos(a);
      latent[i * L + 1] = 0;
      latent[i * L + 2] = Math.sin(a);
    }
  }
  return finish(
    "linkedRings",
    n,
    {
      latent,
      nLatent: L,
      latentNames: ["ring plane x", "ring plane y", "link axis"],
      label,
      labelNames: ["ring A", "ring B"],
      truth,
      truthName: "angle",
      truthCyclic: true,
    },
    0.3,
    "Two loops that cannot both stay closed in 2-D — expect one of them to be cut. The tear is the projection, not the data.",
    rnd,
  );
}

/** A hollow sphere: a surface with no flat representation at all.
 *
 *  Every embedding has to cut it open somewhere, and where the cut lands is arbitrary —
 *  a compact reminder that the position of a gap in a UMAP carries no meaning. */
function sphere(n: number, seed: number): Manifold {
  const rnd = mulberry32(seed);
  const L = 3;
  const latent = new Float32Array(n * L);
  const label = new Uint8Array(n);
  const truth = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    // z uniform in [-1,1] gives a uniform density on the sphere; sampling the polar angle
    // uniformly instead would crowd the poles.
    const z = rnd() * 2 - 1;
    const a = rnd() * Math.PI * 2;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    latent[i * L] = r * Math.cos(a);
    latent[i * L + 1] = r * Math.sin(a);
    latent[i * L + 2] = z;
    truth[i] = (z + 1) / 2;
    label[i] = Math.min(5, Math.floor(((z + 1) / 2) * 6));
  }
  return finish(
    "sphere",
    n,
    {
      latent,
      nLatent: L,
      latentNames: ["axis x", "axis y", "axis z"],
      label,
      labelNames: Array.from({ length: 6 }, (_, b) => `latitude ${b + 1}`),
      truth,
      truthName: "latitude",
    },
    0.3,
    "A disc or a cut-open shell with latitude running edge to centre. Wherever the cut lands is arbitrary.",
    rnd,
  );
}

/** Clusters within clusters — structure at two scales at once.
 *
 *  This is the generator that makes `n_neighbors` mean something: a small value resolves
 *  the sub-clusters, a large one sees only the coarse groups. Both pictures are correct
 *  and they look nothing alike, which is the honest answer to "which is the real
 *  structure". */
function hierarchy(n: number, seed: number, coarse = 3, fine = 4): Manifold {
  const rnd = mulberry32(seed);
  const L = coarse + fine;
  const latent = new Float32Array(n * L);
  const label = new Uint8Array(n);
  // The sub-structure is a shared set of axes at a quarter of the amplitude, so the fine
  // splits sit *inside* each coarse group rather than forming a grid of their own.
  const FINE_AMPLITUDE = 0.25;
  for (let i = 0; i < n; i++) {
    const c = i % coarse;
    const f = Math.floor(i / coarse) % fine;
    label[i] = c * fine + f;
    latent[i * L + c] = 1;
    latent[i * L + coarse + f] = FINE_AMPLITUDE;
  }
  return finish(
    "hierarchy",
    n,
    {
      latent,
      nLatent: L,
      latentNames: [
        ...Array.from({ length: coarse }, (_, c) => `lineage ${c + 1}`),
        ...Array.from({ length: fine }, (_, f) => `state ${f + 1}`),
      ],
      label,
      labelNames: Array.from({ length: coarse * fine }, (_, t) => `L${Math.floor(t / fine) + 1}·S${(t % fine) + 1}`),
    },
    0.5,
    "Groups of sub-groups. Drag n_neighbors low to resolve the sub-clusters and high to see only the coarse ones.",
    rnd,
  );
}

/** A cycle crossed with discrete types — a continuum and clusters in the same dataset.
 *
 *  The cell-cycle case: each type traces its own loop. Tests whether an embedding can hold
 *  a continuous coordinate *within* clusters at the same time as separating them, which is
 *  a routine ask of real data and one that blob-only demos never exercise. */
function cellCycle(n: number, seed: number, types = 3): Manifold {
  const rnd = mulberry32(seed);
  const L = types + 2;
  const latent = new Float32Array(n * L);
  const label = new Uint8Array(n);
  const truth = new Float32Array(n);
  const CYCLE_AMPLITUDE = 0.42;
  for (let i = 0; i < n; i++) {
    const c = i % types;
    const a = rnd() * Math.PI * 2;
    label[i] = c;
    truth[i] = a / (Math.PI * 2);
    latent[i * L + c] = 1;
    latent[i * L + types] = Math.cos(a) * CYCLE_AMPLITUDE;
    latent[i * L + types + 1] = Math.sin(a) * CYCLE_AMPLITUDE;
  }
  return finish(
    "cellCycle",
    n,
    {
      latent,
      nLatent: L,
      latentNames: [...Array.from({ length: types }, (_, c) => `type ${c + 1}`), "cycle cos", "cycle sin"],
      label,
      labelNames: Array.from({ length: types }, (_, c) => `type ${c + 1}`),
      truth,
      truthName: "cycle phase",
      truthCyclic: true,
    },
    0.35,
    "One ring per type. Switch the two cycle programmes off and the rings collapse to dots.",
    rnd,
  );
}

/** One dominant population plus a handful of rare cells.
 *
 *  The case that decides whether an analysis is any use: a rare type is the thing you are
 *  usually looking for, and it is also the thing a neighbourhood graph is most likely to
 *  swallow — with `n_neighbors` above the size of the rare group, its members' neighbours
 *  are mostly majority cells and it dissolves into the bulk. */
function rarePopulation(n: number, seed: number, fraction = 0.01): Manifold {
  const rnd = mulberry32(seed);
  const L = 3;
  const latent = new Float32Array(n * L);
  const label = new Uint8Array(n);
  const rare = Math.max(3, Math.round(n * fraction));
  for (let i = 0; i < n; i++) {
    const isRare = i < rare;
    label[i] = isRare ? 1 : 0;
    if (isRare) {
      latent[i * L] = 1.6;
      latent[i * L + 1] = 1.6;
      latent[i * L + 2] = 0.3 * gaussian(rnd);
    } else {
      // A mildly elongated bulk, so the rare group is not the only thing with a shape.
      latent[i * L] = gaussian(rnd) * 0.9;
      latent[i * L + 1] = gaussian(rnd) * 0.45;
      latent[i * L + 2] = gaussian(rnd) * 0.45;
    }
  }
  return finish(
    "rarePopulation",
    n,
    {
      latent,
      nLatent: L,
      latentNames: ["bulk axis 1", "bulk axis 2", "rare marker"],
      label,
      labelNames: ["bulk", "rare"],
    },
    0.35,
    `A small island of about ${Math.round(fraction * 100)}% of the cells, held apart from the bulk. Raise n_neighbors past its size and watch it dissolve.`,
    rnd,
  );
}

/** Uniform noise — the null control, and the calibration for every other entry.
 *
 *  This exists to answer "how much of what I am looking at did the algorithm invent?" by
 *  showing what the algorithm does when there is nothing to find. The measured answer is
 *  a **featureless disc**: no islands, no clumps, no apparent cell types. That is worth
 *  stating because the folklore says the opposite, and this generator was written expecting
 *  an archipelago — uniform latents at 6, 30 and 80 dimensions were all tried and every one
 *  came out as a plain disc.
 *
 *  So the lesson it teaches is the reverse of the intended one, and more useful: on THIS
 *  algorithm, at these settings, visible islands are not free. The clumping UMAP is
 *  criticised for is real, but it is what happens to CLEAN LOW-DIMENSIONAL structure (see
 *  the module header), not to noise. */
function uniformNoise(n: number, seed: number, dims = 6): Manifold {
  const rnd = mulberry32(seed);
  const latent = new Float32Array(n * dims);
  for (let t = 0; t < latent.length; t++) latent[t] = rnd() * 2 - 1;
  return finish(
    "uniformNoise",
    n,
    {
      latent,
      nLatent: dims,
      latentNames: Array.from({ length: dims }, (_, c) => `noise ${c + 1}`),
      label: new Uint8Array(n),
      labelNames: ["(no structure)"],
    },
    0.05,
    "A plain, featureless disc — no islands. That is the measured answer, and it is the yardstick: structure you see elsewhere is not something UMAP hands out for free.",
    rnd,
  );
}

// --- registry ------------------------------------------------------------------------

export interface ManifoldSpec {
  readonly key: string;
  readonly label: string;
  readonly describe: string;
  readonly make: (n: number, seed: number) => Manifold;
}

/** Ordered so the list reads as an argument: the familiar case, then the shapes that
 *  break it, then the null. */
export const MANIFOLDS: ManifoldSpec[] = [
  { key: "blobs", label: "Separated types", describe: "Isotropic clusters — the easy case.", make: (n, s) => blobs(n, s) },
  {
    key: "branching",
    label: "Branching trajectory",
    describe: "A trunk splitting into arms — the shape single-cell data usually has.",
    make: (n, s) => branching(n, s),
  },
  {
    key: "swissRoll",
    label: "Swiss roll",
    describe: "A flat sheet rolled up in 3-D. Should come out as a rectangle.",
    make: (n, s) => swissRoll(n, s),
  },
  {
    key: "linkedRings",
    label: "Linked rings",
    describe: "Two interlocked loops — 2-D cannot hold both, so one must tear.",
    make: (n, s) => linkedRings(n, s),
  },
  {
    key: "sphere",
    label: "Hollow sphere",
    describe: "A surface with no flat form; the cut lands somewhere arbitrary.",
    make: (n, s) => sphere(n, s),
  },
  {
    key: "hierarchy",
    label: "Clusters of clusters",
    describe: "Structure at two scales — n_neighbors picks which one you see.",
    make: (n, s) => hierarchy(n, s),
  },
  { key: "cellCycle", label: "Cycling types", describe: "A ring per type: a continuum inside clusters.", make: (n, s) => cellCycle(n, s) },
  {
    key: "rarePopulation",
    label: "Rare population",
    describe: "1% of cells apart from the bulk — the group most easily lost.",
    make: (n, s) => rarePopulation(n, s),
  },
  {
    key: "uniformNoise",
    label: "Noise (null control)",
    describe: "No structure at all — the yardstick for everything else.",
    make: (n, s) => uniformNoise(n, s),
  },
];

export function makeManifold(key: string, n: number, seed = 11): Manifold {
  const spec = MANIFOLDS.find((m) => m.key === key);
  if (!spec) throw new Error(`syntheticManifolds: no generator '${key}'`);
  return spec.make(n, seed);
}

// --- expression ----------------------------------------------------------------------

/** A group of gene columns that move together — one latent axis's block. */
export interface FeatureBlock {
  readonly name: string;
  readonly columns: number[];
}

export interface ExpressedManifold {
  /** Row-major `[n, dim]`. */
  readonly values: Float32Array;
  readonly n: number;
  readonly dim: number;
  readonly blocks: FeatureBlock[];
  /** The noise actually applied, as a fraction of the RMS pairwise distance. */
  readonly noise: number;
  /** The gene-space length `noise` was multiplied by — reported so a caller can say what
   *  the figure means rather than quoting a bare number. */
  readonly reference: number;
}

export interface ExpressOptions {
  /** Genes per latent axis. More genes average the noise down without changing geometry. */
  readonly genesPerAxis?: number;
  /** Fraction of the RMS pairwise distance added as independent per-gene noise. Defaults
   *  to the generator's own figure, which is the one tuned for that shape. */
  readonly noise?: number;
  readonly seed?: number;
}

/**
 * Turn a manifold into a gene-expression matrix.
 *
 * Each latent axis gets a block of genes with random positive loadings and a baseline, so
 * a column reads like an expression level rather than a signed coordinate. Loadings are
 * positive on purpose: a gene that is *anti*-correlated with a programme is a perfectly
 * real thing, but mixing signs here would let two blocks cancel and make the programme
 * toggles behave in ways that have nothing to do with the manifold.
 *
 * Noise is calibrated against the noiseless GENE matrix's own RMS pairwise distance — not
 * the latent's — so the requested fraction is honoured whatever the loadings happened to
 * draw.
 */
export function expressManifold(m: Manifold, opts: ExpressOptions = {}): ExpressedManifold {
  const genes = Math.max(1, opts.genesPerAxis ?? 6);
  const noise = opts.noise ?? m.noise;
  const rnd = mulberry32((opts.seed ?? 1) ^ 0x5bf03635);

  const dim = m.nLatent * genes;
  const values = new Float32Array(m.n * dim);
  const blocks: FeatureBlock[] = [];

  for (let l = 0; l < m.nLatent; l++) {
    const columns: number[] = [];
    for (let g = 0; g < genes; g++) {
      const col = l * genes + g;
      columns.push(col);
      const loading = 0.6 + rnd() * 0.8;
      const baseline = rnd() * 0.4;
      for (let i = 0; i < m.n; i++) values[i * dim + col] = baseline + loading * m.latent[i * m.nLatent + l]!;
    }
    blocks.push({ name: m.latentNames[l] ?? `axis ${l + 1}`, columns });
  }

  // A pair's noise displacement has variance 2*sigma^2 per gene over `dim` genes, hence
  // the sqrt(2*dim): it converts the requested displacement into a per-gene sigma.
  const reference = rmsPairwise(values, m.n, dim, rnd);
  const sigma = (noise * reference) / Math.sqrt(2 * dim);
  if (sigma > 0) for (let t = 0; t < values.length; t++) values[t] = values[t]! + sigma * gaussian(rnd);

  return { values, n: m.n, dim, blocks, noise, reference };
}
