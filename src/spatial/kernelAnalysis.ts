// Does smoothing the mark kernel actually buy anything? — a measurement, not an opinion.
//
// The paper's top-hat is the least smooth member of the family in `kernels.ts`. Replacing it is
// only worth doing if the replacement is measurably better at the job the mark is for: separating
// A cells that really do sit in B-rich neighbourhoods from those that do not. So this module
// builds a scene with KNOWN ground truth and scores each kernel on three axes:
//
//   discrimination  AUC of the mark at truly-associated A cells vs the rest. 1 = perfect
//                   separation, 0.5 = useless. Computed with TIE-AVERAGED ranks, which matters
//                   more than it sounds: the top-hat's mark is a count, so it is heavily
//                   discretised, and ties are a real loss of resolution that a naive AUC hides.
//   radius stab.    ‖Δm‖/‖m‖ when the support radius is nudged 2%. The top-hat moves a whole
//                   count when a B cell crosses |u| = r; smooth kernels move by ~0.
//   jitter stab.    ‖Δm‖/‖m‖ when every cell is jittered by a small fraction of the radius —
//                   i.e. robustness to segmentation/centroid error, which is a real property of
//                   the data, not a numerical artefact.
//
// Kernels are always compared at MATCHED SECOND MOMENT (`equivalentRadius`), never at equal r:
// μ₂ = r²/(n+2) shrinks with order, so equal-r comparison would confound "smoother" with "more
// local" and hand the smooth kernels an unearned win on stability.
//
// Everything here is CPU and deterministic (seeded PRNG), so the numbers are reproducible and can
// be asserted in tests rather than eyeballed.

import { equivalentRadius, KERNELS, type KernelSpec, kernelLabel, moment2, roughness } from "./kernels";
import type { CellCloud } from "./tcm";
import { kernelMarks } from "./tcmKernel";

/** mulberry32 — lattice-free, unlike a plain LCG (whose consecutive values fall on 2-D planes and
 *  would fabricate exactly the spatial structure we are trying to measure). */
export function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface AssociationScene {
  readonly A: CellCloud;
  readonly B: CellCloud;
  /** Index-aligned to A: was this cell PLACED in a B patch? The ground truth. */
  readonly associated: boolean[];
  readonly bbox: readonly [number, number, number, number];
  /** Patch radius — the spatial scale the mark has to detect. */
  readonly patchRadius: number;
}

export interface SceneOptions {
  seed?: number;
  /** Side of the square ROI. */
  extent?: number;
  nPatches?: number;
  patchRadius?: number;
  /** B cells per patch, plus a uniform background of B. */
  bPerPatch?: number;
  bBackground?: number;
  /** A cells placed inside patches, and placed uniformly. */
  aAssociated?: number;
  aBackground?: number;
}

/** A scene with known association: B forms dense patches on a uniform background, and a labelled
 *  subset of A is placed inside those patches while the rest is scattered uniformly.
 *
 *  Note the deliberate imperfection: a "background" A cell can land in a patch by chance, so no
 *  estimator can reach AUC = 1. That is the point — the scores are comparable to each other, not
 *  absolute. */
export function makeAssociationScene(opts: SceneOptions = {}): AssociationScene {
  const rnd = mulberry32(opts.seed ?? 0x51ee7);
  const extent = opts.extent ?? 1000;
  const nPatches = opts.nPatches ?? 6;
  const patchRadius = opts.patchRadius ?? 60;
  const bPerPatch = opts.bPerPatch ?? 90;
  const bBackground = opts.bBackground ?? 400;
  const aAssociated = opts.aAssociated ?? 150;
  const aBackground = opts.aBackground ?? 350;

  // Patch centres, kept clear of the ROI boundary so edge effects do not confound the comparison.
  const margin = patchRadius * 1.5;
  const centres = Array.from({ length: nPatches }, () => [margin + rnd() * (extent - 2 * margin), margin + rnd() * (extent - 2 * margin)]);
  /** Uniform point in the disk around a patch centre (sqrt for area-uniformity). */
  const inPatch = (): [number, number] => {
    const c = centres[Math.floor(rnd() * centres.length)]!;
    const th = rnd() * 2 * Math.PI;
    const rr = patchRadius * Math.sqrt(rnd());
    return [c[0]! + rr * Math.cos(th), c[1]! + rr * Math.sin(th)];
  };

  const bx: number[] = [];
  const by: number[] = [];
  for (let p = 0; p < nPatches; p++)
    for (let i = 0; i < bPerPatch; i++) {
      const c = centres[p]!;
      const th = rnd() * 2 * Math.PI;
      const rr = patchRadius * Math.sqrt(rnd());
      bx.push(c[0]! + rr * Math.cos(th));
      by.push(c[1]! + rr * Math.sin(th));
    }
  for (let i = 0; i < bBackground; i++) {
    bx.push(rnd() * extent);
    by.push(rnd() * extent);
  }

  const ax: number[] = [];
  const ay: number[] = [];
  const associated: boolean[] = [];
  for (let i = 0; i < aAssociated; i++) {
    const [x, y] = inPatch();
    ax.push(x);
    ay.push(y);
    associated.push(true);
  }
  for (let i = 0; i < aBackground; i++) {
    ax.push(rnd() * extent);
    ay.push(rnd() * extent);
    associated.push(false);
  }

  return {
    A: { xs: ax, ys: ay },
    B: { xs: bx, ys: by },
    associated,
    bbox: [0, 0, extent, extent],
    patchRadius,
  };
}

/** Area under the ROC curve via the Mann–Whitney statistic, with TIE-AVERAGED ranks.
 *
 *  Tie handling is the substantive detail: the top-hat mark takes only the values
 *  {0, 1, 2, …}/(ρ_B·πr²), so large groups of cells score identically and cannot be ordered. Tied
 *  pairs count as half — which is exactly the credit an estimator that cannot separate them
 *  deserves. */
export function auc(positive: readonly number[], negative: readonly number[]): number {
  const n1 = positive.length;
  const n2 = negative.length;
  if (n1 === 0 || n2 === 0) return 0.5;
  const all = [...positive.map((v) => ({ v, pos: true })), ...negative.map((v) => ({ v, pos: false }))].sort((p, q) => p.v - q.v);
  let rankSum = 0;
  let i = 0;
  while (i < all.length) {
    let j = i;
    while (j + 1 < all.length && all[j + 1]!.v === all[i]!.v) j++;
    const avgRank = (i + j) / 2 + 1; // 1-based, averaged across the tie group
    for (let k = i; k <= j; k++) if (all[k]!.pos) rankSum += avgRank;
    i = j + 1;
  }
  return (rankSum - (n1 * (n1 + 1)) / 2) / (n1 * n2);
}

/** Fraction of A cells whose mark is tied with at least one other — the top-hat's hidden cost. */
function tiedFraction(values: ArrayLike<number>): number {
  const counts = new Map<number, number>();
  for (let i = 0; i < values.length; i++) counts.set(values[i]!, (counts.get(values[i]!) ?? 0) + 1);
  let tied = 0;
  for (const c of counts.values()) if (c > 1) tied += c;
  return tied / Math.max(values.length, 1);
}

const relRms = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    num += (b[i]! - a[i]!) ** 2;
    den += a[i]! ** 2;
  }
  return Math.sqrt(num / Math.max(den, 1e-30));
};

function jitter(c: CellCloud, scale: number, rnd: () => number): CellCloud {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < c.xs.length; i++) {
    // Box–Muller, so the perturbation is isotropic Gaussian rather than boxy.
    const u = Math.max(rnd(), 1e-12);
    const th = rnd() * 2 * Math.PI;
    const g = Math.sqrt(-2 * Math.log(u));
    xs.push(c.xs[i]! + scale * g * Math.cos(th));
    ys.push(c.ys[i]! + scale * g * Math.sin(th));
  }
  return { xs, ys };
}

export interface KernelScore {
  readonly kernel: KernelSpec;
  readonly label: string;
  /** Matched-μ₂ radius actually used. */
  readonly radius: number;
  /** AUC separating associated from background A cells by their mark. Higher is better. */
  readonly auc: number;
  /** Fraction of marks tied with another — resolution lost to discretisation. Lower is better. */
  readonly tied: number;
  /** ‖Δm‖/‖m‖ for a +2% radius change. Lower is better. */
  readonly radiusSensitivity: number;
  /** ‖Δm‖/‖m‖ for an isotropic jitter of 5% of the radius. Lower is better. */
  readonly jitterSensitivity: number;
  /** μ₂ — equal across kernels by construction; reported so the matching is auditable. */
  readonly moment2: number;
  /** R(K) = ∫K². At matched scale this is the variance term of the AMISE.
   *
   *  Compare kernels by `roughness · moment2`, which is dimensionless and scale-invariant —
   *  R(K) alone has units of 1/area, and R(K)·r² uses each kernel's OWN radius, which un-does the
   *  scale matching and reverses the ordering. */
  readonly roughness: number;
}

export interface ScoreOptions {
  /** Top-hat radius the others are matched to. Default: the scene's patch radius. */
  baseRadius?: number;
  seed?: number;
  /** Jitter magnitude as a fraction of the radius. Default 0.05. */
  jitterFraction?: number;
  /** Radius perturbation for the stability probe. Default 0.02. */
  radiusFraction?: number;
}

/** Score every kernel on the same scene, at matched spatial scale. */
export function scoreKernels(scene: AssociationScene, opts: ScoreOptions = {}): KernelScore[] {
  const baseRadius = opts.baseRadius ?? scene.patchRadius;
  const jitterFraction = opts.jitterFraction ?? 0.05;
  const radiusFraction = opts.radiusFraction ?? 0.02;
  // One jittered scene, shared by every kernel: the comparison must not depend on which random
  // perturbation each kernel happened to draw.
  const rnd = mulberry32(opts.seed ?? 0xa11ce);
  const jitteredB = jitter(scene.B, jitterFraction * baseRadius, rnd);
  const jitteredA = jitter(scene.A, jitterFraction * baseRadius, rnd);

  return KERNELS.map((kernel) => {
    const radius = equivalentRadius(kernel, baseRadius);
    const p = { width: 1, height: 1, bbox: scene.bbox, radius, sigma: 1, alpha: 5, kernel };
    const m = kernelMarks(scene.A, scene.B, p);
    const pos: number[] = [];
    const neg: number[] = [];
    for (let i = 0; i < m.length; i++) (scene.associated[i] ? pos : neg).push(m[i]!);
    const wider = kernelMarks(scene.A, scene.B, { ...p, radius: radius * (1 + radiusFraction) });
    const jittered = kernelMarks(jitteredA, jitteredB, p);
    return {
      kernel,
      label: kernelLabel(kernel),
      radius,
      auc: auc(pos, neg),
      tied: tiedFraction(m),
      radiusSensitivity: relRms(m, wider),
      jitterSensitivity: relRms(m, jittered),
      moment2: moment2(kernel, radius),
      roughness: roughness(kernel, radius),
    };
  });
}

/** A fixed-width text table of the scores — for the demo HUD and for eyeballing in a REPL. */
export function formatScores(scores: readonly KernelScore[]): string {
  const head = ["kernel", "radius", "AUC", "tied", "d/dr", "jitter", "R·μ₂"];
  const rows = scores.map((s) => [
    s.label,
    s.radius.toFixed(1),
    s.auc.toFixed(4),
    `${(100 * s.tied).toFixed(1)}%`,
    `${(100 * s.radiusSensitivity).toFixed(2)}%`,
    `${(100 * s.jitterSensitivity).toFixed(2)}%`,
    (s.roughness * s.moment2).toFixed(4),
  ]);
  const widths = head.map((h, i) => Math.max(h.length, ...rows.map((r) => r[i]!.length)));
  const line = (cells: string[]) => cells.map((c, i) => c.padEnd(widths[i]!)).join("  ");
  return [line(head), line(widths.map((w) => "-".repeat(w))), ...rows.map(line)].join("\n");
}
