// The pure per-cell HsPf math — the "reaction" F and the LD derivation — as plain,
// deterministic TypeScript. This is the correctness anchor for the artefact (ADR-0011,
// decision 5): the fiddly parts (offspring-table indexing, the fitness blend, the LD
// denominator) are the class of thing that fails *silently* on the GPU, so they live here,
// CPU-testable in isolation, rather than only inside the fused kernel. The kernel re-derives
// exactly this arithmetic in `"use gpu"` TGSL.
//
// Genotype order throughout is [--, -+, +-, ++] over the two biallelic loci (HbS × the
// Pf-linked locus). See src/gpu/sim/hspf/CONTEXT.md.

/** A genotype vector [--, -+, +-, ++]; on a land cell it sums to 1 after normalisation. */
export type Vec4 = readonly [number, number, number, number];

const add4 = (a: Vec4, b: Vec4): Vec4 => [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
const mul4 = (a: Vec4, b: Vec4): Vec4 => [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]];
const scale4 = (a: Vec4, k: number): Vec4 => [a[0] * k, a[1] * k, a[2] * k, a[3] * k];
export const sum4 = (a: Vec4): number => a[0] + a[1] + a[2] + a[3];

/** The 16×4 offspring/Punnett table: for each ordered parent-genotype pair (row `g1*4 + g2`),
 *  the probabilities of transmitting each of the four genotypes. Every row sums to 1. This is
 *  the mechanism of two-bite recombination. */
export const OFFSPRING: readonly Vec4[] = [
  /* -- × -- */ [1, 0, 0, 0],
  /* -- × -+ */ [0.5, 0.5, 0, 0],
  /* -- × +- */ [0.5, 0, 0.5, 0],
  /* -- × ++ */ [0.25, 0.25, 0.25, 0.25],
  /* -+ × -- */ [0.5, 0.5, 0, 0],
  /* -+ × -+ */ [0, 1, 0, 0],
  /* -+ × +- */ [0.25, 0.25, 0.25, 0.25],
  /* -+ × ++ */ [0, 0.5, 0, 0.5],
  /* +- × -- */ [0.5, 0, 0.5, 0],
  /* +- × -+ */ [0.25, 0.25, 0.25, 0.25],
  /* +- × +- */ [0, 0, 1, 0],
  /* +- × ++ */ [0, 0, 0.5, 0.5],
  /* ++ × -- */ [0.25, 0.25, 0.25, 0.25],
  /* ++ × -+ */ [0, 0.5, 0, 0.5],
  /* ++ × +- */ [0, 0, 0.5, 0.5],
  /* ++ × ++ */ [0, 0, 0, 1],
];

/** The 2×4 fitness matrix: selection coefficients per genotype for the background (`A`) and
 *  sickle (`S`) backgrounds. The original's default. */
export interface FitnessMatrix {
  /** Background (non-sickle) fitness per genotype. */
  A: Vec4;
  /** Sickle-background fitness per genotype. */
  S: Vec4;
}

export const DEFAULT_FITNESS: FitnessMatrix = {
  //         --          -+                +-              ++
  A: [1.0, Math.sqrt(0.9), Math.sqrt(0.9), 0.8],
  S: [0.01, Math.sqrt(0.1), Math.sqrt(0.1), 0.8],
};

/** Selection weights from the local HbS frequency `fs`: `s` is the sickle-allele exposure
 *  (`fs² + 2·fs·(1−fs)`, i.e. 1 − (1−fs)²) and `a = 1 − s`. Used to blend the fitness rows. */
export function selectionWeights(fs: number): { a: number; s: number } {
  const s = fs * fs + 2 * fs * (1 - fs);
  return { a: 1 - s, s };
}

/** Blend the fitness rows by the local selection weights into one per-genotype fitness vector
 *  `a·A + s·S`. */
export function blendFitness(fit: FitnessMatrix, a: number, s: number): Vec4 {
  return add4(scale4(fit.A, a), scale4(fit.S, s));
}

/** Two-bite recombination: given a bite location's genotype vector `pf`, the distribution of
 *  transmitted genotypes when two independent bites from that location recombine —
 *  `Σ_{g1,g2} pf[g1]·pf[g2]·offspring[g1,g2]`. Sums to `(Σ pf)²` (⇒ 1 when `pf` is normalised). */
export function recombine(pf: Vec4): Vec4 {
  let out: Vec4 = [0, 0, 0, 0];
  for (let g1 = 0; g1 < 4; g1++) {
    for (let g2 = 0; g2 < 4; g2++) {
      const row = OFFSPRING[g1 * 4 + g2];
      if (!row) continue; // checked access, no non-null assertion
      out = add4(out, scale4(row, (pf[g1] ?? 0) * (pf[g2] ?? 0)));
    }
  }
  return out;
}

/** The reaction `F` for one bite location: the fitness-weighted mix of single-bite (linear in
 *  `pf`) and two-bite (bilinear, recombining) transmission. `twoBiteRate ∈ [0,1]`. The gathered,
 *  normalised sum of `weight·F` over the neighbourhood is the cell's next genotype vector. */
export function reactBite(pf: Vec4, fit: Vec4, twoBiteRate: number): Vec4 {
  const single = mul4(pf, fit);
  const two = mul4(recombine(pf), fit);
  return add4(scale4(single, 1 - twoBiteRate), scale4(two, twoBiteRate));
}

/** The normalised gather for one cell: `Σ weight·F(pf) / Σ weight·sum(F(pf))` over the
 *  neighbourhood's bite samples. This is the single-cell reference the fused kernel mirrors —
 *  the reaction + normalisation, not a whole-field golden (ADR-0011, decision 5). Bites whose
 *  target or source is off-domain are the caller's responsibility to omit. */
export function gatherCell(bites: ReadonlyArray<{ pf: Vec4; weight: number }>, fit: Vec4, twoBiteRate: number): Vec4 {
  let acc: Vec4 = [0, 0, 0, 0];
  let denom = 0;
  for (const { pf, weight } of bites) {
    const f = reactBite(pf, fit, twoBiteRate);
    acc = add4(acc, scale4(f, weight));
    denom += weight * sum4(f);
  }
  return denom === 0 ? [0, 0, 0, 0] : scale4(acc, 1 / denom);
}

/** Linkage disequilibrium `r ∈ [−1, 1]` from a normalised genotype vector. `f1·` and `f·1` are
 *  the marginal allele frequencies of the two loci; `r` is the normalised covariance, clamped. */
export function ld(value: Vec4): number {
  const f1_ = value[2] + value[3]; // marginal frequency of the first-locus `+`
  const f_1 = value[1] + value[3]; // marginal frequency of the second-locus `+`
  const cov = value[3] - f1_ * f_1;
  const denom = Math.sqrt(f1_ * (1 - f1_) * f_1 * (1 - f_1));
  if (denom === 0) return 0;
  const r = cov / denom;
  return r < -1 ? -1 : r > 1 ? 1 : r;
}
