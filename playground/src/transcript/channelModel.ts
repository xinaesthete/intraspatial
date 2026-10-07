// What the user has picked → the weight vectors `transcriptChannels` takes. A channel is one gene,
// or a gene set counted as one (its transcripts summed). Channels must be disjoint, so a gene that
// belongs to an active set cannot also be picked on its own.

import type { WeightedChannel } from "../../../src/spatial/transcriptChannels";
import type { PointsFeature } from "../datasource/pointsTileLoader";
import geneSetsJson from "./geneSets.json";

export interface GeneSet {
  readonly name: string;
  readonly genes: readonly string[];
}

/** Probe and codeword controls — no gene behind them, so they are the panel's own noise floor. */
export const NEGATIVE_CONTROLS = "Negative controls";
const CONTROL = /^(NegControl|BLANK|Unassigned|Deprecated|Intergenic)/i;

export const isControl = (name: string): boolean => CONTROL.test(name);

export const STARTER_SETS: readonly GeneSet[] = geneSetsJson.sets;

export interface Selection {
  readonly sets: readonly string[];
  readonly genes: readonly string[];
}

export interface BuiltChannels {
  readonly channels: WeightedChannel[];
  /** Set members the store's panel does not have, by set. */
  readonly missing: Readonly<Record<string, readonly string[]>>;
}

/** The gene names each active set covers, controls included. */
export function setMembers(name: string, features: readonly PointsFeature[], sets: readonly GeneSet[] = STARTER_SETS): string[] {
  if (name === NEGATIVE_CONTROLS) return features.filter((f) => isControl(f.name)).map((f) => f.name);
  return [...(sets.find((s) => s.name === name)?.genes ?? [])];
}

export function buildChannels(sel: Selection, features: readonly PointsFeature[], sets: readonly GeneSet[] = STARTER_SETS): BuiltChannels {
  const code = new Map(features.map((f) => [f.name, f.code]));
  const missing: Record<string, string[]> = {};
  const channels: WeightedChannel[] = [];
  for (const name of sel.sets) {
    const weights = new Map<number, number>();
    for (const g of setMembers(name, features, sets)) {
      const c = code.get(g);
      if (c !== undefined) weights.set(c, 1);
      else missing[name] = [...(missing[name] ?? []), g];
    }
    if (weights.size) channels.push({ label: name, weights });
  }
  for (const g of sel.genes) {
    const c = code.get(g);
    if (c !== undefined) channels.push({ label: g, weights: new Map([[c, 1]]) });
  }
  return { channels, missing };
}
