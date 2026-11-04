// PROTOTYPE: the wand's settings (docs/wand-metric-and-regions.md §4): the metric the boundary field is
// measured in, and the rule its patch is bounded by, with presets that each set every fine-tune control.

import { useState } from "react";
import type { PatchRule } from "./wandPatch";

export interface WandSettings {
  /** Main modes (q): the rest are counted at the noise level. */
  q: number;
  /** Weight of what the main modes leave out. */
  w: number;
  /** σ, in kernel radii, of the blur the modes after q are differentiated through. */
  residualSmooth: number;
  /** Draw boundaries as thin lines (off: every place above θ). */
  lines: boolean;
  rule: PatchRule;
  /** σ, in kernel radii, of the smoothing on d before its Laplacian (Marr–Hildreth's σ, on top of r). */
  dSmooth: number;
  /** Also stop where d exceeds this; 0 = off. */
  tol: number;
  /** Opening radius, in kernel radii: parts of the cell narrower than this are dropped; 0 = off. */
  open: number;
  /** The opening graded by neck width (a lobe fades as its neck narrows), or the binary opening.
   *  Off by default: on real data the fill behind a small patch has often percolated over most of the
   *  map, and grading shows that cluster faintly, which swings more than the binary pops (E8). */
  graded: boolean;
  /** Judge the patch at a second, broader radius too: the broad cell picks it, this one places its edge. */
  twoScale: boolean;
  /** The broad radius, as a multiple of r. */
  broad: number;
  /** How far, in r, the edge may sit outside the broad cell. */
  slack: number;
  /** Boundary strength θ; undefined = follow the window's 90th percentile. */
  theta: number | undefined;
}

/** The settings a preset fixes: all but θ, which follows the window, and graded necks, an experiment. */
type PresetValues = Omit<WandSettings, "theta" | "graded">;
export type WandPresetName = "patch" | "tissue" | "fine";
const ONE_SCALE = { twoScale: false, broad: 3, slack: 1 } as const;
export const WAND_PRESETS: Record<WandPresetName, { label: string; hint: string; values: PresetValues }> = {
  patch: {
    label: "The patch the click is in (recommended)",
    hint: "a compact region of the click's mixture; thin branches dropped",
    values: { q: 3, w: 0, residualSmooth: 0, lines: true, rule: "laplacian", dSmooth: 1, tol: 0, open: 0.75, ...ONE_SCALE },
  },
  tissue: {
    label: "Follow the tissue",
    hint: "everywhere connected to the click with the same mixture, branches and all",
    values: { q: 3, w: 0, residualSmooth: 0, lines: true, rule: "laplacian", dSmooth: 1, tol: 0, open: 0, ...ONE_SCALE },
  },
  fine: {
    label: "Fine detail",
    hint: "edges judged at half the radius: smaller patches, closer to the boundary",
    values: { q: 3, w: 0, residualSmooth: 0, lines: true, rule: "laplacian", dSmooth: 0.5, tol: 0, open: 0.5, ...ONE_SCALE },
  },
};

const PRESET_NAMES = Object.keys(WAND_PRESETS) as WandPresetName[];

/** The wand's settings, starting from the recommended preset. */
export function useWandSettings() {
  const [settings, setSettings] = useState<WandSettings>({ ...WAND_PRESETS.patch.values, graded: false, theta: undefined });
  const set = <K extends keyof WandSettings>(key: K, value: WandSettings[K]) => setSettings((s) => ({ ...s, [key]: value }));
  /** The preset every fine-tune control matches, if any; θ is left out, as it follows the window. */
  const preset = PRESET_NAMES.find((name) => {
    const v = WAND_PRESETS[name].values;
    return (Object.keys(v) as (keyof PresetValues)[]).every((k) => v[k] === settings[k]);
  });
  const applyPreset = (name: WandPresetName) => setSettings((s) => ({ ...WAND_PRESETS[name].values, graded: s.graded, theta: undefined }));
  return { settings, set, setSettings, preset, applyPreset };
}

export type WandSettingsState = ReturnType<typeof useWandSettings>;
