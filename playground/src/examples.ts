// Prebuilt example graphs the palette can load. The reaction–diffusion example is a
// feedback loop: a Gray–Scott seed feeds two delay nodes whose state drives the RD
// step, whose outputs feed back into the delays — so it runs over time (Play).
import type { Edge, Node } from "@xyflow/react";
import { defaultParamsFor, getSpec } from "./specs";
import { defScopeId } from "./subgraphs";
import type { DefLibrary, SubgraphDef } from "./subgraphs";
import { registerExtraOps } from "./extraOps";

// Some examples reference opt-in pack ops (fdwt/idwt/…). This module builds its example
// graphs at load via getSpec, which needs those ops registered — and it is imported by
// App before App's own registerExtraOps() runs, so register here too (idempotent).
registerExtraOps();

function node(id: string, opName: string, x: number, y: number): Node {
  const data = { opName, params: defaultParamsFor(getSpec(opName)) };
  return { id, type: "op", position: { x, y }, data: data as unknown as Record<string, unknown> };
}

export interface Example {
  label: string;
  nodes: Node[];
  edges: Edge[];
  /** Reusable named subgraph definitions referenced by `instance` nodes. */
  defs?: DefLibrary;
  /** Node + output port to select and preview. */
  sink: { node: string; port?: string };
}

const e = (id: string, s: string, sh: string, t: string, th: string): Edge => ({ id, source: s, sourceHandle: sh, target: t, targetHandle: th });

export function reactionDiffusionExample(): Example {
  const nodes: Node[] = [
    node("seed", "grayScottSeed", 20, 200),
    node("fbU", "feedback", 230, 120),
    node("fbV", "feedback", 230, 300),
    node("rd", "reactionDiffusionStep", 450, 200),
  ];
  const edges: Edge[] = [
    e("e-su", "seed", "u", "fbU", "init"),
    e("e-sv", "seed", "v", "fbV", "init"),
    e("e-uin", "fbU", "state", "rd", "u"),
    e("e-vin", "fbV", "state", "rd", "v"),
    e("e-uback", "rd", "u", "fbU", "next"),
    e("e-vback", "rd", "v", "fbV", "next"),
  ];
  return { label: "Reaction–diffusion", nodes, edges, sink: { node: "rd", port: "v" } };
}

// A reusable "Hotspots" subgraph (points → KDE density → Getis-Ord Gi*), instantiated
// twice over two different blob sources and combined — the same definition reused, so
// editing it (e.g. the KDE bandwidth) updates both instances at once.
export function reusableSubgraphExample(): Example {
  const def = "Hotspots";
  const scope = defScopeId(def);
  const io = (id: string, ioKind: "input" | "output", label: string, x: number, y: number): Node => ({
    id,
    type: ioKind,
    position: { x, y },
    data: { group: scope, label, io: ioKind } as unknown as Record<string, unknown>,
  });
  const member = (id: string, opName: string, x: number, y: number): Node => {
    const n = node(id, opName, x, y);
    (n.data as { group?: string }).group = scope;
    return n;
  };

  const hotspots: SubgraphDef = {
    name: def,
    label: def,
    nodes: [
      io("hs~in", "input", "points", 20, 140),
      member("hs_kde", "splatDensity", 220, 120),
      member("hs_gi", "getisOrd", 440, 120),
      io("hs~out", "output", "hotspots", 660, 140),
    ],
    edges: [
      e("hs-e1", "hs~in", "out", "hs_kde", "points"),
      e("hs-e2", "hs_kde", "density", "hs_gi", "grid"),
      e("hs-e3", "hs_gi", "z", "hs~out", "in"),
    ],
  };

  const inst = (id: string, x: number, y: number): Node => ({
    id,
    type: "instance",
    position: { x, y },
    data: { def, label: def, group: "__root__" } as unknown as Record<string, unknown>,
  });

  const nodes: Node[] = [
    node("blobsA", "blobPoints", 20, 80),
    node("blobsB", "blobPoints", 20, 320),
    inst("hsA", 280, 80),
    inst("hsB", 280, 320),
    node("sum", "addGrids", 560, 200),
  ];
  const edges: Edge[] = [
    e("ex-a", "blobsA", "points", "hsA", "hs~in"),
    e("ex-b", "blobsB", "points", "hsB", "hs~in"),
    e("ex-sa", "hsA", "hs~out", "sum", "a"),
    e("ex-sb", "hsB", "hs~out", "sum", "b"),
  ];

  return { label: "Reusable subgraph (Hotspots ×2)", nodes, edges, defs: { [def]: hotspots }, sink: { node: "sum", port: "out" } };
}

// Wavelet denoise: a smooth signal + white noise → forward DWT → shrink detail
// coefficients → inverse DWT. With a signal present, thresholding has something
// meaningful to recover (denoised RMSE drops below the noisy input).
//
// Two deliberate parameter choices, both to avoid blocky LL-only reconstructions:
//   - 9/7 kernel, not 5/3. The 9/7's K scaling makes it near-orthonormal, so a single
//     threshold shrinks every subband on a consistent scale. The 5/3 reversible
//     transform is NOT normalised (coarse subbands carry far larger coefficients), so a
//     uniform threshold over-shrinks some bands and a denoise actually *raises* error.
//   - thresh small (detail coeffs of an O(1) field are O(1)). A threshold larger than
//     the largest detail coefficient zeros ALL detail, leaving only the coarse LL band —
//     reconstructed alone that gives the grid artefacts at the LL-cell scale.
// idwt and thresholdDetail carry no kernel/levels params — they read the wavelet
// contract from the field produced by fdwt (ADR-0006).
export function waveletDenoiseExample(): Example {
  const nodes: Node[] = [
    node("signal", "waveGrid", 20, 120),
    node("noise", "noiseGrid", 20, 320),
    node("mix", "addGrids", 250, 220),
    node("fwd", "fdwt", 460, 220),
    node("shrink", "thresholdDetail", 670, 220),
    node("inv", "idwt", 880, 220),
  ];
  (nodes[0]!.data as { params: Record<string, unknown> }).params = { width: 96, height: 96, kind: "radial", freq: 5, angle: 0, amp: 1 };
  (nodes[1]!.data as { params: Record<string, unknown> }).params = { width: 96, height: 96, kind: "white", scale: 12, seed: 7, amp: 0.4 };
  (nodes[2]!.data as { params: Record<string, unknown> }).params = { wa: 1, wb: 1 };
  (nodes[3]!.data as { params: Record<string, unknown> }).params = { kernel: "9/7", levels: 3 };
  (nodes[4]!.data as { params: Record<string, unknown> }).params = { thresh: 0.3, soft: true };
  const edges: Edge[] = [
    e("e-sm", "signal", "out", "mix", "a"),
    e("e-nm", "noise", "out", "mix", "b"),
    e("e-mf", "mix", "out", "fwd", "in"),
    e("e-fs", "fwd", "coeffs", "shrink", "coeffs"),
    e("e-si", "shrink", "out", "inv", "coeffs"),
  ];
  return { label: "Wavelet denoise", nodes, edges, sink: { node: "inv", port: "out" } };
}

export const EXAMPLES: Example[] = [reactionDiffusionExample(), reusableSubgraphExample(), waveletDenoiseExample()];
