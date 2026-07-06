// Prebuilt example graphs the palette can load. The reaction–diffusion example is a
// feedback loop: a Gray–Scott seed feeds two delay nodes whose state drives the RD
// step, whose outputs feed back into the delays — so it runs over time (Play).
import type { Edge, Node } from "@xyflow/react";
import { registerExtraOps } from "./extraOps";
import { defaultParamsFor, getSpec } from "./specs";
import type { DefLibrary, SubgraphDef } from "./subgraphs";
import { defScopeId } from "./subgraphs";

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

const e = (id: string, s: string, sh: string, t: string, th: string): Edge => ({
  id,
  source: s,
  sourceHandle: sh,
  target: t,
  targetHandle: th,
});

// Gray–Scott reaction–diffusion as a feedback loop, with the (U,V) state carried as a
// single complex field (re = U, im = V — ADR-0004). Because it's one field, the loop is
// one seed → one feedback (delay) → one step → back, instead of the two parallel
// feedback nodes the scalar (U,V)-as-two-grids formulation needs. The feedback node
// propagates the complex element around the loop, so every edge stays complex. Preview
// the `state` output and switch the lane selector to `im` to watch V (the classic view).
export function reactionDiffusionExample(): Example {
  const nodes: Node[] = [
    node("seed", "grayScottSeedComplex", 20, 200),
    node("fb", "feedback", 240, 200),
    node("rd", "reactionDiffusionComplex", 460, 200),
  ];
  const edges: Edge[] = [
    e("e-si", "seed", "state", "fb", "init"),
    e("e-in", "fb", "state", "rd", "state"),
    e("e-back", "rd", "state", "fb", "next"),
  ];
  return { label: "Reaction–diffusion (complex)", nodes, edges, sink: { node: "rd", port: "state" } };
}

// Ceilidh dancer — the DANCERL force field *composed from building blocks* rather than
// one monolithic op. A swarm body (rigid-body agents) feeds a delay node; bodyTap exposes
// pos/vel; several force ops (containment, cohesion, separation, orbit) each emit a
// per-agent acceleration; those are summed (addFields) and fed to the rigid-body
// integrator, whose next body closes the loop. Each force node's params are breedable
// traits (src/evo). Preview the integrator's `swarm` output (a top-down scatter); the
// dedicated 3D view + the Ceilidh caller (figures, partner progression) come in the app.
export function ceilidhExample(): Example {
  const nodes: Node[] = [
    node("seed", "swarmSeed", 20, 320),
    node("fb", "feedback", 200, 320),
    node("tap", "bodyTap", 360, 320),
    // the caller's clock (frame counter)
    node("cstart", "clockStart", 20, 560),
    node("cfb", "feedback", 200, 560),
    node("clock", "clock", 360, 560),
    // forces
    node("constrain", "constrain", 560, 40),
    node("cohere", "cohere", 560, 150),
    node("separate", "separate", 560, 260),
    node("caller", "caller", 560, 380),
    node("partner", "partnerOrbit", 560, 520),
    // sum the forces
    node("sum1", "addFields", 780, 100),
    node("sum2", "addFields", 780, 210),
    node("sum3", "addFields", 780, 330),
    node("sum4", "addFields", 780, 450),
    node("integ", "integrate", 1000, 320),
  ];
  const edges: Edge[] = [
    e("e-seed", "seed", "body", "fb", "init"),
    e("e-tap", "fb", "state", "tap", "body"),
    // clock loop
    e("e-cs", "cstart", "t", "cfb", "init"),
    e("e-ck", "cfb", "state", "clock", "prev"),
    e("e-ckb", "clock", "t", "cfb", "next"),
    // force inputs
    e("e-cp", "tap", "pos", "constrain", "pos"),
    e("e-hp", "tap", "pos", "cohere", "pos"),
    e("e-sp", "tap", "pos", "separate", "pos"),
    e("e-lp", "tap", "pos", "caller", "pos"),
    e("e-lv", "tap", "vel", "caller", "vel"),
    e("e-lf", "clock", "t", "caller", "frame"),
    e("e-pp", "tap", "pos", "partner", "pos"),
    e("e-pv", "tap", "vel", "partner", "vel"),
    // sum: (((constrain + cohere) + separate) + caller) + partnerOrbit
    e("e-s1a", "constrain", "force", "sum1", "a"),
    e("e-s1b", "cohere", "force", "sum1", "b"),
    e("e-s2a", "sum1", "out", "sum2", "a"),
    e("e-s2b", "separate", "force", "sum2", "b"),
    e("e-s3a", "sum2", "out", "sum3", "a"),
    e("e-s3b", "caller", "force", "sum3", "b"),
    e("e-s4a", "sum3", "out", "sum4", "a"),
    e("e-s4b", "partner", "force", "sum4", "b"),
    // integrate + close the loop
    e("e-ib", "fb", "state", "integ", "body"),
    e("e-if", "sum4", "out", "integ", "force"),
    e("e-back", "integ", "body", "fb", "next"),
  ];
  return { label: "Ceilidh dancer (DANCERL)", nodes, edges, sink: { node: "integ", port: "swarm" } };
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
  // `nodes` is the fixed 5-element literal above, so indices 0–4 are statically present.
  // `?.` here would be a false guard (it short-circuits `.data` but we still assign `.params`,
  // so it throws just the same if missing) — assert instead, with the invariant stated once.
  // biome-ignore-start lint/style/noNonNullAssertion: fixed-length literal above; indices 0–4 provably in range
  (nodes[0]!.data as { params: Record<string, unknown> }).params = { width: 96, height: 96, kind: "radial", freq: 5, angle: 0, amp: 1 };
  (nodes[1]!.data as { params: Record<string, unknown> }).params = { width: 96, height: 96, kind: "white", scale: 12, seed: 7, amp: 0.4 };
  (nodes[2]!.data as { params: Record<string, unknown> }).params = { wa: 1, wb: 1 };
  (nodes[3]!.data as { params: Record<string, unknown> }).params = { kernel: "9/7", levels: 3 };
  (nodes[4]!.data as { params: Record<string, unknown> }).params = { thresh: 0.3, soft: true };
  // biome-ignore-end lint/style/noNonNullAssertion: end fixed-literal block
  const edges: Edge[] = [
    e("e-sm", "signal", "out", "mix", "a"),
    e("e-nm", "noise", "out", "mix", "b"),
    e("e-mf", "mix", "out", "fwd", "in"),
    e("e-fs", "fwd", "coeffs", "shrink", "coeffs"),
    e("e-si", "shrink", "out", "inv", "coeffs"),
  ];
  return { label: "Wavelet denoise", nodes, edges, sink: { node: "inv", port: "out" } };
}

export const EXAMPLES: Example[] = [ceilidhExample(), reactionDiffusionExample(), reusableSubgraphExample(), waveletDenoiseExample()];
