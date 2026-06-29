// Prebuilt example graphs the palette can load. The reaction–diffusion example is a
// feedback loop: a Gray–Scott seed feeds two delay nodes whose state drives the RD
// step, whose outputs feed back into the delays — so it runs over time (Play).
import type { Edge, Node } from "@xyflow/react";
import { defaultParamsFor, getSpec } from "./specs";
import { defScopeId } from "./subgraphs";
import type { DefLibrary, SubgraphDef } from "./subgraphs";

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

export const EXAMPLES: Example[] = [reactionDiffusionExample(), reusableSubgraphExample()];
