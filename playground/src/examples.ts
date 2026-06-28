// Prebuilt example graphs the palette can load. The reaction–diffusion example is a
// feedback loop: a Gray–Scott seed feeds two delay nodes whose state drives the RD
// step, whose outputs feed back into the delays — so it runs over time (Play).
import type { Edge, Node } from "@xyflow/react";
import { defaultParamsFor, getSpec } from "./specs";

function node(id: string, opName: string, x: number, y: number): Node {
  const data = { opName, params: defaultParamsFor(getSpec(opName)) };
  return { id, type: "op", position: { x, y }, data: data as unknown as Record<string, unknown> };
}

export interface Example {
  label: string;
  nodes: Node[];
  edges: Edge[];
  /** Node + output port to select and preview. */
  sink: { node: string; port?: string };
}

export function reactionDiffusionExample(): Example {
  const nodes: Node[] = [
    node("seed", "grayScottSeed", 20, 200),
    node("fbU", "feedback", 230, 120),
    node("fbV", "feedback", 230, 300),
    node("rd", "reactionDiffusionStep", 450, 200),
  ];
  const e = (id: string, s: string, sh: string, t: string, th: string): Edge => ({ id, source: s, sourceHandle: sh, target: t, targetHandle: th });
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

export const EXAMPLES: Example[] = [reactionDiffusionExample()];
