// Turn the React Flow canvas (nodes + edges) into the runtime's Graph IR, then pull
// or advance a chosen output. The canvas state is the explicit graph; this is the
// adapter that hands it to the executor.
import type { Edge, Node } from "@xyflow/react";
import { Graph, advance, createSimState, pull } from "../../src/gpu/graph";
import type { FeedbackHandle, FieldValue, GpuField, GraphMemo, SimState } from "../../src/gpu/graph";
import { getSource, isSource } from "./sources";
import { getSpec } from "./specs";
import { browserBackend } from "./backend.browser";
import { isGroupNode, isPortStub } from "./grouping";

export interface NodeData {
  opName: string;
  params: Record<string, unknown>;
  [key: string]: unknown;
}

export interface BuildResult {
  graph: Graph;
  produced: Map<string, Record<string, GpuField>>;
}

const FEEDBACK = "feedback";

export function graphHasFeedback(nodes: Node[]): boolean {
  return nodes.some((n) => (n.data as unknown as NodeData).opName === FEEDBACK);
}

/** Build the Graph IR; `produced` maps each canvas node id to its output handles.
 *  Feedback nodes are built from their `init` only; the `next` (loop-closing) edge is
 *  wired in a second pass once its producer exists. */
export function buildGraph(allNodes: Node[], allEdges: Edge[]): BuildResult {
  // Group nodes (and any port stubs) are a UI projection — the real flat graph is
  // what executes. Membership is just a `data.group` tag and edges are never
  // rewritten by grouping, so we simply drop group/stub nodes here.
  const nodes = allNodes.filter((n) => !isGroupNode(n) && !isPortStub(n));
  const edges = allEdges;

  const graph = new Graph();
  const produced = new Map<string, Record<string, GpuField>>();
  const feedbacks = new Map<string, FeedbackHandle>();
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const visiting = new Set<string>();
  const done = new Set<string>();

  const edgeTo = (id: string, port: string) => edges.find((e) => e.target === id && e.targetHandle === port);
  const handleOf = (e: Edge) => produced.get(e.source)?.[e.sourceHandle ?? ""];

  const build = (id: string) => {
    const node = byId.get(id);
    if (!node) throw new Error(`buildGraph: no node ${id}`);
    const data = node.data as unknown as NodeData;
    const params = data.params ?? {};

    if (isSource(data.opName)) {
      produced.set(id, getSource(data.opName)!.make(graph, params));
      return;
    }

    if (data.opName === FEEDBACK) {
      const initEdge = edgeTo(id, "init");
      if (!initEdge) throw new Error("Feedback: its init (seed) is not connected");
      const init = handleOf(initEdge);
      if (!init) throw new Error("Feedback: upstream init output missing");
      const fb = graph.feedback(init, id); // stable key = canvas node id
      feedbacks.set(id, fb);
      produced.set(id, { state: fb.state });
      return; // `next` wired later
    }

    const spec = getSpec(data.opName);
    const inputs: Record<string, GpuField> = {};
    for (const port of spec.inputs) {
      const e = edgeTo(id, port.name);
      if (!e) throw new Error(`${spec.label}: input “${port.name}” is not connected`);
      const handle = handleOf(e);
      if (!handle) throw new Error(`${spec.label}: upstream output “${e.sourceHandle}” missing`);
      inputs[port.name] = handle;
    }
    const outs = graph.op(data.opName, inputs, params);
    const map: Record<string, GpuField> = {};
    spec.outputs.forEach((o, i) => { map[o.name] = outs[i]!; });
    produced.set(id, map);
  };

  const visit = (id: string) => {
    if (done.has(id)) return;
    if (visiting.has(id)) throw new Error("buildGraph: cycle detected (a cycle must pass through a feedback node)");
    visiting.add(id);
    const isFeedback = (byId.get(id)?.data as unknown as NodeData)?.opName === FEEDBACK;
    // For feedback, only follow `init` — the `next` back-edge is cut here too.
    for (const e of edges.filter((ed) => ed.target === id && (!isFeedback || ed.targetHandle === "init"))) visit(e.source);
    visiting.delete(id);
    build(id);
    done.add(id);
  };

  for (const n of nodes) visit(n.id);

  // Second pass: close each feedback loop now that downstream handles exist.
  for (const [id, fb] of feedbacks) {
    const nextEdge = edgeTo(id, "next");
    if (nextEdge) {
      const next = handleOf(nextEdge);
      if (next) fb.close(next);
    }
  }

  return { graph, produced };
}

function pickHandle(produced: BuildResult["produced"], nodeId: string, port?: string): GpuField {
  const map = produced.get(nodeId);
  if (!map) throw new Error("node not in graph");
  const handle = port ? map[port] : Object.values(map)[0];
  if (!handle) throw new Error("node has no output to pull");
  return handle;
}

/** Build + pull one output of one node (one tick from the seed for feedback graphs).
 *  A persistent `cache` reuses unchanged upstream nodes. */
export async function runNode(
  nodes: Node[],
  edges: Edge[],
  nodeId: string,
  port?: string,
  cache?: GraphMemo,
): Promise<FieldValue> {
  const { graph, produced } = buildGraph(nodes, edges);
  return pull(graph, pickHandle(produced, nodeId, port), { ctx: { backend: browserBackend }, cache });
}

/** Build + advance a feedback graph by `steps` ticks, persisting `state` across
 *  calls so the simulation continues frame to frame. */
export async function advanceNode(
  nodes: Node[],
  edges: Edge[],
  nodeId: string,
  port: string | undefined,
  opts: { steps?: number; state: SimState; reset?: boolean },
): Promise<FieldValue> {
  const { graph, produced } = buildGraph(nodes, edges);
  return advance(graph, pickHandle(produced, nodeId, port), {
    ctx: { backend: browserBackend },
    steps: opts.steps,
    state: opts.state,
    reset: opts.reset,
  });
}

export { createSimState };
export type { SimState };
