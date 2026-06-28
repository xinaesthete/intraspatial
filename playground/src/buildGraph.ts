// Turn the React Flow canvas (nodes + edges) into the runtime's Graph IR and pull a
// chosen output. The canvas state is the explicit graph; this is the adapter that
// hands it to the executor.
import type { Edge, Node } from "@xyflow/react";
import { Graph, pull } from "../../src/gpu/graph";
import type { FieldValue, GpuField } from "../../src/gpu/graph";
import { getSource, isSource } from "./sources";
import { getSpec } from "./specs";
import { browserBackend } from "./backend.browser";

export interface NodeData {
  opName: string;
  params: Record<string, unknown>;
  [key: string]: unknown;
}

export interface BuildResult {
  graph: Graph;
  produced: Map<string, Record<string, GpuField>>;
}

/** Build the Graph IR; `produced` maps each canvas node id to its output handles. */
export function buildGraph(nodes: Node[], edges: Edge[]): BuildResult {
  const graph = new Graph();
  const produced = new Map<string, Record<string, GpuField>>();
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const visiting = new Set<string>();
  const done = new Set<string>();

  const build = (id: string) => {
    const node = byId.get(id);
    if (!node) throw new Error(`buildGraph: no node ${id}`);
    const data = node.data as unknown as NodeData;
    const params = data.params ?? {};
    if (isSource(data.opName)) {
      produced.set(id, getSource(data.opName)!.make(graph, params));
      return;
    }
    const spec = getSpec(data.opName);
    const inputs: Record<string, GpuField> = {};
    for (const port of spec.inputs) {
      const e = edges.find((ed) => ed.target === id && ed.targetHandle === port.name);
      if (!e) throw new Error(`${spec.label}: input “${port.name}” is not connected`);
      const up = produced.get(e.source);
      const handle = up?.[e.sourceHandle ?? ""];
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
    if (visiting.has(id)) throw new Error("buildGraph: cycle detected");
    visiting.add(id);
    for (const e of edges.filter((ed) => ed.target === id)) visit(e.source);
    visiting.delete(id);
    build(id);
    done.add(id);
  };

  for (const n of nodes) visit(n.id);
  return { graph, produced };
}

/** Build + pull one output of one node. */
export async function runNode(
  nodes: Node[],
  edges: Edge[],
  nodeId: string,
  port?: string,
): Promise<FieldValue> {
  const { graph, produced } = buildGraph(nodes, edges);
  const map = produced.get(nodeId);
  if (!map) throw new Error("runNode: node not in graph");
  const handle = port ? map[port] : Object.values(map)[0];
  if (!handle) throw new Error("runNode: node has no output to pull");
  return pull(graph, handle, { ctx: { backend: browserBackend } });
}
