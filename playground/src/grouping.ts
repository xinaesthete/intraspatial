// Collapse/expand of subgraphs — a UI projection over the flat graph. The React Flow
// node/edge arrays are the source of truth and hold both real nodes and "group" proxy
// nodes; collapsing hides the members and adds a proxy whose ports are the edges
// crossing the selection boundary. `buildGraph` filters proxies out, so the executor
// always runs the real flat graph regardless of collapse state.
import type { Edge, Node } from "@xyflow/react";
import type { NodeData } from "./buildGraph";
import { getSpec } from "./specs";

export interface BoundaryPort {
  id: string;            // stable handle id on the proxy
  node: string;          // internal node id at the boundary
  port: string;          // internal port name
  label: string;         // user-facing (editable)
}

export interface GroupData {
  label: string;
  members: string[];
  inputs: BoundaryPort[];
  outputs: BoundaryPort[];
  collapsed: boolean;
  [key: string]: unknown;
}

export const isGroupNode = (n: Node): boolean => n.type === "group";
export const isProxyEdge = (e: Edge): boolean => Boolean((e.data as { proxy?: boolean } | undefined)?.proxy);

const inPortId = (node: string, port: string) => `gi:${node}:${port}`;
const outPortId = (node: string, port: string) => `go:${node}:${port}`;

function shortLabel(node: Node | undefined, port: string): string {
  if (!node) return port;
  const op = (node.data as unknown as NodeData).opName;
  const spec = getSpec(op);
  return `${spec.label.split(" ")[0]}·${port}`;
}

/** Inputs = edges entering the selection from outside; outputs = edges leaving it.
 *  Deduped by the internal (node, port) they attach to. */
export function computeBoundary(nodes: Node[], edges: Edge[], memberIds: string[]): { inputs: BoundaryPort[]; outputs: BoundaryPort[] } {
  const members = new Set(memberIds);
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const inputs = new Map<string, BoundaryPort>();
  const outputs = new Map<string, BoundaryPort>();
  for (const e of edges) {
    if (isProxyEdge(e)) continue;
    const sIn = members.has(e.source);
    const tIn = members.has(e.target);
    if (tIn && !sIn) {
      const port = e.targetHandle ?? "in";
      const k = `${e.target}:${port}`;
      if (!inputs.has(k)) inputs.set(k, { id: inPortId(e.target, port), node: e.target, port, label: shortLabel(byId.get(e.target), port) });
    } else if (sIn && !tIn) {
      const port = e.sourceHandle ?? "out";
      const k = `${e.source}:${port}`;
      if (!outputs.has(k)) outputs.set(k, { id: outPortId(e.source, port), node: e.source, port, label: shortLabel(byId.get(e.source), port) });
    }
  }
  // Also expose *dangling* member outputs (no consumer at all) — otherwise grouping a
  // terminal subgraph would have no outputs and couldn't be previewed.
  const realEdges = edges.filter((e) => !isProxyEdge(e));
  for (const id of memberIds) {
    const node = byId.get(id);
    if (!node) continue;
    const data = node.data as unknown as NodeData;
    const spec = getSpec(data.opName);
    for (const o of spec.outputs) {
      const k = `${id}:${o.name}`;
      if (outputs.has(k)) continue;
      const consumed = realEdges.some((e) => e.source === id && (e.sourceHandle ?? "out") === o.name);
      if (!consumed) outputs.set(k, { id: outPortId(id, o.name), node: id, port: o.name, label: shortLabel(node, o.name) });
    }
  }
  return { inputs: [...inputs.values()], outputs: [...outputs.values()] };
}

const centroid = (nodes: Node[]) => {
  const n = nodes.length || 1;
  const sx = nodes.reduce((a, m) => a + m.position.x, 0) / n;
  const sy = nodes.reduce((a, m) => a + m.position.y, 0) / n;
  return { x: sx, y: sy };
};

/** Collapse `memberIds` into a new proxy node `groupId`. Hides members + their edges
 *  and adds proxy edges connecting external nodes to the proxy's boundary ports. */
export function collapse(
  nodes: Node[],
  edges: Edge[],
  memberIds: string[],
  groupId: string,
  label: string,
): { nodes: Node[]; edges: Edge[] } {
  const members = new Set(memberIds);
  const { inputs, outputs } = computeBoundary(nodes, edges, memberIds);
  const inByKey = new Map(inputs.map((p) => [`${p.node}:${p.port}`, p]));
  const outByKey = new Map(outputs.map((p) => [`${p.node}:${p.port}`, p]));

  const proxy: Node = {
    id: groupId,
    type: "group",
    position: centroid(nodes.filter((n) => members.has(n.id))),
    data: { label, members: memberIds, inputs, outputs, collapsed: true } as GroupData as unknown as Record<string, unknown>,
  };

  const nextNodes = nodes.map((n) => (members.has(n.id) ? { ...n, hidden: true, selected: false } : n));
  nextNodes.push(proxy);

  const proxyEdges: Edge[] = [];
  const nextEdges = edges.map((e) => {
    if (isProxyEdge(e)) return e;
    const sIn = members.has(e.source), tIn = members.has(e.target);
    if (sIn || tIn) {
      if (sIn && !tIn) {
        const p = outByKey.get(`${e.source}:${e.sourceHandle ?? "out"}`)!;
        proxyEdges.push({ id: `px:${e.id}`, source: groupId, sourceHandle: p.id, target: e.target, targetHandle: e.targetHandle, data: { proxy: true } });
      } else if (tIn && !sIn) {
        const p = inByKey.get(`${e.target}:${e.targetHandle ?? "in"}`)!;
        proxyEdges.push({ id: `px:${e.id}`, source: e.source, sourceHandle: e.sourceHandle, target: groupId, targetHandle: p.id, data: { proxy: true } });
      }
      return { ...e, hidden: true }; // internal or boundary → hidden behind the proxy
    }
    return e;
  });

  return { nodes: nextNodes, edges: [...nextEdges, ...proxyEdges] };
}

/** Toggle a group open/closed, keeping the group itself. Collapsed → proxy visible,
 *  members hidden (boundary recomputed from the current wiring). Expanded → proxy
 *  hidden, members visible. The group persists either way. */
export function setCollapsed(nodes: Node[], edges: Edge[], groupId: string, collapsed: boolean): { nodes: Node[]; edges: Edge[] } {
  const proxy = nodes.find((n) => n.id === groupId);
  if (!proxy) return { nodes, edges };
  const members = new Set((proxy.data as unknown as GroupData).members);

  if (!collapsed) {
    // EXPAND: hide the proxy, show members, drop this group's proxy edges.
    const nextNodes = nodes.map((n) =>
      n.id === groupId
        ? { ...n, hidden: true, data: { ...(n.data as object), collapsed: false } as Record<string, unknown> }
        : members.has(n.id) ? { ...n, hidden: false } : n,
    );
    const nextEdges = edges
      .filter((e) => !(isProxyEdge(e) && (e.source === groupId || e.target === groupId)))
      .map((e) => (members.has(e.source) || members.has(e.target) ? { ...e, hidden: false } : e));
    return { nodes: nextNodes, edges: nextEdges };
  }

  // COLLAPSE: recompute the boundary from the current real edges, show the proxy,
  // hide members, and add fresh proxy edges.
  const realEdges = edges.filter((e) => !isProxyEdge(e));
  const { inputs, outputs } = computeBoundary(nodes, realEdges, [...members]);
  const inByKey = new Map(inputs.map((p) => [`${p.node}:${p.port}`, p]));
  const outByKey = new Map(outputs.map((p) => [`${p.node}:${p.port}`, p]));

  const nextNodes = nodes.map((n) =>
    n.id === groupId
      ? { ...n, hidden: false, data: { ...(n.data as object), inputs, outputs, collapsed: true } as Record<string, unknown> }
      : members.has(n.id) ? { ...n, hidden: true, selected: false } : n,
  );

  const proxyEdges: Edge[] = [];
  const nextEdges = realEdges.map((e) => {
    const sIn = members.has(e.source), tIn = members.has(e.target);
    if (sIn || tIn) {
      if (sIn && !tIn) {
        const p = outByKey.get(`${e.source}:${e.sourceHandle ?? "out"}`)!;
        proxyEdges.push({ id: `px:${e.id}`, source: groupId, sourceHandle: p.id, target: e.target, targetHandle: e.targetHandle, data: { proxy: true } });
      } else if (tIn && !sIn) {
        const p = inByKey.get(`${e.target}:${e.targetHandle ?? "in"}`)!;
        proxyEdges.push({ id: `px:${e.id}`, source: e.source, sourceHandle: e.sourceHandle, target: groupId, targetHandle: p.id, data: { proxy: true } });
      }
      return { ...e, hidden: true };
    }
    return e;
  });
  return { nodes: nextNodes, edges: [...nextEdges, ...proxyEdges] };
}

/** Dissolve a group entirely: remove the proxy + proxy edges, unhide members. */
export function ungroup(nodes: Node[], edges: Edge[], groupId: string): { nodes: Node[]; edges: Edge[] } {
  const proxy = nodes.find((n) => n.id === groupId);
  if (!proxy) return { nodes, edges };
  const members = new Set((proxy.data as unknown as GroupData).members);
  const nextNodes = nodes
    .filter((n) => n.id !== groupId)
    .map((n) => (members.has(n.id) ? { ...n, hidden: false } : n));
  const nextEdges = edges
    .filter((e) => !(isProxyEdge(e) && (e.source === groupId || e.target === groupId)))
    .map((e) => (members.has(e.source) || members.has(e.target) ? { ...e, hidden: false } : e));
  return { nodes: nextNodes, edges: nextEdges };
}

/** Resolve a (proxy node, output port id) to the internal (node, port) it stands for,
 *  so a collapsed group can still be pulled/previewed. */
export function resolveGroupOutput(proxy: Node, portId?: string): { node: string; port: string } | undefined {
  const outs = (proxy.data as unknown as GroupData).outputs;
  const p = portId ? outs.find((o) => o.id === portId) : outs[0];
  return p ? { node: p.node, port: p.port } : undefined;
}
