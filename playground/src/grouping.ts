// Subgraph grouping with explicit, materialised Input/Output interface nodes
// (the Houdini-subnet / Blueprint-function model), and scope navigation.
//
// Node kinds (all live in the one flat logical array, each tagged with the scope it
// lives in via `data.group`):
//   • op                – a real operation (executes); includes source generators.
//   • group             – the subnet marker in the PARENT scope; its ports are its
//                         interface nodes.
//   • input / output    – interface nodes INSIDE a group; you wire them to members.
//
// Because grouping rewires each boundary-crossing edge to pass through the group
// node's port (parent half) and an interface node (child half), **no edge ever spans
// two scopes** — every edge has both endpoints in one scope. So the per-scope view
// is just a filter, and execution flattens by splicing interface/group nodes out
// (`resolveSource`). buildGraph consumes that.
import type { Edge, Node } from "@xyflow/react";
import type { NodeData } from "./buildGraph";
import { getSpec } from "./specs";
// Instance nodes reference reusable subgraph definitions; their ports derive from the
// def interface. (Cycle with subgraphs.ts is safe — only referenced inside functions.)
import type { DefLibrary } from "./subgraphs";
import { instanceDefName, instancePorts, isInstanceNode } from "./subgraphs";

export const ROOT_SCOPE = "__root__";
export type IOType = "input" | "output";

export interface IODataShape {
  group: string;
  label: string;
  io: IOType;
  [key: string]: unknown;
}
export interface GroupData {
  label: string;
  group: string;
  inputs?: Port[];
  outputs?: Port[];
  members?: number;
  [key: string]: unknown;
}
export interface Port {
  id: string; // = the interface node id (the handle on the group node)
  node: string; // the member this interface attaches to (for resolution)
  port: string;
  kind: string;
  label: string;
}

export const isGroupNode = (n: Node): boolean => n.type === "group";
export const isInputNode = (n: Node): boolean => n.type === "input";
export const isOutputNode = (n: Node): boolean => n.type === "output";
export const isInterfaceNode = (n: Node): boolean => n.type === "input" || n.type === "output";
export const isOpNode = (n: Node): boolean => !n.type || n.type === "op";
export const scopeOf = (n: Node): string => (n.data as { group?: string } | undefined)?.group ?? ROOT_SCOPE;

const opName = (n: Node): string => (n.data as unknown as NodeData).opName;

function opPortKind(node: Node | undefined, port: string, dir: "in" | "out", defs: DefLibrary = {}): string {
  if (!node) return "any";
  if (isInstanceNode(node)) {
    const def = defs[instanceDefName(node)];
    if (!def) return "any";
    const ports = dir === "in" ? instancePorts(def, defs).inputs : instancePorts(def, defs).outputs;
    return ports.find((p) => p.id === port)?.kind ?? "any";
  }
  if (!isOpNode(node)) return "any";
  const spec = getSpec(opName(node));
  return (dir === "in" ? spec.inputs : spec.outputs).find((p) => p.name === port)?.kind ?? "any";
}

// ── Flatten / resolution ────────────────────────────────────────────────────
// resolveSource walks back from a connection point to the real op output that
// ultimately produces its value, splicing input/output/group pass-throughs. Used by
// buildGraph (execution), output-port kinds, and group-output preview.
export function resolveSource(
  nodes: Node[],
  edges: Edge[],
  nodeId: string,
  handle: string,
  seen = new Set<string>(),
): { node: string; port: string } | null {
  const guard = `${nodeId}:${handle}`;
  if (seen.has(guard)) return null;
  seen.add(guard);
  const byId = idMap(nodes);
  const n = byId.get(nodeId);
  if (!n) return null;
  if (isOpNode(n)) return { node: nodeId, port: handle };
  if (isInputNode(n)) {
    // fed by the parent edge into the group port named after this input node
    const e = edges.find((ed) => ed.target === scopeOf(n) && ed.targetHandle === nodeId);
    return e ? resolveSource(nodes, edges, e.source, e.sourceHandle ?? "out", seen) : null;
  }
  if (isGroupNode(n)) {
    // the group's output port `handle` is an output interface node; it's fed by a member
    const e = edges.find((ed) => ed.target === handle && (ed.targetHandle ?? "in") === "in");
    return e ? resolveSource(nodes, edges, e.source, e.sourceHandle ?? "out", seen) : null;
  }
  return null; // output node as a source: not a producer
}

// Consumer-side kind for an input interface node (what it feeds), for colouring.
function inputKind(
  nodes: Node[],
  edges: Edge[],
  inputNodeId: string,
  defs: DefLibrary,
  seen = new Set<string>(),
): { node: string; port: string; kind: string } {
  if (seen.has(inputNodeId)) return { node: "", port: "", kind: "any" };
  seen.add(inputNodeId);
  const byId = idMap(nodes);
  const e = edges.find((ed) => ed.source === inputNodeId && (ed.sourceHandle ?? "out") === "out");
  if (!e) return { node: "", port: "", kind: "any" };
  const tgt = byId.get(e.target);
  if (tgt && isGroupNode(tgt)) return inputKind(nodes, edges, e.targetHandle ?? "", defs, seen); // nested group port
  return { node: e.target, port: e.targetHandle ?? "in", kind: opPortKind(tgt, e.targetHandle ?? "in", "in", defs) };
}

const idMap = (nodes: Node[]) => new Map(nodes.map((n) => [n.id, n]));

/** The ports a group node exposes, derived from its interface nodes. */
export function groupPorts(nodes: Node[], edges: Edge[], groupId: string, defs: DefLibrary = {}): { inputs: Port[]; outputs: Port[] } {
  const byId = idMap(nodes);
  const inputs: Port[] = [];
  const outputs: Port[] = [];
  for (const n of nodes) {
    if (scopeOf(n) !== groupId) continue;
    if (isInputNode(n)) {
      const k = inputKind(nodes, edges, n.id, defs);
      inputs.push({ id: n.id, node: k.node, port: k.port, kind: k.kind, label: (n.data as unknown as IODataShape).label });
    } else if (isOutputNode(n)) {
      const e = edges.find((ed) => ed.target === n.id && (ed.targetHandle ?? "in") === "in");
      const src = e ? byId.get(e.source) : undefined;
      const kind = e ? opPortKind(src, e.sourceHandle ?? "out", "out", defs) : "any";
      outputs.push({ id: n.id, node: e?.source ?? "", port: e?.sourceHandle ?? "", kind, label: (n.data as unknown as IODataShape).label });
    }
  }
  return { inputs, outputs };
}

// ── Build a group (materialise boundary into interface nodes) ────────────────
function ioNode(id: string, io: IOType, group: string, label: string, position: { x: number; y: number }): Node {
  return { id, type: io, position, data: { group, label, io } as IODataShape as unknown as Record<string, unknown> };
}

export function createGroup(
  nodes: Node[],
  edges: Edge[],
  memberIds: string[],
  groupId: string,
  label: string,
  scope: string,
): { nodes: Node[]; edges: Edge[] } {
  const members = new Set(memberIds);
  const byId = idMap(nodes);
  const sx = nodes.filter((n) => members.has(n.id));
  const cx = sx.reduce((a, m) => a + m.position.x, 0) / (sx.length || 1);
  const cy = sx.reduce((a, m) => a + m.position.y, 0) / (sx.length || 1);
  const minX = Math.min(...sx.map((m) => m.position.x), cx);
  const maxX = Math.max(...sx.map((m) => m.position.x), cx);

  const groupNode: Node = {
    id: groupId,
    type: "group",
    position: { x: cx, y: cy },
    data: { label, group: scope } as GroupData as unknown as Record<string, unknown>,
  };
  const newNodes: Node[] = [groupNode];
  const nextEdges: Edge[] = [];
  let inK = 0,
    outK = 0;
  const outFor = new Map<string, string>(); // member:port -> output node id (dedup multiple consumers)

  for (const e of edges) {
    const sIn = members.has(e.source),
      tIn = members.has(e.target);
    if (tIn && !sIn) {
      // boundary input: ext.src -> member.tgt. Materialise an input node.
      const iid = `${groupId}~in${inK++}`;
      const m = byId.get(e.target);
      newNodes.push(ioNode(iid, "input", groupId, e.targetHandle ?? "in", { x: minX - 200, y: cy + inK * 70 }));
      nextEdges.push({ ...e, target: groupId, targetHandle: iid }); // parent half
      nextEdges.push({ id: `${e.id}~c`, source: iid, sourceHandle: "out", target: e.target, targetHandle: e.targetHandle }); // child half
      void m;
    } else if (sIn && !tIn) {
      // boundary output: member.src -> ext.tgt. One output node per (member, port).
      const key = `${e.source}:${e.sourceHandle ?? "out"}`;
      let oid = outFor.get(key);
      if (!oid) {
        oid = `${groupId}~out${outK++}`;
        outFor.set(key, oid);
        newNodes.push(ioNode(oid, "output", groupId, e.sourceHandle ?? "out", { x: maxX + 240, y: cy + outK * 70 }));
        nextEdges.push({ id: `${e.id}~c`, source: e.source, sourceHandle: e.sourceHandle, target: oid, targetHandle: "in" }); // child half
      }
      nextEdges.push({ ...e, source: groupId, sourceHandle: oid }); // parent half
    } else {
      nextEdges.push(e); // internal or unrelated
    }
  }

  // dangling member outputs → expose as outputs too
  for (const id of memberIds) {
    const node = byId.get(id);
    if (!node || !isOpNode(node)) continue;
    for (const o of getSpec(opName(node)).outputs) {
      const key = `${id}:${o.name}`;
      if (outFor.has(key)) continue;
      if (nextEdges.some((e) => e.source === id && (e.sourceHandle ?? "out") === o.name)) continue;
      const oid = `${groupId}~out${outK++}`;
      newNodes.push(ioNode(oid, "output", groupId, o.name, { x: maxX + 240, y: cy + outK * 70 }));
      nextEdges.push({ id: `dang:${oid}`, source: id, sourceHandle: o.name, target: oid, targetHandle: "in" });
    }
  }

  const nextNodes = nodes
    .map((n) =>
      members.has(n.id) ? { ...n, selected: false, data: { ...(n.data as object), group: groupId } as Record<string, unknown> } : n,
    )
    .concat(newNodes);
  return { nodes: nextNodes, edges: nextEdges };
}

/** Add a fresh, unwired interface node inside a group scope. */
export function addInterface(scope: string, io: IOType, id: string, position: { x: number; y: number }): Node {
  return ioNode(id, io, scope, io === "input" ? "in" : "out", position);
}

/** Dissolve a group: splice its interface nodes (reconnect external↔member directly),
 *  remove the group + interface nodes, move members to the group's parent scope. */
export function ungroup(nodes: Node[], edges: Edge[], groupId: string): { nodes: Node[]; edges: Edge[] } {
  const g = nodes.find((n) => n.id === groupId);
  if (!g) return { nodes, edges };
  const parent = scopeOf(g);
  const ioIds = new Set(nodes.filter((n) => isInterfaceNode(n) && scopeOf(n) === groupId).map((n) => n.id));

  const spliced: Edge[] = [];
  const drop = new Set<string>();
  for (const n of nodes) {
    if (!isInterfaceNode(n) || scopeOf(n) !== groupId) continue;
    if (isInputNode(n)) {
      const parentE = edges.find((e) => e.target === groupId && e.targetHandle === n.id);
      const childEs = edges.filter((e) => e.source === n.id);
      for (const ce of childEs) {
        if (parentE)
          spliced.push({
            id: `ug:${ce.id}`,
            source: parentE.source,
            sourceHandle: parentE.sourceHandle,
            target: ce.target,
            targetHandle: ce.targetHandle,
          });
        drop.add(ce.id);
      }
      if (parentE) drop.add(parentE.id);
    } else {
      const childE = edges.find((e) => e.target === n.id && (e.targetHandle ?? "in") === "in");
      const parentEs = edges.filter((e) => e.source === groupId && e.sourceHandle === n.id);
      for (const pe of parentEs) {
        if (childE)
          spliced.push({
            id: `ug:${pe.id}`,
            source: childE.source,
            sourceHandle: childE.sourceHandle,
            target: pe.target,
            targetHandle: pe.targetHandle,
          });
        drop.add(pe.id);
      }
      if (childE) drop.add(childE.id);
    }
  }

  const nextEdges = edges.filter((e) => !drop.has(e.id)).concat(spliced);
  const nextNodes = nodes
    .filter((n) => n.id !== groupId && !ioIds.has(n.id))
    .map((n) => (scopeOf(n) === groupId ? { ...n, data: { ...(n.data as object), group: parent } as Record<string, unknown> } : n));
  return { nodes: nextNodes, edges: nextEdges };
}

/** React Flow nodes/edges for the given scope: just a filter (edges never cross
 *  scopes), with group nodes augmented with their derived ports. */
export function deriveDisplay(nodes: Node[], edges: Edge[], scope: string, defs: DefLibrary = {}): { nodes: Node[]; edges: Edge[] } {
  const byId = idMap(nodes);
  // kinds of this scope's own interface nodes (if it's a group/def) for colouring
  const selfPorts = scope === ROOT_SCOPE ? { inputs: [], outputs: [] } : groupPorts(nodes, edges, scope, defs);
  const kindOf = new Map<string, string>([...selfPorts.inputs, ...selfPorts.outputs].map((p) => [p.id, p.kind]));
  const dispNodes = nodes
    .filter((n) => scopeOf(n) === scope)
    .map((n) => {
      if (isInstanceNode(n)) {
        const def = defs[instanceDefName(n)];
        const { inputs, outputs } = def ? instancePorts(def, defs) : { inputs: [], outputs: [] };
        const members = def ? def.nodes.filter((m) => isOpNode(m)).length : 0;
        return { ...n, data: { ...(n.data as object), inputs, outputs, members, def: instanceDefName(n) } as Record<string, unknown> };
      }
      if (isGroupNode(n)) {
        const { inputs, outputs } = groupPorts(nodes, edges, n.id, defs);
        const members = nodes.filter((m) => scopeOf(m) === n.id && isOpNode(m)).length;
        return { ...n, data: { ...(n.data as object), inputs, outputs, members } as Record<string, unknown> };
      }
      if (isInterfaceNode(n)) return { ...n, data: { ...(n.data as object), kind: kindOf.get(n.id) ?? "any" } as Record<string, unknown> };
      return n;
    });
  const inScope = new Set(dispNodes.map((n) => n.id));
  const dispEdges = edges.filter((e) => inScope.has(e.source) && inScope.has(e.target) && byId.has(e.source) && byId.has(e.target));
  return { nodes: dispNodes, edges: dispEdges };
}

/** Resolve a group output port to the real (op, port) it produces, for preview. */
export function resolveGroupOutput(
  nodes: Node[],
  edges: Edge[],
  groupId: string,
  portId?: string,
): { node: string; port: string } | undefined {
  const outs = groupPorts(nodes, edges, groupId).outputs;
  const p = portId ? outs.find((o) => o.id === portId) : outs[0];
  if (!p) return undefined;
  return resolveSource(nodes, edges, groupId, p.id) ?? undefined;
}
