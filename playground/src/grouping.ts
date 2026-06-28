// Subgraph grouping via scope navigation (drill-in + breadcrumb), not inline
// collapse. Each node carries `data.group` = the scope it lives in (the root scope,
// or a group id). The logical graph (real nodes + real edges) is NEVER rewritten by
// grouping — membership is just a tag — so `buildGraph` only has to skip group
// nodes. The canvas shows ONE scope at a time, derived from the flat graph:
//
//   • root scope      → top-level nodes + group nodes (child groups appear as a
//                       single node whose ports are the edges crossing into them).
//   • inside a group  → its members + boundary "port" stubs for the connections that
//                       enter/leave the group.
//
// Double-clicking a group node pushes its scope (navigate in); the breadcrumb pops
// back up.  (v1: groups nest one level — members are op nodes, not groups.)
import type { Edge, Node } from "@xyflow/react";
import type { NodeData } from "./buildGraph";
import { getSpec } from "./specs";

export const ROOT_SCOPE = "__root__";

export interface BoundaryPort {
  id: string;     // handle id on the group node / key for the stub
  node: string;   // internal member id at the boundary
  port: string;   // internal port name
  label: string;
}

export interface GroupData {
  label: string;
  group: string;                       // the scope this group node lives in
  portLabels?: Record<string, string>; // user overrides, keyed by port id
  // attached by deriveDisplay for rendering:
  inputs?: BoundaryPort[];
  outputs?: BoundaryPort[];
  members?: number;
  [key: string]: unknown;
}

export const isGroupNode = (n: Node): boolean => n.type === "group";
export const isPortStub = (n: Node): boolean => n.type === "groupPort";
export const scopeOf = (n: Node): string => ((n.data as { group?: string } | undefined)?.group) ?? ROOT_SCOPE;

const inId = (node: string, port: string) => `gi:${node}:${port}`;
const outId = (node: string, port: string) => `go:${node}:${port}`;

/** Decode a group-node handle id back to the internal (member, port) it stands for. */
export function decodePort(handle: string | null | undefined): { dir: "in" | "out"; node: string; port: string } | null {
  if (!handle) return null;
  const m = /^g([io]):([^:]+):(.+)$/.exec(handle);
  if (!m) return null;
  return { dir: m[1] === "i" ? "in" : "out", node: m[2]!, port: m[3]! };
}

function shortLabel(node: Node | undefined, port: string): string {
  if (!node) return port;
  const op = (node.data as unknown as NodeData).opName;
  return `${getSpec(op).label.split(" ")[0]}·${port}`;
}

/** Tag the selected nodes as members of a new group; create the group node in the
 *  current scope. Edges are left untouched. */
export function createGroup(
  nodes: Node[],
  edges: Edge[],
  memberIds: string[],
  groupId: string,
  label: string,
  scope: string,
): { nodes: Node[]; edges: Edge[] } {
  const members = new Set(memberIds);
  const sx = nodes.filter((n) => members.has(n.id));
  const cx = sx.reduce((a, m) => a + m.position.x, 0) / (sx.length || 1);
  const cy = sx.reduce((a, m) => a + m.position.y, 0) / (sx.length || 1);
  const groupNode: Node = {
    id: groupId,
    type: "group",
    position: { x: cx, y: cy },
    data: { label, group: scope, portLabels: {} } as GroupData as unknown as Record<string, unknown>,
  };
  const nextNodes = nodes.map((n) =>
    members.has(n.id)
      ? { ...n, selected: false, data: { ...(n.data as object), group: groupId } as Record<string, unknown> }
      : n,
  );
  nextNodes.push(groupNode);
  return { nodes: nextNodes, edges };
}

/** Dissolve a group: its members move back to the group's own scope; the group node
 *  is removed. Edges are untouched. */
export function ungroup(nodes: Node[], edges: Edge[], groupId: string): { nodes: Node[]; edges: Edge[] } {
  const g = nodes.find((n) => n.id === groupId);
  if (!g) return { nodes, edges };
  const parent = scopeOf(g);
  const nextNodes = nodes
    .filter((n) => n.id !== groupId)
    .map((n) => (scopeOf(n) === groupId ? { ...n, data: { ...(n.data as object), group: parent } as Record<string, unknown> } : n));
  return { nodes: nextNodes, edges };
}

/** The boundary ports of a group: edges crossing in/out, plus dangling member
 *  outputs (so a terminal subgraph stays previewable). Labels merge user overrides. */
export function boundaryOf(nodes: Node[], edges: Edge[], groupId: string): { inputs: BoundaryPort[]; outputs: BoundaryPort[] } {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const groupNode = byId.get(groupId);
  const overrides = (groupNode?.data as unknown as GroupData)?.portLabels ?? {};
  const members = new Set(nodes.filter((n) => scopeOf(n) === groupId).map((n) => n.id));
  const inputs = new Map<string, BoundaryPort>();
  const outputs = new Map<string, BoundaryPort>();
  const lbl = (id: string, dflt: string) => overrides[id] ?? dflt;

  for (const e of edges) {
    const sIn = members.has(e.source), tIn = members.has(e.target);
    if (tIn && !sIn) {
      const port = e.targetHandle ?? "in", id = inId(e.target, port);
      if (!inputs.has(id)) inputs.set(id, { id, node: e.target, port, label: lbl(id, shortLabel(byId.get(e.target), port)) });
    } else if (sIn && !tIn) {
      const port = e.sourceHandle ?? "out", id = outId(e.source, port);
      if (!outputs.has(id)) outputs.set(id, { id, node: e.source, port, label: lbl(id, shortLabel(byId.get(e.source), port)) });
    }
  }
  // dangling member outputs
  for (const id of members) {
    const node = byId.get(id);
    if (!node) continue;
    for (const o of getSpec((node.data as unknown as NodeData).opName).outputs) {
      const pid = outId(id, o.name);
      if (outputs.has(pid)) continue;
      if (!edges.some((e) => e.source === id && (e.sourceHandle ?? "out") === o.name)) {
        outputs.set(pid, { id: pid, node: id, port: o.name, label: lbl(pid, shortLabel(node, o.name)) });
      }
    }
  }
  return { inputs: [...inputs.values()], outputs: [...outputs.values()] };
}

interface Located { kind: "in" | "child" | "outside"; group?: string }
function locate(byId: Map<string, Node>, scope: string, nodeId: string): Located {
  const x = byId.get(nodeId);
  const sx = x ? scopeOf(x) : ROOT_SCOPE;
  if (sx === scope) return { kind: "in" };
  const g = byId.get(sx);
  if (g && isGroupNode(g) && scopeOf(g) === scope) return { kind: "child", group: sx };
  return { kind: "outside" };
}

/** Build the React Flow nodes/edges for the given scope from the flat graph. */
export function deriveDisplay(nodes: Node[], edges: Edge[], scope: string): { nodes: Node[]; edges: Edge[] } {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const dispNodes: Node[] = [];

  // 1. nodes that live directly in this scope (op nodes + child group nodes)
  for (const n of nodes) {
    if (scopeOf(n) !== scope) continue;
    if (isGroupNode(n)) {
      const b = boundaryOf(nodes, edges, n.id);
      const members = nodes.filter((m) => scopeOf(m) === n.id).length;
      dispNodes.push({ ...n, data: { ...(n.data as object), inputs: b.inputs, outputs: b.outputs, members } as Record<string, unknown> });
    } else {
      dispNodes.push(n);
    }
  }

  // 2. inside a group: boundary port stubs at the sides
  let stubs: { inputs: BoundaryPort[]; outputs: BoundaryPort[] } = { inputs: [], outputs: [] };
  if (scope !== ROOT_SCOPE) {
    stubs = boundaryOf(nodes, edges, scope);
    const members = nodes.filter((n) => scopeOf(n) === scope);
    const minX = Math.min(...members.map((m) => m.position.x), 0);
    const maxX = Math.max(...members.map((m) => m.position.x), 0);
    const baseY = members.length ? members.reduce((a, m) => a + m.position.y, 0) / members.length : 0;
    stubs.inputs.forEach((p, i) => dispNodes.push({
      id: `stub:${p.id}`, type: "groupPort", position: { x: minX - 220, y: baseY + (i - stubs.inputs.length / 2) * 70 },
      data: { label: p.label, side: "in" } as Record<string, unknown>,
    }));
    stubs.outputs.forEach((p, i) => dispNodes.push({
      id: `stub:${p.id}`, type: "groupPort", position: { x: maxX + 260, y: baseY + (i - stubs.outputs.length / 2) * 70 },
      data: { label: p.label, side: "out" } as Record<string, unknown>,
    }));
  }

  // 3. edges, routed to representatives within this scope
  const dispEdges: Edge[] = [];
  const stubInByKey = new Map(stubs.inputs.map((p) => [`${p.node}:${p.port}`, p]));
  const stubOutByKey = new Map(stubs.outputs.map((p) => [`${p.node}:${p.port}`, p]));
  for (const e of edges) {
    const aPort = e.sourceHandle ?? "out", bPort = e.targetHandle ?? "in";
    const ra = locate(byId, scope, e.source), rb = locate(byId, scope, e.target);
    const mk = (src: string, sh: string, tgt: string, th: string) =>
      dispEdges.push({ id: `d:${e.id}`, source: src, sourceHandle: sh, target: tgt, targetHandle: th });

    if (ra.kind === "in" && rb.kind === "in") mk(e.source, aPort, e.target, bPort);
    else if (ra.kind === "in" && rb.kind === "child") mk(e.source, aPort, rb.group!, inId(e.target, bPort));
    else if (ra.kind === "child" && rb.kind === "in") mk(ra.group!, outId(e.source, aPort), e.target, bPort);
    else if (ra.kind === "child" && rb.kind === "child") {
      if (ra.group !== rb.group) mk(ra.group!, outId(e.source, aPort), rb.group!, inId(e.target, bPort));
      // same group → internal edge, hidden at this level
    } else if (scope !== ROOT_SCOPE && ra.kind === "in" && rb.kind === "outside") {
      const p = stubOutByKey.get(`${e.source}:${aPort}`);
      if (p) mk(e.source, aPort, `stub:${p.id}`, "in");
    } else if (scope !== ROOT_SCOPE && ra.kind === "outside" && rb.kind === "in") {
      const p = stubInByKey.get(`${e.target}:${bPort}`);
      if (p) mk(`stub:${p.id}`, "out", e.target, bPort);
    }
  }

  // Dangling outputs have no real edge — draw a synthetic member→stub link so the
  // group's output is visibly connected inside.
  if (scope !== ROOT_SCOPE) {
    const targeted = new Set(dispEdges.map((e) => e.target));
    for (const p of stubs.outputs) {
      const stubId = `stub:${p.id}`;
      if (!targeted.has(stubId)) dispEdges.push({ id: `ds:${p.id}`, source: p.node, sourceHandle: p.port, target: stubId, targetHandle: "in" });
    }
  }

  return { nodes: dispNodes, edges: dispEdges };
}

/** Resolve a group node's output port id to the internal (node, port) it represents,
 *  for pull/preview. Defaults to the first output. */
export function resolveGroupOutput(nodes: Node[], edges: Edge[], groupId: string, portId?: string): { node: string; port: string } | undefined {
  const outs = boundaryOf(nodes, edges, groupId).outputs;
  const p = portId ? outs.find((o) => o.id === portId) : outs[0];
  return p ? { node: p.node, port: p.port } : undefined;
}

/** Decode a connection made onto a group node port into a real (member, port). */
export function decodeGroupConnection(handle: string | null | undefined): { node: string; port: string } | null {
  const d = decodePort(handle);
  return d ? { node: d.node, port: d.port } : null;
}
