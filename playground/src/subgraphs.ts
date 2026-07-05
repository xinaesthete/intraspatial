// Reusable, named subgraph DEFINITIONS — the live-linked counterpart to the one-off
// `group` (subnet) in grouping.ts. A definition is authored once and referenced by
// any number of `instance` nodes; editing the definition updates every instance.
//
// Representation reuses the group model wholesale: a definition's interior is exactly
// the same shape a group's interior is — op nodes + `input`/`output` interface nodes,
// each tagged with `data.group === defScopeId(name)`. So `groupPorts`, `resolveSource`
// and `deriveDisplay` operate on a definition's interior unchanged.
//
// The ONE difference from a group: a group's interior lives inline in the flat node
// array (one copy); a definition's interior lives in the DefLibrary and is *expanded*
// — copied and namespaced (`instanceId/origId`) — once per instance at execution time.
// After `expandInstances`, every instance has become an ordinary group + interior, so
// the existing flatten (buildGraph → resolveSource) needs no change.
import type { Edge, Node } from "@xyflow/react";
import type { GroupData, Port } from "./grouping";
import { groupPorts, scopeOf } from "./grouping";

/** A named, reusable subgraph. `nodes`/`edges` are its interior (scope `def:<name>`). */
export interface SubgraphDef {
  name: string; // unique key (the def id)
  label: string; // user-facing name (editable; defaults to name)
  nodes: Node[];
  edges: Edge[];
}

/** name -> definition. Held in App state, serialised with the graph. */
export type DefLibrary = Record<string, SubgraphDef>;

/** Data carried by an `instance` node in the flat graph. */
export interface InstanceData {
  def: string; // definition name it references
  group: string; // the scope this instance lives in
  label?: string;
  // ports are NOT stored — derived from the def via instancePorts()
  [key: string]: unknown;
}

export const DEF_SCOPE_PREFIX = "def:";
export const defScopeId = (name: string): string => `${DEF_SCOPE_PREFIX}${name}`;
export const isDefScope = (scope: string): boolean => scope.startsWith(DEF_SCOPE_PREFIX);
export const defNameOfScope = (scope: string): string => scope.slice(DEF_SCOPE_PREFIX.length);

export const isInstanceNode = (n: Node): boolean => n.type === "instance";
export const instanceDefName = (n: Node): string => (n.data as unknown as InstanceData).def;

/** The ports an instance exposes — derived from its definition's interface. Same shape
 *  groupPorts returns, so GroupNode/InstanceNode and edge wiring treat them alike. */
export function instancePorts(def: SubgraphDef, defs: DefLibrary): { inputs: Port[]; outputs: Port[] } {
  return groupPorts(def.nodes, def.edges, defScopeId(def.name), defs);
}

// ── Promote a group into a reusable definition ───────────────────────────────
/** Convert an existing one-off group into a reusable definition + its first instance.
 *  The group's interior (members + interface nodes + internal edges) moves into the
 *  library under `name`; the group node becomes an `instance` referencing it. Because
 *  the interface-node ids are preserved, the group's external (parent-scope) edges keep
 *  connecting to the instance's matching ports unchanged. */
export function promoteToDef(
  nodes: Node[],
  edges: Edge[],
  defs: DefLibrary,
  groupId: string,
  name: string,
  label = name,
): { nodes: Node[]; edges: Edge[]; defs: DefLibrary } {
  const group = nodes.find((n) => n.id === groupId);
  if (!group) throw new Error(`promoteToDef: no group "${groupId}"`);
  if (defs[name]) throw new Error(`A reusable subgraph named "${name}" already exists`);

  const newScope = defScopeId(name);
  const interiorIds = new Set(nodes.filter((n) => scopeOf(n) === groupId).map((n) => n.id));

  // Interior nodes move to the def, retagged to the def scope.
  const defNodes = nodes
    .filter((n) => interiorIds.has(n.id))
    .map((n) => ({ ...n, selected: false, data: { ...(n.data as object), group: newScope } as Record<string, unknown> }));
  // Internal edges (both endpoints inside the interior) move to the def.
  const defEdges = edges.filter((e) => interiorIds.has(e.source) && interiorIds.has(e.target));

  const def: SubgraphDef = { name, label, nodes: defNodes, edges: defEdges };

  // The group node becomes an instance; its parent-scope edges (which reference the
  // group/interface port ids) stay valid since those ids are unchanged in the def.
  const parentScope = scopeOf(group);
  const instance: Node = {
    ...group,
    type: "instance",
    data: { def: name, group: parentScope, label } as InstanceData as unknown as Record<string, unknown>,
  };

  const movedEdgeIds = new Set(defEdges.map((e) => e.id));
  const nextNodes = nodes.filter((n) => !interiorIds.has(n.id)).map((n) => (n.id === groupId ? instance : n));
  const nextEdges = edges.filter((e) => !movedEdgeIds.has(e.id));
  return { nodes: nextNodes, edges: nextEdges, defs: { ...defs, [name]: def } };
}

/** Build a fresh instance node referencing `name`, placed in `scope`. */
export function instantiate(name: string, label: string, scope: string, id: string, position: { x: number; y: number }): Node {
  return { id, type: "instance", position, data: { def: name, group: scope, label } as InstanceData as unknown as Record<string, unknown> };
}

// ── Expansion (instances → inline groups) for execution / preview ────────────
const nsId = (instId: string, id: string): string => `${instId}/${id}`;

/** Recursively replace every `instance` node with a namespaced copy of its
 *  definition's interior, producing a flat graph of only op/group/interface nodes —
 *  the shape buildGraph/resolveSource already understand. Throws on recursive defs. */
export function expandInstances(
  nodes: Node[],
  edges: Edge[],
  defs: DefLibrary,
  seen: ReadonlySet<string> = new Set(),
): { nodes: Node[]; edges: Edge[] } {
  const instIds = new Set(nodes.filter(isInstanceNode).map((n) => n.id));
  const outNodes: Node[] = [];
  const outEdges: Edge[] = [];

  // Edges touching an instance: rewrite the instance-side handle to its namespaced
  // interface id (instId/portId), matching how the interior is namespaced below.
  for (const e of edges) {
    const ne: Edge = { ...e };
    if (instIds.has(e.source) && e.sourceHandle != null) ne.sourceHandle = nsId(e.source, e.sourceHandle);
    if (instIds.has(e.target) && e.targetHandle != null) ne.targetHandle = nsId(e.target, e.targetHandle);
    outEdges.push(ne);
  }

  for (const n of nodes) {
    if (!isInstanceNode(n)) {
      outNodes.push(n);
      continue;
    }
    const name = instanceDefName(n);
    const def = defs[name];
    if (!def) throw new Error(`expandInstances: instance "${n.id}" references missing subgraph "${name}"`);
    if (seen.has(name)) throw new Error(`Recursive reusable subgraph: "${name}" contains itself`);

    const instId = n.id;
    const ns = (id: string): string => nsId(instId, id);
    const defScope = defScopeId(name);
    const defNodeIds = new Set(def.nodes.map((m) => m.id));
    // A handle that names an interface node (a def node id) refers to a nested
    // group/interface port → namespace it; op port names ("density", "in") pass through.
    const nsh = (h: string | null | undefined): string | null | undefined => (h != null && defNodeIds.has(h) ? ns(h) : h);

    // The instance stands in as an ordinary group node (same id → parent edges hold).
    const groupNode: Node = {
      id: instId,
      type: "group",
      position: n.position,
      data: { label: (n.data as InstanceData).label ?? def.label, group: scopeOf(n) } as GroupData as unknown as Record<string, unknown>,
    };

    // Clone the interior. Members at the def root re-home to this instance's scope
    // (group === instId); deeper nodes keep their (namespaced) nested scope.
    const interiorNodes: Node[] = def.nodes.map((m) => {
      const mScope = scopeOf(m);
      return {
        ...m,
        id: ns(m.id),
        selected: false,
        data: { ...(m.data as object), group: mScope === defScope ? instId : ns(mScope) } as Record<string, unknown>,
      };
    });
    const interiorEdges: Edge[] = def.edges.map((e) => ({
      ...e,
      id: ns(e.id),
      source: ns(e.source),
      target: ns(e.target),
      sourceHandle: nsh(e.sourceHandle),
      targetHandle: nsh(e.targetHandle),
    }));

    // Recurse to expand any nested instances inside the interior.
    const sub = expandInstances(interiorNodes, interiorEdges, defs, new Set([...seen, name]));
    outNodes.push(groupNode, ...sub.nodes);
    outEdges.push(...sub.edges);
  }

  return { nodes: outNodes, edges: outEdges };
}

/** True if a definition (transitively) contains a feedback node — so the UI knows to
 *  offer the simulation transport when such a def is instantiated. */
export function defHasFeedback(def: SubgraphDef, defs: DefLibrary, seen: ReadonlySet<string> = new Set()): boolean {
  if (seen.has(def.name)) return false;
  const next = new Set([...seen, def.name]);
  for (const n of def.nodes) {
    if ((n.data as { opName?: string }).opName === "feedback") return true;
    if (isInstanceNode(n)) {
      const d = defs[instanceDefName(n)];
      if (d && defHasFeedback(d, defs, next)) return true;
    }
  }
  return false;
}
