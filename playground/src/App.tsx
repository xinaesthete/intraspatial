import type { Connection, Edge, Node, NodeTypes } from "@xyflow/react";
import { addEdge, Background, Controls, ReactFlow, useEdgesState, useNodesState } from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "@xyflow/react/dist/style.css";
import "./styles.css";
import type { EdgeChange, NodeChange } from "@xyflow/react";
import { applyEdgeChanges, applyNodeChanges } from "@xyflow/react";
import type { FieldValue, GraphMemo, SimState } from "../../src/gpu/graph";
import { createMemo, createSimState } from "../../src/gpu/graph";
import type { NodeData } from "./buildGraph";
import { advanceNode, graphHasFeedback, runNode } from "./buildGraph";
import { CommandPalette } from "./CommandPalette";
import type { Example } from "./examples";
import { EXAMPLES } from "./examples";
import { registerExtraOps } from "./extraOps";
import { FieldTooltip } from "./FieldTooltip";
import { GroupNode } from "./GroupNode";
import type { GroupData, IOType } from "./grouping";
import {
  addInterface,
  createGroup,
  deriveDisplay,
  groupPorts,
  isGroupNode,
  isOpNode,
  ROOT_SCOPE,
  resolveGroupOutput,
  scopeOf,
  ungroup,
} from "./grouping";
import { HelpTooltip } from "./HelpTooltip";
import { InstanceNode } from "./InstanceNode";
import { InterfaceNode } from "./InterfaceNode";
import { MathTex } from "./Math";
import { OpNode } from "./OpNode";
import { CATEGORY_ORDER } from "./opMeta";
import { PortHoverContext } from "./PortHover";
import { Preview } from "./Preview";
import { kindColor } from "./portKinds";
import type { NodeSpec } from "./specs";
import { defaultParamsFor, getSpec, listOpSpecs, listSourceSpecs } from "./specs";
import type { DefLibrary } from "./subgraphs";
import {
  defHasFeedback,
  defNameOfScope,
  defScopeId,
  expandInstances,
  instanceDefName,
  instancePorts,
  instantiate,
  isDefScope,
  isInstanceNode,
  promoteToDef,
} from "./subgraphs";

// Register the element-algebra + wavelet op packs synchronously, before any render,
// into the same registry the palette reads (see extraOps.ts for why not the async path).
registerExtraOps();

const nodeTypes: NodeTypes = { op: OpNode, group: GroupNode, instance: InstanceNode, input: InterfaceNode, output: InterfaceNode };

function mkNode(id: string, opName: string, x: number, y: number): Node {
  const spec = getSpec(opName);
  const data: NodeData = { opName, params: defaultParamsFor(spec) };
  return { id, type: "op", position: { x, y }, data: data as unknown as Record<string, unknown> };
}

/** Group node specs by category, filtered by a query (matches label/name/category),
 *  ordered by CATEGORY_ORDER then alphabetically. Empty categories are dropped. */
function groupByCategory(specs: NodeSpec[], filter: string): { category: string; specs: NodeSpec[] }[] {
  const f = filter.trim().toLowerCase();
  const match = (s: NodeSpec) =>
    !f || s.label.toLowerCase().includes(f) || s.name.toLowerCase().includes(f) || s.category.toLowerCase().includes(f);
  const byCat = new Map<string, NodeSpec[]>();
  for (const s of specs) {
    if (!match(s)) continue;
    const arr = byCat.get(s.category) ?? [];
    arr.push(s);
    byCat.set(s.category, arr);
  }
  const rank = (c: string) => {
    const i = CATEGORY_ORDER.indexOf(c);
    return i < 0 ? CATEGORY_ORDER.length : i;
  };
  return [...byCat.keys()].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b)).map((c) => ({ category: c, specs: byCat.get(c) ?? [] }));
}

// Default example: blob clusters -> KDE density -> Getis-Ord hotspots.
const initialNodes: Node[] = [
  mkNode("s1", "blobPoints", 20, 160),
  mkNode("n1", "splatDensity", 250, 120),
  mkNode("n2", "getisOrd", 500, 120),
];
const initialEdges: Edge[] = [
  { id: "e1", source: "s1", sourceHandle: "points", target: "n1", targetHandle: "points" },
  { id: "e2", source: "n1", sourceHandle: "density", target: "n2", targetHandle: "grid" },
];

export default function App() {
  const [nodes, setNodes] = useNodesState(initialNodes);
  const [edges, setEdges] = useEdgesState(initialEdges);
  const [selectedId, setSelectedId] = useState<string | null>("n2");
  const [selectedPort, setSelectedPort] = useState<string | null>(null);
  const [value, setValue] = useState<FieldValue | null>(null);
  const [stale, setStale] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [live, setLive] = useState(false);
  const [playing, setPlaying] = useState(false);
  const idSeq = useRef(100);
  const memo = useRef<GraphMemo>(createMemo());
  const sim = useRef<SimState>(createSimState());
  const rafRef = useRef<number | undefined>(undefined);
  const liveRef = useRef(live);
  liveRef.current = live;

  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [path, setPath] = useState<string[]>([ROOT_SCOPE]);
  const scope = path[path.length - 1];
  if (scope === undefined) throw new Error(`Couldn't resolve scope, expected path with at least ['${ROOT_SCOPE}']`);
  // Reusable named subgraph definitions (live-linked: editing one updates all instances).
  const [defs, setDefs] = useState<DefLibrary>({});
  // Palette filtering + the `/` command palette (insert a node at the cursor).
  const [paletteFilter, setPaletteFilter] = useState("");
  const [cmdOpen, setCmdOpen] = useState(false);
  const flowRef = useRef<{ screenToFlowPosition: (p: { x: number; y: number }) => { x: number; y: number } } | null>(null);
  const paneScreenPos = useRef<{ x: number; y: number } | null>(null);
  // Per-port values captured during the last run (key "nodeId:port"), driving the
  // port/edge hover tooltip's data view.
  const portValues = useRef<Map<string, FieldValue>>(new Map());
  const [inspect, setInspect] = useState<{ title: string; kind: string; value?: FieldValue; rect: DOMRect } | null>(null);
  const inspectTimer = useRef<number | undefined>(undefined);
  // All insertable node specs (sources + ops), grouped for the palette + command list.
  const allSpecs = useMemo(() => [...listSourceSpecs(), ...listOpSpecs()], []);
  // Foldable palette categories + a rich hover tooltip (description + math).
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [hoverHelp, setHoverHelp] = useState<{ spec: NodeSpec; rect: DOMRect } | null>(null);
  const hoverTimer = useRef<number | undefined>(undefined);
  const toggleCategory = useCallback((cat: string) => {
    setCollapsed((s) => {
      const n = new Set(s);
      if (n.has(cat)) n.delete(cat);
      else n.add(cat);
      return n;
    });
  }, []);
  const showHelp = useCallback((spec: NodeSpec, el: HTMLElement) => {
    window.clearTimeout(hoverTimer.current);
    const rect = el.getBoundingClientRect();
    hoverTimer.current = window.setTimeout(() => setHoverHelp({ spec, rect }), 220);
  }, []);
  const hideHelp = useCallback(() => {
    window.clearTimeout(hoverTimer.current);
    setHoverHelp(null);
  }, []);

  // Port/edge type+data inspector. An output port shows its own value; an input port
  // shows the value of the edge feeding it; an edge shows its source-port value.
  const portValueAt = useCallback(
    (nodeId: string, port: string, isInput: boolean): FieldValue | undefined => {
      if (!isInput) return portValues.current.get(`${nodeId}:${port}`);
      const edge = edges.find((ed) => ed.target === nodeId && ed.targetHandle === port);
      return edge ? portValues.current.get(`${edge.source}:${edge.sourceHandle}`) : undefined;
    },
    [edges],
  );
  const portHover = useMemo(
    () => ({
      onPortEnter: (nodeId: string, port: string, isInput: boolean, kind: string, rect: DOMRect) => {
        window.clearTimeout(inspectTimer.current);
        const value = portValueAt(nodeId, port, isInput);
        inspectTimer.current = window.setTimeout(() => setInspect({ title: port, kind, value, rect }), 120);
      },
      onPortLeave: () => {
        window.clearTimeout(inspectTimer.current);
        setInspect(null);
      },
    }),
    [portValueAt],
  );
  const onEdgeEnter = useCallback(
    (e: React.MouseEvent, edge: Edge) => {
      const srcNode = nodes.find((n) => n.id === edge.source);
      const opName = srcNode ? (srcNode.data as unknown as NodeData).opName : undefined;
      let kind = "?";
      try {
        if (opName) kind = getSpec(opName).outputs.find((o) => o.name === edge.sourceHandle)?.kind ?? "?";
      } catch {
        /* non-op node (instance/interface) — leave kind unknown */
      }
      const x = e.clientX,
        y = e.clientY;
      const rect = { right: x, left: x, top: y, bottom: y, width: 0, height: 0, x, y } as DOMRect;
      window.clearTimeout(inspectTimer.current);
      setInspect({ title: edge.sourceHandle ?? "out", kind, value: portValues.current.get(`${edge.source}:${edge.sourceHandle}`), rect });
    },
    [nodes],
  );
  const onEdgeLeave = useCallback(() => setInspect(null), []);

  const hasFeedback = useMemo(() => graphHasFeedback(nodes) || Object.values(defs).some((d) => defHasFeedback(d, defs)), [nodes, defs]);
  const inflight = useRef(false);
  const pending = useRef(false);

  // The "active container" is what the canvas edits: either the top-level flat graph,
  // or — when we've navigated into a reusable-subgraph definition — that definition's
  // interior. The governing def is the LAST `def:` segment on the path; nested groups
  // below it still live in that def's container.
  const activeDefName = useMemo(() => {
    for (let i = path.length - 1; i >= 0; i--) {
      const seg = path[i];
      if (seg !== undefined && isDefScope(seg)) return defNameOfScope(seg);
    }
    return null;
  }, [path]);
  const containerNodes = activeDefName ? (defs[activeDefName]?.nodes ?? []) : nodes;
  const containerEdges = activeDefName ? (defs[activeDefName]?.edges ?? []) : edges;

  const setContainerNodes = useCallback(
    (updater: (prev: Node[]) => Node[]) => {
      if (activeDefName) {
        setDefs((ds) => {
          const d = ds[activeDefName];
          return d ? { ...ds, [activeDefName]: { ...d, nodes: updater(d.nodes) } } : ds;
        });
      } else setNodes(updater);
    },
    [activeDefName, setNodes],
  );
  const setContainerEdges = useCallback(
    (updater: (prev: Edge[]) => Edge[]) => {
      if (activeDefName) {
        setDefs((ds) => {
          const d = ds[activeDefName];
          return d ? { ...ds, [activeDefName]: { ...d, edges: updater(d.edges) } } : ds;
        });
      } else setEdges(updater);
    },
    [activeDefName, setEdges],
  );

  // The nodes/edges React Flow renders are derived per current scope from the active
  // container; grouping/instancing never rewrites the logical graph.
  //
  // deriveDisplay allocates fresh objects for group/instance/interface nodes (their
  // ports/members are derived), so a naive recompute would hand React Flow a brand-new
  // identity for every such node on every render. During a marquee drag that re-render
  // storm makes RF thrash — selection flickers across nodes (most visibly with the two
  // instance nodes in the Hotspots example). We stabilise identity: a node keeps its
  // previous object reference whenever its derived content (position/type/selected/data)
  // is unchanged, so RF only sees the nodes that actually changed.
  const displayCache = useRef(new Map<string, { sig: string; node: Node }>());
  const display = useMemo(() => {
    const raw = deriveDisplay(containerNodes, containerEdges, scope, defs);
    const prev = displayCache.current;
    const next = new Map<string, { sig: string; node: Node }>();
    const nodes = raw.nodes.map((n) => {
      const sig = JSON.stringify({ p: n.position, t: n.type, s: n.selected ?? false, d: n.data });
      const hit = prev.get(n.id);
      if (hit && hit.sig === sig) {
        next.set(n.id, hit);
        return hit.node;
      }
      next.set(n.id, { sig, node: n });
      return n;
    });
    displayCache.current = next;
    return { nodes, edges: raw.edges };
  }, [containerNodes, containerEdges, scope, defs]);

  // Edges never cross scopes, so a connection drawn in the current scope is a real
  // logical edge as-is (a group-node port handle is its interface-node id; the
  // executor's flatten resolves it). No translation needed.
  const onConnect = useCallback(
    (c: Connection) => setContainerEdges((eds) => addEdge({ ...c, id: `e${idSeq.current++}` }, eds)),
    [setContainerEdges],
  );

  // Display edges ARE the logical edges for this scope (same ids), so changes apply
  // straight through.
  const onDisplayEdgesChange = useCallback(
    (changes: EdgeChange[]) => setContainerEdges((eds) => applyEdgeChanges(changes, eds)),
    [setContainerEdges],
  );

  // Memoised + identity-stable: return the same array when the selection is unchanged
  // so React Flow's selection effect doesn't re-fire into an update loop.
  const onSelectionChange = useCallback(({ nodes: sel }: { nodes: Node[] }) => {
    setSelectedIds((prev) => {
      const ids = sel.map((n) => n.id);
      return ids.length === prev.length && ids.every((v, i) => v === prev[i]) ? prev : ids;
    });
  }, []);

  // We render DERIVED display nodes; persist position/remove/select back to the logical
  // graph. Selection MUST be applied (React Flow is controlled — dropping `select`
  // changes desyncs RF's internal selection from the nodes prop and makes the marquee
  // flicker). `dimensions` is still left to RF, which measures via ResizeObserver and
  // merges by node id; combined with the identity-stable `display` above, applying
  // `select` no longer churns node objects, so there is no recompute loop.
  const onDisplayNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const apply = changes.filter((c) => c.type === "position" || c.type === "remove" || c.type === "select");
      if (apply.length) setContainerNodes((ns) => applyNodeChanges(apply, ns));
    },
    [setContainerNodes],
  );

  // Group the current multi-selection (≥2 nodes in this scope), then navigate in.
  // Anything that carries a value — ops, one-off groups, and reusable-subgraph
  // instances — can be grouped; only the scope's own boundary interface nodes can't.
  const groupableIds = useMemo(
    () =>
      selectedIds.filter((id) => {
        const n = containerNodes.find((x) => x.id === id);
        return !!n && scopeOf(n) === scope && (isOpNode(n) || isGroupNode(n) || isInstanceNode(n));
      }),
    [selectedIds, containerNodes, scope],
  );
  const addIONode = useCallback(
    (io: IOType) => {
      if (scope === ROOT_SCOPE) return;
      const id = `io#${idSeq.current++}`;
      setContainerNodes((ns) => ns.concat(addInterface(scope, io, id, { x: io === "input" ? 40 : 480, y: 80 + (ns.length % 5) * 60 })));
    },
    [scope, setContainerNodes],
  );
  const groupSelection = useCallback(() => {
    if (groupableIds.length < 2) return;
    const gid = `group#${idSeq.current++}`;
    const r = createGroup(containerNodes, containerEdges, groupableIds, gid, "Group", scope);
    setContainerNodes(() => r.nodes);
    setContainerEdges(() => r.edges);
    setSelectedId(gid);
    setSelectedPort(null);
    setValue(null);
    setStale(false);
  }, [groupableIds, containerNodes, containerEdges, scope, setContainerNodes, setContainerEdges]);

  const enterGroup = useCallback((groupId: string) => {
    setPath((p) => [...p, groupId]);
    setSelectedId(null);
    setSelectedPort(null);
    setValue(null);
  }, []);

  // Open a reusable subgraph instance → edit its SHARED definition (live-linked).
  const enterInstance = useCallback(
    (instId: string) => {
      const inst = containerNodes.find((n) => n.id === instId);
      if (!inst || !isInstanceNode(inst)) return;
      setPath((p) => [...p, defScopeId(instanceDefName(inst))]);
      setSelectedId(null);
      setSelectedPort(null);
      setValue(null);
    },
    [containerNodes],
  );

  const navigateTo = useCallback((index: number) => {
    setPath((p) => p.slice(0, index + 1));
    setSelectedId(null);
    setValue(null);
  }, []);

  const dissolveGroup = useCallback(
    (groupId: string) => {
      const r = ungroup(containerNodes, containerEdges, groupId);
      setContainerNodes(() => r.nodes);
      setContainerEdges(() => r.edges);
      setPath((p) => (p.includes(groupId) ? p.slice(0, p.indexOf(groupId)) : p)); // pop if we're inside it
      setSelectedId(null);
      setValue(null);
    },
    [containerNodes, containerEdges, setContainerNodes, setContainerEdges],
  );

  // Promote a group (or the current ≥2-op selection) into a reusable named subgraph.
  const promoteToReusable = useCallback(() => {
    let workNodes = containerNodes;
    let workEdges = containerEdges;
    let groupId: string;
    const sel = containerNodes.find((n) => n.id === selectedId);
    if (sel && isGroupNode(sel)) {
      groupId = sel.id;
    } else if (groupableIds.length >= 2) {
      groupId = `group#${idSeq.current++}`;
      const g = createGroup(workNodes, workEdges, groupableIds, groupId, "Group", scope);
      workNodes = g.nodes;
      workEdges = g.edges;
    } else {
      return;
    }
    const name = window.prompt("Name this reusable subgraph:", "MySubgraph");
    if (!name) return;
    if (defs[name]) {
      window.alert(`A reusable subgraph named "${name}" already exists.`);
      return;
    }
    // nb, this is not the same as SubgraphDef, would be good to be clearer on that.
    let result: ReturnType<typeof promoteToDef>;
    try {
      result = promoteToDef(workNodes, workEdges, defs, groupId, name, name);
    } catch (e) {
      window.alert(e instanceof Error ? e.message : String(e));
      return;
    }
    if (activeDefName) {
      // The container is itself a def: write its updated interior AND the new def.
      // biome-ignore lint/style/noNonNullAssertion: ds[activeDefName] unchecked but should be correct
      setDefs((ds) => ({ ...result.defs, [activeDefName]: { ...ds[activeDefName]!, nodes: result.nodes, edges: result.edges } }));
    } else {
      setNodes(result.nodes);
      setEdges(result.edges);
      setDefs(result.defs);
    }
    setSelectedId(groupId); // the group node id is reused as the instance id
    setSelectedPort(null);
    setValue(null);
    setStale(false);
  }, [containerNodes, containerEdges, defs, selectedId, groupableIds, scope, activeDefName, setNodes, setEdges]);

  // Drop another instance of an existing definition into the current scope.
  const instantiateDef = useCallback(
    (name: string) => {
      const id = `inst${idSeq.current++}`;
      const node = instantiate(name, defs[name]?.label ?? name, scope, id, {
        x: 160 + (containerNodes.length % 5) * 40,
        y: 220 + (containerNodes.length % 4) * 40,
      });
      setContainerNodes((ns) => ns.concat(node));
    },
    [defs, scope, containerNodes.length, setContainerNodes],
  );

  // Rename a definition's shared label (updates every instance's default display).
  const renameDef = useCallback(
    // biome-ignore lint/style/noNonNullAssertion: ds[name] unchecked but should be safe
    (name: string, label: string) => setDefs((ds) => (ds[name] ? { ...ds, [name]: { ...ds[name]!, label } } : ds)),
    [],
  );

  const onNodeDoubleClick = useCallback(
    (_: unknown, node: Node) => {
      if (isInstanceNode(node)) enterInstance(node.id);
      else if (isGroupNode(node)) enterGroup(node.id);
    },
    [enterGroup, enterInstance],
  );

  // Resolve the pull target against the EXPANDED top-level graph (the same one
  // buildGraph runs): a group/instance node stands in for an internal (node, port).
  // Instances become groups whose port ids are namespaced `instId/portId`.
  const resolveSink = useCallback((): { id: string; port?: string } => {
    if (!selectedId) throw new Error("Unexpted falsey selectedId");
    const sel = nodes.find((n) => n.id === selectedId);
    if (sel && (isGroupNode(sel) || isInstanceNode(sel))) {
      const exp = expandInstances(nodes, edges, defs);
      let portId = selectedPort ?? undefined;
      if (isInstanceNode(sel)) {
        const def = defs[instanceDefName(sel)];
        const base = selectedPort ?? (def ? instancePorts(def, defs).outputs[0]?.id : undefined);
        portId = base ? `${sel.id}/${base}` : undefined;
      }
      const r = resolveGroupOutput(exp.nodes, exp.edges, sel.id, portId);
      if (r) return { id: r.node, port: r.port };
    }
    return { id: selectedId, port: selectedPort ?? undefined };
  }, [nodes, edges, defs, selectedId, selectedPort]);

  // Rename a group's label or one of its boundary ports (stored as portLabels).
  // Rename any node's label (a group, or an interface node — which IS a group port).
  const renameNode = useCallback(
    (id: string, label: string) => {
      setContainerNodes((ns) =>
        ns.map((n) => (n.id === id ? { ...n, data: { ...(n.data as object), label } as Record<string, unknown> } : n)),
      );
    },
    [setContainerNodes],
  );

  const addNode = useCallback(
    (opName: string, pos?: { x: number; y: number }) => {
      const id = `x${idSeq.current++}`;
      const x = pos ? pos.x : 120 + (containerNodes.length % 5) * 40;
      const y = pos ? pos.y : 360 + (containerNodes.length % 4) * 30;
      const n = mkNode(id, opName, x, y);
      (n.data as { group?: string }).group = scope; // new node lives in the current scope
      setContainerNodes((ns) => ns.concat(n));
    },
    [containerNodes.length, scope, setContainerNodes],
  );

  // Insert a node chosen in the command palette at the last cursor position over the
  // canvas (converted to flow coords), falling back to the default cascade.
  const insertFromCommand = useCallback(
    (opName: string) => {
      const screen = paneScreenPos.current;
      const pos = screen && flowRef.current ? flowRef.current.screenToFlowPosition(screen) : undefined;
      addNode(opName, pos);
      setCmdOpen(false);
    },
    [addNode],
  );

  // `/` opens the command palette (unless typing in a field).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "/" || cmdOpen) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      e.preventDefault();
      setCmdOpen(true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cmdOpen]);

  // Breadcrumb label for a scope segment: "Main", a definition's label, or a (possibly
  // nested) group node's label found in whichever container holds it.
  const segLabel = useCallback(
    (seg: string): string => {
      if (seg === ROOT_SCOPE) return "Main";
      if (isDefScope(seg)) return defs[defNameOfScope(seg)]?.label ?? "subgraph";
      const all = [nodes, ...Object.values(defs).map((d) => d.nodes)].flat();
      return (all.find((n) => n.id === seg)?.data as unknown as GroupData)?.label ?? "group";
    },
    [nodes, defs],
  );

  const selected = useMemo(() => containerNodes.find((n) => n.id === selectedId) ?? null, [containerNodes, selectedId]);
  const selectedGroup = selected && isGroupNode(selected) ? (selected.data as unknown as GroupData) : null;
  const selectedInstance = selected && isInstanceNode(selected) ? selected : null;
  const selectedInterface = selected && (selected.type === "input" || selected.type === "output") ? selected : null;
  const selectedSpec = selected && isOpNode(selected) ? getSpec((selected.data as unknown as NodeData).opName) : null;
  // Group/instance boundary + member count are derived (not stored on the logical node).
  const selectedGroupBoundary = useMemo(() => {
    if (!selected) return { inputs: [], outputs: [] };
    if (selectedInstance) {
      const d = defs[instanceDefName(selected)];
      return d ? instancePorts(d, defs) : { inputs: [], outputs: [] };
    }
    if (selectedGroup) return groupPorts(containerNodes, containerEdges, selected.id, defs);
    return { inputs: [], outputs: [] };
  }, [selectedGroup, selectedInstance, selected, containerNodes, containerEdges, defs]);
  const selectedGroupMembers = useMemo(() => {
    if (!selected) return 0;
    if (selectedInstance) {
      const d = defs[instanceDefName(selected)];
      return d ? d.nodes.filter(isOpNode).length : 0;
    }
    return containerNodes.filter((n) => scopeOf(n) === selected.id).length;
  }, [selected, selectedInstance, containerNodes, defs]);

  const setParam = useCallback(
    (key: string, v: unknown) => {
      if (!selectedId) return;
      setContainerNodes((ns) =>
        ns.map((n) => {
          if (n.id !== selectedId) return n;
          const d = n.data as unknown as NodeData;
          return { ...n, data: { ...d, params: { ...d.params, [key]: v } } as unknown as Record<string, unknown> };
        }),
      );
      if (!liveRef.current) setStale(true); // keep the last preview visible, mark it stale
    },
    [selectedId, setContainerNodes],
  );

  const runRef = useRef<() => void>(() => {});
  const run = useCallback(async () => {
    if (!selectedId) return;
    if (inflight.current) {
      pending.current = true;
      return;
    } // coalesce while busy
    inflight.current = true;
    setRunning(true);
    setError(null);
    try {
      const sink = resolveSink();
      const out = await runNode(nodes, edges, sink.id, sink.port, memo.current, defs, (k, v) => portValues.current.set(k, v));
      setValue(out);
      setStale(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setValue(null);
    } finally {
      inflight.current = false;
      setRunning(false);
      if (pending.current) {
        pending.current = false;
        runRef.current();
      } // run the latest
    }
  }, [nodes, edges, defs, selectedId, resolveSink]);
  runRef.current = run;

  // Signature of the run-relevant state (params + wiring + selection), excluding
  // node positions so dragging doesn't trigger a re-run.
  const sig = useMemo(
    () =>
      JSON.stringify({
        n: nodes.map((n) => ({ id: n.id, op: (n.data as unknown as NodeData).opName, p: (n.data as unknown as NodeData).params })),
        e: edges.map((e) => ({ s: e.source, sh: e.sourceHandle, t: e.target, th: e.targetHandle })),
        defs, // re-run when a referenced definition's interior changes (live-link)
        sel: selectedId,
        port: selectedPort,
      }),
    [nodes, edges, defs, selectedId, selectedPort],
  );

  // Auto-run: when live, re-pull on every signature change. We deliberately do NOT
  // trailing-debounce (that only fires after you stop dragging, an onMouseUp feel) —
  // the in-flight guard paces runs to GPU throughput and always runs the latest
  // state, so a slider drag updates continuously in real time. Disabled for feedback
  // graphs, which use the transport (play/step) below instead.
  useEffect(() => {
    sig; // this is the signal to run the effect
    if (!live || !selectedId || hasFeedback) return;
    runRef.current();
  }, [sig, live, selectedId, hasFeedback]);

  // Simulation transport: advance one tick (a graph step; the RD op may batch many
  // Euler steps inside it) and show the result.
  const step = useCallback(
    async (reset = false) => {
      if (!selectedId) return;
      try {
        const sink = resolveSink();
        const out = await advanceNode(nodes, edges, sink.id, sink.port, {
          steps: 1,
          state: sim.current,
          reset,
          defs,
          onValue: (k, v) => portValues.current.set(k, v),
        });
        setValue(out);
        setStale(false);
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setPlaying(false);
      }
    },
    [nodes, edges, defs, selectedId, resolveSink],
  );

  const resetSim = useCallback(() => {
    setPlaying(false);
    sim.current.clear();
    setValue(null);
    setError(null);
  }, []);

  const loadExample = useCallback(
    (ex: Example) => {
      setPlaying(false);
      sim.current.clear();
      setPath([ROOT_SCOPE]);
      setNodes(ex.nodes);
      setEdges(ex.edges);
      setDefs(ex.defs ?? {});
      setSelectedId(ex.sink.node);
      setSelectedPort(ex.sink.port ?? null);
      setLive(false);
      setValue(null);
      setError(null);
    },
    [setNodes, setEdges],
  );

  // The play loop: advance one tick per animation frame while `playing`. Awaiting
  // each tick paces the loop to GPU throughput. Restarts if the graph/selection
  // changes (picks up edited params).
  useEffect(() => {
    if (!playing || !selectedId || !hasFeedback) return;
    let cancelled = false;
    const tick = async () => {
      if (cancelled) return;
      try {
        const sink = resolveSink();
        const out = await advanceNode(nodes, edges, sink.id, sink.port, {
          steps: 1,
          state: sim.current,
          defs,
          onValue: (k, v) => portValues.current.set(k, v),
        });
        if (cancelled) return;
        setValue(out);
        setStale(false); // each animated frame is freshly computed, never stale
        rafRef.current = requestAnimationFrame(tick);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setPlaying(false);
      }
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, hasFeedback, selectedId, nodes, edges, defs, resolveSink]);

  return (
    <PortHoverContext.Provider value={portHover}>
      <div className="app">
        <aside className="palette">
          <h1>GPU graph composer</h1>
          <p className="muted">Wire ops, select a node, pull its output. Edges only connect matching port types.</p>
          <h2>Examples</h2>
          {EXAMPLES.map((ex) => (
            <button key={ex.label} className="palette-btn" title="Load this graph" onClick={() => loadExample(ex)}>
              {ex.label}
            </button>
          ))}
          <input
            className="palette-filter"
            placeholder="Filter nodes…  (press / to insert)"
            value={paletteFilter}
            onChange={(e) => setPaletteFilter(e.target.value)}
          />
          {groupByCategory(allSpecs, paletteFilter).map((group) => {
            // While filtering, force every matching group open.
            const isCollapsed = collapsed.has(group.category) && !paletteFilter.trim();
            return (
              <div key={group.category} className="palette-group">
                <button className="palette-group-head" onClick={() => toggleCategory(group.category)}>
                  <span className="fold">{isCollapsed ? "▸" : "▾"}</span>
                  <span className="palette-group-name">{group.category}</span>
                  <span className="palette-group-count">{group.specs.length}</span>
                </button>
                {!isCollapsed &&
                  group.specs.map((s) => (
                    <button
                      key={s.name}
                      className={`palette-btn${s.isSource ? " source" : ""}`}
                      draggable
                      onDragStart={(e) => {
                        e.dataTransfer.setData("application/op-name", s.name);
                        e.dataTransfer.effectAllowed = "move";
                        hideHelp();
                      }}
                      onClick={() => addNode(s.name)}
                      onMouseEnter={(e) => showHelp(s, e.currentTarget)}
                      onMouseLeave={hideHelp}
                    >
                      {s.label}
                    </button>
                  ))}
              </div>
            );
          })}
          {Object.keys(defs).length > 0 && (
            <>
              <h2>Subgraphs</h2>
              {Object.values(defs).map((d) => (
                <button
                  key={d.name}
                  className="palette-btn subgraph"
                  title={d.name === activeDefName ? "Can't place a subgraph inside itself" : `Place an instance of "${d.label}"`}
                  disabled={d.name === activeDefName}
                  onClick={() => instantiateDef(d.name)}
                >
                  ⬡ {d.label}
                </button>
              ))}
            </>
          )}
        </aside>

        <main
          className="canvas"
          onMouseMove={(e) => {
            paneScreenPos.current = { x: e.clientX, y: e.clientY };
          }}
        >
          <div className="canvas-toolbar">
            <div className="breadcrumb">
              {path.map((seg, i) => {
                const label = segLabel(seg);
                return (
                  <span key={seg} className="bc-item">
                    {i > 0 && <span className="bc-sep">›</span>}
                    <button
                      className={`bc-seg ${isDefScope(seg) ? "def" : ""}`}
                      disabled={i === path.length - 1}
                      onClick={() => navigateTo(i)}
                    >
                      {label}
                    </button>
                  </span>
                );
              })}
            </div>
            {groupableIds.length >= 2 && (
              <>
                <button className="tb-primary" onClick={groupSelection}>
                  ▦ Group {groupableIds.length} nodes
                </button>
                <button className="tb-primary" onClick={promoteToReusable} title="Group these nodes and save as a reusable named subgraph">
                  ⬡ Save as subgraph
                </button>
              </>
            )}
            {scope !== ROOT_SCOPE && (
              <span className="tb-group">
                <button onClick={() => addIONode("input")} title="Add an input to this subgraph">
                  ＋ Input
                </button>
                <button onClick={() => addIONode("output")} title="Add an output to this subgraph">
                  ＋ Output
                </button>
              </span>
            )}
            {activeDefName && (
              <span
                className="tb-def-badge"
                title="You are editing a shared reusable subgraph definition — changes apply to every instance"
              >
                ⬡ editing definition · live-linked
              </span>
            )}
          </div>
          <ReactFlow
            nodes={display.nodes}
            edges={display.edges}
            nodeTypes={nodeTypes}
            onNodesChange={onDisplayNodesChange}
            onEdgesChange={onDisplayEdgesChange}
            onConnect={onConnect}
            onSelectionChange={onSelectionChange}
            onNodeClick={(_, n) => {
              setSelectedId(n.id);
              setSelectedPort(null);
              setValue(null);
              setStale(false);
              setError(null);
            }}
            onNodeDoubleClick={onNodeDoubleClick}
            onEdgeMouseEnter={onEdgeEnter}
            onEdgeMouseLeave={onEdgeLeave}
            onInit={(inst) => {
              flowRef.current = inst;
            }}
            onDragOver={(e) => {
              e.preventDefault();
              e.dataTransfer.dropEffect = "move";
            }}
            onDrop={(e) => {
              e.preventDefault();
              const name = e.dataTransfer.getData("application/op-name");
              if (!name) return;
              const pos = flowRef.current?.screenToFlowPosition({ x: e.clientX, y: e.clientY });
              addNode(name, pos);
            }}
            fitView
          >
            <Background />
            <Controls />
          </ReactFlow>
        </main>

        <aside className="inspector">
          {selected && selectedInterface ? (
            <>
              <h2>{selectedInterface.type === "input" ? "▸ Input" : "Output ▸"}</h2>
              <label className="row">
                <span>name</span>
                <input
                  type="text"
                  value={(selectedInterface.data as { label: string }).label}
                  onChange={(e) => renameNode(selected.id, e.target.value)}
                  style={{ width: 130 }}
                />
              </label>
              <p className="muted">
                A {selectedInterface.type} port of this subgraph. Wire it to a member; the group node gets a matching port.
              </p>
            </>
          ) : selected && selectedInstance ? (
            <>
              <h2>⬡ Reusable subgraph</h2>
              <label className="row">
                <span>definition</span>
                <input
                  type="text"
                  value={defs[instanceDefName(selected)]?.label ?? instanceDefName(selected)}
                  onChange={(e) => renameDef(instanceDefName(selected), e.target.value)}
                  style={{ width: 130 }}
                />
              </label>
              <p className="muted">{selectedGroupMembers} nodes · live-linked · edits apply to every instance</p>
              <div className="transport-row">
                <button onClick={() => enterInstance(selected.id)}>⬡ Edit definition</button>
              </div>
              {selectedGroupBoundary.outputs.length > 1 && (
                <label className="row">
                  <span>preview</span>
                  <select value={selectedPort ?? selectedGroupBoundary.outputs[0]?.id} onChange={(e) => setSelectedPort(e.target.value)}>
                    {selectedGroupBoundary.outputs.map((o) => (
                      <option key={o.id} value={o.id}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {activeDefName ? (
                <p className="muted">
                  Open an instance in <strong>Main</strong> to run — a definition has no bound inputs.
                </p>
              ) : (
                <>
                  <button className="run-btn" onClick={run} disabled={running}>
                    {running ? "Running…" : "▶ Run / pull"}
                  </button>
                  <Preview value={value} error={error} stale={stale} />
                </>
              )}
            </>
          ) : selected && selectedGroup ? (
            <>
              <h2>▦ Group</h2>
              <label className="row">
                <span>name</span>
                <input
                  type="text"
                  value={selectedGroup.label}
                  onChange={(e) => renameNode(selected.id, e.target.value)}
                  style={{ width: 130 }}
                />
              </label>
              <p className="muted">{selectedGroupMembers} nodes · double-click to open</p>
              <div className="transport-row">
                <button onClick={() => enterGroup(selected.id)}>▦ Open</button>
                <button onClick={() => dissolveGroup(selected.id)}>✕ Ungroup</button>
              </div>
              <div className="transport-row">
                <button onClick={promoteToReusable} title="Save this group as a reusable named subgraph (live-linked)">
                  ⬡ Save as reusable subgraph
                </button>
              </div>
              {[...selectedGroupBoundary.inputs, ...selectedGroupBoundary.outputs].length > 0 && (
                <div className="params">
                  <h2>Ports</h2>
                  {selectedGroupBoundary.inputs.map((p) => (
                    <label className="row" key={p.id}>
                      <span>
                        in <em style={{ color: kindColor(p.kind), fontStyle: "normal" }}>{p.kind}</em>
                      </span>
                      <input type="text" value={p.label} onChange={(e) => renameNode(p.id, e.target.value)} style={{ width: 110 }} />
                    </label>
                  ))}
                  {selectedGroupBoundary.outputs.map((p) => (
                    <label className="row" key={p.id}>
                      <span>
                        out <em style={{ color: kindColor(p.kind), fontStyle: "normal" }}>{p.kind}</em>
                      </span>
                      <input type="text" value={p.label} onChange={(e) => renameNode(p.id, e.target.value)} style={{ width: 110 }} />
                    </label>
                  ))}
                </div>
              )}
              {selectedGroupBoundary.outputs.length > 1 && (
                <label className="row">
                  <span>preview</span>
                  <select value={selectedPort ?? selectedGroupBoundary.outputs[0]?.id} onChange={(e) => setSelectedPort(e.target.value)}>
                    {selectedGroupBoundary.outputs.map((o) => (
                      <option key={o.id} value={o.id}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {activeDefName ? (
                <p className="muted">
                  Open an instance in <strong>Main</strong> to run — a definition has no bound inputs.
                </p>
              ) : (
                <>
                  <button className="run-btn" onClick={run} disabled={running}>
                    {running ? "Running…" : "▶ Run / pull"}
                  </button>
                  <Preview value={value} error={error} stale={stale} />
                </>
              )}
            </>
          ) : selected && selectedSpec ? (
            <>
              <h2>{selectedSpec.label}</h2>
              {selectedSpec.describe && <p className="muted">{selectedSpec.describe}</p>}
              {selectedSpec.help && (
                <div className="op-help">
                  {selectedSpec.help.detail && <p className="op-help-detail">{selectedSpec.help.detail}</p>}
                  {selectedSpec.help.math && (
                    <div className="op-help-math">
                      <MathTex tex={selectedSpec.help.math} />
                    </div>
                  )}
                </div>
              )}
              <div className="params">
                {selectedSpec.params.map((p) => (
                  <ParamControl
                    key={p.name}
                    spec={p}
                    value={(selected.data as unknown as NodeData).params[p.name]}
                    onChange={(v) => setParam(p.name, v)}
                  />
                ))}
                {selectedSpec.params.length === 0 && <span className="muted">no parameters</span>}
              </div>

              {selectedSpec.outputs.length > 1 && (
                <label className="row">
                  <span>output</span>
                  <select value={selectedPort ?? selectedSpec.outputs[0]?.name} onChange={(e) => setSelectedPort(e.target.value)}>
                    {selectedSpec.outputs.map((o) => (
                      <option key={o.name} value={o.name}>
                        {o.name}
                      </option>
                    ))}
                  </select>
                </label>
              )}

              {activeDefName ? (
                <p className="muted">
                  Editing the <strong>{segLabel(defScopeId(activeDefName))}</strong> definition. Open an instance in <strong>Main</strong>{" "}
                  to run — a definition has no bound inputs.
                </p>
              ) : hasFeedback ? (
                <>
                  <div className="transport">
                    <button className={`run-btn ${playing ? "live" : ""}`} onClick={() => setPlaying((p) => !p)}>
                      {playing ? "⏸ Pause" : "▶ Play"}
                    </button>
                    <div className="transport-row">
                      <button onClick={() => step(false)} disabled={playing} title="Advance one tick">
                        ⏭ Step
                      </button>
                      <button onClick={resetSim} title="Reset to the seed">
                        ⟲ Reset
                      </button>
                    </div>
                    <p className="muted">Feedback loop — runs over time. One tick = one graph step.</p>
                  </div>
                  <Preview value={value} error={error} stale={stale} />
                </>
              ) : (
                <>
                  <label className="row live-row" title="Re-pull automatically when params or wiring change">
                    <span>Auto-run (live)</span>
                    <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} />
                  </label>
                  <button className={`run-btn ${live ? "live" : ""}`} onClick={run} disabled={running}>
                    {running ? "Running…" : live ? "● Live — re-running on change" : "▶ Run / pull"}
                  </button>
                  <Preview value={value} error={error} stale={stale} />
                </>
              )}
            </>
          ) : (
            <p className="muted">Click a node to edit its parameters and pull its output.</p>
          )}
        </aside>
        {cmdOpen && (
          <CommandPalette
            items={allSpecs.map((s) => ({ name: s.name, label: s.label, category: s.category, describe: s.describe }))}
            onPick={insertFromCommand}
            onClose={() => setCmdOpen(false)}
          />
        )}
        {hoverHelp && <HelpTooltip spec={hoverHelp.spec} rect={hoverHelp.rect} />}
        {inspect && <FieldTooltip title={inspect.title} kind={inspect.kind} value={inspect.value} rect={inspect.rect} />}
      </div>
    </PortHoverContext.Provider>
  );
}

function ParamControl({
  spec,
  value,
  onChange,
}: {
  spec: import("../../src/gpu/graph").ParamSpec;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const v = value ?? spec.default;
  if (spec.type === "bool") {
    return (
      <label className="row">
        <span title={spec.describe}>{spec.name}</span>
        <input type="checkbox" checked={Boolean(v)} onChange={(e) => onChange(e.target.checked)} />
      </label>
    );
  }
  if (spec.type === "enum") {
    return (
      <label className="row">
        <span title={spec.describe}>{spec.name}</span>
        <select value={String(v)} onChange={(e) => onChange(e.target.value)}>
          {(spec.options ?? []).map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      </label>
    );
  }
  const step = spec.step ?? (spec.type === "int" ? 1 : 0.01);
  return (
    <label className="row">
      <span title={spec.describe}>{spec.name}</span>
      <span className="num">
        <input
          type="range"
          min={spec.min ?? 0}
          max={spec.max ?? 1}
          step={step}
          value={Number(v)}
          onChange={(e) => onChange(spec.type === "int" ? Math.round(+e.target.value) : +e.target.value)}
        />
        <input
          type="number"
          min={spec.min}
          max={spec.max}
          step={step}
          value={Number(v)}
          onChange={(e) => onChange(spec.type === "int" ? Math.round(+e.target.value) : +e.target.value)}
        />
      </span>
    </label>
  );
}
