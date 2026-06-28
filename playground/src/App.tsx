import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  addEdge,
  useNodesState,
  useEdgesState,
} from "@xyflow/react";
import type { Connection, Edge, Node, NodeTypes } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import "./styles.css";
import type { FieldValue, GraphMemo, SimState } from "../../src/gpu/graph";
import { createMemo, createSimState } from "../../src/gpu/graph";
import { applyEdgeChanges, applyNodeChanges } from "@xyflow/react";
import type { EdgeChange, NodeChange } from "@xyflow/react";
import { OpNode } from "./OpNode";
import { GroupNode } from "./GroupNode";
import { GroupPortNode } from "./GroupPortNode";
import { advanceNode, graphHasFeedback, runNode } from "./buildGraph";
import type { NodeData } from "./buildGraph";
import {
  ROOT_SCOPE,
  boundaryOf,
  createGroup,
  decodePort,
  deriveDisplay,
  isGroupNode,
  isPortStub,
  resolveGroupOutput,
  scopeOf,
  ungroup,
} from "./grouping";
import type { GroupData } from "./grouping";
import { defaultParamsFor, getSpec, listOpSpecs, listSourceSpecs } from "./specs";
import { EXAMPLES } from "./examples";
import type { Example } from "./examples";
import { Preview } from "./Preview";

const nodeTypes: NodeTypes = { op: OpNode, group: GroupNode, groupPort: GroupPortNode };

function mkNode(id: string, opName: string, x: number, y: number): Node {
  const spec = getSpec(opName);
  const data: NodeData = { opName, params: defaultParamsFor(spec) };
  return { id, type: "op", position: { x, y }, data: data as unknown as Record<string, unknown> };
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
  const scope = path[path.length - 1]!;
  const hasFeedback = useMemo(() => graphHasFeedback(nodes), [nodes]);
  const inflight = useRef(false);
  const pending = useRef(false);

  // The nodes/edges React Flow renders are derived per current scope from the flat
  // logical graph; grouping never rewrites the logical graph.
  const display = useMemo(() => deriveDisplay(nodes, edges, scope), [nodes, edges, scope]);

  // Map a display endpoint back to a real (node, port): group-node ports decode to
  // their internal member; stubs aren't real nodes → null.
  const resolveEndpoint = useCallback(
    (id: string | null, handle: string | null | undefined): { node: string; port: string } | null => {
      const n = nodes.find((x) => x.id === id);
      if (!n) return null;
      if (isGroupNode(n)) { const d = decodePort(handle); return d ? { node: d.node, port: d.port } : null; }
      return { node: id!, port: handle ?? "" };
    },
    [nodes],
  );

  const onConnect = useCallback(
    (c: Connection) => {
      const src = resolveEndpoint(c.source, c.sourceHandle);
      const tgt = resolveEndpoint(c.target, c.targetHandle);
      if (!src || !tgt) return; // boundary stub / unresolved
      setEdges((eds) => addEdge({ id: `e${idSeq.current++}`, source: src.node, sourceHandle: src.port, target: tgt.node, targetHandle: tgt.port }, eds));
    },
    [resolveEndpoint, setEdges],
  );

  // Display edges carry ids "d:<realId>"; translate removals/selection to the logical
  // edge.
  const onDisplayEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      setEdges((eds) => applyEdgeChanges(changes.map((c) => ("id" in c && typeof c.id === "string" && c.id.startsWith("d:") ? { ...c, id: c.id.slice(2) } : c)), eds));
    },
    [setEdges],
  );

  // Memoised + identity-stable: return the same array when the selection is unchanged
  // so React Flow's selection effect doesn't re-fire into an update loop.
  const onSelectionChange = useCallback(({ nodes: sel }: { nodes: Node[] }) => {
    setSelectedIds((prev) => {
      const ids = sel.map((n) => n.id);
      return ids.length === prev.length && ids.every((v, i) => v === prev[i]) ? prev : ids;
    });
  }, []);

  // We render DERIVED display nodes; only persist position/remove back to the logical
  // graph. Letting dimension/selection changes flow into the logical array would
  // recompute the display, re-emit changes, and loop — React Flow owns those itself.
  const onDisplayNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const apply = changes.filter((c) => c.type === "position" || c.type === "remove");
      if (apply.length) setNodes((ns) => applyNodeChanges(apply, ns));
    },
    [setNodes],
  );

  // Group the current multi-selection (≥2 op nodes in this scope), then navigate in.
  const groupableIds = useMemo(
    () => selectedIds.filter((id) => { const n = nodes.find((x) => x.id === id); return n && !isGroupNode(n) && !isPortStub(n) && scopeOf(n) === scope; }),
    [selectedIds, nodes, scope],
  );
  const groupSelection = useCallback(() => {
    if (groupableIds.length < 2) return;
    const gid = `group#${idSeq.current++}`;
    const r = createGroup(nodes, edges, groupableIds, gid, "Group", scope);
    setNodes(r.nodes);
    setEdges(r.edges);
    setSelectedId(gid);
    setSelectedPort(null);
    setValue(null);
    setStale(false);
  }, [groupableIds, nodes, edges, scope, setNodes, setEdges]);

  const enterGroup = useCallback((groupId: string) => {
    setPath((p) => [...p, groupId]);
    setSelectedId(null);
    setSelectedPort(null);
    setValue(null);
  }, []);

  const navigateTo = useCallback((index: number) => {
    setPath((p) => p.slice(0, index + 1));
    setSelectedId(null);
    setValue(null);
  }, []);

  const dissolveGroup = useCallback(
    (groupId: string) => {
      const r = ungroup(nodes, edges, groupId);
      setNodes(r.nodes);
      setEdges(r.edges);
      setPath((p) => (p.includes(groupId) ? p.slice(0, p.indexOf(groupId)) : p)); // pop if we're inside it
      setSelectedId(null);
      setValue(null);
    },
    [nodes, edges, setNodes, setEdges],
  );

  const onNodeDoubleClick = useCallback(
    (_: unknown, node: Node) => { if (isGroupNode(node)) enterGroup(node.id); },
    [enterGroup],
  );

  // Resolve the pull target: a group node stands in for an internal (node, port).
  const resolveSink = useCallback((): { id: string; port?: string } => {
    const sel = nodes.find((n) => n.id === selectedId);
    if (sel && isGroupNode(sel)) {
      const r = resolveGroupOutput(nodes, edges, sel.id, selectedPort ?? undefined);
      if (r) return { id: r.node, port: r.port };
    }
    return { id: selectedId!, port: selectedPort ?? undefined };
  }, [nodes, edges, selectedId, selectedPort]);

  // Rename a group's label or one of its boundary ports (stored as portLabels).
  const renameGroup = useCallback(
    (groupId: string, change: { label?: string; portId?: string; portLabel?: string }) => {
      setNodes((ns) =>
        ns.map((n) => {
          if (n.id !== groupId) return n;
          const d = n.data as unknown as GroupData;
          const next: GroupData = { ...d, portLabels: { ...(d.portLabels ?? {}) } };
          if (change.label !== undefined) next.label = change.label;
          if (change.portId !== undefined && change.portLabel !== undefined) next.portLabels![change.portId] = change.portLabel;
          return { ...n, data: next as unknown as Record<string, unknown> };
        }),
      );
    },
    [setNodes],
  );

  const addNode = useCallback(
    (opName: string) => {
      const id = `x${idSeq.current++}`;
      const n = mkNode(id, opName, 120 + (nodes.length % 5) * 40, 360 + (nodes.length % 4) * 30);
      (n.data as { group?: string }).group = scope; // new node lives in the current scope
      setNodes((ns) => ns.concat(n));
    },
    [nodes.length, scope, setNodes],
  );

  const selected = useMemo(() => nodes.find((n) => n.id === selectedId) ?? null, [nodes, selectedId]);
  const selectedGroup = selected && isGroupNode(selected) ? (selected.data as unknown as GroupData) : null;
  const selectedSpec = selected && !selectedGroup ? getSpec((selected.data as unknown as NodeData).opName) : null;
  // Group boundary + member count are derived (not stored on the logical node).
  const selectedGroupBoundary = useMemo(
    () => (selectedGroup && selected ? boundaryOf(nodes, edges, selected.id) : { inputs: [], outputs: [] }),
    [selectedGroup, selected, nodes, edges],
  );
  const selectedGroupMembers = useMemo(
    () => (selected ? nodes.filter((n) => scopeOf(n) === selected.id).length : 0),
    [selected, nodes],
  );

  const setParam = useCallback(
    (key: string, v: unknown) => {
      if (!selectedId) return;
      setNodes((ns) =>
        ns.map((n) => {
          if (n.id !== selectedId) return n;
          const d = n.data as unknown as NodeData;
          return { ...n, data: { ...d, params: { ...d.params, [key]: v } } as unknown as Record<string, unknown> };
        }),
      );
      if (!liveRef.current) setStale(true); // keep the last preview visible, mark it stale
    },
    [selectedId, setNodes],
  );

  const runRef = useRef<() => void>(() => {});
  const run = useCallback(async () => {
    if (!selectedId) return;
    if (inflight.current) { pending.current = true; return; } // coalesce while busy
    inflight.current = true;
    setRunning(true);
    setError(null);
    try {
      const sink = resolveSink();
      const out = await runNode(nodes, edges, sink.id, sink.port, memo.current);
      setValue(out);
      setStale(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setValue(null);
    } finally {
      inflight.current = false;
      setRunning(false);
      if (pending.current) { pending.current = false; runRef.current(); } // run the latest
    }
  }, [nodes, edges, selectedId, selectedPort, resolveSink]);
  runRef.current = run;

  // Signature of the run-relevant state (params + wiring + selection), excluding
  // node positions so dragging doesn't trigger a re-run.
  const sig = useMemo(
    () =>
      JSON.stringify({
        n: nodes.map((n) => ({ id: n.id, op: (n.data as unknown as NodeData).opName, p: (n.data as unknown as NodeData).params })),
        e: edges.map((e) => ({ s: e.source, sh: e.sourceHandle, t: e.target, th: e.targetHandle })),
        sel: selectedId,
        port: selectedPort,
      }),
    [nodes, edges, selectedId, selectedPort],
  );

  // Auto-run: when live, re-pull on every signature change. We deliberately do NOT
  // trailing-debounce (that only fires after you stop dragging, an onMouseUp feel) —
  // the in-flight guard paces runs to GPU throughput and always runs the latest
  // state, so a slider drag updates continuously in real time. Disabled for feedback
  // graphs, which use the transport (play/step) below instead.
  useEffect(() => {
    if (!live || !selectedId || hasFeedback) return;
    runRef.current();
  }, [sig, live, selectedId, hasFeedback]);

  // Simulation transport: advance one tick (a graph step; the RD op may batch many
  // Euler steps inside it) and show the result.
  const step = useCallback(async (reset = false) => {
    if (!selectedId) return;
    try {
      const sink = resolveSink();
      const out = await advanceNode(nodes, edges, sink.id, sink.port, {
        steps: 1,
        state: sim.current,
        reset,
      });
      setValue(out);
      setStale(false);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPlaying(false);
    }
  }, [nodes, edges, selectedId, selectedPort, resolveSink]);

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
        const out = await advanceNode(nodes, edges, sink.id, sink.port, { steps: 1, state: sim.current });
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
  }, [playing, hasFeedback, selectedId, selectedPort, nodes, edges]);

  return (
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
        <h2>Sources</h2>
        {listSourceSpecs().map((s) => (
          <button key={s.name} className="palette-btn source" title={s.describe} onClick={() => addNode(s.name)}>
            {s.label}
          </button>
        ))}
        <h2>Operations</h2>
        {listOpSpecs().map((s) => (
          <button key={s.name} className="palette-btn" title={s.describe} onClick={() => addNode(s.name)}>
            {s.label}
          </button>
        ))}
      </aside>

      <main className="canvas">
        <div className="canvas-toolbar">
          <div className="breadcrumb">
            {path.map((seg, i) => {
              const label = seg === ROOT_SCOPE ? "Main" : ((nodes.find((n) => n.id === seg)?.data as unknown as GroupData)?.label ?? "group");
              return (
                <span key={seg} className="bc-item">
                  {i > 0 && <span className="bc-sep">›</span>}
                  <button className="bc-seg" disabled={i === path.length - 1} onClick={() => navigateTo(i)}>{label}</button>
                </span>
              );
            })}
          </div>
          {groupableIds.length >= 2 && (
            <button className="tb-primary" onClick={groupSelection}>▦ Group {groupableIds.length} nodes</button>
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
          onNodeClick={(_, n) => { if (isPortStub(n)) return; setSelectedId(n.id); setSelectedPort(null); setValue(null); setStale(false); setError(null); }}
          onNodeDoubleClick={onNodeDoubleClick}
          fitView
        >
          <Background />
          <Controls />
        </ReactFlow>
      </main>

      <aside className="inspector">
        {selected && selectedGroup ? (
          <>
            <h2>▦ Group</h2>
            <label className="row">
              <span>name</span>
              <input
                type="text"
                value={selectedGroup.label}
                onChange={(e) => renameGroup(selected.id, { label: e.target.value })}
                style={{ width: 130 }}
              />
            </label>
            <p className="muted">{selectedGroupMembers} nodes</p>
            <div className="transport-row">
              <button onClick={() => enterGroup(selected.id)}>▦ Open</button>
              <button onClick={() => dissolveGroup(selected.id)}>✕ Ungroup</button>
            </div>
            {[...selectedGroupBoundary.inputs, ...selectedGroupBoundary.outputs].length > 0 && (
              <div className="params">
                <h2>Ports</h2>
                {selectedGroupBoundary.inputs.map((p) => (
                  <label className="row" key={p.id}><span>in</span>
                    <input type="text" value={p.label} onChange={(e) => renameGroup(selected.id, { portId: p.id, portLabel: e.target.value })} style={{ width: 120 }} />
                  </label>
                ))}
                {selectedGroupBoundary.outputs.map((p) => (
                  <label className="row" key={p.id}><span>out</span>
                    <input type="text" value={p.label} onChange={(e) => renameGroup(selected.id, { portId: p.id, portLabel: e.target.value })} style={{ width: 120 }} />
                  </label>
                ))}
              </div>
            )}
            {selectedGroupBoundary.outputs.length > 1 && (
              <label className="row">
                <span>preview</span>
                <select value={selectedPort ?? selectedGroupBoundary.outputs[0]!.id} onChange={(e) => setSelectedPort(e.target.value)}>
                  {selectedGroupBoundary.outputs.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
                </select>
              </label>
            )}
            <button className="run-btn" onClick={run} disabled={running}>{running ? "Running…" : "▶ Run / pull"}</button>
            <Preview value={value} error={error} stale={stale} />
          </>
        ) : selected && selectedSpec ? (
          <>
            <h2>{selectedSpec.label}</h2>
            {selectedSpec.describe && <p className="muted">{selectedSpec.describe}</p>}
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
                <select value={selectedPort ?? selectedSpec.outputs[0]!.name} onChange={(e) => setSelectedPort(e.target.value)}>
                  {selectedSpec.outputs.map((o) => <option key={o.name} value={o.name}>{o.name}</option>)}
                </select>
              </label>
            )}

            {hasFeedback ? (
              <div className="transport">
                <button className={`run-btn ${playing ? "live" : ""}`} onClick={() => setPlaying((p) => !p)}>
                  {playing ? "⏸ Pause" : "▶ Play"}
                </button>
                <div className="transport-row">
                  <button onClick={() => step(false)} disabled={playing} title="Advance one tick">⏭ Step</button>
                  <button onClick={resetSim} title="Reset to the seed">⟲ Reset</button>
                </div>
                <p className="muted">Feedback loop — runs over time. One tick = one graph step.</p>
              </div>
            ) : (
              <>
                <label className="row live-row" title="Re-pull automatically when params or wiring change">
                  <span>Auto-run (live)</span>
                  <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} />
                </label>
                <button className={`run-btn ${live ? "live" : ""}`} onClick={run} disabled={running}>
                  {running ? "Running…" : live ? "● Live — re-running on change" : "▶ Run / pull"}
                </button>
              </>
            )}
            <Preview value={value} error={error} stale={stale} />
          </>
        ) : (
          <p className="muted">Click a node to edit its parameters and pull its output.</p>
        )}
      </aside>
    </div>
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
          {(spec.options ?? []).map((o) => <option key={o} value={o}>{o}</option>)}
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
