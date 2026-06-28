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
import { OpNode } from "./OpNode";
import { GroupNode } from "./GroupNode";
import { advanceNode, graphHasFeedback, runNode } from "./buildGraph";
import type { NodeData } from "./buildGraph";
import { collapse, isGroupNode, resolveGroupOutput, setCollapsed, ungroup } from "./grouping";
import type { GroupData } from "./grouping";
import { defaultParamsFor, getSpec, listOpSpecs, listSourceSpecs } from "./specs";
import { EXAMPLES } from "./examples";
import type { Example } from "./examples";
import { Preview } from "./Preview";

const nodeTypes: NodeTypes = { op: OpNode, group: GroupNode };

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
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
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
  const hasFeedback = useMemo(() => graphHasFeedback(nodes), [nodes]);
  // Serialise an in-flight pull so overlapping auto-runs don't race on the shared
  // GPU pools; a change during a run sets `pending` and re-runs once it finishes.
  const inflight = useRef(false);
  const pending = useRef(false);

  // Collapse the current multi-selection (≥2 non-group nodes) into a group proxy.
  const groupableIds = useMemo(
    () => selectedIds.filter((id) => { const n = nodes.find((x) => x.id === id); return n && !isGroupNode(n); }),
    [selectedIds, nodes],
  );
  const groupSelection = useCallback(() => {
    if (groupableIds.length < 2) return;
    const gid = `group#${idSeq.current++}`;
    const r = collapse(nodes, edges, groupableIds, gid, "Group");
    setNodes(r.nodes);
    setEdges(r.edges);
    setSelectedId(gid);
    setSelectedPort(null);
    setValue(null);
    setStale(false);
  }, [groupableIds, nodes, edges, setNodes, setEdges]);

  // Toggle a group open/closed (the group persists either way).
  const toggleGroup = useCallback(
    (groupId: string, collapsed: boolean) => {
      const r = setCollapsed(nodes, edges, groupId, collapsed);
      setNodes(r.nodes);
      setEdges(r.edges);
      setSelectedId(collapsed ? groupId : null);
      setValue(null);
    },
    [nodes, edges, setNodes, setEdges],
  );

  const dissolveGroup = useCallback(
    (groupId: string) => {
      const r = ungroup(nodes, edges, groupId);
      setNodes(r.nodes);
      setEdges(r.edges);
      setSelectedId(null);
      setValue(null);
    },
    [nodes, edges, setNodes, setEdges],
  );

  const onNodeDoubleClick = useCallback(
    (_: unknown, node: Node) => {
      if (isGroupNode(node)) toggleGroup(node.id, false); // open the group
    },
    [toggleGroup],
  );

  // Every group (collapsed or expanded), for the toolbar that lets you navigate
  // in/out — the missing "way out" once a group is opened.
  const groups = useMemo(
    () => nodes.filter(isGroupNode).map((n) => ({ id: n.id, data: n.data as unknown as GroupData })),
    [nodes],
  );

  // Resolve the pull target: a group proxy stands in for an internal (node, port).
  const resolveSink = useCallback((): { id: string; port?: string } => {
    const sel = nodes.find((n) => n.id === selectedId);
    if (sel && isGroupNode(sel)) {
      const r = resolveGroupOutput(sel, selectedPort ?? undefined);
      if (r) return { id: r.node, port: r.port };
    }
    return { id: selectedId!, port: selectedPort ?? undefined };
  }, [nodes, selectedId, selectedPort]);

  // Rename a group's label or one of its boundary ports.
  const renameGroup = useCallback(
    (groupId: string, change: { label?: string; portId?: string; portLabel?: string }) => {
      setNodes((ns) =>
        ns.map((n) => {
          if (n.id !== groupId) return n;
          const d = n.data as unknown as GroupData;
          const next: GroupData = { ...d };
          if (change.label !== undefined) next.label = change.label;
          if (change.portId !== undefined && change.portLabel !== undefined) {
            const relabel = (ps: GroupData["inputs"]) => ps.map((p) => (p.id === change.portId ? { ...p, label: change.portLabel! } : p));
            next.inputs = relabel(d.inputs);
            next.outputs = relabel(d.outputs);
          }
          return { ...n, data: next as unknown as Record<string, unknown> };
        }),
      );
    },
    [setNodes],
  );

  const onConnect = useCallback(
    (c: Connection) => setEdges((eds) => addEdge({ ...c, id: `e${idSeq.current++}` }, eds)),
    [setEdges],
  );

  const addNode = useCallback(
    (opName: string) => {
      const id = `x${idSeq.current++}`;
      setNodes((ns) => ns.concat(mkNode(id, opName, 120 + (ns.length % 5) * 40, 360 + (ns.length % 4) * 30)));
    },
    [setNodes],
  );

  const selected = useMemo(() => nodes.find((n) => n.id === selectedId) ?? null, [nodes, selectedId]);
  const selectedGroup = selected && isGroupNode(selected) ? (selected.data as unknown as GroupData) : null;
  const selectedSpec = selected && !selectedGroup ? getSpec((selected.data as unknown as NodeData).opName) : null;

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
        {(groupableIds.length >= 2 || groups.length > 0) && (
          <div className="canvas-toolbar">
            {groupableIds.length >= 2 && (
              <button className="tb-primary" onClick={groupSelection}>▦ Group {groupableIds.length} nodes</button>
            )}
            {groups.map((g) => (
              <span key={g.id} className="tb-group">
                <button title={g.data.collapsed ? "Open group" : "Collapse group"} onClick={() => toggleGroup(g.id, !g.data.collapsed)}>
                  {g.data.collapsed ? "▢" : "▣"} {g.data.label}
                </button>
                <button className="tb-x" title="Ungroup (dissolve)" onClick={() => dissolveGroup(g.id)}>✕</button>
              </span>
            ))}
          </div>
        )}
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onSelectionChange={({ nodes: sel }) => setSelectedIds(sel.map((n) => n.id))}
          onNodeClick={(_, n) => { setSelectedId(n.id); setSelectedPort(null); setValue(null); setStale(false); setError(null); }}
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
            <p className="muted">{selectedGroup.members.length} nodes · double-click to open; use the toolbar ▣/▢ to collapse or ✕ to ungroup.</p>
            {[...selectedGroup.inputs, ...selectedGroup.outputs].length > 0 && (
              <div className="params">
                <h2>Ports</h2>
                {selectedGroup.inputs.map((p) => (
                  <label className="row" key={p.id}><span>in</span>
                    <input type="text" value={p.label} onChange={(e) => renameGroup(selected.id, { portId: p.id, portLabel: e.target.value })} style={{ width: 120 }} />
                  </label>
                ))}
                {selectedGroup.outputs.map((p) => (
                  <label className="row" key={p.id}><span>out</span>
                    <input type="text" value={p.label} onChange={(e) => renameGroup(selected.id, { portId: p.id, portLabel: e.target.value })} style={{ width: 120 }} />
                  </label>
                ))}
              </div>
            )}
            {selectedGroup.outputs.length > 1 && (
              <label className="row">
                <span>preview</span>
                <select value={selectedPort ?? selectedGroup.outputs[0]!.id} onChange={(e) => setSelectedPort(e.target.value)}>
                  {selectedGroup.outputs.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
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
