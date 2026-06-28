// The explicit, serialisable graph IR + a thin builder that returns lazy
// `GpuField` handles. This is the single source of truth the React Flow canvas
// edits and the executor runs (resolving the "implicit vs explicit" open question
// in docs/gpu-resource-sync.md in favour of explicit — a canvas cannot edit a
// closure-based implicit trace).
//
// Edges are *derived* from each node's `inputs` map (a RAW dependency on the
// referenced producer); they are never hand-declared.
import type { FieldValue, GpuField, NodeId, Shape } from "./handle";
import type { Params } from "./op";
import { getOp } from "./registry";

/** A reference to a producer node's output port. */
export interface EdgeRef {
  node: NodeId;
  port: string;
}

export interface GraphNode {
  id: NodeId;
  /** Registry op name, or the builtins "source" / "feedback". */
  op: string;
  params: Params;
  /** Port name -> producer reference. Empty for sources. */
  inputs: Record<string, EdgeRef>;
  /** For "source" nodes: the host value this node yields (not serialised). */
  source?: FieldValue;
  /** For "feedback" (delay) nodes: a key, stable across graph rebuilds, under which
   *  this node's state is stored between ticks (the React Flow node id in the UI). */
  stableKey?: string;
}

/** Handle returned by `Graph.feedback`: the delayed `state` output, plus `close` to
 *  wire the value fed back for the next tick (the edge that closes the loop). */
export interface FeedbackHandle {
  state: GpuField;
  close(next: GpuField): void;
}

let fieldSeq = 0;

export class Graph {
  readonly nodes = new Map<NodeId, GraphNode>();
  private seq = 0;

  private nextNodeId(op: string): NodeId {
    return `${op}#${this.seq++}`;
  }

  /** Add a source node carrying a host value; returns its lazy handle. */
  source(value: FieldValue, label = "source"): GpuField {
    const id = this.nextNodeId(label);
    this.nodes.set(id, { id, op: "source", params: {}, inputs: {}, source: value });
    return makeField(id, "out", value.shape, value.dtype ?? "f32");
  }

  /** A points source from parallel x/y arrays, packed as [x0,y0,x1,y1,...]. */
  points(xs: ArrayLike<number>, ys: ArrayLike<number>): GpuField {
    const n = xs.length;
    if (ys.length !== n) throw new Error("graph.points: xs and ys length mismatch");
    const data = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      data[i * 2] = xs[i]!;
      data[i * 2 + 1] = ys[i]!;
    }
    return this.source({ shape: { kind: "points", n }, dtype: "f32", data }, "points");
  }

  /** A grid source from a row-major width*height array. */
  grid(data: ArrayLike<number>, width: number, height: number): GpuField {
    if (data.length !== width * height) throw new Error("graph.grid: data length != width*height");
    return this.source(
      { shape: { kind: "grid", width, height }, dtype: "f32", data: Float32Array.from(data) },
      "grid",
    );
  }

  /** A feedback (unit-delay) node: it outputs the *previous* tick's value (seeded
   *  by `init`), breaking what would otherwise be a cycle. Call `close(next)` with
   *  the value to feed back. `key` is the store key, stable across rebuilds. */
  feedback(init: GpuField, key?: string): FeedbackHandle {
    const id = this.nextNodeId("feedback");
    const node: GraphNode = {
      id,
      op: "feedback",
      params: {},
      inputs: { init: { node: init.producer, port: init.outPort } },
      stableKey: key ?? id,
    };
    this.nodes.set(id, node);
    const state = makeField(id, "state", init.shape, init.dtype);
    return {
      state,
      close: (next: GpuField) => {
        node.inputs.next = { node: next.producer, port: next.outPort };
      },
    };
  }

  /** Instantiate an op node. Returns one lazy handle per declared output port. */
  op(name: string, inputs: Record<string, GpuField>, params: Params = {}): GpuField[] {
    const def = getOp(name);
    const inputRefs: Record<string, EdgeRef> = {};
    const inShapes: Shape[] = [];
    for (const spec of def.inputs) {
      const f = inputs[spec.name];
      if (!f) throw new Error(`graph.op(${name}): missing input "${spec.name}"`);
      inputRefs[spec.name] = { node: f.producer, port: f.outPort };
      inShapes.push(f.shape);
    }
    // Declared defaults first, then overlay everything the caller supplied — this
    // keeps undeclared pass-through params (e.g. an explicit `bbox`) that ops read
    // but the palette doesn't surface.
    const merged: Params = {};
    for (const p of def.params) merged[p.name] = p.default;
    Object.assign(merged, params);
    const outShapes = def.inferShapes(inShapes, merged);
    const id = this.nextNodeId(name);
    this.nodes.set(id, { id, op: name, params: merged, inputs: inputRefs });
    return def.outputs.map((o, i) => makeField(id, o.name, outShapes[i]!, o.dtype ?? "f32"));
  }

  /** Convenience for single-output ops. */
  op1(name: string, inputs: Record<string, GpuField>, params: Params = {}): GpuField {
    const outs = this.op(name, inputs, params);
    if (outs.length !== 1) throw new Error(`graph.op1(${name}): expected 1 output, got ${outs.length}`);
    return outs[0]!;
  }

  getNode(id: NodeId): GraphNode {
    const n = this.nodes.get(id);
    if (!n) throw new Error(`graph: no node "${id}"`);
    return n;
  }

  /** Structural serialisation (nodes, params, edges). Raw-data sources are emitted
   *  without their `source` payload — the UI uses generator ops, not uploads. */
  toJSON(): unknown {
    return {
      nodes: [...this.nodes.values()].map((n) => ({
        id: n.id,
        op: n.op,
        params: n.params,
        inputs: n.inputs,
      })),
    };
  }
}

function makeField(producer: NodeId, outPort: string, shape: Shape, dtype: GpuField["dtype"]): GpuField {
  return { id: fieldSeq++, shape, dtype, producer, outPort, version: 0 };
}
