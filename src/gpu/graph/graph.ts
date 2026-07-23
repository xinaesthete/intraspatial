// The explicit, serialisable graph IR + a thin builder that returns lazy
// `GpuField` handles. This is the single source of truth the React Flow canvas
// edits and the executor runs (resolving the "implicit vs explicit" open question
// in docs/gpu-resource-sync.md in favour of explicit — a canvas cannot edit a
// closure-based implicit trace).
//
// Edges are *derived* from each node's `inputs` map (a RAW dependency on the
// referenced producer); they are never hand-declared.
import type { Basis, ElementType, FieldRole, FieldValue, GpuField, NodeId, ResolvedPlacement, Shape, TensorAxis } from "./handle";
import { SCALAR, SPATIAL } from "./handle";
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
  /** Inferred output bases (ADR-0006), positional by output port. The executor stamps
   *  these onto produced values so the basis propagates without each op setting it. */
  outBases?: Basis[];
  /** Inferred output open-axes (ADR-0004/0015), positional by output port. Stamped by the
   *  executor exactly like `outBases`, so a channel axis propagates without each op setting it.
   *  A `undefined` entry ⇒ that output has no axes. */
  outAxes?: (readonly TensorAxis[] | undefined)[];
  /** Inferred output polarity (ADR-0015), positional by output port. Stamped by the executor
   *  like `outBases`. A `undefined` entry ⇒ that output is `intensity` (the default). */
  outRoles?: (FieldRole | undefined)[];
  /** Inferred output world placement (ADR-0018), positional by output port. Stamped by the
   *  executor like `outBases`, so placement survives generic ops. A `undefined` entry ⇒ that
   *  output is array-space (unplaced). */
  outPlacements?: (ResolvedPlacement | undefined)[];
}

/** Handle returned by `Graph.feedback`: the delayed `state` output, plus `close` to
 *  wire the value fed back for the next tick (the edge that closes the loop). */
export interface FeedbackHandle {
  state: GpuField;
  close(next: GpuField): void;
}

/** Handle returned by `Graph.delay`: the `depth`-ticks-delayed `out` output, plus `close`
 *  to wire the value pushed into the history each tick. */
export interface DelayHandle {
  out: GpuField;
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
    // Carry the source value's axes/role/placement onto the handle and node, exactly like basis —
    // otherwise a source's coordinate facets are lost at build (the ADR-0015 known gap).
    this.nodes.set(id, {
      id,
      op: "source",
      params: {},
      inputs: {},
      source: value,
      outBases: [value.basis ?? SPATIAL],
      outAxes: [value.axes],
      outRoles: [value.role],
      outPlacements: [value.placement],
    });
    return makeField(
      id,
      "out",
      value.shape,
      value.dtype ?? "f32",
      value.element ?? SCALAR,
      value.basis ?? SPATIAL,
      value.axes,
      value.role,
      value.placement,
    );
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
    return this.source({ shape: { kind: "grid", width, height }, dtype: "f32", data: Float32Array.from(data) }, "grid");
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
      outBases: [init.basis ?? SPATIAL],
      outAxes: [init.axes],
      outRoles: [init.role],
      outPlacements: [init.placement],
    };
    this.nodes.set(id, node);
    const state = makeField(
      id,
      "state",
      init.shape,
      init.dtype,
      init.element ?? SCALAR,
      init.basis ?? SPATIAL,
      init.axes,
      init.role,
      init.placement,
    );
    return {
      state,
      close: (next: GpuField) => {
        node.inputs.next = { node: next.producer, port: next.outPort };
      },
    };
  }

  /** A `depth`-deep delay (z⁻ᵏ) — generalises `feedback` (which is `depth` 1). Outputs the
   *  value from `depth` ticks ago (seeded by `init`); `close(next)` wires what is pushed
   *  into its history ring each tick. Holds `depth` copies of the field resident (see
   *  `simStateBytes`). `key` is the store key, stable across rebuilds. */
  delay(init: GpuField, depth: number, key?: string): DelayHandle {
    const id = this.nextNodeId("delay");
    const node: GraphNode = {
      id,
      op: "delay",
      params: { depth },
      inputs: { init: { node: init.producer, port: init.outPort } },
      stableKey: key ?? id,
      outBases: [init.basis ?? SPATIAL],
      outAxes: [init.axes],
      outRoles: [init.role],
      outPlacements: [init.placement],
    };
    this.nodes.set(id, node);
    const out = makeField(
      id,
      "out",
      init.shape,
      init.dtype,
      init.element ?? SCALAR,
      init.basis ?? SPATIAL,
      init.axes,
      init.role,
      init.placement,
    );
    return {
      out,
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
    const inElements: ElementType[] = [];
    const inBases: Basis[] = [];
    const inAxes: (readonly TensorAxis[] | undefined)[] = [];
    const inRoles: (FieldRole | undefined)[] = [];
    const inPlacements: (ResolvedPlacement | undefined)[] = [];
    for (const spec of def.inputs) {
      const f = inputs[spec.name];
      if (!f) throw new Error(`graph.op(${name}): missing input "${spec.name}"`);
      inputRefs[spec.name] = { node: f.producer, port: f.outPort };
      inShapes.push(f.shape);
      inElements.push(f.element ?? SCALAR);
      inBases.push(f.basis ?? SPATIAL);
      inAxes.push(f.axes);
      inRoles.push(f.role);
      inPlacements.push(f.placement);
    }
    // Declared defaults first, then overlay everything the caller supplied — this
    // keeps undeclared pass-through params (e.g. an explicit `bbox`) that ops read
    // but the palette doesn't surface.
    const merged: Params = {};
    for (const p of def.params) merged[p.name] = p.default;
    Object.assign(merged, params);
    const outShapes = def.inferShapes(inShapes, merged);
    // Element inference is opt-in; ops that don't declare it keep the legacy
    // all-scalar contract (ADR-0004). Rejection of a wrong element happens here.
    const outElements = def.inferElements ? def.inferElements(inElements, merged) : def.outputs.map(() => SCALAR);
    // Basis inference is opt-in; ops that don't declare it pass the first input's basis
    // through to every output (a source ⇒ spatial). Rejection of a wrong basis (e.g.
    // idwt on a spatial field) happens in the op's inferBasis (ADR-0006).
    const passThrough = inBases[0] ?? SPATIAL;
    const outBases = def.inferBasis ? def.inferBasis(inBases, merged) : def.outputs.map(() => passThrough);
    // Axes/role/placement thread through the identical mechanism as basis (ADR-0015 known gap +
    // ADR-0018): opt-in hook, else pass through the first input's value to every output — the
    // "a convolve does not move the grid" default. A source (no inputs) passes through `undefined`.
    const outAxes = def.inferAxes ? def.inferAxes(inAxes, merged) : def.outputs.map(() => inAxes[0]);
    const outRoles = def.inferRole ? def.inferRole(inRoles, merged) : def.outputs.map(() => inRoles[0]);
    const outPlacements = def.inferPlacement ? def.inferPlacement(inPlacements, merged) : def.outputs.map(() => inPlacements[0]);
    const id = this.nextNodeId(name);
    this.nodes.set(id, { id, op: name, params: merged, inputs: inputRefs, outBases, outAxes, outRoles, outPlacements });
    return def.outputs.map((o, i) =>
      makeField(
        id,
        o.name,
        outShapes[i]!,
        o.dtype ?? "f32",
        outElements[i] ?? SCALAR,
        outBases[i] ?? SPATIAL,
        outAxes[i],
        outRoles[i],
        outPlacements[i],
      ),
    );
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

function makeField(
  producer: NodeId,
  outPort: string,
  shape: Shape,
  dtype: GpuField["dtype"],
  element: ElementType = SCALAR,
  basis: Basis = SPATIAL,
  axes?: readonly TensorAxis[],
  role?: FieldRole,
  placement?: ResolvedPlacement,
): GpuField {
  // `axes`/`role`/`placement` are threaded exactly like `element`/`basis`: absent ⇒ the
  // field-level default (no axes / intensity / array-space), so an existing op is untouched.
  return { id: fieldSeq++, shape, dtype, element, basis, axes, role, placement, producer, outPort, version: 0 };
}
