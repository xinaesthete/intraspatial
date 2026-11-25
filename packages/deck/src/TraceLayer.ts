// The history trace: where the chain has been. A `PositionHistory` ring keeps past snapshots of the
// positions on the GPU, and this layer draws them one of two ways. WGSL only.
//
//   - "sheet" (FoldSynth's and CSynth's history trace): the surface the chain sweeps out through
//     time. One edge is the chain as it is now; going across the sheet is going back in time. A
//     stretch that has stopped moving leaves no sheet, and one that is still folding leaves a wide one.
//   - "paths": the line each particle has travelled.
//
// Both fade with age, and colour runs along the chain through a ramp, as on the tube.
//
// The trace can be moved as it ages, so that it leaves the chain instead of piling up where the
// chain is: `drift` shifts each older snapshot by a fixed step in the positions' own units, and
// `recede` pushes it straight away from the camera, so the past trails off into the distance.

import { color, type DefaultProps, Layer, type LayerProps, project32, type UpdateParameters } from "@deck.gl/core";
import type { Buffer } from "@luma.gl/core";
import { Model } from "@luma.gl/engine";
import type { ShaderModule } from "@luma.gl/shadertools";
import type { Color } from "./DensityLayer";
import { HandleCache } from "./handles";
import type { PositionHistory } from "./history";
import { syncPalette } from "./palette";
import { MATERIAL_WGSL, RAMP_WGSL, VIEW_FRAME_WGSL } from "./shadingWgsl";

export type TraceLayerProps = {
  /** The ring of snapshots. Its `head` and `filled` are read at every draw. */
  history: PositionHistory | null;
  /** The swept surface, or one line per particle. */
  mode?: "sheet" | "paths";
  /** Sheet: 0 flat colour, 1 lit. */
  shading?: number;
  /** Paths: draw every `particleStep`-th particle's path: 1 is all of them. */
  particleStep?: number;
  /** Colour by particle index, first to last. */
  colors?: readonly Color[];
  /** Shift per snapshot of age, in the positions' units: the trace streams off in this direction. */
  drift?: readonly [number, number, number];
  /** Distance each snapshot of age is pushed away from the camera, in the positions' units. */
  recede?: number;
  /** How fast a path fades with age: opacity is (1 - age)^fade, age running 0 (now) to 1 (oldest kept). */
  fade?: number;
} & LayerProps;

const defaultProps: DefaultProps<TraceLayerProps> = {
  history: { type: "object", value: null, compare: false },
  mode: "sheet",
  shading: { type: "number", value: 1, min: 0, max: 1 },
  particleStep: { type: "number", value: 1, min: 1 },
  colors: { type: "object", value: [[255, 255, 255, 255]], compare: true },
  fade: { type: "number", value: 1.5, min: 0 },
  drift: { type: "array", value: [0, 0, 0], compare: true },
  recede: { type: "number", value: 0 },
};

type TraceUniforms = {
  /** xyz: drift per snapshot of age; w: recede per snapshot of age. */
  drift: [number, number, number, number];
  stride: number;
  count: number;
  capacity: number;
  head: number;
  filled: number;
  particleStep: number;
  stops: number;
  fade: number;
  opacity: number;
  shading: number;
  pad1: number;
  pad2: number;
};

const traceModule: ShaderModule<TraceUniforms> = {
  name: "trace",
  source: /* wgsl */ `\
struct TraceUniforms {
  drift: vec4<f32>,
  stride: f32, count: f32, capacity: f32, head: f32,
  filled: f32, particleStep: f32, stops: f32, fade: f32,
  opacity: f32, shading: f32, pad1: f32, pad2: f32,
};
@group(0) @binding(auto) var<uniform> trace: TraceUniforms;
/** capacity snapshots of count particles, slot after slot. */
@group(0) @binding(auto) var<storage, read> history: array<f32>;
@group(0) @binding(auto) var<storage, read> palette: array<vec4<f32>>;
`,
  uniformTypes: {
    drift: "vec4<f32>",
    stride: "f32",
    count: "f32",
    capacity: "f32",
    head: "f32",
    filled: "f32",
    particleStep: "f32",
    stops: "f32",
    fade: "f32",
    opacity: "f32",
    shading: "f32",
    pad1: "f32",
    pad2: "f32",
  },
};

/** A particle's position `age` snapshots back (0 is the newest), in the positions' own units. */
const TRACE_POINT_WGSL = /* wgsl */ `
fn trace_point(particle: u32, age: u32) -> vec3<f32> {
  let capacity = u32(trace.capacity);
  let slot = (u32(trace.head) + 2u * capacity - 1u - age) % capacity;
  let p = (slot * u32(trace.count) + particle) * u32(trace.stride);
  return vec3<f32>(history[p], history[p + 1u], history[p + 2u]);
}

/** The same point in deck's common space, moved by its age: drifted, and pushed away from the camera. */
fn trace_aged(particle: u32, age: u32, toward: vec3<f32>) -> vec3<f32> {
  let moved = project_position_vec3_f32(trace_point(particle, age) + trace.drift.xyz * f32(age));
  return moved - toward * project_size_float(trace.drift.w * f32(age));
}

/** Opacity at an age: 1 now, 0 at the oldest snapshot kept. */
fn trace_fade(age: f32) -> f32 {
  return pow(max(1.0 - age / (trace.capacity - 1.0), 0.0), trace.fade);
}
`;

const pathSource = /* wgsl */ `\
${VIEW_FRAME_WGSL}
${RAMP_WGSL}
${TRACE_POINT_WGSL}

struct Varyings {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instance: u32) -> Varyings {
  let spans = u32(trace.capacity) - 1u;
  let particle = (instance / spans) * u32(trace.particleStep);
  let age = instance % spans + vertexIndex;
  var output: Varyings;
  if (age >= u32(trace.filled) || particle >= u32(trace.count)) {
    // Not recorded yet. Outside the clip volume: the line is dropped.
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    return output;
  }
  output.position = project_common_position_to_clipspace(vec4<f32>(trace_aged(particle, age, view_frame().toward), 1.0));
  let base = ramp_at(f32(particle) / max(trace.count - 1.0, 1.0), u32(trace.stops));
  output.color = vec4<f32>(base.rgb, base.a * trace.opacity * trace_fade(f32(age)));
  return output;
}

@fragment
fn fragmentMain(input: Varyings) -> @location(0) vec4<f32> {
  return deckgl_premultiplied_alpha(input.color);
}
`;

const sheetSource = /* wgsl */ `\
${VIEW_FRAME_WGSL}
${MATERIAL_WGSL}
${RAMP_WGSL}
${TRACE_POINT_WGSL}

struct Varyings {
  @builtin(position) position: vec4<f32>,
  /** The quad's normal in view space, the same at all its corners. */
  @location(0) @interpolate(flat) normal: vec3<f32>,
  /** Position along the chain, 0..1. */
  @location(1) along: f32,
  /** Snapshots back in time. */
  @location(2) age: f32,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> Varyings {
  // (particle, age) offsets of a quad's six vertices.
  var corners = array<vec2<u32>, 6>(
    vec2<u32>(0u, 0u), vec2<u32>(1u, 0u), vec2<u32>(1u, 1u),
    vec2<u32>(0u, 0u), vec2<u32>(1u, 1u), vec2<u32>(0u, 1u)
  );
  let links = u32(trace.count) - 1u;
  let quad = vertexIndex / 6u;
  let particle = quad % links;
  let age = quad / links;
  let corner = corners[vertexIndex % 6u];
  var output: Varyings;
  if (age + 1u >= u32(trace.filled)) {
    // Not recorded yet. Outside the clip volume: the quad is dropped.
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    return output;
  }
  let view = view_frame();
  let here = trace_aged(particle + corner.x, age + corner.y, view.toward);
  // The quad's own corner and its two neighbours give one normal for the whole quad.
  let a = trace_aged(particle, age, view.toward);
  let b = trace_aged(particle + 1u, age, view.toward);
  let c = trace_aged(particle, age + 1u, view.toward);
  let across = cross(b - a, c - a);
  output.position = project_common_position_to_clipspace(vec4<f32>(here, 1.0));
  // Where the chain has not moved the quad has no area and no normal: face the camera.
  output.normal = select(vec3<f32>(0.0, 0.0, 1.0), to_view(normalize(across), view), dot(across, across) > 1e-12);
  output.along = f32(particle + corner.x) / f32(max(links, 1u));
  output.age = f32(age + corner.y);
  return output;
}

@fragment
fn fragmentMain(input: Varyings) -> @location(0) vec4<f32> {
  let base = ramp_at(input.along, u32(trace.stops));
  // Lit the same from either side: turn the normal to face the camera.
  let normal = input.normal * select(-1.0, 1.0, input.normal.z >= 0.0);
  let alpha = base.a * trace.opacity * trace_fade(input.age);
  if (alpha < 1.0 / 255.0) {
    discard;
  }
  return deckgl_premultiplied_alpha(vec4<f32>(material(base.rgb, normal, trace.shading), alpha));
}
`;

/** Vertices the sheet needs: two triangles per chain link per step back in time. */
export function sheetVertexCount(count: number, capacity: number): number {
  return count < 2 || capacity < 2 ? 0 : (count - 1) * (capacity - 1) * 6;
}

type State = {
  paths?: Model;
  sheet?: Model;
  handles?: HandleCache;
  palette?: Buffer;
  ready: boolean;
};

export class TraceLayer extends Layer<TraceLayerProps> {
  static override layerName = "TraceLayer";
  static override defaultProps = defaultProps;

  declare state: State;

  override getShaders(source: string = sheetSource) {
    return super.getShaders({ source, modules: [color, project32, traceModule] });
  }

  override initializeState(): void {
    // A state object whatever happens next: see DensityLayer.initializeState.
    const s: State = { ready: false };
    this.state = s;
    const device = this.context.device;
    if (device.type !== "webgpu") {
      this.raiseError(new Error("TraceLayer needs deck on WebGPU (deviceProps: { type: 'webgpu' })"), "TraceLayer");
      return;
    }
    const shared = { bufferLayout: [], shaderAssembler: this.context.shaderAssembler };
    s.paths = new Model(device, {
      ...this.getShaders(pathSource),
      ...shared,
      id: `${this.props.id}-paths`,
      topology: "line-list",
      isInstanced: true,
      vertexCount: 2,
    });
    s.sheet = new Model(device, {
      ...this.getShaders(sheetSource),
      ...shared,
      id: `${this.props.id}-sheet`,
      topology: "triangle-list",
      isInstanced: false,
      vertexCount: 0,
    });
    s.handles = new HandleCache(device);
    s.ready = true;
  }

  override updateState({ props, oldProps }: UpdateParameters<this>): void {
    const s = this.state;
    if (!s.ready) return;
    if (props.colors !== oldProps.colors || !s.palette) {
      s.palette = syncPalette(this.context.device, `${this.props.id}:palette`, s.palette, props.colors ?? []);
    }
  }

  override finalizeState(context: Parameters<Layer["finalizeState"]>[0]): void {
    const s = this.state;
    s.handles?.destroy(); // wrappers only: the ring is its owner's
    s.palette?.destroy();
    s.paths?.destroy();
    s.sheet?.destroy();
    super.finalizeState(context);
  }

  /** Both models, so deck's per-draw parameters and attachment formats reach whichever is drawn. */
  override getModels(): Model[] {
    return [this.state.paths, this.state.sheet].filter((m): m is Model => m !== undefined);
  }

  override draw(): void {
    const s = this.state;
    const { handles, palette } = s;
    const history = this.props.history;
    const sheet = this.props.mode !== "paths";
    const model = sheet ? s.sheet : s.paths;
    if (!s.ready || !model || !handles || !palette || !history || history.filled < 2) return;
    const particleStep = Math.max(1, Math.floor(this.props.particleStep ?? 1));
    model.setBindings({ history: handles.wrap(history.buffer), palette });
    const [dx, dy, dz] = this.props.drift ?? [0, 0, 0];
    const uniforms: TraceUniforms = {
      drift: [dx, dy, dz, this.props.recede ?? 0],
      stride: history.stride,
      count: history.count,
      capacity: history.capacity,
      head: history.head,
      filled: history.filled,
      particleStep,
      stops: Math.max(1, this.props.colors?.length ?? 1),
      fade: this.props.fade ?? 1.5,
      opacity: this.props.opacity ?? 1,
      shading: this.props.shading ?? 1,
      pad1: 0,
      pad2: 0,
    };
    model.shaderInputs.setProps({ trace: uniforms });
    if (sheet) model.setVertexCount(sheetVertexCount(history.count, history.capacity));
    else model.setInstanceCount(Math.ceil(history.count / particleStep) * (history.capacity - 1));
    model.draw(this.context.renderPass);
  }
}
