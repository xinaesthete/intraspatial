// PROTOTYPE: patch rectangles, one quad each, from two flat arrays: `rects` (x0, y0, x1, y1 in
// pixels, half-open) and `flags` (bit 0 kept, bit 1 edge). Every candidate gets a faint fill and a
// hairline; kept patches an amber outline, and kept edge patches a dashed cyan one. Outlines are
// measured in screen pixels (distance to the rectangle's edge over `fwidth`), so they stay thin at
// any zoom; a patch only a few screen pixels across is filled instead. Same world frame as
// `MaskLayer`.

import { type DefaultProps, Layer, type LayerProps, project32 } from "@deck.gl/core";
import { Buffer } from "@luma.gl/core";
import { Model } from "@luma.gl/engine";
import type { ShaderModule } from "@luma.gl/shadertools";

export const KEPT = 1;
export const EDGE = 2;

type RectUniforms = { candidates: number; kept: number; selected: number; opacity: number };

const rectUniforms: ShaderModule<RectUniforms> = {
  name: "R",
  source: /* wgsl */ `\
struct RectUniforms { candidates: f32, kept: f32, selected: f32, opacity: f32 };
@group(0) @binding(auto) var<uniform> R: RectUniforms;
@group(0) @binding(auto) var<storage, read> rects: array<vec4<f32>>;
@group(0) @binding(auto) var<storage, read> flags: array<u32>;
`,
  uniformTypes: { candidates: "f32", kept: "f32", selected: "f32", opacity: "f32" },
};

const source = /* wgsl */ `\
struct Varyings {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) @interpolate(flat) size: vec2<f32>,
  @location(2) @interpolate(flat) flag: u32,
  @location(3) @interpolate(flat) index: u32,
};

@vertex
fn vertexMain(@builtin(vertex_index) v: u32) -> Varyings {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
  );
  let index = v / 6u;
  let r = rects[index];
  let c = corners[v % 6u];
  let size = r.zw - r.xy;
  var o: Varyings;
  o.position = project_position_to_clipspace(vec3<f32>(r.xy + c * size, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  o.local = c * size;
  o.size = size;
  o.flag = flags[index];
  o.index = index;
  return o;
}

@fragment
fn fragmentMain(input: Varyings) -> @location(0) vec4<f32> {
  // Screen pixels to the nearest edge; derivatives before any branch.
  let fw = max(fwidth(input.local), vec2<f32>(1e-6));
  let toEdge = min(input.local, input.size - input.local) / fw;
  let d = min(toEdge.x, toEdge.y);
  // Patches only a few screen pixels across become fills: their outlines would merge into one.
  let sidePx = min(input.size.x / fw.x, input.size.y / fw.y);
  let outlined = smoothstep(4.0, 10.0, sidePx);
  let kept = (input.flag & 1u) != 0u;
  let edge = (input.flag & 2u) != 0u;
  let line = 1.0 - smoothstep(0.75, 1.5, d);
  let selected = f32(input.index) == R.selected;

  var rgb = vec3<f32>(0.0);
  var alpha = 0.0;
  if (R.candidates > 0.5) {
    rgb = vec3<f32>(0.85, 0.9, 1.0);
    alpha = mix(0.035, mix(0.035, 0.22, outlined), line);
  }
  if (kept && R.kept > 0.5) {
    // Edge patches dashed, in screen pixels so the dash reads the same at any zoom.
    let dash = select(1.0, step(0.5, fract((input.position.x + input.position.y) / 10.0)), edge);
    let ink = (1.0 - smoothstep(1.0, 1.8, d)) * dash * outlined;
    rgb = select(vec3<f32>(1.0, 0.72, 0.15), vec3<f32>(0.25, 0.9, 1.0), edge);
    alpha = max(max(alpha, mix(0.2, 0.06, outlined)), 0.95 * ink);
  }
  if (selected) {
    let ink = 1.0 - smoothstep(2.0, 3.0, d);
    rgb = mix(rgb, vec3<f32>(1.0), ink);
    alpha = max(max(alpha, ink), 0.18);
  }
  alpha = alpha * R.opacity;
  if (alpha <= 0.0) { discard; }
  return vec4<f32>(rgb * alpha, alpha);
}
`;

export type RectLayerProps = {
  /** `n × 4` f32 `[x0, y0, x1, y1]`; uploaded when its identity changes. */
  rects: Float32Array | null;
  /** `n` u32 of `KEPT | EDGE` bits, with `rects`. */
  flags: Uint32Array | null;
  showCandidates: boolean;
  showKept: boolean;
  /** Index of a rectangle to outline in white, or -1. */
  selected: number;
} & LayerProps;

const defaultProps: DefaultProps<RectLayerProps> = {
  rects: { type: "object", value: null, compare: true },
  flags: { type: "object", value: null, compare: true },
  showCandidates: { type: "boolean", value: true },
  showKept: { type: "boolean", value: true },
  selected: { type: "number", value: -1 },
};

type Slot = { buf?: Buffer; src?: ArrayBufferView | null };
type State = { model?: Model; rects: Slot; flags: Slot };

export class RectLayer extends Layer<RectLayerProps> {
  static override layerName = "RectLayer";
  static override defaultProps = defaultProps;
  declare state: State;

  override getShaders() {
    return super.getShaders({ source, modules: [project32, rectUniforms] });
  }

  override initializeState(): void {
    const model = new Model(this.context.device, {
      ...this.getShaders(),
      id: this.props.id,
      topology: "triangle-list",
      bufferLayout: [],
      isInstanced: false,
      vertexCount: 0,
      shaderAssembler: this.context.shaderAssembler,
    });
    this.setState({ model, rects: {}, flags: {} });
  }

  override finalizeState(context: Parameters<Layer["finalizeState"]>[0]): void {
    this.state.rects.buf?.destroy();
    this.state.flags.buf?.destroy();
    this.state.model?.destroy();
    super.finalizeState(context);
  }

  override draw(): void {
    const { model } = this.state;
    const { rects, flags, showCandidates, showKept, selected, opacity } = this.props;
    const n = rects ? rects.length / 4 : 0;
    if (!model || !rects || !flags || n === 0 || flags.length !== n) return;
    this.upload(model, "rects", rects);
    this.upload(model, "flags", flags);
    model.setVertexCount(6 * n);
    model.shaderInputs.setProps({
      R: { candidates: showCandidates ? 1 : 0, kept: showKept ? 1 : 0, selected, opacity: opacity ?? 1 },
    });
    model.draw(this.context.renderPass);
  }

  /** Mutates state rather than `setState`: a state change from draw would ask for another draw. */
  private upload(model: Model, name: "rects" | "flags", src: Float32Array | Uint32Array): void {
    const slot = this.state[name];
    if (slot.src === src) return;
    slot.buf?.destroy();
    slot.buf = this.context.device.createBuffer({ data: src, usage: Buffer.STORAGE | Buffer.COPY_DST });
    slot.src = src;
    model.setBindings({ [name]: slot.buf });
  }
}
