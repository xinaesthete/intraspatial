// PROTOTYPE: a binary mask drawn as a flat tint, from one bit per pixel. A 1 µm/px canvas of 62 Mpx
// is 7.8 MB this way, against 248 MB as f32. World units are pixels: the image covers
// [0, width] × [0, height], row 0 at y = 0 (an OrthographicView with y down).

import { type DefaultProps, Layer, type LayerProps, project32 } from "@deck.gl/core";
import { Buffer } from "@luma.gl/core";
import { Model } from "@luma.gl/engine";
import type { ShaderModule } from "@luma.gl/shadertools";
import type { MaskRuns } from "../../../src/spatial/patchGrid";

/** The mask's pixels as bits: pixel `i` is bit `i & 31` of word `i >> 5`. */
export function packRuns(runs: MaskRuns): Uint32Array {
  const bits = new Uint32Array(Math.max(1, Math.ceil((runs.width * runs.height) / 32)));
  for (let y = 0; y < runs.height; y++) {
    const row = y * runs.width;
    for (let k = runs.rowStart[y]!; k < runs.rowStart[y + 1]!; k++) {
      for (let i = row + runs.x0[k]!; i < row + runs.x1[k]!; i++) bits[i >>> 5]! |= 1 << (i & 31);
    }
  }
  return bits;
}

type MaskUniforms = { colour: [number, number, number, number]; width: number; height: number; pad0: number; pad1: number };

const maskUniforms: ShaderModule<MaskUniforms> = {
  name: "M",
  source: /* wgsl */ `\
struct MaskUniforms { colour: vec4<f32>, width: f32, height: f32, pad0: f32, pad1: f32 };
@group(0) @binding(auto) var<uniform> M: MaskUniforms;
@group(0) @binding(auto) var<storage, read> bits: array<u32>;
`,
  uniformTypes: { colour: "vec4<f32>", width: "f32", height: "f32", pad0: "f32", pad1: "f32" },
};

const source = /* wgsl */ `\
struct Varyings { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vertexMain(@builtin(vertex_index) i: u32) -> Varyings {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
  );
  let c = corners[i];
  var o: Varyings;
  o.position = project_position_to_clipspace(vec3<f32>(c.x * M.width, c.y * M.height, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  o.uv = c;
  return o;
}

@fragment
fn fragmentMain(input: Varyings) -> @location(0) vec4<f32> {
  let size = vec2<u32>(u32(M.width), u32(M.height));
  let p = min(vec2<u32>(input.uv * vec2<f32>(size)), size - vec2<u32>(1u));
  let i = p.y * size.x + p.x;
  let on = (bits[i >> 5u] >> (i & 31u)) & 1u;
  if (on == 0u) { discard; }
  return vec4<f32>(M.colour.rgb * M.colour.a, M.colour.a);
}
`;

export type MaskLayerProps = {
  /** `packRuns` of the mask; uploaded when its identity changes. */
  bits: Uint32Array | null;
  width: number;
  height: number;
  /** RGBA in [0, 1]. */
  colour: [number, number, number, number];
} & LayerProps;

const defaultProps: DefaultProps<MaskLayerProps> = {
  bits: { type: "object", value: null, compare: true },
  width: { type: "number", value: 0 },
  height: { type: "number", value: 0 },
  colour: { type: "array", value: [0.3, 0.55, 0.65, 0.55], compare: true },
};

type State = { model?: Model; buf?: Buffer; src?: Uint32Array | null };

export class MaskLayer extends Layer<MaskLayerProps> {
  static override layerName = "MaskLayer";
  static override defaultProps = defaultProps;
  declare state: State;

  override getShaders() {
    return super.getShaders({ source, modules: [project32, maskUniforms] });
  }

  override initializeState(): void {
    const model = new Model(this.context.device, {
      ...this.getShaders(),
      id: this.props.id,
      topology: "triangle-list",
      bufferLayout: [],
      isInstanced: false,
      vertexCount: 6,
      shaderAssembler: this.context.shaderAssembler,
    });
    this.setState({ model });
  }

  override finalizeState(context: Parameters<Layer["finalizeState"]>[0]): void {
    this.state.buf?.destroy();
    this.state.model?.destroy();
    super.finalizeState(context);
  }

  override draw(): void {
    const { model } = this.state;
    const { bits, width, height, colour, opacity } = this.props;
    if (!model || !bits || width === 0 || height === 0) return;
    if (this.state.src !== bits) {
      this.state.buf?.destroy();
      // Mutated, not setState: a state change from draw would ask for another draw.
      this.state.buf = this.context.device.createBuffer({ data: bits, usage: Buffer.STORAGE | Buffer.COPY_DST });
      this.state.src = bits;
      model.setBindings({ bits: this.state.buf });
    }
    const [r, g, b, a] = colour;
    model.shaderInputs.setProps({ M: { colour: [r, g, b, a * (opacity ?? 1)], width, height, pad0: 0, pad1: 0 } });
    model.draw(this.context.renderPass);
  }
}
