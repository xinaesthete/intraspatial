// Bonds between particles, drawn as lines straight from a bond-slot table and a position buffer
// that something else is writing. WebGPU only; the companion of ParticleLayer.
//
// The table has `slotsPerParticle` slots per particle, each `slotWords` u32 wide. Word 0 of a slot
// is `partner | kind << 24`, or all ones when the slot is empty. A bond appears in both ends' rows
// and is drawn from the lower-numbered end only, so once (see `BOND_WGSL`).
//
// One line instance per slot: empty slots and the second copy of each bond are moved outside the
// clip volume in the vertex shader, so the host never compacts or counts the table.

import { color, type DefaultProps, Layer, type LayerProps, project32, type UpdateParameters } from "@deck.gl/core";
import { Buffer } from "@luma.gl/core";
import { Model } from "@luma.gl/engine";
import type { ShaderModule } from "@luma.gl/shadertools";
import type { Color } from "./DensityLayer";
import { type BufferSource, HandleCache, resolveBuffer } from "./handles";
import { packPalette } from "./palette";
import { BOND_WGSL } from "./particleWgsl";

export type BondLayerProps = {
  /** `positionStride` f32 per particle, x y z first. A function is asked again at every draw. */
  positions: BufferSource;
  /** The slot table. */
  slots: BufferSource;
  /** Particles: the table has `count * slotsPerParticle` slots. */
  count: number;
  slotsPerParticle: number;
  /** u32 words per slot. */
  slotWords?: number;
  positionStride?: number;
  /** Colour per bond kind; the last stop serves any higher kind. Alpha 0 hides a kind. */
  colors?: readonly Color[];
} & LayerProps;

const defaultProps: DefaultProps<BondLayerProps> = {
  positions: { type: "object", value: null, compare: false },
  slots: { type: "object", value: null, compare: false },
  count: { type: "number", value: 0, min: 0 },
  slotsPerParticle: { type: "number", value: 8, min: 1 },
  slotWords: { type: "number", value: 4, min: 1 },
  positionStride: { type: "number", value: 4, min: 3 },
  colors: { type: "object", value: [[255, 255, 255, 120]], compare: true },
};

type BondUniforms = {
  positionStride: number;
  slotsPerParticle: number;
  slotWords: number;
  stops: number;
  opacity: number;
  pad0: number;
  pad1: number;
  pad2: number;
};

const bondModule: ShaderModule<BondUniforms> = {
  name: "bond",
  source: /* wgsl */ `\
struct BondUniforms {
  positionStride: f32, slotsPerParticle: f32, slotWords: f32, stops: f32,
  opacity: f32, pad0: f32, pad1: f32, pad2: f32,
};
@group(0) @binding(auto) var<uniform> bond: BondUniforms;
@group(0) @binding(auto) var<storage, read> positions: array<f32>;
@group(0) @binding(auto) var<storage, read> slots: array<u32>;
@group(0) @binding(auto) var<storage, read> palette: array<vec4<f32>>;
`,
  uniformTypes: {
    positionStride: "f32",
    slotsPerParticle: "f32",
    slotWords: "f32",
    stops: "f32",
    opacity: "f32",
    pad0: "f32",
    pad1: "f32",
    pad2: "f32",
  },
};

const source = /* wgsl */ `\
${BOND_WGSL}

struct Varyings {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) color: vec4<f32>,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) slot: u32) -> Varyings {
  let e = bond_ends(slots[slot * u32(bond.slotWords)], slot, u32(bond.slotsPerParticle));
  let rgba = palette[min(e.kind, u32(bond.stops) - 1u)];
  var output: Varyings;
  output.color = vec4<f32>(rgba.rgb, rgba.a * bond.opacity);
  if (!e.draw || rgba.a <= 0.0) {
    // Outside the clip volume: the line is dropped.
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    return output;
  }
  let p = select(e.a, e.b, vertexIndex == 1u) * u32(bond.positionStride);
  let world = vec3<f32>(positions[p], positions[p + 1u], positions[p + 2u]);
  output.position = project_position_to_clipspace(world, vec3<f32>(0.0), vec3<f32>(0.0));
  return output;
}

@fragment
fn fragmentMain(input: Varyings) -> @location(0) vec4<f32> {
  return deckgl_premultiplied_alpha(input.color);
}
`;

type State = {
  model?: Model;
  handles?: HandleCache;
  palette?: Buffer;
  ready: boolean;
};

export class BondLayer extends Layer<BondLayerProps> {
  static override layerName = "BondLayer";
  static override defaultProps = defaultProps;

  declare state: State;

  override getShaders() {
    return super.getShaders({ source, modules: [color, project32, bondModule] });
  }

  override initializeState(): void {
    // A state object whatever happens next: see DensityLayer.initializeState.
    const s: State = { ready: false };
    this.state = s;
    const device = this.context.device;
    if (device.type !== "webgpu") {
      this.raiseError(new Error("BondLayer needs deck on WebGPU (deviceProps: { type: 'webgpu' })"), "BondLayer");
      return;
    }
    s.model = new Model(device, {
      ...this.getShaders(),
      id: this.props.id,
      topology: "line-list",
      bufferLayout: [],
      isInstanced: true,
      vertexCount: 2,
      shaderAssembler: this.context.shaderAssembler,
    });
    s.handles = new HandleCache(device);
    s.ready = true;
  }

  override updateState({ props, oldProps }: UpdateParameters<this>): void {
    const s = this.state;
    if (!s.ready) return;
    if (props.colors !== oldProps.colors || !s.palette) {
      const data = packPalette(props.colors ?? []);
      if (s.palette && s.palette.byteLength >= data.byteLength) s.palette.write(data);
      else {
        s.palette?.destroy();
        s.palette = this.context.device.createBuffer({
          id: `${this.props.id}:palette`,
          data,
          usage: Buffer.STORAGE | Buffer.COPY_DST,
        });
      }
    }
  }

  override finalizeState(context: Parameters<Layer["finalizeState"]>[0]): void {
    const s = this.state;
    s.handles?.destroy(); // wrappers only: the buffers are their owner's
    s.palette?.destroy();
    s.model?.destroy();
    super.finalizeState(context);
  }

  override draw(): void {
    const s = this.state;
    const { model, handles, palette } = s;
    if (!s.ready || !model || !handles || !palette) return;
    const positions = resolveBuffer(this.props.positions);
    const slots = resolveBuffer(this.props.slots);
    const instances = Math.floor(this.props.count) * Math.floor(this.props.slotsPerParticle);
    if (!positions || !slots || instances <= 0) return;
    model.setBindings({ positions: handles.wrap(positions), slots: handles.wrap(slots), palette });
    const uniforms: BondUniforms = {
      positionStride: this.props.positionStride ?? 4,
      slotsPerParticle: this.props.slotsPerParticle,
      slotWords: this.props.slotWords ?? 4,
      stops: Math.max(1, this.props.colors?.length ?? 1),
      opacity: this.props.opacity ?? 1,
      pad0: 0,
      pad1: 0,
      pad2: 0,
    };
    model.shaderInputs.setProps({ bond: uniforms });
    model.setInstanceCount(instances);
    model.draw(this.context.renderPass);
  }
}
