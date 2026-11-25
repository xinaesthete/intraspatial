// A triangle mesh over a position buffer something else is writing: the triangles are index
// triples into the particles, so the surface moves with them and nothing is read back. WGSL only.
//
// Made for surfaces built from a bond table (`bondFaces`: sheets look like sheets), but it knows
// only positions and indices. Each triangle is lit flat, the same from either side, and coloured
// along the chain by its corners' indices through a ramp.

import { color, type DefaultProps, Layer, type LayerProps, project32, type UpdateParameters } from "@deck.gl/core";
import { Buffer } from "@luma.gl/core";
import { Model } from "@luma.gl/engine";
import type { ShaderModule } from "@luma.gl/shadertools";
import type { Color } from "./DensityLayer";
import { type BufferSource, HandleCache, resolveBuffer } from "./handles";
import { syncPalette } from "./palette";
import { MATERIAL_WGSL, RAMP_WGSL, VIEW_FRAME_WGSL } from "./shadingWgsl";

export type SurfaceLayerProps = {
  /** `positionStride` f32 per particle, x y z first. A function is asked again at every draw. */
  positions: BufferSource;
  /** Three particle indices per triangle. */
  indices: Uint32Array | null;
  /** Particles the indices range over: sets where on the colour ramp an index falls. */
  count: number;
  positionStride?: number;
  /** Colour by particle index, first to last. */
  colors?: readonly Color[];
  /** 0: flat colour. 1: lit. */
  shading?: number;
} & LayerProps;

const defaultProps: DefaultProps<SurfaceLayerProps> = {
  positions: { type: "object", value: null, compare: false },
  indices: { type: "object", value: null, compare: false },
  count: { type: "number", value: 0, min: 0 },
  positionStride: { type: "number", value: 4, min: 3 },
  colors: { type: "object", value: [[255, 255, 255, 255]], compare: true },
  shading: { type: "number", value: 1, min: 0, max: 1 },
};

type SurfaceUniforms = {
  positionStride: number;
  count: number;
  stops: number;
  shading: number;
  opacity: number;
  pad0: number;
  pad1: number;
  pad2: number;
};

const surfaceModule: ShaderModule<SurfaceUniforms> = {
  name: "surface",
  source: /* wgsl */ `\
struct SurfaceUniforms {
  positionStride: f32, count: f32, stops: f32, shading: f32,
  opacity: f32, pad0: f32, pad1: f32, pad2: f32,
};
@group(0) @binding(auto) var<uniform> surface: SurfaceUniforms;
@group(0) @binding(auto) var<storage, read> positions: array<f32>;
@group(0) @binding(auto) var<storage, read> indices: array<u32>;
@group(0) @binding(auto) var<storage, read> palette: array<vec4<f32>>;
`,
  uniformTypes: {
    positionStride: "f32",
    count: "f32",
    stops: "f32",
    shading: "f32",
    opacity: "f32",
    pad0: "f32",
    pad1: "f32",
    pad2: "f32",
  },
};

const source = /* wgsl */ `\
${VIEW_FRAME_WGSL}
${MATERIAL_WGSL}
${RAMP_WGSL}

fn surface_point(index: u32) -> vec3<f32> {
  let p = index * u32(surface.positionStride);
  return project_position_vec3_f32(vec3<f32>(positions[p], positions[p + 1u], positions[p + 2u]));
}

struct Varyings {
  @builtin(position) position: vec4<f32>,
  /** The triangle's normal in view space, the same at its three corners. */
  @location(0) @interpolate(flat) normal: vec3<f32>,
  @location(1) along: f32,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> Varyings {
  let first = vertexIndex - vertexIndex % 3u;
  let index = indices[vertexIndex];
  let a = surface_point(indices[first]);
  let b = surface_point(indices[first + 1u]);
  let c = surface_point(indices[first + 2u]);
  let across = cross(b - a, c - a);
  // A triangle squashed to a line has no normal: face the camera.
  let normal = select(vec3<f32>(0.0, 0.0, 1.0), to_view(normalize(across), view_frame()), dot(across, across) > 1e-12);

  var output: Varyings;
  output.position = project_common_position_to_clipspace(vec4<f32>(surface_point(index), 1.0));
  output.normal = normal;
  output.along = f32(index) / max(surface.count - 1.0, 1.0);
  return output;
}

@fragment
fn fragmentMain(input: Varyings) -> @location(0) vec4<f32> {
  let base = ramp_at(input.along, u32(surface.stops));
  // Lit the same from either side: turn the normal to face the camera.
  let normal = input.normal * select(-1.0, 1.0, input.normal.z >= 0.0);
  return deckgl_premultiplied_alpha(vec4<f32>(material(base.rgb, normal, surface.shading), base.a * surface.opacity));
}
`;

type State = {
  model?: Model;
  handles?: HandleCache;
  palette?: Buffer;
  indices?: Buffer;
  /** Indices in `indices`: three per triangle. */
  indexCount: number;
  ready: boolean;
};

export class SurfaceLayer extends Layer<SurfaceLayerProps> {
  static override layerName = "SurfaceLayer";
  static override defaultProps = defaultProps;

  declare state: State;

  override getShaders() {
    return super.getShaders({ source, modules: [color, project32, surfaceModule] });
  }

  override initializeState(): void {
    // A state object whatever happens next: see DensityLayer.initializeState.
    const s: State = { ready: false, indexCount: 0 };
    this.state = s;
    const device = this.context.device;
    if (device.type !== "webgpu") {
      this.raiseError(new Error("SurfaceLayer needs deck on WebGPU (deviceProps: { type: 'webgpu' })"), "SurfaceLayer");
      return;
    }
    s.model = new Model(device, {
      ...this.getShaders(),
      id: this.props.id,
      topology: "triangle-list",
      bufferLayout: [],
      isInstanced: false,
      vertexCount: 0,
      shaderAssembler: this.context.shaderAssembler,
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
    if (props.indices !== oldProps.indices || !s.indices) {
      const wanted = props.indices ?? new Uint32Array(0);
      // Whole triangles only, and never an empty buffer: a binding must always be filled.
      s.indexCount = wanted.length - (wanted.length % 3);
      const data = wanted.length > 0 ? wanted : new Uint32Array(4);
      if (s.indices && s.indices.byteLength >= data.byteLength) s.indices.write(data);
      else {
        s.indices?.destroy();
        s.indices = this.context.device.createBuffer({
          id: `${this.props.id}:indices`,
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
    s.indices?.destroy();
    s.model?.destroy();
    super.finalizeState(context);
  }

  override draw(): void {
    const s = this.state;
    const { model, handles, palette, indices } = s;
    if (!s.ready || !model || !handles || !palette || !indices) return;
    const positions = resolveBuffer(this.props.positions);
    if (!positions || s.indexCount === 0) return;
    model.setBindings({ positions: handles.wrap(positions), indices, palette });
    const uniforms: SurfaceUniforms = {
      positionStride: this.props.positionStride ?? 4,
      count: this.props.count,
      stops: Math.max(1, this.props.colors?.length ?? 1),
      shading: this.props.shading ?? 1,
      opacity: this.props.opacity ?? 1,
      pad0: 0,
      pad1: 0,
      pad2: 0,
    };
    model.shaderInputs.setProps({ surface: uniforms });
    model.setVertexCount(s.indexCount);
    model.draw(this.context.renderPass);
  }
}
