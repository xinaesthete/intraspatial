// Particles drawn as spheres, straight from a storage buffer something else is writing: a
// simulation's positions, never read back to the host. WebGPU only.
//
// Each particle is one camera-facing quad shaded as a sphere (an "impostor"), and it writes the
// sphere's own depth, so spheres intersect each other and other layers correctly in a 3D view. In a
// flat view the same layer draws discs.
//
// INPUTS ARE BUFFERS. `positions` holds `positionStride` floats per particle with x, y, z first;
// `attributes` (optional) holds `attributeStride` words per particle, one of them an integer
// category and one a radius. Nothing here knows which simulation wrote them.
//
// COLOUR IS A PALETTE indexed by the category. A stop with alpha 0 hides its category; the category
// `deadCategory` is never drawn, which is how a fixed-capacity simulation marks unused slots.
//
// The shader is in two parts, after ADR-0014's surface program: `surface` (where the sphere is, its
// normal, whether the pixel is on it) and `material` (palette lookup and lighting).

import { color, type DefaultProps, Layer, type LayerProps, project32, type UpdateParameters } from "@deck.gl/core";
import { Buffer } from "@luma.gl/core";
import { Model } from "@luma.gl/engine";
import type { ShaderModule } from "@luma.gl/shadertools";
import type { Color } from "./DensityLayer";
import { type BufferSource, HandleCache, resolveBuffer } from "./handles";
import { packPalette } from "./palette";
import { CATEGORY_WGSL } from "./particleWgsl";
import { MATERIAL_WGSL } from "./shadingWgsl";

export type ParticleLayerProps = {
  /** `positionStride` f32 per particle, x y z first. A function is asked again at every draw. */
  positions: BufferSource;
  /** Particles to draw (some may be hidden or dead), starting at `first`. */
  count: number;
  /** Index of the first particle drawn: with `count: 1`, one particle picked out of the buffer. */
  first?: number;
  /** A lit sphere, or a flat marker drawn at the sphere's front: a ring round it, or a disc. */
  style?: "sphere" | "ring" | "disc";
  positionStride?: number;
  /** Per-particle words: f32 or u32, `attributeStride` per particle. Absent: one category, one radius. */
  attributes?: BufferSource;
  attributeStride?: number;
  /** Word holding the category within a particle's attributes. */
  categoryOffset?: number;
  /** How the category is stored: a u32, or an f32 holding a whole number. */
  categoryFormat?: "u32" | "f32";
  /** Word holding the radius (f32) within a particle's attributes, or -1 to use `radius` for all. */
  radiusOffset?: number;
  /** One f32 per particle: its radius, for a simulation that keeps radii in a buffer of their own.
   *  Used instead of `radiusOffset` when given. */
  radii?: BufferSource;
  /** The radius when there is no `radii` and `radiusOffset` is -1, in the positions' units. */
  radius?: number;
  /** Multiplies every radius. */
  radiusScale?: number;
  /** Smallest drawn radius, in CSS pixels, so a zoomed-out particle stays visible. */
  radiusMinPixels?: number;
  /** Colour per category; the last stop serves any higher category. Alpha 0 hides a category. */
  colors?: readonly Color[];
  /** This category is never drawn. */
  deadCategory?: number;
  /** 0: flat discs. 1: lit spheres. */
  shading?: number;
} & LayerProps;

const defaultProps: DefaultProps<ParticleLayerProps> = {
  positions: { type: "object", value: null, compare: false },
  count: { type: "number", value: 0, min: 0 },
  first: { type: "number", value: 0, min: 0 },
  style: "sphere",
  positionStride: { type: "number", value: 4, min: 3 },
  attributes: { type: "object", value: null, compare: false },
  radii: { type: "object", value: null, compare: false },
  attributeStride: { type: "number", value: 1, min: 1 },
  categoryOffset: { type: "number", value: 0, min: 0 },
  categoryFormat: "u32",
  radiusOffset: { type: "number", value: -1 },
  radius: { type: "number", value: 0.5, min: 0 },
  radiusScale: { type: "number", value: 1, min: 0 },
  radiusMinPixels: { type: "number", value: 0, min: 0 },
  colors: { type: "object", value: [[255, 255, 255, 255]], compare: true },
  deadCategory: { type: "number", value: 255 },
  shading: { type: "number", value: 1, min: 0, max: 1 },
};

type ParticleUniforms = {
  positionStride: number;
  attributeStride: number;
  categoryOffset: number;
  /** -1: no per-particle radius. */
  radiusOffset: number;
  radius: number;
  radiusScale: number;
  radiusMinPixels: number;
  stops: number;
  dead: number;
  shading: number;
  opacity: number;
  /** 1: attributes are bound. */
  hasAttributes: number;
  /** 1: the category is an f32. */
  categoryIsFloat: number;
  first: number;
  /** 0 sphere, 1 ring, 2 disc. */
  style: number;
  /** 1: `radii` is bound. */
  hasRadii: number;
};

/** The uniforms for a set of props: what the shader needs to find a particle's fields. */
export function particleUniforms(props: ParticleLayerProps, hasAttributes: boolean, hasRadii = false): ParticleUniforms {
  return {
    positionStride: props.positionStride ?? 4,
    attributeStride: props.attributeStride ?? 1,
    categoryOffset: props.categoryOffset ?? 0,
    radiusOffset: hasAttributes ? (props.radiusOffset ?? -1) : -1,
    radius: props.radius ?? 0.5,
    radiusScale: props.radiusScale ?? 1,
    radiusMinPixels: props.radiusMinPixels ?? 0,
    stops: Math.max(1, props.colors?.length ?? 1),
    dead: props.deadCategory ?? 255,
    shading: props.shading ?? 1,
    opacity: props.opacity ?? 1,
    hasAttributes: hasAttributes ? 1 : 0,
    categoryIsFloat: props.categoryFormat === "f32" ? 1 : 0,
    first: Math.floor(props.first ?? 0),
    style: props.style === "ring" ? 1 : props.style === "disc" ? 2 : 0,
    hasRadii: hasRadii ? 1 : 0,
  };
}

// The module and the WGSL variable share a name: luma binds a module's uniform block by it.
const particleModule: ShaderModule<ParticleUniforms> = {
  name: "particle",
  source: /* wgsl */ `\
struct ParticleUniforms {
  positionStride: f32, attributeStride: f32, categoryOffset: f32, radiusOffset: f32,
  radius: f32, radiusScale: f32, radiusMinPixels: f32, stops: f32,
  dead: f32, shading: f32, opacity: f32, hasAttributes: f32,
  categoryIsFloat: f32, first: f32, style: f32, hasRadii: f32,
};
@group(0) @binding(auto) var<uniform> particle: ParticleUniforms;
@group(0) @binding(auto) var<storage, read> positions: array<f32>;
/** Words: a field is read as u32 or reinterpreted as f32. */
@group(0) @binding(auto) var<storage, read> attributes: array<u32>;
@group(0) @binding(auto) var<storage, read> palette: array<vec4<f32>>;
@group(0) @binding(auto) var<storage, read> radii: array<f32>;
`,
  uniformTypes: {
    positionStride: "f32",
    attributeStride: "f32",
    categoryOffset: "f32",
    radiusOffset: "f32",
    radius: "f32",
    radiusScale: "f32",
    radiusMinPixels: "f32",
    stops: "f32",
    dead: "f32",
    shading: "f32",
    opacity: "f32",
    hasAttributes: "f32",
    categoryIsFloat: "f32",
    first: "f32",
    style: "f32",
    hasRadii: "f32",
  },
};

const source = /* wgsl */ `\
${CATEGORY_WGSL}

struct Varyings {
  @builtin(position) position: vec4<f32>,
  /** Position on the quad, the sphere's silhouette being the unit circle. */
  @location(0) uv: vec2<f32>,
  @location(1) @interpolate(flat) color: vec4<f32>,
  /** Depth of the sphere's nearest point minus depth of its centre (negative). */
  @location(2) @interpolate(flat) depthSpan: f32,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instance: u32) -> Varyings {
  let i = instance + u32(particle.first);
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  let p = i * u32(particle.positionStride);
  let world = vec3<f32>(positions[p], positions[p + 1u], positions[p + 2u]);

  var category = 0u;
  var radius = particle.radius;
  if (particle.hasAttributes > 0.5) {
    let a = i * u32(particle.attributeStride);
    category = particle_category(attributes[a + u32(particle.categoryOffset)], particle.categoryIsFloat > 0.5);
    if (particle.radiusOffset >= 0.0) {
      radius = bitcast<f32>(attributes[a + u32(particle.radiusOffset)]);
    }
  }
  if (particle.hasRadii > 0.5) {
    radius = radii[i];
  }
  let rgba = palette[min(category, u32(particle.stops) - 1u)];

  // Common space is where deck's camera matrix applies. Its first two rows are the screen's right
  // and up directions there, in any view, flat or orbiting; their cross product faces the camera
  // or away from it, and abs() below makes the sign irrelevant.
  let centre = project_position_vec3_f32(world);
  let m = project.viewProjectionMatrix;
  let right = normalize(vec3<f32>(m[0][0], m[1][0], m[2][0]));
  let up = normalize(vec3<f32>(m[0][1], m[1][1], m[2][1]));
  let r = max(project_size_float(radius * particle.radiusScale), particle.radiusMinPixels / project.scale);

  let clipCentre = project_common_position_to_clipspace(vec4<f32>(centre, 1.0));
  let clipNear = project_common_position_to_clipspace(vec4<f32>(centre + cross(right, up) * r, 1.0));

  var output: Varyings;
  output.position = project_common_position_to_clipspace(vec4<f32>(centre + (right * corner.x + up * corner.y) * r, 1.0));
  output.uv = corner;
  output.color = vec4<f32>(rgba.rgb, rgba.a * particle.opacity);
  output.depthSpan = -abs(clipNear.z / clipNear.w - clipCentre.z / clipCentre.w);
  if (category == u32(particle.dead) || rgba.a <= 0.0) {
    // Outside the clip volume: the quad is dropped.
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  }
  return output;
}

struct Surface {
  /** Unit normal in view space: x right, y up, z toward the camera. */
  normal: vec3<f32>,
  depth: f32,
  valid: bool,
};

/** Where this pixel meets the sphere. The quad lies in the plane through the sphere's centre, so
 *  its own depth is the centre's, and the surface is nearer by the normal's z share of the span.
 *  Exact in an orthographic view; in perspective the silhouette is that of the centre plane. */
fn surface(uv: vec2<f32>, quadDepth: f32, depthSpan: f32) -> Surface {
  var s: Surface;
  let d2 = dot(uv, uv);
  s.valid = d2 <= 1.0;
  let nz = sqrt(max(1.0 - d2, 0.0));
  s.normal = vec3<f32>(uv, nz);
  s.depth = clamp(quadDepth + nz * depthSpan, 0.0, 1.0);
  return s;
}

${MATERIAL_WGSL}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fragmentMain(input: Varyings) -> FragmentOutput {
  let s = surface(input.uv, input.position.z, input.depthSpan);
  if (!s.valid) {
    discard;
  }
  var output: FragmentOutput;
  if (particle.style > 0.5) {
    // A flat marker at the sphere's front: a ring is the outer quarter of the disc.
    if (particle.style < 1.5 && dot(input.uv, input.uv) < 0.5625) {
      discard;
    }
    output.color = deckgl_premultiplied_alpha(input.color);
    output.depth = clamp(input.position.z + input.depthSpan, 0.0, 1.0);
    return output;
  }
  output.color = deckgl_premultiplied_alpha(vec4<f32>(material(input.color.rgb, s.normal, particle.shading), input.color.a));
  output.depth = s.depth;
  return output;
}
`;

type State = {
  model?: Model;
  handles?: HandleCache;
  palette?: Buffer;
  /** Bound in place of `attributes` when there are none: a binding must always be filled. */
  noAttributes?: Buffer;
  ready: boolean;
};

export class ParticleLayer extends Layer<ParticleLayerProps> {
  static override layerName = "ParticleLayer";
  static override defaultProps = defaultProps;

  declare state: State;

  override getShaders() {
    return super.getShaders({ source, modules: [color, project32, particleModule] });
  }

  override initializeState(): void {
    // A state object whatever happens next: see DensityLayer.initializeState.
    const s: State = { ready: false };
    this.state = s;
    const device = this.context.device;
    if (device.type !== "webgpu") {
      this.raiseError(new Error("ParticleLayer needs deck on WebGPU (deviceProps: { type: 'webgpu' })"), "ParticleLayer");
      return;
    }
    // The fragment shader writes depth, so the pipeline needs a depth format from its first build.
    // deck only tells a model its target's formats at the first draw, by which time a pipeline
    // without one has already failed validation (harmlessly, but loudly).
    const target = device.getDefaultCanvasContext().getCurrentFramebuffer();
    s.model = new Model(device, {
      ...this.getShaders(),
      id: this.props.id,
      topology: "triangle-list",
      bufferLayout: [],
      isInstanced: true,
      vertexCount: 6,
      shaderAssembler: this.context.shaderAssembler,
      depthStencilAttachmentFormat: target.depthStencilAttachment?.texture.format,
      parameters: { depthWriteEnabled: true, depthCompare: "less-equal" },
    });
    s.handles = new HandleCache(device);
    s.noAttributes = device.createBuffer({
      id: `${this.props.id}:no-attributes`,
      data: new Uint32Array(4),
      usage: Buffer.STORAGE | Buffer.COPY_DST,
    });
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
    s.noAttributes?.destroy();
    s.model?.destroy();
    super.finalizeState(context);
  }

  override draw(): void {
    const s = this.state;
    const { model, handles, palette, noAttributes } = s;
    if (!s.ready || !model || !handles || !palette || !noAttributes) return;
    const positions = resolveBuffer(this.props.positions);
    const count = Math.floor(this.props.count);
    if (!positions || count <= 0) return;
    const attributes = resolveBuffer(this.props.attributes);
    const radii = resolveBuffer(this.props.radii);
    // Rebound every draw: the positions handle changes whenever the simulation has stepped.
    model.setBindings({
      positions: handles.wrap(positions),
      attributes: attributes ? handles.wrap(attributes) : noAttributes,
      palette,
      radii: radii ? handles.wrap(radii) : noAttributes,
    });
    model.shaderInputs.setProps({ particle: particleUniforms(this.props, attributes !== null, radii !== null) });
    model.setInstanceCount(count);
    model.draw(this.context.renderPass);
  }
}
