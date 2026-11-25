// A chain of particles drawn as a tube: real extruded geometry, generated in the vertex shader from
// a position buffer something else is writing. WGSL only.
//
// The centreline is a Catmull-Rom curve through the first `count` particles, in order. Rings of
// `sides` vertices are placed along it, `subdivisions` to each particle-to-particle span. Unless a
// profile says otherwise the section is a circle, set square to the curve with one axis across the
// line of sight; how a circle is rolled about the curve cannot be seen, and this choice never
// twists the tube between rings. Colour runs along the chain through a ramp.
//
// The radius can vary along the chain: `taper` closes each end over a few particles (a rounded
// point, as on the dancer's swept creature), and an optional per-particle value scales it.
//
// A `profile` changes the section along the chain: wide and thin, square, coloured, with its broad
// side facing a direction named by other particles (see `tubeProfile.ts`). That is how a protein's
// cartoon view is drawn: `cartoonProfile` makes one. Where a profile names no direction the section
// faces the viewer as before. A new profile is reached over `transition` milliseconds, from
// wherever the section had got to. After FoldSynth's `FluidCartoonView` (Peter Todd, Stephen Todd):
// two radii per point of the curve, a direction from the structure, values eased toward a target.

import { color, type DefaultProps, Layer, type LayerProps, project32, type UpdateParameters } from "@deck.gl/core";
import { Buffer } from "@luma.gl/core";
import { Model } from "@luma.gl/engine";
import type { ShaderModule } from "@luma.gl/shadertools";
import type { Color } from "./DensityLayer";
import { type BufferSource, HandleCache, resolveBuffer } from "./handles";
import { syncPalette } from "./palette";
import { MATERIAL_WGSL, RAMP_WGSL, VIEW_FRAME_WGSL } from "./shadingWgsl";
import { ease, mixProfiles, ORIENT_WORDS, packTransition, roundProfile, SHAPE_FLOATS, type TubeProfile } from "./tubeProfile";

export type TubeLayerProps = {
  /** `positionStride` f32 per particle, x y z first. A function is asked again at every draw. */
  positions: BufferSource;
  /** Particles in the chain: the first `count` in the buffer, in order. */
  count: number;
  positionStride?: number;
  /** Tube radius, in the positions' units. */
  radius?: number;
  /** Vertices around the tube; an odd number is rounded up. */
  sides?: number;
  /** Rings per span between neighbouring particles. */
  subdivisions?: number;
  /** Particles over which each end narrows to a point; 0 leaves the ends open. */
  taper?: number;
  /** Per-particle words (`attributeStride` each), one of them an f32 that scales the radius there. */
  attributes?: BufferSource;
  attributeStride?: number;
  /** Word holding the radius scale within a particle's attributes. */
  radiusOffset?: number;
  /** The section along the chain; null is a round tube. Must cover `count` particles. */
  profile?: TubeProfile | null;
  /** Milliseconds a new `profile` takes to arrive; 0 is at once. */
  transition?: number;
  /** Colour along the chain, first particle to last. */
  colors?: readonly Color[];
  /** 0: flat colour. 1: lit. */
  shading?: number;
} & LayerProps;

const defaultProps: DefaultProps<TubeLayerProps> = {
  positions: { type: "object", value: null, compare: false },
  count: { type: "number", value: 0, min: 0 },
  positionStride: { type: "number", value: 4, min: 3 },
  radius: { type: "number", value: 0.3, min: 0 },
  sides: { type: "number", value: 12, min: 3 },
  subdivisions: { type: "number", value: 6, min: 1 },
  taper: { type: "number", value: 2, min: 0 },
  attributes: { type: "object", value: null, compare: false },
  attributeStride: { type: "number", value: 1, min: 1 },
  radiusOffset: { type: "number", value: 0, min: 0 },
  profile: { type: "object", value: null, compare: false },
  transition: { type: "number", value: 600, min: 0 },
  colors: { type: "object", value: [[255, 255, 255, 255]], compare: true },
  shading: { type: "number", value: 1, min: 0, max: 1 },
};

type TubeUniforms = {
  positionStride: number;
  count: number;
  radius: number;
  sides: number;
  subdivisions: number;
  stops: number;
  shading: number;
  opacity: number;
  taper: number;
  /** 1: per-particle radius scales are bound. */
  hasAttributes: number;
  attributeStride: number;
  radiusOffset: number;
  /** How far the section has got from the profile it is leaving (0) to the one it is reaching (1). */
  blend: number;
  pad0: number;
  pad1: number;
  pad2: number;
};

/** Vertices the tube needs: two triangles per side per ring-to-ring step. */
export function tubeVertexCount(count: number, sides: number, subdivisions: number): number {
  return count < 2 ? 0 : (count - 1) * subdivisions * sides * 6;
}

const tubeModule: ShaderModule<TubeUniforms> = {
  name: "tube",
  source: /* wgsl */ `\
struct TubeUniforms {
  positionStride: f32, count: f32, radius: f32, sides: f32,
  subdivisions: f32, stops: f32, shading: f32, opacity: f32,
  taper: f32, hasAttributes: f32, attributeStride: f32, radiusOffset: f32,
  blend: f32, pad0: f32, pad1: f32, pad2: f32,
};
@group(0) @binding(auto) var<uniform> tube: TubeUniforms;
@group(0) @binding(auto) var<storage, read> positions: array<f32>;
@group(0) @binding(auto) var<storage, read> attributes: array<u32>;
@group(0) @binding(auto) var<storage, read> palette: array<vec4<f32>>;
// Per particle, the profile being left then the one being reached: three vec4 each (sizes, colour,
// form) in shapes, one vec4 of particle indices each in orients.
@group(0) @binding(auto) var<storage, read> shapes: array<vec4<f32>>;
@group(0) @binding(auto) var<storage, read> orients: array<vec4<u32>>;
`,
  uniformTypes: {
    positionStride: "f32",
    count: "f32",
    radius: "f32",
    sides: "f32",
    subdivisions: "f32",
    stops: "f32",
    shading: "f32",
    opacity: "f32",
    taper: "f32",
    hasAttributes: "f32",
    attributeStride: "f32",
    radiusOffset: "f32",
    blend: "f32",
    pad0: "f32",
    pad1: "f32",
    pad2: "f32",
  },
};

const source = /* wgsl */ `\
${VIEW_FRAME_WGSL}
${MATERIAL_WGSL}
${RAMP_WGSL}

/** Particle i in common space; indices past either end repeat the end, which stops the curve there. */
fn tube_point(i: i32) -> vec3<f32> {
  let p = u32(clamp(i, 0, i32(tube.count) - 1)) * u32(tube.positionStride);
  return project_position_vec3_f32(vec3<f32>(positions[p], positions[p + 1u], positions[p + 2u]));
}

/** The radius scale at particle i: 1 unless per-particle scales are bound. */
fn tube_scale(i: i32) -> f32 {
  if (tube.hasAttributes < 0.5) {
    return 1.0;
  }
  let a = u32(clamp(i, 0, i32(tube.count) - 1)) * u32(tube.attributeStride) + u32(tube.radiusOffset);
  return bitcast<f32>(attributes[a]);
}

/** How much of the radius is left at a distance (in particles) from an end: a quarter circle, so
 *  the end is a rounded point. */
fn tube_taper(fromEnd: f32) -> f32 {
  if (tube.taper <= 0.0) {
    return 1.0;
  }
  let x = clamp(fromEnd / tube.taper, 0.0, 1.0);
  return sqrt(1.0 - (1.0 - x) * (1.0 - x));
}

struct Section {
  /** widthIn, widthOut, thickIn, thickOut: multiples of the radius. */
  size: vec4<f32>,
  /** rgb, and how far it replaces the ramp's colour. */
  colour: vec4<f32>,
  /** x: 0 an ellipse, 1 a rectangle. */
  form: vec4<f32>,
};

/** The section at particle i, as far as the change of profile has got. */
fn tube_section(i: i32) -> Section {
  let p = u32(clamp(i, 0, i32(tube.count) - 1)) * 6u;
  var s: Section;
  s.size = mix(shapes[p], shapes[p + 3u], tube.blend);
  s.colour = mix(shapes[p + 1u], shapes[p + 4u], tube.blend);
  s.form = mix(shapes[p + 2u], shapes[p + 5u], tube.blend);
  return s;
}

/** The direction four particles name, (P[y] - P[x]) + (P[w] - P[z]), or "otherwise" when they name none. */
fn tube_named(o: vec4<u32>, otherwise: vec3<f32>) -> vec3<f32> {
  let d = (tube_point(i32(o.y)) - tube_point(i32(o.x))) + (tube_point(i32(o.w)) - tube_point(i32(o.z)));
  let named = o.x != 0xFFFFFFFFu && dot(d, d) > 1e-12;
  return select(otherwise, d / max(length(d), 1e-6), named);
}

/** Which way the broad side of the section faces at particle i: what its profile names, else
 *  across the line of sight, so that an unnamed section is seen flat on. */
fn tube_up(i: i32, view: ViewFrame) -> vec3<f32> {
  let step = tube_point(i + 1) - tube_point(i - 1);
  let along = step / max(length(step), 1e-6);
  var facing = cross(along, view.toward);
  // End-on to the camera: any axis across the line of sight will do.
  facing = select(facing, view.right, dot(facing, facing) < 1e-8);
  facing = normalize(facing);
  let p = u32(clamp(i, 0, i32(tube.count) - 1)) * 2u;
  let a = tube_named(orients[p], facing);
  var b = tube_named(orients[p + 1u], facing);
  // The section looks the same turned half way round, so take the nearer of the two.
  b = select(b, -b, dot(a, b) < 0.0);
  let up = mix(a, b, tube.blend);
  return up / max(length(up), 1e-6);
}

/** |x|^k with x's sign: turns a circle's cosine and sine into a squarer outline as k falls below 1. */
fn tube_power(x: f32, k: f32) -> f32 {
  return sign(x) * pow(abs(x), k);
}

struct Varyings {
  @builtin(position) position: vec4<f32>,
  /** View space: x right, y up, z toward the camera. */
  @location(0) normal: vec3<f32>,
  /** 0 at the first particle, 1 at the last. */
  @location(1) along: f32,
  /** The profile's colour, and how far it replaces the ramp's. */
  @location(2) colour: vec4<f32>,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> Varyings {
  // (ring, side) offsets of a quad's six vertices.
  var corners = array<vec2<u32>, 6>(
    vec2<u32>(0u, 0u), vec2<u32>(1u, 0u), vec2<u32>(1u, 1u),
    vec2<u32>(0u, 0u), vec2<u32>(1u, 1u), vec2<u32>(0u, 1u)
  );
  let sides = u32(tube.sides);
  let sub = u32(tube.subdivisions);
  let quad = vertexIndex / 6u;
  let corner = corners[vertexIndex % 6u];
  let side = quad % sides + corner.y;
  // The span is the quad's, particle i to i + 1, and k counts its rings 0..sub. The ring two spans
  // share is worked out from each side's own span: the same outline either way, but a side's own
  // choice of which way is up, so no quad is ever twisted by the other span's choice.
  let i = i32(quad / sides / sub);
  let k = (quad / sides) % sub + corner.x;
  let f = f32(k) / f32(sub);

  // Catmull-Rom through particles i-1 .. i+2, at fraction f of the span i .. i+1.
  let p0 = tube_point(i - 1);
  let p1 = tube_point(i);
  let p2 = tube_point(i + 1);
  let p3 = tube_point(i + 2);
  let a = 2.0 * p1;
  let b = p2 - p0;
  let c = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3;
  let d = -p0 + 3.0 * p1 - 3.0 * p2 + p3;
  let centre = 0.5 * (a + b * f + c * f * f + d * f * f * f);
  let velocity = 0.5 * (b + 2.0 * c * f + 3.0 * d * f * f);

  let view = view_frame();
  let tangent = velocity / max(length(velocity), 1e-6);
  let up0 = tube_up(i, view);
  var up1 = tube_up(i + 1, view);
  up1 = select(up1, -up1, dot(up0, up1) < 0.0);
  let up = mix(up0, up1, f);
  // Two axes across the tube: "broad" is "up" made square to the curve, "thin" is square to both.
  // Where "up" runs nearly along the curve it says nothing about which way to face (a helix not
  // yet wound names its own direction of travel), so there the section turns to face the viewer.
  var facing = cross(tangent, view.toward);
  facing = select(facing, view.right, dot(facing, facing) < 1e-8);
  var flat = normalize(cross(tangent, normalize(facing)));
  let named = cross(tangent, up);
  let sure = smoothstep(0.03, 0.25, dot(named, named));
  flat = select(flat, -flat, dot(flat, named) < 0.0);
  var thin = mix(flat, named / max(length(named), 1e-6), sure);
  thin = normalize(thin - tangent * dot(thin, tangent));
  let broad = cross(thin, tangent);

  // Sizes leave particle i at its Out values and reach i + 1's In values one ring early; the last
  // ring is i + 1's Out, which is where the next span starts. A step between In and Out is
  // therefore a wall one ring long, not a hole.
  let s0 = tube_section(i);
  let s1 = tube_section(i + 1);
  let g = min(f32(k) / f32(max(sub - 1u, 1u)), 1.0);
  var size = mix(s0.size.yw, s1.size.xz, g);
  size = select(size, s1.size.yw, k == sub);
  size = max(size, vec2<f32>(1e-4));

  // Round the section: an ellipse, squared off as the form asks.
  let power = 1.0 - 0.8 * clamp(mix(s0.form.x, s1.form.x, f), 0.0, 1.0);
  let angle = 6.28318530718 * f32(side) / f32(sides);
  let cs = vec2<f32>(cos(angle), sin(angle));
  let outline = vec2<f32>(tube_power(cs.x, power), tube_power(cs.y, power));
  let outward = vec2<f32>(tube_power(cs.x, 2.0 - power), tube_power(cs.y, 2.0 - power)) / size.yx;
  let normal = normalize(thin * outward.x + broad * outward.y);

  var output: Varyings;
  // Along the chain in particles, and from there the radius: the profile, times the nearer end's taper.
  let at = f32(i) + f;
  let radius = project_size_float(tube.radius * mix(tube_scale(i), tube_scale(i + 1), f) * tube_taper(min(at, tube.count - 1.0 - at)));
  let offset = thin * (outline.x * size.y) + broad * (outline.y * size.x);
  output.position = project_common_position_to_clipspace(vec4<f32>(centre + offset * radius, 1.0));
  output.normal = to_view(normal, view);
  output.along = (f32(i) + f) / max(tube.count - 1.0, 1.0);
  output.colour = mix(s0.colour, s1.colour, f);
  return output;
}

@fragment
fn fragmentMain(input: Varyings) -> @location(0) vec4<f32> {
  let ramp = ramp_at(input.along, u32(tube.stops));
  let base = mix(ramp.rgb, input.colour.rgb, clamp(input.colour.a, 0.0, 1.0));
  return deckgl_premultiplied_alpha(vec4<f32>(material(base, normalize(input.normal), tube.shading), ramp.a * tube.opacity));
}
`;

type State = {
  model?: Model;
  handles?: HandleCache;
  palette?: Buffer;
  /** Bound in place of `attributes` when there are none: a binding must always be filled. */
  noAttributes?: Buffer;
  /** Both ends of the change of profile under way, and the buffers holding them. */
  from?: TubeProfile;
  to?: TubeProfile;
  /** `performance.now()` when the change began, and how long it takes. */
  started: number;
  duration: number;
  shapes?: Buffer;
  orients?: Buffer;
  ready: boolean;
};

export class TubeLayer extends Layer<TubeLayerProps> {
  static override layerName = "TubeLayer";
  static override defaultProps = defaultProps;

  declare state: State;

  override getShaders() {
    return super.getShaders({ source, modules: [color, project32, tubeModule] });
  }

  override initializeState(): void {
    // A state object whatever happens next: see DensityLayer.initializeState.
    const s: State = { ready: false, started: 0, duration: 0 };
    this.state = s;
    const device = this.context.device;
    if (device.type !== "webgpu") {
      this.raiseError(new Error("TubeLayer needs deck on WebGPU (deviceProps: { type: 'webgpu' })"), "TubeLayer");
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
      s.palette = syncPalette(this.context.device, `${this.props.id}:palette`, s.palette, props.colors ?? []);
    }
    const count = Math.max(0, Math.floor(props.count));
    if (props.profile !== oldProps.profile || s.to?.shapes.length !== count * SHAPE_FLOATS) this.retarget(count);
  }

  /** How far the change of profile under way has got, 0..1. */
  private blend(): number {
    const s = this.state;
    return s.duration <= 0 ? 1 : ease((performance.now() - s.started) / s.duration);
  }

  /** Start changing to the profile in the props, from wherever the section has got to. */
  private retarget(count: number): void {
    const s = this.state;
    const given = this.props.profile;
    const fits = given && given.shapes.length === count * SHAPE_FLOATS && given.orients.length === count * ORIENT_WORDS;
    const to = fits ? given : roundProfile(count);
    const duration = this.props.transition ?? 600;
    const continues = s.from && s.to && s.to.shapes.length === to.shapes.length && duration > 0;
    s.from = continues && s.from && s.to ? mixProfiles(s.from, s.to, this.blend()) : to;
    s.to = to;
    s.started = performance.now();
    s.duration = continues ? duration : 0;
    const packed = packTransition(s.from, s.to);
    // Never empty: a binding must always be filled.
    const shapes = packed.shapes.length > 0 ? packed.shapes : new Float32Array(SHAPE_FLOATS * 2);
    const orients = packed.orients.length > 0 ? packed.orients : new Uint32Array(ORIENT_WORDS * 2);
    s.shapes = this.fill(s.shapes, "shapes", shapes);
    s.orients = this.fill(s.orients, "orients", orients);
  }

  /** Keep one of the layer's buffers holding `data`: rewritten in place while it fits, else replaced. */
  private fill(current: Buffer | undefined, name: string, data: Float32Array | Uint32Array): Buffer {
    if (current && current.byteLength === data.byteLength) {
      current.write(data);
      return current;
    }
    current?.destroy();
    return this.context.device.createBuffer({ id: `${this.props.id}:${name}`, data, usage: Buffer.STORAGE | Buffer.COPY_DST });
  }

  override finalizeState(context: Parameters<Layer["finalizeState"]>[0]): void {
    const s = this.state;
    s.handles?.destroy(); // wrappers only: the buffers are their owner's
    s.palette?.destroy();
    s.noAttributes?.destroy();
    s.shapes?.destroy();
    s.orients?.destroy();
    s.model?.destroy();
    super.finalizeState(context);
  }

  override draw(): void {
    const s = this.state;
    const { model, handles, palette, noAttributes, shapes, orients } = s;
    if (!s.ready || !model || !handles || !palette || !noAttributes || !shapes || !orients) return;
    const positions = resolveBuffer(this.props.positions);
    const count = Math.floor(this.props.count);
    // Even, so that the outline is the same turned half way round (see the vertex shader).
    const sides = Math.max(4, Math.ceil((this.props.sides ?? 12) / 2) * 2);
    const subdivisions = Math.max(1, Math.floor(this.props.subdivisions ?? 6));
    const vertices = tubeVertexCount(count, sides, subdivisions);
    if (!positions || vertices === 0) return;
    const attributes = resolveBuffer(this.props.attributes);
    model.setBindings({
      positions: handles.wrap(positions),
      attributes: attributes ? handles.wrap(attributes) : noAttributes,
      palette,
      shapes,
      orients,
    });
    const blend = this.blend();
    // Still changing: deck must draw again even if nothing else asks it to.
    if (blend < 1) this.setNeedsRedraw();
    const uniforms: TubeUniforms = {
      positionStride: this.props.positionStride ?? 4,
      count,
      radius: this.props.radius ?? 0.3,
      sides,
      subdivisions,
      stops: Math.max(1, this.props.colors?.length ?? 1),
      shading: this.props.shading ?? 1,
      opacity: this.props.opacity ?? 1,
      taper: this.props.taper ?? 2,
      hasAttributes: attributes ? 1 : 0,
      attributeStride: this.props.attributeStride ?? 1,
      radiusOffset: this.props.radiusOffset ?? 0,
      blend,
      pad0: 0,
      pad1: 0,
      pad2: 0,
    };
    model.shaderInputs.setProps({ tube: uniforms });
    model.setVertexCount(vertices);
    model.draw(this.context.renderPass);
  }
}
