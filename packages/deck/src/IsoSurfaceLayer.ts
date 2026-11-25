// The surface where a 3D field crosses a level, drawn by marching a ray through the field for
// every pixel. No mesh is made: the field is a buffer something else writes (`splatVolume` for
// metaballs or a 3D density), and changing the level or the field costs nothing extra. WGSL only.
//
// The field's box is drawn as twelve triangles. Each pixel of it walks its line of sight through
// the box in steps of half a voxel, stops at the first step where the field is at or above the
// level, and places the surface between that step and the one before. It writes that point's
// depth, so the surface hides and is hidden by other layers' geometry correctly, and lights it
// from the field's slope there.
//
// Field layout, as `splatVolume` writes it: one f32 per voxel, x fastest
// (`field[x + nx * (y + ny * z)]`), voxel (x, y, z) centred at `origin + (x + 0.5, …) * voxel`.
// Values between voxel centres are interpolated; past the outermost centres the edge value holds.
//
// The march is where a mask belongs when there is one (a second buffer the same shape, asked at
// each step whether this voxel counts), and a mesh for export is a different consumer of the same
// field. Neither is here yet.

import { color, type DefaultProps, Layer, type LayerProps, project32 } from "@deck.gl/core";
import { Model } from "@luma.gl/engine";
import type { ShaderModule } from "@luma.gl/shadertools";
import type { Color } from "./DensityLayer";
import { type BufferSource, HandleCache, resolveBuffer } from "./handles";
import { ISO_MARCH_WGSL } from "./isoMarchWgsl";
import { MATERIAL_WGSL, VIEW_FRAME_WGSL } from "./shadingWgsl";

export type IsoSurfaceLayerProps = {
  /** One f32 per voxel, x fastest. A function is asked again at every draw. */
  field: BufferSource;
  /** Voxels along x, y and z. */
  dims: readonly [number, number, number];
  /** The field's low corner, in the layer's coordinates. */
  origin: readonly [number, number, number];
  /** Edge of a voxel. */
  voxel: number;
  /** The level: the surface is where the field equals this, with higher values inside. */
  threshold?: number;
  color?: Color;
  /** 0: flat colour. 1: lit. */
  shading?: number;
  /** The most steps a pixel may take through the box. */
  maxSteps?: number;
} & LayerProps;

const defaultProps: DefaultProps<IsoSurfaceLayerProps> = {
  field: { type: "object", value: null, compare: false },
  dims: { type: "array", value: [1, 1, 1], compare: true },
  origin: { type: "array", value: [0, 0, 0], compare: true },
  voxel: { type: "number", value: 1, min: 0 },
  threshold: { type: "number", value: 0.5 },
  color: { type: "color", value: [255, 255, 255, 255] },
  shading: { type: "number", value: 1, min: 0, max: 1 },
  maxSteps: { type: "number", value: 512, min: 1 },
};

type Vec4 = [number, number, number, number];

type IsoUniforms = {
  /** nx, ny, nz, and the level. */
  dims: Vec4;
  /** The low corner, and the voxel size. */
  origin: Vec4;
  tint: Vec4;
  /** Most steps, shading, opacity. */
  march: Vec4;
};

const isoModule: ShaderModule<IsoUniforms> = {
  name: "iso",
  source: /* wgsl */ `\
struct IsoUniforms {
  dims: vec4<f32>,
  origin: vec4<f32>,
  tint: vec4<f32>,
  march: vec4<f32>,
};
@group(0) @binding(auto) var<uniform> iso: IsoUniforms;
@group(0) @binding(auto) var<storage, read> field: array<f32>;
`,
  uniformTypes: {
    dims: "vec4<f32>",
    origin: "vec4<f32>",
    tint: "vec4<f32>",
    march: "vec4<f32>",
  },
};

const source = /* wgsl */ `\
${VIEW_FRAME_WGSL}
${MATERIAL_WGSL}

struct Varyings {
  @builtin(position) position: vec4<f32>,
  /** Where on the box this pixel is, in deck's common space. */
  @location(0) place: vec3<f32>,
  /** The box's low corner in common space, and the three rows that turn a common-space offset from
   *  it into box coordinates (0..1 along each edge). */
  @location(1) @interpolate(flat) corner: vec3<f32>,
  @location(2) @interpolate(flat) row0: vec3<f32>,
  @location(3) @interpolate(flat) row1: vec3<f32>,
  @location(4) @interpolate(flat) row2: vec3<f32>,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> Varyings {
  // The box's eight corners are the three bits of an index; these are its twelve triangles.
  var corners = array<u32, 36>(
    0u, 1u, 3u, 0u, 3u, 2u, 4u, 6u, 7u, 4u, 7u, 5u, 0u, 4u, 5u, 0u, 5u, 1u,
    2u, 3u, 7u, 2u, 7u, 6u, 0u, 2u, 6u, 0u, 6u, 4u, 1u, 5u, 7u, 1u, 7u, 3u
  );
  let c = corners[vertexIndex];
  let unit = vec3<f32>(f32(c & 1u), f32((c >> 1u) & 1u), f32((c >> 2u) & 1u));
  let size = iso.dims.xyz * iso.origin.w;
  let o = project_position_vec3_f32(iso.origin.xyz);
  let a = project_position_vec3_f32(iso.origin.xyz + vec3<f32>(size.x, 0.0, 0.0)) - o;
  let b = project_position_vec3_f32(iso.origin.xyz + vec3<f32>(0.0, size.y, 0.0)) - o;
  let d = project_position_vec3_f32(iso.origin.xyz + vec3<f32>(0.0, 0.0, size.z)) - o;
  let det = dot(a, cross(b, d));
  var output: Varyings;
  output.place = o + a * unit.x + b * unit.y + d * unit.z;
  output.position = project_common_position_to_clipspace(vec4<f32>(output.place, 1.0));
  output.corner = o;
  output.row0 = cross(b, d) / det;
  output.row1 = cross(d, a) / det;
  output.row2 = cross(a, b) / det;
  return output;
}

${ISO_MARCH_WGSL}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fragmentMain(input: Varyings) -> FragmentOutput {
  let view = view_frame();
  let m = project.viewProjectionMatrix;
  // A view with no perspective has parallel lines of sight; otherwise they meet at the camera.
  let parallel = abs(m[0][3]) + abs(m[1][3]) + abs(m[2][3]) < 1e-6;
  let fromCamera = input.place - project.cameraPosition;
  let dir = select(normalize(fromCamera), -view.toward, parallel);

  // The line of sight in box coordinates, with t = 0 at this pixel's point on the box.
  let u0 = vec3<f32>(dot(input.row0, input.place - input.corner), dot(input.row1, input.place - input.corner), dot(input.row2, input.place - input.corner));
  let du = vec3<f32>(dot(input.row0, dir), dot(input.row1, dir), dot(input.row2, dir));
  let safe = select(du, vec3<f32>(1e-9), abs(du) < vec3<f32>(1e-9));
  let ta = (vec3<f32>(0.0) - u0) / safe;
  let tb = (vec3<f32>(1.0) - u0) / safe;
  let lo = min(ta, tb);
  let hi = max(ta, tb);
  var enter = max(lo.x, max(lo.y, lo.z));
  let leave = min(hi.x, min(hi.y, hi.z));
  // Every line of sight crosses the box's skin twice. Only the far crossing draws, so a pixel is
  // marched once, and the surface is still there when the camera is inside the box.
  if (abs(leave) > abs(enter)) {
    discard;
  }
  // Nothing nearer than the near plane is seen. Clip z is linear along the line of sight, so where it
  // is 0 is one division away. Starting there, the view may be inside the surface: see iso_march.
  let clip0 = project_common_position_to_clipspace(vec4<f32>(input.place, 1.0));
  let clipPerT = project_common_position_to_clipspace(vec4<f32>(input.place + dir, 1.0)) - clip0;
  var fromNear = false;
  if (clipPerT.z > 0.0) {
    let near = -clip0.z / clipPerT.z;
    if (near > enter) {
      enter = near;
      fromNear = true;
    }
  }
  if (enter >= leave) {
    discard;
  }
  let found = iso_march(u0, du, enter, leave, fromNear);
  if (!found.hit) {
    discard;
  }
  let at = found.at;

  let u = u0 + du * at;
  // Sloped over a whole voxel either side: between neighbouring voxels the interpolated field's own
  // slope jumps, which would show as facets.
  let e = vec3<f32>(1.0) / iso.dims.xyz;
  let slope = vec3<f32>(
    iso_field(u + vec3<f32>(e.x, 0.0, 0.0)) - iso_field(u - vec3<f32>(e.x, 0.0, 0.0)),
    iso_field(u + vec3<f32>(0.0, e.y, 0.0)) - iso_field(u - vec3<f32>(0.0, e.y, 0.0)),
    iso_field(u + vec3<f32>(0.0, 0.0, e.z)) - iso_field(u - vec3<f32>(0.0, 0.0, e.z))
  ) / (2.0 * e);
  // Outward is downhill. A flat field (the box's own face cutting through) faces the viewer.
  var outward = -(input.row0 * slope.x + input.row1 * slope.y + input.row2 * slope.z);
  outward = select(outward, -dir, dot(outward, outward) < 1e-12);
  // Seen from inside, the wall is lit on the side facing the viewer.
  outward = select(outward, -outward, found.exiting);
  let normal = to_view(normalize(outward), view);

  let clip = project_common_position_to_clipspace(vec4<f32>(input.place + dir * at, 1.0));
  var output: FragmentOutput;
  output.color = deckgl_premultiplied_alpha(vec4<f32>(material(iso.tint.rgb, normal, iso.march.y), iso.tint.a * iso.march.z));
  output.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  return output;
}
`;

type State = {
  model?: Model;
  handles?: HandleCache;
  ready: boolean;
};

export class IsoSurfaceLayer extends Layer<IsoSurfaceLayerProps> {
  static override layerName = "IsoSurfaceLayer";
  static override defaultProps = defaultProps;

  declare state: State;

  override getShaders() {
    return super.getShaders({ source, modules: [color, project32, isoModule] });
  }

  override initializeState(): void {
    // A state object whatever happens next: see DensityLayer.initializeState.
    const s: State = { ready: false };
    this.state = s;
    const device = this.context.device;
    if (device.type !== "webgpu") {
      this.raiseError(new Error("IsoSurfaceLayer needs deck on WebGPU (deviceProps: { type: 'webgpu' })"), "IsoSurfaceLayer");
      return;
    }
    // The fragment shader writes depth: see ParticleLayer for why the format is given here.
    const target = device.getDefaultCanvasContext().getCurrentFramebuffer();
    s.model = new Model(device, {
      ...this.getShaders(),
      id: this.props.id,
      topology: "triangle-list",
      bufferLayout: [],
      isInstanced: false,
      vertexCount: 36,
      shaderAssembler: this.context.shaderAssembler,
      depthStencilAttachmentFormat: target.depthStencilAttachment?.texture.format,
      parameters: { depthWriteEnabled: true, depthCompare: "less-equal", cullMode: "none" },
    });
    s.handles = new HandleCache(device);
    s.ready = true;
  }

  override finalizeState(context: Parameters<Layer["finalizeState"]>[0]): void {
    const s = this.state;
    s.handles?.destroy(); // wrappers only: the field is its owner's
    s.model?.destroy();
    super.finalizeState(context);
  }

  override draw(): void {
    const s = this.state;
    const { model, handles } = s;
    if (!s.ready || !model || !handles) return;
    const field = resolveBuffer(this.props.field);
    const [nx, ny, nz] = this.props.dims.map((v) => Math.floor(v)) as [number, number, number];
    // A field smaller than its dimensions say would be read past its end.
    if (!field || nx < 1 || ny < 1 || nz < 1 || field.size < nx * ny * nz * 4 || !(this.props.voxel > 0)) return;
    const [r, g, b, a = 255] = this.props.color ?? [255, 255, 255, 255];
    model.setBindings({ field: handles.wrap(field) });
    const uniforms: IsoUniforms = {
      dims: [nx, ny, nz, this.props.threshold ?? 0.5],
      origin: [this.props.origin[0], this.props.origin[1], this.props.origin[2], this.props.voxel],
      tint: [r / 255, g / 255, b / 255, a / 255],
      march: [this.props.maxSteps ?? 512, this.props.shading ?? 1, this.props.opacity ?? 1, 0],
    };
    model.shaderInputs.setProps({ iso: uniforms });
    model.draw(this.context.renderPass);
  }
}
