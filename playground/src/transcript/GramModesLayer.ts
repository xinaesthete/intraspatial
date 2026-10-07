// The co-location mode map as a deck.gl layer (WebGPU only): `paintGramModes`' picture, drawn from
// the buffer `gramMatrixGpu` left on the device — no readback, no copy — and placed by deck's own
// projection, so it pans, zooms and registers with sd.js's image layer beneath it.
//
// Not a second implementation. The host maths is `modeParams` and the shader is built from the same
// WGSL (`sampleWhitened`, `selectionOver`, `markerOver`, `oklabToSrgb`), so a colour means the same
// here as on the canvas map. What differs is the image: the canvas map blends it in OKLab before
// the sRGB conversion; here the image is its own layer and the modes go over it with `opacity`.
//
// The rasters come from `useTranscriptGram`'s snapshot (`TranscriptGram.rasters`), not from
// `res.resident`, which the next compute overwrites while deck may still be drawing. They can be
// bound at all only because deck's device IS this library's (see `adoptDevice`).

import { color, type DefaultProps, Layer, type LayerProps, project32, type UpdateParameters } from "@deck.gl/core";
import { Buffer } from "@luma.gl/core";
import { Model } from "@luma.gl/engine";
import type { ShaderModule } from "@luma.gl/shadertools";
import type { GramMatrixGpuResult } from "../../../src/gpu/spatial/gramMatrix";
import { MODE_LOOK, type ModeBasis, modeParams, OKLAB_TO_SRGB_WGSL } from "../../../src/gpu/spatial/gramModes";
import { MARKER_WGSL } from "../../../src/gpu/spatial/markerWgsl";
import { SIMILARITY_WGSL } from "../../../src/gpu/spatial/similarityWgsl";

type ModeUniforms = {
  /** Raster bounds in the layer's own units: [minX, minY, maxX, maxY]; row 0 is maxY. */
  bounds: [number, number, number, number];
  /** Per-mode scale (L, a, b), unused. */
  scales: [number, number, number, number];
  /** baseL, spanL, chroma, opacity. */
  look: [number, number, number, number];
  /** Marker col, row (raster pixels), on, half-width in screen pixels. */
  marker: [number, number, number, number];
  width: number;
  height: number;
  rowWords: number;
  K: number;
  m: number;
  selTol: number;
  selOn: number;
  /** 1 when `rasters` holds packed f16 (two texels per word), 0 for plain f32. */
  half: number;
};

// Named `U` because the shared mixins read `U.K` and `U.m`; luma binds a module's uniform block by
// the module's name, so the WGSL variable and the module must agree.
const modeUniforms: ShaderModule<ModeUniforms> = {
  name: "U",
  source: /* wgsl */ `\
struct ModeUniforms {
  bounds: vec4<f32>,
  scales: vec4<f32>,
  look: vec4<f32>,
  marker: vec4<f32>,
  width: f32, height: f32, rowWords: f32, K: f32,
  m: f32, selTol: f32, selOn: f32, half: f32,
};
@group(0) @binding(auto) var<uniform> U: ModeUniforms;
/** Words, not floats: at f16 one word holds two texels (see gramMatrix's ResidentRasters). */
@group(0) @binding(auto) var<storage, read> rasters: array<u32>;
/** Per channel: mean, 1/sd and the three mode loadings — see similarityWgsl. */
@group(0) @binding(auto) var<storage, read> chan: array<f32>;
/** K floats of wand reference z, then m*K of the whitening matrix — see similarityWgsl. */
@group(0) @binding(auto) var<storage, read> wand: array<f32>;
`,
  // Order must match the WGSL struct: luma lays the buffer out from this table.
  uniformTypes: {
    bounds: "vec4<f32>",
    scales: "vec4<f32>",
    look: "vec4<f32>",
    marker: "vec4<f32>",
    width: "f32",
    height: "f32",
    rowWords: "f32",
    K: "f32",
    m: "f32",
    selTol: "f32",
    selOn: "f32",
    half: "f32",
  },
};

const source = /* wgsl */ `\
fn fetch(a: u32, col: u32, row: u32) -> f32 {
  let rowWords = u32(U.rowWords);
  let base = a * u32(U.height) * rowWords + row * rowWords;
  if (U.half != 0.0) {
    let pair = unpack2x16float(rasters[base + (col >> 1u)]);
    return select(pair.y, pair.x, (col & 1u) == 0u);
  }
  return bitcast<f32>(rasters[base + col]);
}

${MARKER_WGSL}
${SIMILARITY_WGSL}
${OKLAB_TO_SRGB_WGSL}

struct Varyings {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> Varyings {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
  );
  let corner = corners[vertexIndex];
  // uv.y = 0 is raster row 0, which is the bbox's maxY.
  let x = mix(U.bounds.x, U.bounds.z, corner.x);
  let y = mix(U.bounds.w, U.bounds.y, corner.y);
  var output: Varyings;
  output.position = project_position_to_clipspace(vec3<f32>(x, y, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  output.uv = corner;
  return output;
}

@fragment
fn fragmentMain(input: Varyings) -> @location(0) vec4<f32> {
  // As paintGramModes: bilinear between pixel centres, the field being band-limited by the kernel.
  let size = vec2<f32>(U.width, U.height);
  let p = input.uv * size;
  let q = clamp(p - vec2<f32>(0.5), vec2<f32>(0.0), size - vec2<f32>(1.0));
  // Raster pixels per screen pixel, taken before any branch (derivatives need uniform control flow):
  // the marker keeps a constant on-screen weight at any zoom.
  let perPixel = fwidth(q);
  let c0 = vec2<u32>(floor(q));
  let c1 = min(c0 + vec2<u32>(1u), vec2<u32>(size) - vec2<u32>(1u));
  let f = q - vec2<f32>(c0);
  let s = mix(
    mix(sampleWhitened(c0.x, c0.y), sampleWhitened(c1.x, c0.y), f.x),
    mix(sampleWhitened(c0.x, c1.y), sampleWhitened(c1.x, c1.y), f.x),
    f.y,
  );

  let t = clamp(s.xyz * U.scales.xyz, vec3<f32>(-1.0), vec3<f32>(1.0));
  let lab = vec3<f32>(U.look.x + U.look.y * t.x, U.look.z * t.y, U.look.z * t.z);
  let rgb = selectionOver(oklabToSrgb(lab), s.w, U.selTol, U.selOn);
  let marked = markerOver(rgb, q - U.marker.xy, perPixel * U.marker.w, U.marker.z);
  return deckgl_premultiplied_alpha(vec4<f32>(marked, U.look.w));
}
`;

export type GramModesLayerProps = {
  res: GramMatrixGpuResult | null;
  /** `res`'s rasters, in `res.resident`'s layout, in a buffer nothing will overwrite while drawn. */
  rasters: GPUBuffer | null;
  /** Mode-major eigenvectors — `vectors[k*K + a]`. */
  vectors: Float64Array | null;
  /** See `ModePaintOptions`. */
  saturate?: number;
  chromaWeight?: number;
  reference?: Float64Array | null;
  modesUsed?: number;
  tolerance?: number;
  /** Wand sample, in raster pixels. */
  marker?: { col: number; row: number } | null;
  /** Paint with this fixed projection rather than `res`'s own; `vectors` must then be its modes. */
  basis?: ModeBasis | null;
} & LayerProps;

const defaultProps: DefaultProps<GramModesLayerProps> = {
  res: { type: "object", value: null, compare: false },
  rasters: { type: "object", value: null, compare: false },
  vectors: { type: "object", value: null, compare: false },
  saturate: { type: "number", value: 2.5 },
  chromaWeight: { type: "number", value: 1 },
  reference: { type: "object", value: null, compare: false },
  modesUsed: { type: "number", value: 3 },
  tolerance: { type: "number", value: 1.2 },
  marker: { type: "object", value: null, compare: 1 },
  basis: { type: "object", value: null, compare: false },
};

type State = {
  model?: Model;
  rasters?: Buffer;
  handle?: GPUBuffer;
  chan?: Buffer;
  wand?: Buffer;
  params?: ReturnType<typeof modeParams>;
};

/** A storage buffer holding `data`, reused while it is large enough. */
function storage(layer: GramModesLayer, current: Buffer | undefined, data: Float32Array): Buffer {
  if (current && current.byteLength >= data.byteLength) {
    current.write(data);
    return current;
  }
  current?.destroy();
  return layer.context.device.createBuffer({ data, usage: Buffer.STORAGE | Buffer.COPY_DST });
}

// we'd like to have some better visual feedback of loading progress etc.
// could this subclass our points layer (which has sublayers for 'debug' tile loading status etc)
// but with the actual point rendering replaced with our gram stuff (which is currently very much outside of this)
// Conceptually, want to be modelling as "points layer, but rendering the tiles means this gpu graph"
export class GramModesLayer extends Layer<GramModesLayerProps> {
  static override layerName = "GramModesLayer";
  static override defaultProps = defaultProps;

  declare state: State;

  override getShaders() {
    return super.getShaders({ source, modules: [color, project32, modeUniforms] });
  }

  override initializeState(): void {
    if (this.context.device.type !== "webgpu") throw new Error("GramModesLayer is WGSL-only");
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

  override updateState({ props, oldProps }: UpdateParameters<this>): void {
    const { res, vectors } = props;
    if (!res || !vectors) return;
    const changed =
      res !== oldProps.res ||
      vectors !== oldProps.vectors ||
      props.saturate !== oldProps.saturate ||
      props.chromaWeight !== oldProps.chromaWeight ||
      props.reference !== oldProps.reference ||
      props.modesUsed !== oldProps.modesUsed ||
      props.basis !== oldProps.basis;
    if (!changed && this.state.params) return;
    const params = modeParams(res, {
      vectors,
      saturate: props.saturate,
      chromaWeight: props.chromaWeight,
      reference: props.reference ?? undefined,
      modesUsed: props.modesUsed,
      basis: props.basis ?? undefined,
    });
    // Assigned rather than setState: these are GPU resources, not render inputs deck should diff.
    this.state.chan = storage(this, this.state.chan, params.chan);
    this.state.wand = storage(this, this.state.wand, params.wand);
    this.state.params = params;
    this.state.model?.setBindings({ chan: this.state.chan, wand: this.state.wand });
  }

  override finalizeState(context: Parameters<Layer["finalizeState"]>[0]): void {
    this.state.rasters?.destroy(); // a wrapper: luma never destroys a supplied handle
    this.state.chan?.destroy();
    this.state.wand?.destroy();
    this.state.model?.destroy();
    super.finalizeState(context);
  }

  override draw(): void {
    const { model, params } = this.state;
    const { res, rasters, marker, reference, tolerance, opacity } = this.props;
    if (!model || !res || !rasters || !params) return;
    const handle = rasters;
    if (this.state.handle !== handle) {
      // Every result has its own pooled snapshot, so the handle changes with each one. The pool
      // never rewrites a snapshot while its result, or the one after it, is on screen — so however
      // fast the inputs change, this buffer holds `res`'s rasters (see useTranscriptGram).
      // Assigned, not setState: draw() must not schedule another update.
      this.state.rasters?.destroy();
      this.state.rasters = this.context.device.createBuffer({ handle, byteLength: handle.size, usage: Buffer.STORAGE });
      this.state.handle = handle;
      model.setBindings({ rasters: this.state.rasters });
    }
    const [minX, minY, maxX, maxY] = res.bbox;
    const [sL, sA, sB] = params.scales;
    const uniforms: ModeUniforms = {
      bounds: [minX, minY, maxX, maxY],
      scales: [sL, sA, sB, 0],
      look: [MODE_LOOK.baseL, MODE_LOOK.spanL, MODE_LOOK.chroma, opacity ?? 1],
      marker: [marker?.col ?? 0, marker?.row ?? 0, marker ? 1 : 0, 0.75],
      width: res.width,
      height: res.height,
      rowWords: res.resident.rowWords,
      half: res.resident.precision === "f16" ? 1 : 0,
      K: res.labels.length,
      m: params.m,
      selTol: tolerance ?? 1.2,
      selOn: reference ? 1 : 0,
    };
    model.shaderInputs.setProps({ U: uniforms });
    model.draw(this.context.renderPass);
  }
}
