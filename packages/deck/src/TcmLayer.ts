// A topographical correlation map (TCM) between two cell types, drawn as a diverging field — on
// WebGPU, computed on the GPU deck draws with and never read back.
//
// Γ_AB(x) (Bull et al.'s TCM) says where A cells sit among more B cells than chance (Γ > 0, A
// clustered with B) or fewer (Γ < 0, A excluded from B). Both types are WEIGHTINGS OF ONE POINT SET —
// a table's x/y and, per type, a weight column (0/1 for membership) — so a layer per pair shares the
// coordinates and costs two weight arrays. See `intraspatial/gpu/spatial/tcmRender` for the
// statistic and its two-pass render.
//
// The pipeline per view: points and both weightings (uploaded once per data change) → the plan for
// the Γ grid (`fieldPlan`, with σ as the bandwidth) and the mark grid behind it (`tcmPlan`) → one
// `TcmRenderer.renderResident`, which renders ρ̂_B and Γ and copies Γ and max |Γ| into buffers this
// layer owns → one quad over the grid, whose fragment shader maps Γ through a diverging ramp and
// draws iso-lines.
//
// UNITS ARE THE DATA'S. `radius` (the mark neighbourhood r) and `sigma` (Γ's Gaussian) are in the
// units of `x`/`y`. ρ_B, the density an A cell's neighbourhood is compared with, is B's total weight
// over the area of `region` (default: the bounds of all the points) — so the same layer means the
// same thing at any zoom. Domains (ρ_B · |disk ∩ W| rather than ρ_B · πr²) are not supported yet.
//
// DEVICES: the layer renders on deck's own GPUDevice, with a `TcmRenderer` of its own, so any number
// of TCM layers (and Decks) coexist. Deck must be on WebGPU, and the device must have
// `float32-blendable` (both passes blend into r32float); without it the layer reports an error and
// draws nothing.
//
// TEARDOWN: a render is synchronous — it writes, encodes and submits before `refresh` returns — so
// the layer never has GPU work waiting on the host. Finalizing destroys everything at once; WebGPU
// lets work already submitted finish. (`DensityLayer`'s splat awaits between its submits, which is
// why it has to retire buffers while busy; nothing here can be in that state.)

import { color, type DefaultProps, Layer, type LayerProps, project32, type UpdateParameters } from "@deck.gl/core";
import { Buffer } from "@luma.gl/core";
import { Model } from "@luma.gl/engine";
import type { ShaderModule } from "@luma.gl/shadertools";
import { TcmRenderer, tcmRowFloats } from "intraspatial/gpu/spatial/tcmRender";
import { type KernelSpec, TOPHAT } from "intraspatial/spatial/kernels";
import type { Color } from "./DensityLayer";
import { type Bounds, type FieldPlan, type PlanInput, planField, planServes } from "./fieldPlan";
import { fitSide, planMarks } from "./tcmPlan";
import { densityOver, packTcmPoints } from "./tcmPoints";

export type TcmLayerProps = {
  /** Point x coordinates, in the layer's own units. */
  x: ArrayLike<number>;
  /** Point y coordinates, same length as `x`. */
  y: ArrayLike<number>;
  /** Type A's weight per point: 1 for a cell of type A, 0 otherwise (default: every point). Γ is
   *  about where these cells sit. A non-finite or negative weight counts as 0, and a point with a
   *  non-finite coordinate is left out of both types. */
  weightsA?: ArrayLike<number> | null;
  /** Type B's weight per point (default: every point). */
  weightsB?: ArrayLike<number> | null;
  /** The mark neighbourhood radius r: how near a B cell must be to count for an A cell. */
  radius: number;
  /** Γ's Gaussian σ: how far each A cell's evidence spreads. */
  sigma: number;
  /** The extreme threshold α (> 1): an A cell with α times the expected B neighbours (or 1/α times)
   *  counts fully as clustered (or excluded). Default 5, the paper's. */
  alpha?: number;
  /** The mark kernel. Default `TOPHAT`, the paper's disk. */
  kernel?: KernelSpec;
  /** The region ρ_B is averaged over (Mode 1's ROI). Default: the bounds of every point. */
  region?: Bounds | null;
  /** `"relative"`: levels are fractions of max |Γ| over the rendered grid. `"absolute"`: levels are Γ. */
  scale?: "relative" | "absolute";
  /** The ramp spans [−colorDomain, +colorDomain], in `scale`'s units. Default: max |Γ|. */
  colorDomain?: number | null;
  /** A diverging ramp, exclusion (Γ < 0) to clustering (Γ > 0), interpolated linearly; its middle is
   *  Γ = 0. Stops' alpha is honoured, which is how the default leaves Γ ≈ 0 transparent. */
  colorRange?: readonly Color[];
  /** Opacity of the fill. */
  fillOpacity?: number;
  /** Iso-lines every this much Γ (in `scale`'s units), on both sides of 0, not at 0. `null`: none. */
  contourInterval?: number | null;
  /** Opacity of the iso-lines; they fade where they would crowd, as `DensityLayer`'s do. */
  contourOpacity?: number;
  /** Line colour. Default: the ramp's colour at the line, opaque. */
  lineColor?: Color | null;
  /** Line width, in CSS pixels. */
  lineWidth?: number;
  /** Γ grid resolution when zoomed out: screen pixels per cell. */
  pixelsPerCell?: number;
  /** Γ grid resolution cap when zoomed in: cells per σ. */
  cellsPerSigma?: number;
  /** Mark grid resolution: cells per r, at least. Higher is closer to the exact statistic (the
   *  top-hat's step is the slowest to converge). */
  markCellsPerRadius?: number;
} & LayerProps;

/** Blue (exclusion) through transparent to red (clustering): RdBu, with the middle cleared. */
const DEFAULT_RAMP: readonly Color[] = [
  [33, 102, 172, 255],
  [103, 169, 207, 200],
  [247, 247, 247, 0],
  [239, 138, 98, 200],
  [178, 24, 43, 255],
];

const defaultProps: DefaultProps<TcmLayerProps> = {
  x: { type: "object", value: [], compare: false },
  y: { type: "object", value: [], compare: false },
  weightsA: { type: "object", value: null, compare: false },
  weightsB: { type: "object", value: null, compare: false },
  radius: { type: "number", value: 1, min: 0 },
  sigma: { type: "number", value: 1, min: 0 },
  alpha: { type: "number", value: 5 },
  kernel: { type: "object", value: TOPHAT, compare: true },
  region: { type: "object", value: null, compare: true },
  scale: "relative",
  colorDomain: { type: "object", value: null },
  colorRange: { type: "object", value: DEFAULT_RAMP, compare: true },
  fillOpacity: { type: "number", value: 0.8, min: 0, max: 1 },
  contourInterval: { type: "object", value: 0.25 },
  contourOpacity: { type: "number", value: 0.9, min: 0, max: 1 },
  lineColor: { type: "object", value: null, compare: true },
  lineWidth: { type: "number", value: 1, min: 0 },
  pixelsPerCell: { type: "number", value: 1, min: 0.25 },
  cellsPerSigma: { type: "number", value: 4, min: 2 },
  markCellsPerRadius: { type: "number", value: 8, min: 1 },
};

/** Props that change the uploaded points. */
const DATA_PROPS = ["x", "y", "weightsA", "weightsB"] as const;
/** Props that change Γ or the grids it is rendered on: every prop `planInput` and `refresh` read. */
const FIELD_PROPS = ["radius", "sigma", "alpha", "kernel", "region", "cellsPerSigma", "pixelsPerCell", "markCellsPerRadius"] as const;

function differs(keys: readonly (keyof TcmLayerProps)[], a: Partial<TcmLayerProps>, b: Partial<TcmLayerProps>): boolean {
  return keys.some((k) => a[k] !== b[k]);
}

/** Γ's support in σ, the renderer's default, restated because the plans need it too. */
const RADIUS_SIGMA = 4;
/** Cells in either grid, at most. */
const MAX_CELLS = 4_000_000;
/** Up to this many ramp stops are honoured. */
const MAX_STOPS = 16;

type TcmUniforms = {
  /** The grid's extent in layer units: [minX, minY, maxX, maxY]; row 0 is maxY. */
  bounds: [number, number, number, number];
  /** domain magnitude, contour interval, unused, unused — in `scale`'s units. */
  levels: [number, number, number, number];
  /** fill opacity, contour opacity, layer opacity, line width in device pixels. */
  look: [number, number, number, number];
  lineColor: [number, number, number, number];
  width: number;
  height: number;
  /** Floats between rows of the field (copies out of a texture pad rows to 256 bytes). */
  rowFloats: number;
  relative: number;
  stops: number;
  hasContours: number;
  /** 1: no colorDomain given — the ramp spans ±(1 or max |Γ|), resolved on the GPU. */
  autoDomain: number;
  hasLineColor: number;
};

// Named `tcm` to match the WGSL variable: luma binds a module's uniform block by the module's name.
const tcmUniforms: ShaderModule<TcmUniforms> = {
  name: "tcm",
  source: /* wgsl */ `\
struct TcmUniforms {
  bounds: vec4<f32>,
  levels: vec4<f32>,
  look: vec4<f32>,
  lineColor: vec4<f32>,
  width: f32, height: f32, rowFloats: f32, relative: f32,
  stops: f32, hasContours: f32, autoDomain: f32, hasLineColor: f32,
};
@group(0) @binding(auto) var<uniform> tcm: TcmUniforms;
/** Γ, rows \`rowFloats\` apart, row 0 at maxY. */
@group(0) @binding(auto) var<storage, read> field: array<f32>;
/** [0]: max |Γ| over the grid, as f32 bits. */
@group(0) @binding(auto) var<storage, read> fieldMax: array<u32>;
/** The ramp's stops, rgba in 0..1. */
@group(0) @binding(auto) var<storage, read> ramp: array<vec4<f32>>;
`,
  // Order must match the WGSL struct: luma lays the buffer out from this table.
  uniformTypes: {
    bounds: "vec4<f32>",
    levels: "vec4<f32>",
    look: "vec4<f32>",
    lineColor: "vec4<f32>",
    width: "f32",
    height: "f32",
    rowFloats: "f32",
    relative: "f32",
    stops: "f32",
    hasContours: "f32",
    autoDomain: "f32",
    hasLineColor: "f32",
  },
};

const source = /* wgsl */ `\
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
  // uv.y = 0 is field row 0, which is the grid's maxY.
  let x = mix(tcm.bounds.x, tcm.bounds.z, corner.x);
  let y = mix(tcm.bounds.w, tcm.bounds.y, corner.y);
  var output: Varyings;
  output.position = project_position_to_clipspace(vec3<f32>(x, y, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  output.uv = corner;
  return output;
}

fn at(col: i32, row: i32) -> f32 {
  let c = u32(clamp(col, 0, i32(tcm.width) - 1));
  let r = u32(clamp(row, 0, i32(tcm.height) - 1));
  return field[r * u32(tcm.rowFloats) + c];
}

/** Uniform cubic B-spline weights for the samples at offsets -1, 0, 1, 2 from floor(q). */
fn bspline(t: f32) -> vec4<f32> {
  let t2 = t * t;
  let t3 = t2 * t;
  let u = 1.0 - t;
  return vec4<f32>(u * u * u, 3.0 * t3 - 6.0 * t2 + 4.0, -3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0, t3) / 6.0;
}

fn rampAt(t: f32) -> vec4<f32> {
  let n = u32(tcm.stops);
  let s = clamp(t, 0.0, 1.0) * f32(n - 1u);
  let i = min(u32(floor(s)), n - 2u);
  return mix(ramp[i], ramp[i + 1u], s - f32(i));
}

fn lineCover(d: f32, dd: f32, widthPx: f32) -> f32 {
  let px = d / max(dd, 1e-20);
  return 1.0 - smoothstep(widthPx * 0.5 - 0.5, widthPx * 0.5 + 0.5, px);
}

@fragment
fn fragmentMain(input: Varyings) -> @location(0) vec4<f32> {
  // Cubic B-spline reconstruction between cell centres, as DensityLayer does, so iso-lines stay
  // smooth when zoomed in. It blurs by a variance of h²/3 (h the cell): at the default cap of σ/4,
  // Γ's width grows by about 1%. Unlike the density, Γ is not re-splatted to compensate, since its
  // normalisation is the paper's and depends on σ.
  let size = vec2<f32>(tcm.width, tcm.height);
  let q = clamp(input.uv * size - vec2<f32>(0.5), vec2<f32>(0.0), size - vec2<f32>(1.0));
  let base = vec2<i32>(floor(q));
  let wx = bspline(q.x - f32(base.x));
  let wy = bspline(q.y - f32(base.y));
  var sum = 0.0;
  for (var j = 0; j < 4; j++) {
    var row = 0.0;
    for (var i = 0; i < 4; i++) {
      row += wx[i] * at(base.x + i - 1, base.y + j - 1);
    }
    sum += wy[j] * row;
  }

  let peak = bitcast<f32>(fieldMax[0]);
  let v = select(sum, sum / max(peak, 1e-30), tcm.relative > 0.5);

  // Derivatives before any branch: fwidth needs uniform control flow.
  let interval = max(tcm.levels.y, 1e-30);
  let k = v / interval;
  let dk = fwidth(k);

  let lineW = tcm.look.w;
  let hi = select(tcm.levels.x, select(peak, 1.0, tcm.relative > 0.5), tcm.autoDomain > 0.5);
  let t = 0.5 + 0.5 * v / max(hi, 1e-30);
  let rampColor = rampAt(t);
  let fill = vec4<f32>(rampColor.rgb, rampColor.a * tcm.look.x);
  // Iso-lines at every nonzero multiple of the interval, faded out where they crowd.
  let spacing = 1.0 / max(dk, 1e-20);
  let uncrowded = smoothstep(2.0 * lineW, 6.0 * lineW, spacing);
  let iso = select(0.0, lineCover(abs(fract(k + 0.5) - 0.5), dk, lineW), tcm.hasContours > 0.5 && abs(k) > 0.5) * uncrowded;
  let lineRgb = select(rampColor.rgb, tcm.lineColor.rgb, tcm.hasLineColor > 0.5);
  let lineA = iso * tcm.look.y * select(1.0, tcm.lineColor.a, tcm.hasLineColor > 0.5);

  // Line over fill, straight alpha.
  let a = lineA + fill.a * (1.0 - lineA);
  let rgb = select(vec3<f32>(0.0), (lineRgb * lineA + fill.rgb * fill.a * (1.0 - lineA)) / max(a, 1e-6), a > 0.0);
  if (a * tcm.look.z < 1.0 / 512.0) {
    discard;
  }
  return deckgl_premultiplied_alpha(vec4<f32>(rgb, a * tcm.look.z));
}
`;

type State = {
  model?: Model;
  device?: GPUDevice;
  renderer?: TcmRenderer;
  /** [x, y] per point, shared by both weightings. */
  xy?: GPUBuffer;
  wA?: GPUBuffer;
  wB?: GPUBuffer;
  pointCount: number;
  /** Bounds of the A cells: where Γ can be non-zero, before its support. */
  boundsA: Bounds | null;
  /** Bounds of every placed point: ρ_B's default region. */
  boundsAll: Bounds | null;
  totalB: number;
  /** Γ as the renderer copied it out (rows `tcmRowFloats(width)` apart). */
  field?: GPUBuffer;
  fieldWrap?: Buffer;
  max?: GPUBuffer;
  maxWrap?: Buffer;
  ramp?: GPUBuffer;
  rampWrap?: Buffer;
  /** The plan whose Γ is in `field` — what `draw` describes. */
  shown: FieldPlan | null;
  /** Bumped by every change that invalidates the field (data, weights, any parameter of Γ). */
  version: number;
  shownVersion: number;
  /** Configured and initialised: GPU work may run. False after a configuration error. */
  ready: boolean;
  finalized: boolean;
};

const toUnit = (c: Color): [number, number, number, number] => [c[0] / 255, c[1] / 255, c[2] / 255, (c[3] ?? 255) / 255];

export class TcmLayer extends Layer<TcmLayerProps> {
  static override layerName = "TcmLayer";
  static override defaultProps = defaultProps;

  declare state: State;

  override getShaders() {
    return super.getShaders({ source, modules: [color, project32, tcmUniforms] });
  }

  override initializeState(): void {
    // A state object whatever happens next, so a failed setup leaves an inert layer rather than
    // later lifecycle calls failing on a missing state (see DensityLayer).
    const s: State = {
      pointCount: 0,
      boundsA: null,
      boundsAll: null,
      totalB: 0,
      shown: null,
      version: 0,
      shownVersion: -1,
      ready: false,
      finalized: false,
    };
    this.state = s;
    const device = this.context.device;
    const handle = device.handle;
    if (device.type !== "webgpu" || !(handle instanceof GPUDevice)) {
      this.fail("TcmLayer needs deck on WebGPU (deviceProps: { type: 'webgpu' })");
      return;
    }
    try {
      s.renderer = new TcmRenderer(handle, this.label("tcm"));
    } catch (e) {
      this.fail(e instanceof Error ? e.message : String(e));
      return;
    }
    s.device = handle;
    s.model = new Model(device, {
      ...this.getShaders(),
      id: this.props.id,
      topology: "triangle-list",
      bufferLayout: [],
      isInstanced: false,
      vertexCount: 6,
      shaderAssembler: this.context.shaderAssembler,
    });
    s.max = handle.createBuffer({
      label: this.label("max"),
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    s.maxWrap = device.createBuffer({ id: this.label("max"), handle: s.max, byteLength: 4, usage: Buffer.STORAGE });
    s.model.setBindings({ fieldMax: s.maxWrap });
    s.ready = true;
  }

  /** A GPU resource's label: the layer id first, so a host can attribute a device error to its layer. */
  private label(resource: string): string {
    return `${this.props.id}:${resource}`;
  }

  private fail(message: string): void {
    this.raiseError(new Error(message), "TcmLayer");
  }

  // Re-plan on view changes too, not just prop changes.
  override shouldUpdateState({ changeFlags }: UpdateParameters<this>): boolean {
    return Boolean(changeFlags.somethingChanged);
  }

  override updateState({ props, oldProps }: UpdateParameters<this>): void {
    const s = this.state;
    if (!s.ready) return;
    if (differs(DATA_PROPS, props, oldProps)) {
      this.uploadPoints();
      s.version++;
    }
    if (differs(FIELD_PROPS, props, oldProps)) s.version++;
    if (props.colorRange !== oldProps.colorRange || !s.ramp) this.uploadRamp();
    this.refresh();
  }

  override finalizeState(context: Parameters<Layer["finalizeState"]>[0]): void {
    const s = this.state;
    s.finalized = true;
    // Nothing is waiting to be submitted (renders are synchronous), so everything goes now. luma never
    // destroys a handle it was given, so the raw buffers are freed here as well as their wrappers.
    s.fieldWrap?.destroy();
    s.maxWrap?.destroy();
    s.rampWrap?.destroy();
    for (const b of [s.field, s.max, s.ramp, s.xy, s.wA, s.wB]) b?.destroy();
    s.renderer?.destroy();
    s.model?.destroy();
    super.finalizeState(context);
  }

  private uploadPoints(): void {
    const s = this.state;
    const { x, y, weightsA, weightsB } = this.props;
    const p = packTcmPoints(x, y, weightsA, weightsB);
    s.xy = this.write(s.xy, "xy", p.xy);
    s.wA = this.write(s.wA, "weightsA", p.wA);
    s.wB = this.write(s.wB, "weightsB", p.wB);
    s.pointCount = p.n;
    s.boundsA = p.boundsA;
    s.boundsAll = p.boundsAll;
    s.totalB = p.totalB;
  }

  /** Write `data` into `buf`, replacing it if too small. The old one can be destroyed at once: every
   *  render that used it has been submitted. */
  private write(buf: GPUBuffer | undefined, name: string, data: Float32Array): GPUBuffer {
    const s = this.state;
    let b = buf;
    if (!b || b.size < data.byteLength) {
      b?.destroy();
      b = s.device!.createBuffer({
        label: this.label(name),
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    s.device!.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  }

  private uploadRamp(): void {
    const s = this.state;
    const range = (this.props.colorRange ?? DEFAULT_RAMP).slice(0, MAX_STOPS);
    const stops = range.length >= 2 ? range : [range[0] ?? [0, 0, 0], range[0] ?? [0, 0, 0]];
    const data = new Float32Array(stops.flatMap(toUnit));
    if (!s.ramp) {
      s.ramp = s.device!.createBuffer({
        label: this.label("ramp"),
        size: MAX_STOPS * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      s.rampWrap = this.context.device.createBuffer({
        id: this.label("ramp"),
        handle: s.ramp,
        byteLength: s.ramp.size,
        usage: Buffer.STORAGE,
      });
      s.model?.setBindings({ ramp: s.rampWrap });
    }
    s.device!.queue.writeBuffer(s.ramp, 0, data);
  }

  /** The Γ-grid plan input for the current view, or null when there is nothing to plan for. */
  private planInput(): PlanInput | null {
    const s = this.state;
    const { viewport } = this.context;
    const { sigma, radius } = this.props;
    if (!s.boundsA || s.pointCount === 0 || !(sigma > 0) || !(radius > 0)) return null;
    const corners = [
      [0, 0],
      [viewport.width, 0],
      [0, viewport.height],
      [viewport.width, viewport.height],
    ].map((p) => toLayerUnits(this.props.modelMatrix, viewport.unproject(p)));
    const xs = corners.map((c) => c[0]);
    const ys = corners.map((c) => c[1]);
    return {
      visible: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
      pixels: [viewport.width, viewport.height],
      data: s.boundsA,
      sigma,
      radiusSigma: RADIUS_SIGMA,
      cellsPerSigma: this.props.cellsPerSigma ?? 4,
      pixelsPerCell: this.props.pixelsPerCell ?? 1,
      margin: 0.25,
      maxCells: MAX_CELLS,
    };
  }

  /** Render Γ again if the field on screen no longer serves the view. Synchronous: see TEARDOWN. */
  private refresh(): void {
    const s = this.state;
    if (!s.ready || s.finalized) return;
    const input = this.planInput();
    if (!input) {
      s.shown = null;
      return;
    }
    const maxSide = s.device!.limits.maxTextureDimension2D;
    const fit = (p: FieldPlan) => fitSide(p, maxSide);
    if (s.shown && s.shownVersion === s.version && planServes(s.shown, input, fit)) return;
    const planned = planField(input);
    const plan = planned && fit(planned);
    const marks =
      plan &&
      planMarks({
        plan,
        boundsA: s.boundsA!,
        support: this.props.sigma * RADIUS_SIGMA,
        radius: this.props.radius,
        cellsPerRadius: this.props.markCellsPerRadius ?? 8,
        maxCells: MAX_CELLS,
        maxSide,
      });
    if (!plan || !marks) {
      s.shown = null;
      return;
    }
    try {
      this.ensureField(tcmRowFloats(plan.width) * plan.height * 4);
      const a = { points: s.xy!, weights: s.wA!, count: s.pointCount };
      const b = { points: s.xy!, weights: s.wB!, count: s.pointCount };
      s.renderer!.renderResident(
        { a, b, rhoB: densityOver(s.totalB, this.props.region ?? s.boundsAll) },
        {
          radius: this.props.radius,
          sigma: this.props.sigma,
          alpha: this.props.alpha ?? 5,
          kernel: this.props.kernel ?? TOPHAT,
          radiusSigma: RADIUS_SIGMA,
          grid: plan,
          markGrid: marks,
          out: { buffer: s.field!, max: s.max! },
        },
      );
    } catch (e) {
      s.shown = null;
      this.raiseError(e instanceof Error ? e : new Error(String(e)), "TcmLayer render");
      return;
    }
    s.shown = plan;
    s.shownVersion = s.version;
    this.setNeedsRedraw();
  }

  private ensureField(bytes: number): void {
    const s = this.state;
    if (s.field && s.field.size >= bytes) return;
    s.fieldWrap?.destroy();
    s.field?.destroy();
    s.field = s.device!.createBuffer({
      label: this.label("field"),
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    s.fieldWrap = this.context.device.createBuffer({
      id: this.label("field"),
      handle: s.field,
      byteLength: s.field.size,
      usage: Buffer.STORAGE,
    });
    s.model?.setBindings({ field: s.fieldWrap });
  }

  override draw(): void {
    const { model, shown, ready } = this.state;
    if (!ready || !model || !shown) return;
    const p = this.props;
    const relative = (p.scale ?? "relative") === "relative";
    const ramp = (p.colorRange ?? DEFAULT_RAMP).slice(0, MAX_STOPS);
    const dpr = globalThis.devicePixelRatio ?? 1;
    const uniforms: TcmUniforms = {
      bounds: shown.bbox,
      levels: [p.colorDomain ?? 1, p.contourInterval ?? 1, 0, 0],
      look: [p.fillOpacity ?? 0.8, p.contourOpacity ?? 0.9, p.opacity ?? 1, (p.lineWidth ?? 1) * dpr],
      lineColor: p.lineColor ? toUnit(p.lineColor) : [0, 0, 0, 1],
      width: shown.width,
      height: shown.height,
      rowFloats: tcmRowFloats(shown.width),
      relative: relative ? 1 : 0,
      stops: Math.max(2, ramp.length),
      hasContours: p.contourInterval == null ? 0 : 1,
      autoDomain: p.colorDomain == null ? 1 : 0,
      hasLineColor: p.lineColor ? 1 : 0,
    };
    model.shaderInputs.setProps({ tcm: uniforms });
    model.draw(this.context.renderPass);
  }
}

/** World → layer units, inverting the 2-D affine part of a column-major 4×4 model matrix. */
function toLayerUnits(m: ArrayLike<number> | null | undefined, world: number[]): [number, number] {
  const wx = world[0] ?? 0;
  const wy = world[1] ?? 0;
  if (!m) return [wx, wy];
  const a = m[0]!;
  const b = m[1]!;
  const c = m[4]!;
  const d = m[5]!;
  const tx = m[12]!;
  const ty = m[13]!;
  const det = a * d - b * c || 1;
  const x = wx - tx;
  const y = wy - ty;
  return [(d * x - c * y) / det, (-b * x + a * y) / det];
}
