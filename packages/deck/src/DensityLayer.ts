// A kernel density estimate of a point cloud, drawn as filled, contoured shapes — on WebGPU, with
// the field computed on the GPU deck draws with and never read back.
//
// The pipeline per view: points (uploaded once per data change) → `splatDensityResident` (the
// additive-blend KDE, accumulated in f32) into a storage buffer this layer owns → `fieldMaxResident`
// (for relative levels) → one quad over the field's extent, whose fragment shader turns density
// into colour, a fill above a level, an edge at that level, and iso-lines at regular intervals.
//
// UNITS ARE THE DATA'S. `bandwidth` is the Gaussian's σ in the same units as `x`/`y` (before any
// `modelMatrix`), and density is weight per unit area: Σᵢ wᵢ · exp(−|p−pᵢ|²/2σ²) / 2πσ². With
// `scale: "relative"` (the default) every level — `colorDomain`, `fillLevel`, `contourInterval` —
// is a fraction of the field's maximum over what was splatted; with `scale: "absolute"` they are
// densities. Nothing depends on zoom: the same σ means the same shapes at any scale. (Screen-pixel
// bandwidths, as deck's HeatmapLayer has, are a host-side conversion and belong to the host.)
//
// Which grid gets splatted, and when it is reused, is `fieldPlan.ts`. Cartesian views only: σ in
// degrees of longitude would not be a circle.
//
// DEVICES: the layer hands deck's own GPUDevice to the kernels it calls, which keep their state per
// device. It never adopts the device into intraspatial, so a page may have any number of Decks, and
// a canvas remounted with a new device (deck does not destroy the old one) just works.

import { color, type DefaultProps, Layer, type LayerProps, project32, type UpdateParameters } from "@deck.gl/core";
import { Buffer } from "@luma.gl/core";
import { Model } from "@luma.gl/engine";
import type { ShaderModule } from "@luma.gl/shadertools";
import { fieldMaxResident } from "intraspatial/gpu/spatial/fieldMax";
import { splatDensityResident } from "intraspatial/gpu/spatial/splatDensity";
import { type Bounds, type FieldPlan, type PlanInput, planField, planServes } from "./fieldPlan";

/** An RGB or RGBA colour, components 0–255 (deck's convention). */
export type Color = readonly [number, number, number] | readonly [number, number, number, number];

export type DensityLayerProps = {
  /** Point x coordinates, in the layer's own units. */
  x: ArrayLike<number>;
  /** Point y coordinates, same length as `x`. */
  y: ArrayLike<number>;
  /** Per-point weight (default 1). A weight of 0 leaves a point out — how a filter or a category
   *  selects its points without the host copying them. */
  weights?: ArrayLike<number> | null;
  /** Gaussian σ, in the same units as `x` and `y`. */
  bandwidth: number;
  /** The points' bounds, when the host already knows them (they are computed otherwise). */
  bounds?: Bounds | null;
  /** `"relative"`: levels are fractions of the field's maximum. `"absolute"`: levels are densities
   *  (weight per unit area). */
  scale?: "relative" | "absolute";
  /** The density range the colour ramp spans, in `scale`'s units. Default `[0, max]`. */
  colorDomain?: readonly [number, number] | null;
  /** The ramp, low to high, interpolated linearly. */
  colorRange?: readonly Color[];
  /** Fill where density ≥ this level (in `scale`'s units), with an edge line along it. `null`: no
   *  fill and no edge. */
  fillLevel?: number | null;
  fillOpacity?: number;
  /** Iso-lines every this much density (in `scale`'s units). `null`: none. */
  contourInterval?: number | null;
  /** Opacity of every line: the iso-lines and the fill's edge. */
  contourOpacity?: number;
  /** Line colour for contours and the fill edge. Default: the top of the ramp. */
  lineColor?: Color | null;
  /** Line width, in CSS pixels. */
  lineWidth?: number;
  /** Grid resolution when zoomed out: screen pixels per cell. */
  pixelsPerCell?: number;
  /** Grid resolution cap when zoomed in: cells per σ. */
  cellsPerSigma?: number;
} & LayerProps;

const DEFAULT_RAMP: readonly Color[] = [
  [255, 255, 204],
  [161, 218, 180],
  [65, 182, 196],
  [44, 127, 184],
  [37, 52, 148],
];

const defaultProps: DefaultProps<DensityLayerProps> = {
  x: { type: "object", value: [], compare: false },
  y: { type: "object", value: [], compare: false },
  weights: { type: "object", value: null, compare: false },
  bandwidth: { type: "number", value: 1 },
  bounds: { type: "object", value: null, compare: true },
  scale: "relative",
  colorDomain: { type: "object", value: null, compare: true },
  colorRange: { type: "object", value: DEFAULT_RAMP, compare: true },
  fillLevel: { type: "object", value: 0.1 },
  fillOpacity: { type: "number", value: 0.6, min: 0, max: 1 },
  contourInterval: { type: "object", value: 0.2 },
  contourOpacity: { type: "number", value: 0.9, min: 0, max: 1 },
  lineColor: { type: "object", value: null, compare: true },
  lineWidth: { type: "number", value: 1.25, min: 0 },
  pixelsPerCell: { type: "number", value: 1, min: 0.25 },
  cellsPerSigma: { type: "number", value: 4, min: 2 },
};

/** Kernel support in σ — the splat's default, restated because the plan needs it too. */
const RADIUS_SIGMA = 4;
/** The resident splat's limit is ~4.19M cells (a 1-D de-pad dispatch); stay under it. */
const MAX_CELLS = 4_000_000;
/** Up to this many ramp stops are honoured. */
const MAX_STOPS = 16;

type DensityUniforms = {
  /** The field's extent in layer units: [minX, minY, maxX, maxY]; row 0 is maxY. */
  bounds: [number, number, number, number];
  /** domain lo, domain hi, fill level, contour interval — all in `scale`'s units. */
  levels: [number, number, number, number];
  /** fill opacity, contour opacity, layer opacity, line width in device pixels. */
  look: [number, number, number, number];
  lineColor: [number, number, number, number];
  width: number;
  height: number;
  /** 1/(2πσ²): kernel sums to density. */
  norm: number;
  relative: number;
  stops: number;
  hasFill: number;
  hasContours: number;
  /** 1: no colorDomain given — the ramp spans [0, the field's maximum], resolved on the GPU. */
  autoDomain: number;
};

// Named `U` to match the WGSL variable: luma binds a module's uniform block by the module's name.
const densityUniforms: ShaderModule<DensityUniforms> = {
  name: "U",
  source: /* wgsl */ `\
struct DensityUniforms {
  bounds: vec4<f32>,
  levels: vec4<f32>,
  look: vec4<f32>,
  lineColor: vec4<f32>,
  width: f32, height: f32, norm: f32, relative: f32,
  stops: f32, hasFill: f32, hasContours: f32, autoDomain: f32,
};
@group(0) @binding(auto) var<uniform> U: DensityUniforms;
/** The field, row-major width*height, as the splat left it (kernel sums; row 0 is maxY). */
@group(0) @binding(auto) var<storage, read> field: array<f32>;
/** [0]: the field's maximum kernel sum, as f32 bits (fieldMaxResident). */
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
    norm: "f32",
    relative: "f32",
    stops: "f32",
    hasFill: "f32",
    hasContours: "f32",
    autoDomain: "f32",
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
  // uv.y = 0 is field row 0, which is the bbox's maxY.
  let x = mix(U.bounds.x, U.bounds.z, corner.x);
  let y = mix(U.bounds.w, U.bounds.y, corner.y);
  var output: Varyings;
  output.position = project_position_to_clipspace(vec3<f32>(x, y, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  output.uv = corner;
  return output;
}

fn at(col: i32, row: i32) -> f32 {
  // Clamp to the edge: the grid's border is either beyond the kernel support or off screen.
  let c = u32(clamp(col, 0, i32(U.width) - 1));
  let r = u32(clamp(row, 0, i32(U.height) - 1));
  return field[r * u32(U.width) + c];
}

/** Uniform cubic B-spline weights for the samples at offsets -1, 0, 1, 2 from floor(q). */
fn bspline(t: f32) -> vec4<f32> {
  let t2 = t * t;
  let t3 = t2 * t;
  let u = 1.0 - t;
  return vec4<f32>(u * u * u, 3.0 * t3 - 6.0 * t2 + 4.0, -3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0, t3) / 6.0;
}

fn rampAt(t: f32) -> vec4<f32> {
  let n = u32(U.stops);
  let s = clamp(t, 0.0, 1.0) * f32(n - 1u);
  let i = min(u32(floor(s)), n - 2u);
  return mix(ramp[i], ramp[i + 1u], s - f32(i));
}

/** Coverage of a line at distance d (in units of the quantity) from it, d per pixel being dd. */
fn lineCover(d: f32, dd: f32, widthPx: f32) -> f32 {
  let px = d / max(dd, 1e-20);
  return 1.0 - smoothstep(widthPx * 0.5 - 0.5, widthPx * 0.5 + 0.5, px);
}

@fragment
fn fragmentMain(input: Varyings) -> @location(0) vec4<f32> {
  // Cubic B-spline reconstruction between cell centres, not bilinear. Bilinear gets the VALUES
  // right but its gradient jumps at every cell edge, and iso-lines follow the gradient: zoomed in,
  // where the cell is a fixed fraction of σ, every contour became a polygon. The B-spline is C²,
  // never overshoots (its weights are non-negative, so no negative density and no spurious lines
  // near zero), and blurs by a known variance h²/3 — which the splat has already taken out of σ.
  let size = vec2<f32>(U.width, U.height);
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

  // Density, then into the levels' units: a fraction of the maximum, or density as it is.
  let density = sum * U.norm;
  let peak = bitcast<f32>(fieldMax[0]) * U.norm;
  let v = select(density, density / max(peak, 1e-30), U.relative > 0.5);

  // Derivatives before any branch: fwidth needs uniform control flow.
  let dv = fwidth(v);
  let interval = max(U.levels.w, 1e-30);
  let k = v / interval;
  let dk = fwidth(k);

  let lineW = U.look.w;
  // The ramp's top: as given, or the maximum in v's units (1 when relative, the peak density when not).
  let hi = select(U.levels.y, select(peak, 1.0, U.relative > 0.5), U.autoDomain > 0.5);
  let t = (v - U.levels.x) / max(hi - U.levels.x, 1e-30);
  var fill = rampAt(t);
  // Fill above the level, antialiased across one pixel.
  let fillCover = select(0.0, smoothstep(-0.5, 0.5, (v - U.levels.z) / max(dv, 1e-20)), U.hasFill > 0.5);
  fill = vec4<f32>(fill.rgb, fill.a * fillCover * U.look.x);
  // The fill's edge, and iso-lines at every multiple of the interval from the first one up (the
  // zeroth would trace the empty background).
  let edge = select(0.0, lineCover(abs(v - U.levels.z), dv, lineW), U.hasFill > 0.5);
  let iso = select(0.0, lineCover(abs(fract(k + 0.5) - 0.5), dk, lineW), U.hasContours > 0.5 && k > 0.5);
  // Lines take the line opacity, not the fill's: drawn at the fill's own opacity, the edge of a
  // single-colour fill would vanish into it.
  let lineA = max(edge, iso) * U.look.y * U.lineColor.a;

  // Line over fill, straight alpha.
  let a = lineA + fill.a * (1.0 - lineA);
  let rgb = select(vec3<f32>(0.0), (U.lineColor.rgb * lineA + fill.rgb * fill.a * (1.0 - lineA)) / max(a, 1e-6), a > 0.0);
  if (a * U.look.z < 1.0 / 512.0) {
    discard;
  }
  return deckgl_premultiplied_alpha(vec4<f32>(rgb, a * U.look.z));
}
`;

type State = {
  model?: Model;
  device?: GPUDevice;
  /** [x, y, w] per point. */
  points?: GPUBuffer;
  pointCount: number;
  dataBounds: Bounds | null;
  field?: GPUBuffer;
  fieldWrap?: Buffer;
  max?: GPUBuffer;
  maxWrap?: Buffer;
  ramp?: GPUBuffer;
  rampWrap?: Buffer;
  /** The plan whose field is in `field` — what `draw` describes. */
  shown: FieldPlan | null;
  /** A splat is on the queue's way; a newer request waits for it rather than racing it. */
  busy: boolean;
  /** Set while busy when the field must be redone again (new data, a new view). */
  again: boolean;
  /** Buffers replaced while a splat was in flight, freed once it has been submitted. */
  retired: GPUBuffer[];
  /** Bumped by every data or σ change: a plan made before it is no longer a cache of anything. */
  version: number;
  shownVersion: number;
  /** The σ the shown field was splatted with — see `splatSigma`. */
  shownSigma: number;
  /** Configured and initialised: GPU work may run. False after a configuration error, and the
   *  layer then draws nothing (having said why, once). */
  ready: boolean;
};

const toUnit = (c: Color): [number, number, number, number] => [c[0] / 255, c[1] / 255, c[2] / 255, (c[3] ?? 255) / 255];

export class DensityLayer extends Layer<DensityLayerProps> {
  static override layerName = "DensityLayer";
  static override defaultProps = defaultProps;

  declare state: State;

  override getShaders() {
    return super.getShaders({ source, modules: [color, project32, densityUniforms] });
  }

  override initializeState(): void {
    // A state object whatever happens next. deck logs an error thrown here and carries on with the
    // layer, so the later lifecycle calls would then fail on a missing state — and THAT error escapes
    // the render loop and takes the whole canvas down. Failing inert keeps it to this layer.
    const s: State = {
      pointCount: 0,
      dataBounds: null,
      shown: null,
      busy: false,
      again: false,
      retired: [],
      version: 0,
      shownVersion: -1,
      shownSigma: 1,
      ready: false,
    };
    this.state = s;
    const device = this.context.device;
    const handle = device.handle;
    if (device.type !== "webgpu" || !(handle instanceof GPUDevice)) {
      this.fail("DensityLayer needs deck on WebGPU (deviceProps: { type: 'webgpu' })");
      return;
    }
    if (!handle.features.has("float32-blendable")) {
      this.fail("DensityLayer needs the 'float32-blendable' device feature (deviceProps.optionalFeatures)");
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
    s.max = handle.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    s.maxWrap = device.createBuffer({ handle: s.max, byteLength: 4, usage: Buffer.STORAGE });
    s.model.setBindings({ fieldMax: s.maxWrap });
    s.ready = true;
  }

  /** Report a configuration problem once and stay inert. */
  private fail(message: string): void {
    this.raiseError(new Error(message), "DensityLayer");
  }

  // Re-plan on view changes too, not just prop changes.
  override shouldUpdateState({ changeFlags }: UpdateParameters<this>): boolean {
    return Boolean(changeFlags.somethingChanged);
  }

  override updateState({ props, oldProps }: UpdateParameters<this>): void {
    const s = this.state;
    if (!s.ready) return;
    if (props.x !== oldProps.x || props.y !== oldProps.y || props.weights !== oldProps.weights || props.bounds !== oldProps.bounds) {
      this.uploadPoints();
      s.version++;
    }
    if (
      props.bandwidth !== oldProps.bandwidth ||
      props.cellsPerSigma !== oldProps.cellsPerSigma ||
      props.pixelsPerCell !== oldProps.pixelsPerCell
    ) {
      s.version++;
    }
    if (props.colorRange !== oldProps.colorRange || !s.ramp) this.uploadRamp();
    // Cheap when the field on screen still serves: `planServes` decides.
    this.refresh();
  }

  override finalizeState(context: Parameters<Layer["finalizeState"]>[0]): void {
    const s = this.state;
    // Wrappers first: luma never destroys a handle it was given, so the raw buffers are ours to free.
    s.fieldWrap?.destroy();
    s.maxWrap?.destroy();
    s.rampWrap?.destroy();
    s.field?.destroy();
    s.max?.destroy();
    s.ramp?.destroy();
    s.points?.destroy();
    for (const b of s.retired.splice(0)) b.destroy();
    s.model?.destroy();
    super.finalizeState(context);
  }

  private uploadPoints(): void {
    const s = this.state;
    const { x, y, weights, bounds } = this.props;
    const n = Math.min(x.length, y.length);
    const xyw = new Float32Array(Math.max(1, n) * 3);
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      const px = x[i]!;
      const py = y[i]!;
      xyw[3 * i] = px;
      xyw[3 * i + 1] = py;
      xyw[3 * i + 2] = weights ? weights[i]! : 1;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }
    if (!s.points || s.points.size < xyw.byteLength) {
      // A splat in flight may not have submitted yet (its first pipeline compile is async), and it
      // holds this buffer: destroying it now would fail that submit. Retire it until the splat ends.
      if (s.points) {
        if (s.busy) s.retired.push(s.points);
        else s.points.destroy();
      }
      s.points = s.device!.createBuffer({ size: xyw.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    s.device!.queue.writeBuffer(s.points, 0, xyw.buffer, xyw.byteOffset, xyw.byteLength);
    s.pointCount = n;
    s.dataBounds = bounds ?? (n > 0 ? [minX, minY, maxX, maxY] : null);
  }

  private uploadRamp(): void {
    const s = this.state;
    const range = (this.props.colorRange ?? DEFAULT_RAMP).slice(0, MAX_STOPS);
    const stops = range.length >= 2 ? range : [range[0] ?? [0, 0, 0], range[0] ?? [0, 0, 0]];
    const data = new Float32Array(stops.flatMap(toUnit));
    if (!s.ramp || s.ramp.size < data.byteLength) {
      s.rampWrap?.destroy();
      s.ramp?.destroy();
      s.ramp = s.device!.createBuffer({ size: MAX_STOPS * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      s.rampWrap = this.context.device.createBuffer({ handle: s.ramp, byteLength: s.ramp.size, usage: Buffer.STORAGE });
      s.model?.setBindings({ ramp: s.rampWrap });
    }
    s.device!.queue.writeBuffer(s.ramp, 0, data);
  }

  /** The plan input for the current view, or null when there is nothing to plan for. */
  private planInput(): PlanInput | null {
    const s = this.state;
    const { viewport } = this.context;
    if (!s.dataBounds || s.pointCount === 0 || this.props.bandwidth <= 0) return null;
    // Screen corners → world → layer units. Only the 2-D affine part of modelMatrix applies here.
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
      data: s.dataBounds,
      sigma: this.props.bandwidth,
      radiusSigma: RADIUS_SIGMA,
      cellsPerSigma: this.props.cellsPerSigma ?? 4,
      pixelsPerCell: this.props.pixelsPerCell ?? 1,
      margin: 0.25,
      maxCells: MAX_CELLS,
    };
  }

  /** Splat again if the field on screen no longer serves the view; at most one splat in flight. */
  private refresh(): void {
    const s = this.state;
    if (s.busy) {
      s.again = true;
      return;
    }
    const input = this.planInput();
    if (!input) {
      s.shown = null;
      return;
    }
    if (s.shown && s.shownVersion === s.version && planServes(s.shown, input)) return;
    const plan = planField(input);
    if (!plan) {
      s.shown = null;
      return;
    }
    s.busy = true;
    this.splat(plan, s.version)
      .catch((e: unknown) => this.raiseError(e instanceof Error ? e : new Error(String(e)), "DensityLayer splat"))
      .finally(() => {
        s.busy = false;
        for (const b of s.retired.splice(0)) b.destroy();
        if (s.again) {
          s.again = false;
          this.refresh();
        }
      });
  }

  private async splat(plan: FieldPlan, version: number): Promise<void> {
    const s = this.state;
    const cells = plan.width * plan.height;
    if (!s.field || s.field.size < cells * 4) {
      // Only one splat is ever in flight and this one hasn't been submitted, so the old field is
      // referenced by nothing but the draw — which is rebound below, before the next frame.
      s.fieldWrap?.destroy();
      s.field?.destroy();
      s.field = s.device!.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      s.fieldWrap = this.context.device.createBuffer({ handle: s.field, byteLength: s.field.size, usage: Buffer.STORAGE });
      s.model?.setBindings({ field: s.fieldWrap });
      s.shown = null; // the new buffer holds nothing yet
    }
    await splatDensityResident(s.points!, s.pointCount, s.field, {
      width: plan.width,
      height: plan.height,
      sigma: splatSigma(this.props.bandwidth, plan.cell),
      radiusSigma: RADIUS_SIGMA,
      bbox: plan.bbox,
      stride: 3,
      device: s.device,
    });
    await fieldMaxResident(s.field, cells, s.max!, s.device);
    // Both submitted; queue order puts them before any draw recorded from here on.
    s.shown = plan;
    s.shownVersion = version;
    s.shownSigma = splatSigma(this.props.bandwidth, plan.cell);
    this.setNeedsRedraw();
  }

  override draw(): void {
    const { model, shown, ready } = this.state;
    if (!ready || !model || !shown) return;
    const p = this.props;
    const relative = (p.scale ?? "relative") === "relative";
    const domain = p.colorDomain ?? [0, 1];
    const ramp = (p.colorRange ?? DEFAULT_RAMP).slice(0, MAX_STOPS);
    const line = toUnit(p.lineColor ?? ramp[ramp.length - 1] ?? [0, 0, 0]);
    const dpr = globalThis.devicePixelRatio ?? 1;
    const uniforms: DensityUniforms = {
      bounds: shown.bbox,
      levels: [domain[0], domain[1], p.fillLevel ?? 0, p.contourInterval ?? 1],
      look: [p.fillOpacity ?? 0.6, p.contourOpacity ?? 0.9, p.opacity ?? 1, (p.lineWidth ?? 1.25) * dpr],
      lineColor: line,
      width: shown.width,
      height: shown.height,
      // The splatted kernel's normalisation: mass is what the B-spline preserves, so this stays a density.
      norm: 1 / (2 * Math.PI * this.state.shownSigma * this.state.shownSigma),
      relative: relative ? 1 : 0,
      stops: Math.max(2, ramp.length),
      hasFill: p.fillLevel == null ? 0 : 1,
      hasContours: p.contourInterval == null ? 0 : 1,
      autoDomain: p.colorDomain == null ? 1 : 0,
    };
    model.shaderInputs.setProps({ U: uniforms });
    model.draw(this.context.renderPass);
  }
}

/**
 * The σ to splat with so that, after the shader's cubic B-spline reconstruction, the field shown has
 * width `sigma`. The B-spline is close to a Gaussian of variance h²/3 per axis (h the cell), and
 * Gaussian variances add, so splatting at √(σ² − h²/3) and reconstructing gives σ. At the zoomed-in
 * cap (h = σ/4) that is a 2% narrower splat; zoomed out (h a screen pixel ≪ σ) nothing. Past h = σ
 * the field is under-resolved anyway, and no compensation is applied.
 */
function splatSigma(sigma: number, cell: number): number {
  return cell < sigma ? Math.sqrt(sigma * sigma - (cell * cell) / 3) : sigma;
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
