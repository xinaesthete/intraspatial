// PROTOTYPE: draws `boundaryStrength`'s field over the mode map. Same placement as GramModesLayer,
// which it sits on top of. A faint white ramp always shows where the mixture changes (should be controlable).
// On top of it:
//
//   * bands: solid amber wherever strength ≥ θ, the pixels an edge-bounded fill would refuse;
//   * lines: the zero-crossings of the strength's derivative across the edge, where strength ≥ θ.
//     Thin, and drawn at screen resolution rather than raster resolution: the direction comes from
//     the interpolated tensor (sign-free), and one Newton step along it, |D|/|D''|, gives the
//     distance to the crossing, which fwidth turns into screen pixels.
//
// The persistent mask built from strokes is filled amber. The preview (what a press would paint: the
// wand's patch or the brush's disc) is shown as the change it would make, by op: cyan where the mask
// would grow, red hatching over its amber where it would shrink. Both have a thin outline.
//
// This shouldn't really belong in 'transcripts', it's more general than that.

import { type DefaultProps, Layer, type LayerProps, project32 } from "@deck.gl/core";
import { Buffer } from "@luma.gl/core";
import { Model } from "@luma.gl/engine";
import type { ShaderModule } from "@luma.gl/shadertools";

type BoundaryUniforms = {
  bounds: [number, number, number, number];
  width: number;
  height: number;
  theta: number;
  opacity: number;
  lines: number;
  cellOn: number;
  maskOn: number;
  /** The op a press would apply: 0 new mask, 1 add, 2 subtract, 3 intersect. */
  op: number;
  /** 1: quadratic B-spline reconstruction and antialiased edges; 0: bilinear and hard thresholds.
   *  Not `smooth`: that is a reserved word in WGSL. */
  aa: number;
  /** 1: the preview fades by the `weight` buffer (graded necks); 0: drawn at full strength. */
  graded: number;
  pad1: number;
  pad2: number;
};

const boundaryUniforms: ShaderModule<BoundaryUniforms> = {
  name: "B",
  source: /* wgsl */ `\
struct BoundaryUniforms {
  bounds: vec4<f32>,
  width: f32, height: f32, theta: f32, opacity: f32,
  lines: f32, cellOn: f32, maskOn: f32, op: f32,
  aa: f32, graded: f32, pad1: f32, pad2: f32,
};
@group(0) @binding(auto) var<uniform> B: BoundaryUniforms;
/** Per pixel: strength, tensor xx, yy, xy. */
@group(0) @binding(auto) var<storage, read> field: array<vec4<f32>>;
/** Per pixel: 1 inside the filled cell. */
@group(0) @binding(auto) var<storage, read> cell: array<f32>;
/** Per pixel: 1 inside the persistent mask. */
@group(0) @binding(auto) var<storage, read> mask: array<f32>;
/** Per pixel: the preview's membership in [0, 1], when graded. */
@group(0) @binding(auto) var<storage, read> weight: array<f32>;
`,
  uniformTypes: {
    bounds: "vec4<f32>",
    width: "f32",
    height: "f32",
    theta: "f32",
    opacity: "f32",
    lines: "f32",
    cellOn: "f32",
    maskOn: "f32",
    op: "f32",
    aa: "f32",
    graded: "f32",
    pad1: "f32",
    pad2: "f32",
  },
};

// could be a general-purpose utility function, but would mean more litigation over semantics.
/** A quadratic B-spline sampler for a 0/1 raster: its 0.5 contour is smooth, where bilinear
 *  interpolation leaves the pixel grid's staircase. 3×3 taps, edge-clamped, pixel centres at whole
 *  numbers. Generated per buffer: WGSL cannot take a storage array as a parameter. */
const quadratic = (name: string, buffer: string) => /* wgsl */ `
fn ${name}(p: vec2<f32>) -> f32 {
  let size = vec2<i32>(i32(B.width), i32(B.height));
  let i = vec2<i32>(round(p));
  let t = p - vec2<f32>(i);
  let wx = vec3<f32>(0.5 * (0.5 - t.x) * (0.5 - t.x), 0.75 - t.x * t.x, 0.5 * (0.5 + t.x) * (0.5 + t.x));
  let wy = vec3<f32>(0.5 * (0.5 - t.y) * (0.5 - t.y), 0.75 - t.y * t.y, 0.5 * (0.5 + t.y) * (0.5 + t.y));
  var v = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    let y = u32(clamp(i.y + dy, 0, size.y - 1));
    for (var dx = -1; dx <= 1; dx++) {
      let x = u32(clamp(i.x + dx, 0, size.x - 1));
      v += wx[dx + 1] * wy[dy + 1] * ${buffer}[y * u32(size.x) + x];
    }
  }
  return v;
}
`;

/** The bilinear sampler it replaced, kept for comparison (`smooth` off). */
const bilinear = (name: string, buffer: string) => /* wgsl */ `
fn ${name}(p: vec2<f32>) -> f32 {
  let size = vec2<f32>(B.width, B.height);
  let q = clamp(p, vec2<f32>(0.0), size - vec2<f32>(1.0));
  let c0 = vec2<u32>(floor(q));
  let c1 = min(c0 + vec2<u32>(1u), vec2<u32>(size) - vec2<u32>(1u));
  let f = q - vec2<f32>(c0);
  return mix(mix(${buffer}[idx(c0)], ${buffer}[idx(vec2<u32>(c1.x, c0.y))], f.x), mix(${buffer}[idx(vec2<u32>(c0.x, c1.y))], ${buffer}[idx(c1)], f.x), f.y);
}
`;

// What keeps us from using TypeGPU more widely?
// How well does it work with luma/deck? Is this something we should discuss with the community?
// Would be a significant change now for template-function type things `maskQ(p)` etc.
// What we have is not ugly, but it'd be nice if it was better understood in editors, which would be harder.
const source = /* wgsl */ `\
struct Varyings { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex //common boilerplate
fn vertexMain(@builtin(vertex_index) i: u32) -> Varyings {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
  );
  let c = corners[i];
  let x = mix(B.bounds.x, B.bounds.z, c.x);
  let y = mix(B.bounds.w, B.bounds.y, c.y);
  var o: Varyings;
  o.position = project_position_to_clipspace(vec3<f32>(x, y, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  o.uv = c;
  return o;
}

fn idx(c: vec2<u32>) -> u32 { return c.y * u32(B.width) + c.x; }

/** Bilinear, edge-clamped, in raster coordinates with pixel centres at whole numbers. */
fn T(p: vec2<f32>) -> vec4<f32> {
  let size = vec2<f32>(B.width, B.height);
  let q = clamp(p, vec2<f32>(0.0), size - vec2<f32>(1.0));
  let c0 = vec2<u32>(floor(q));
  let c1 = min(c0 + vec2<u32>(1u), vec2<u32>(size) - vec2<u32>(1u));
  let f = q - vec2<f32>(c0);
  return mix(
    mix(field[idx(c0)], field[idx(vec2<u32>(c1.x, c0.y))], f.x),
    mix(field[idx(vec2<u32>(c0.x, c1.y))], field[idx(c1)], f.x),
    f.y,
  );
}

${quadratic("maskQ", "mask")}
${quadratic("cellQ", "cell")}
${bilinear("maskB", "mask")}
${bilinear("cellB", "cell")}
${bilinear("weightAt", "weight")}
fn maskAt(p: vec2<f32>) -> f32 { return select(maskB(p), maskQ(p), B.aa > 0.5); }
fn cellAt(p: vec2<f32>) -> f32 { return select(cellB(p), cellQ(p), B.aa > 0.5); }

/** The level set v = 0.5 as a coverage, antialiased over one screen pixel (or, unsmoothed, the old
 *  fixed ramp in v). */
fn coverage(v: f32) -> f32 {
  let w = max(fwidth(v), 1e-6) * 0.5;
  return select(smoothstep(0.4, 0.6, v), smoothstep(0.5 - w, 0.5 + w, v), B.aa > 0.5);
}

@fragment
fn fragmentMain(input: Varyings) -> @location(0) vec4<f32> {
  let size = vec2<f32>(B.width, B.height);
  let q = input.uv * size - vec2<f32>(0.5);
  // Raster pixels per screen pixel, before any branch (derivatives need uniform control flow).
  let perPixel = max(length(fwidth(q)) * 0.70710678, 1e-6);
  let t = T(q);
  let rel = t.x / max(B.theta, 1e-6);
  let wid = max(fwidth(rel), 1e-6);
  let over = smoothstep(1.0 - wid, 1.0 + wid, rel);
  let faint = clamp(rel, 0.0, 1.0);

  // Leading direction from the tensor, then the strength one raster pixel either side.
  let ang = 0.5 * atan2(2.0 * t.w, t.y - t.z);
  let n = vec2<f32>(cos(ang), sin(ang));
  let sp = T(q + n).x;
  let sm = T(q - n).x;
  let d1 = 0.5 * (sp - sm);
  let d2 = sp - 2.0 * t.x + sm;
  let distR = select(1e6, abs(d1) / max(-d2, 1e-6), d2 < 0.0);
  let line = (1.0 - smoothstep(0.6, 1.4, distR / perPixel)) * select(0.0, 1.0, distR < 1.0) * over;

  let cv = cellAt(q);
  // Graded necks: a lobe hanging on a narrowing neck fades out rather than vanishing at once.
  let fade = select(1.0, clamp(weightAt(q), 0.0, 1.0), B.graded > 0.5);
  let inCell = coverage(cv) * B.cellOn * fade;
  // The cell's edge as a ~1 px white line: the 0.5 level set of the interpolated mask.
  let cw = max(fwidth(cv), 1e-6);
  let cellEdge = (1.0 - smoothstep(cw * 0.5, cw * 1.5, abs(cv - 0.5))) * B.cellOn * fade;
  let mv = maskAt(q);
  let inMask = coverage(mv) * B.maskOn;
  let mw = max(fwidth(mv), 1e-6);
  let maskEdge = (1.0 - smoothstep(mw * 0.5, mw * 1.5, abs(mv - 0.5))) * B.maskOn;

  var rgb = vec3<f32>(1.0);
  var alpha = 0.6 * faint * faint;
  let mark = select(over, line, B.lines > 0.5);
  rgb = mix(rgb, vec3<f32>(1.0, 0.72, 0.1), mark);
  alpha = mix(alpha, 0.95, mark);
  // The preview as the change a press would make: what the mask would gain (cyan) and lose (red
  // hatching over its amber), by op. Where nothing would change only the preview's outline shows.
  let op = u32(B.op + 0.5);
  let outside = inCell * (1.0 - inMask);
  let gain = select(0.0, outside, op <= 1u);
  // Only while there is a preview: with none, a press would paint nothing.
  let loss = select(select(inMask * (1.0 - inCell), inMask * inCell, op == 2u), 0.0, op == 1u) * B.cellOn;
  let kept = inMask - loss;
  // Hatching in screen pixels, so it reads the same at any zoom; each stripe's edge antialiased over a
  // pixel (a triangle wave thresholded with its own screen derivative).
  let hs = (input.position.x + input.position.y) / 8.0;
  let hw = max(fwidth(hs), 1e-6) * 0.5;
  let hatch = select(step(0.5, fract(hs)), smoothstep(0.25 - hw, 0.25 + hw, abs(fract(hs) - 0.5)), B.aa > 0.5);

  rgb = mix(rgb, vec3<f32>(1.0, 0.75, 0.1), kept * (1.0 - mark));
  alpha = mix(alpha, 0.5, kept * (1.0 - mark));
  let lossInk = loss * (1.0 - mark);
  rgb = mix(rgb, mix(vec3<f32>(1.0, 0.75, 0.1), vec3<f32>(0.95, 0.15, 0.2), hatch), lossInk);
  alpha = mix(alpha, mix(0.2, 0.75, hatch), lossInk);
  rgb = mix(rgb, vec3<f32>(1.0, 0.8, 0.2), maskEdge);
  alpha = mix(alpha, 1.0, maskEdge);
  rgb = mix(rgb, vec3<f32>(0.2, 0.85, 1.0), gain * (1.0 - mark));
  // A preview, not a selection: light enough that the map under it still reads.
  alpha = mix(alpha, 0.3, gain * (1.0 - mark));
  rgb = mix(rgb, vec3<f32>(1.0), cellEdge);
  alpha = mix(alpha, 1.0, cellEdge);
  alpha = alpha * B.opacity;
  return vec4<f32>(rgb * alpha, alpha);
}
`;

export type BoundaryLayerProps = {
  /** `BoundaryField.buffer`: per raster pixel strength and tensor, row-major, row 0 at bbox maxY. */
  strength: GPUBuffer | null;
  /** Bumped whenever `strength`'s contents change: the buffer is reused, so its identity does not. */
  version: number;
  bbox: readonly [number, number, number, number] | null;
  width: number;
  height: number;
  theta: number;
  /** Draw zero-crossing lines rather than threshold bands. */
  lines: boolean;
  /** The preview, 1 per pixel inside, or null for none. */
  cell: Float32Array | null;
  /** The persistent mask on this raster, 1 per pixel inside, or null for none. */
  mask: Float32Array | null;
  /** The preview's membership in [0, 1] per pixel, which fades it; null draws it at full strength. */
  weight: Float32Array | null;
  /** The op a press would apply, for showing the preview as a change: 0 new, 1 add, 2 subtract, 3 intersect. */
  op: number;
  /** Quadratic reconstruction and antialiased edges (default), or the bilinear rendering, to compare. */
  smooth: boolean;
} & LayerProps;

const defaultProps: DefaultProps<BoundaryLayerProps> = {
  strength: { type: "object", value: null, compare: false },
  version: { type: "number", value: 0 },
  bbox: { type: "object", value: null, compare: false },
  width: { type: "number", value: 0 },
  height: { type: "number", value: 0 },
  theta: { type: "number", value: 1 },
  lines: { type: "boolean", value: true },
  cell: { type: "object", value: null, compare: true },
  mask: { type: "object", value: null, compare: true },
  weight: { type: "object", value: null, compare: true },
  op: { type: "number", value: 0 },
  smooth: { type: "boolean", value: true },
};

type Slot = { buf?: Buffer; src?: Float32Array | null };
type State = { model?: Model; wrapped?: Buffer; handle?: GPUBuffer; cell: Slot; mask: Slot; weight: Slot };

export class BoundaryLayer extends Layer<BoundaryLayerProps> {
  static override layerName = "BoundaryLayer";
  static override defaultProps = defaultProps;
  declare state: State;

  override getShaders() {
    return super.getShaders({ source, modules: [project32, boundaryUniforms] });
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
    this.setState({ model, cell: {}, mask: {}, weight: {} });
  }

  override finalizeState(context: Parameters<Layer["finalizeState"]>[0]): void {
    this.state.wrapped?.destroy();
    this.state.cell.buf?.destroy();
    this.state.mask.buf?.destroy();
    this.state.weight.buf?.destroy();
    this.state.model?.destroy();
    super.finalizeState(context);
  }

  override draw(): void {
    const { model } = this.state;
    const { strength, bbox, width, height, theta, opacity, lines, cell, mask, weight, op, smooth } = this.props;
    if (!model || !strength || !bbox || width === 0) return;
    if (this.state.handle !== strength) {
      this.state.wrapped?.destroy();
      this.state.wrapped = this.context.device.createBuffer({ handle: strength, byteLength: strength.size, usage: Buffer.STORAGE });
      this.state.handle = strength;
      model.setBindings({ field: this.state.wrapped });
    }
    // Always bound (a zero mask when there is none), sized for this raster.
    const n = width * height;
    this.upload(model, "cell", cell, n);
    this.upload(model, "mask", mask, n);
    this.upload(model, "weight", weight, n);
    const [minX, minY, maxX, maxY] = bbox;
    model.shaderInputs.setProps({
      B: {
        bounds: [minX, minY, maxX, maxY],
        width,
        height,
        theta,
        opacity: opacity ?? 1,
        lines: lines ? 1 : 0,
        cellOn: cell && cell.length === n ? 1 : 0,
        maskOn: mask && mask.length === n ? 1 : 0,
        op,
        aa: smooth ? 1 : 0,
        graded: weight && weight.length === n ? 1 : 0,
        pad1: 0,
        pad2: 0,
      },
    });
    model.draw(this.context.renderPass);
  }

  private upload(model: Model, name: "cell" | "mask" | "weight", src: Float32Array | null, n: number): void {
    const slot = this.state[name];
    if (slot.buf && slot.buf.byteLength >= n * 4 && slot.src === src) return;
    const data = src && src.length === n ? src : new Float32Array(n);
    if (slot.buf && slot.buf.byteLength >= data.byteLength) slot.buf.write(data);
    else {
      slot.buf?.destroy();
      slot.buf = this.context.device.createBuffer({ data, usage: Buffer.STORAGE | Buffer.COPY_DST });
    }
    slot.src = src;
    model.setBindings({ [name]: slot.buf });
  }
}
