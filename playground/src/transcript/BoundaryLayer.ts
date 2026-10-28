// PROTOTYPE: draws `boundaryStrength`'s field over the mode map. Same placement as GramModesLayer,
// which it sits on top of. A faint white ramp always shows where the mixture changes. On top of it:
//
//   * bands: solid amber wherever strength ≥ θ, the pixels an edge-bounded fill would refuse;
//   * lines: the zero-crossings of the strength's derivative across the edge, where strength ≥ θ.
//     Thin, and drawn at screen resolution rather than raster resolution: the direction comes from
//     the interpolated tensor (sign-free), and one Newton step along it, |D|/|D''|, gives the
//     distance to the crossing, which fwidth turns into screen pixels.
//
// A filled cell (the ridge-bounded region holding a click) is tinted cyan.

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
  pad0: number;
  pad1: number;
};

const boundaryUniforms: ShaderModule<BoundaryUniforms> = {
  name: "B",
  source: /* wgsl */ `\
struct BoundaryUniforms {
  bounds: vec4<f32>,
  width: f32, height: f32, theta: f32, opacity: f32,
  lines: f32, cellOn: f32, pad0: f32, pad1: f32,
};
@group(0) @binding(auto) var<uniform> B: BoundaryUniforms;
/** Per pixel: strength, tensor xx, yy, xy. */
@group(0) @binding(auto) var<storage, read> field: array<vec4<f32>>;
/** Per pixel: 1 inside the filled cell. */
@group(0) @binding(auto) var<storage, read> cell: array<f32>;
`,
  uniformTypes: {
    bounds: "vec4<f32>",
    width: "f32",
    height: "f32",
    theta: "f32",
    opacity: "f32",
    lines: "f32",
    cellOn: "f32",
    pad0: "f32",
    pad1: "f32",
  },
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

fn cellAt(p: vec2<f32>) -> f32 {
  let size = vec2<f32>(B.width, B.height);
  let q = clamp(p, vec2<f32>(0.0), size - vec2<f32>(1.0));
  let c0 = vec2<u32>(floor(q));
  let c1 = min(c0 + vec2<u32>(1u), vec2<u32>(size) - vec2<u32>(1u));
  let f = q - vec2<f32>(c0);
  return mix(mix(cell[idx(c0)], cell[idx(vec2<u32>(c1.x, c0.y))], f.x), mix(cell[idx(vec2<u32>(c0.x, c1.y))], cell[idx(c1)], f.x), f.y);
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
  let inCell = smoothstep(0.4, 0.6, cv) * B.cellOn;
  // The cell's edge as a ~1 px white line: the 0.5 level set of the interpolated mask.
  let cw = max(fwidth(cv), 1e-6);
  let cellEdge = (1.0 - smoothstep(cw * 0.5, cw * 1.5, abs(cv - 0.5))) * B.cellOn;

  var rgb = vec3<f32>(1.0);
  var alpha = 0.6 * faint * faint;
  let mark = select(over, line, B.lines > 0.5);
  rgb = mix(rgb, vec3<f32>(1.0, 0.72, 0.1), mark);
  alpha = mix(alpha, 0.95, mark);
  rgb = mix(rgb, vec3<f32>(0.2, 0.85, 1.0), inCell * (1.0 - mark));
  alpha = mix(alpha, 0.45, inCell * (1.0 - mark));
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
  /** The filled cell, 1 per pixel inside, or null for none. */
  cell: Float32Array | null;
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
};

type State = { model?: Model; wrapped?: Buffer; handle?: GPUBuffer; cellBuf?: Buffer; cellSrc?: Float32Array | null };

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
    this.setState({ model });
  }

  override finalizeState(context: Parameters<Layer["finalizeState"]>[0]): void {
    this.state.wrapped?.destroy();
    this.state.cellBuf?.destroy();
    this.state.model?.destroy();
    super.finalizeState(context);
  }

  override draw(): void {
    const { model } = this.state;
    const { strength, bbox, width, height, theta, opacity, lines, cell } = this.props;
    if (!model || !strength || !bbox || width === 0) return;
    if (this.state.handle !== strength) {
      this.state.wrapped?.destroy();
      this.state.wrapped = this.context.device.createBuffer({ handle: strength, byteLength: strength.size, usage: Buffer.STORAGE });
      this.state.handle = strength;
      model.setBindings({ field: this.state.wrapped });
    }
    // Always bound (a zero mask when there is no cell), sized for this raster.
    const n = width * height;
    if (!this.state.cellBuf || this.state.cellBuf.byteLength < n * 4 || this.state.cellSrc !== cell) {
      const data = cell && cell.length === n ? cell : new Float32Array(n);
      if (this.state.cellBuf && this.state.cellBuf.byteLength >= data.byteLength) this.state.cellBuf.write(data);
      else {
        this.state.cellBuf?.destroy();
        this.state.cellBuf = this.context.device.createBuffer({ data, usage: Buffer.STORAGE | Buffer.COPY_DST });
      }
      this.state.cellSrc = cell;
      model.setBindings({ cell: this.state.cellBuf });
    }
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
        pad0: 0,
        pad1: 0,
      },
    });
    model.draw(this.context.renderPass);
  }
}
