// The co-location mode map: project every pixel onto the leading eigenvectors and paint the result
// through OKLab, in one fragment pass with no readback.
//
// ## What is being drawn
//
// `gram.ts` builds `corr`, the spatial correlation of the smoothed channel densities, and its
// eigenvectors are co-location modes: signed weightings over channels whose recombined field
//
//     y_k(x) = Σ_a v_ka · (M_a(x) − μ_a) / σ_a
//
// carries the k-th largest share of spatial variance. That is a *change of basis on rasters already
// on the device* — no re-splatting, no neighbour search — so the map costs one pass over the same
// buffer the matrix was reduced from.
//
// ## Why OKLab rather than three ramps
//
// The natural question of a mode map is "where else does the tissue look like it does here?", and
// that is a question about **distance in mode space**. OKLab is (approximately) perceptually
// uniform, so equal steps in `(L, a, b)` are equal perceived colour differences — which makes
// perceived colour distance on the map stand in for actual distance between co-location profiles.
// Painting the three modes as independent red/green/blue ramps would not: sRGB's channels differ
// wildly in perceived weight (a step in blue is far less visible than the same step in green), so
// mode 3 would look like weak structure purely because of where it landed.
//
// The assignment is mode 1 → **L**, modes 2 and 3 → **a** and **b**. Mode 1 carries the most
// variance and lightness is the channel the eye resolves best, so the dominant structure survives
// greyscale printing and low-vision viewing; the two chroma axes then separate profiles that agree
// on the dominant mode. Modes are scaled by their own robust spread rather than a shared constant,
// because eigenvalues fall off fast and a shared scale would render modes 2–3 as flat grey.
//
// Sign is not arbitrary here: `eigenSym` canonicalises each eigenvector's largest loading positive,
// so re-running the analysis cannot flip the map's colours (see that module).

import type { Oklab } from "../../color/oklab";
import { compileShader, getDevice } from "../device";
import type { GramMatrixGpuResult } from "./gramMatrix";
import { IMAGE_OVERLAY_WGSL, type ImageOverlay, overlayResources } from "./imageOverlayWgsl";
import { MARKER_WGSL } from "./markerWgsl";

/** Fragment shader constants: the OKLab→linear-sRGB matrices, matching `src/color/oklab.ts`. */
const PAINT_SHADER = /* wgsl */ `
struct Uni {
  width: f32, height: f32, rowFloats: f32, k: f32,
  scaleL: f32, scaleA: f32, scaleB: f32, baseL: f32,
  spanL: f32, chroma: f32, markerX: f32, markerY: f32,
  markerOn: f32, lineW: f32, imageMix: f32, pad0: f32,
  uv0: vec3f, pad1: f32,
  uv1: vec3f, pad2: f32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> rasters: array<f32>;
/** Per-channel mean, sd, and the three mode loadings: 5 floats per channel, interleaved so one
 *  channel's whole contribution is a single contiguous fetch. */
@group(0) @binding(2) var<storage, read> chan: array<f32>;
@group(0) @binding(3) var ctxImage: texture_2d<f32>;
@group(0) @binding(4) var ctxSampler: sampler;

${MARKER_WGSL}
${IMAGE_OVERLAY_WGSL}

fn oklabToSrgb(lab: vec3f) -> vec3f {
  let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
  let l = l_ * l_ * l_;
  let m = m_ * m_ * m_;
  let s = s_ * s_ * s_;
  let lin = vec3f(
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
   -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
   -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
  // Linear -> sRGB, per channel, clamped. Out-of-gamut is clipped here rather than chroma-reduced:
  // the host caps chroma to a conservative radius first (see MAX_CHROMA), so clipping is a
  // backstop for the corners rather than the normal path.
  let c = clamp(lin, vec3f(0.0), vec3f(1.0));
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  // Fullscreen triangle-strip quad; uv in [0,1] with v flipped so row 0 of the raster (which is the
  // TOP of the bbox, as copyTextureToBuffer left it) lands at the top of the canvas.
  let c = vec2f(f32(vi & 1u), f32((vi >> 1u) & 1u));
  var o: VOut;
  o.pos = vec4f(c.x * 2.0 - 1.0, 1.0 - c.y * 2.0, 0.0, 1.0);
  o.uv = c;
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let W = u32(U.width);
  let H = u32(U.height);
  let rowFloats = u32(U.rowFloats);
  let col = min(W - 1u, u32(in.uv.x * U.width));
  let row = min(H - 1u, u32(in.uv.y * U.height));
  let K = u32(U.k);

  var y = vec3f(0.0);
  for (var a = 0u; a < K; a = a + 1u) {
    let mu = chan[a * 5u];
    let inv = chan[a * 5u + 1u];          // 1/sd, precomputed on the host (sd == 0 -> 0)
    let z = (rasters[a * H * rowFloats + row * rowFloats + col] - mu) * inv;
    y = y + z * vec3f(chan[a * 5u + 2u], chan[a * 5u + 3u], chan[a * 5u + 4u]);
  }

  // Each mode is scaled by its OWN robust spread: eigenvalues fall off fast, so a shared scale
  // would flatten modes 2-3 to grey.
  let t = clamp(y * vec3f(U.scaleL, U.scaleA, U.scaleB), vec3f(-1.0), vec3f(1.0));
  var lab = vec3f(U.baseL + U.spanL * t.x, U.chroma * t.y, U.chroma * t.z);

  // Blend the context image in OKLab, before the sRGB conversion — see imageOverlayWgsl. The
  // sample is unconditional and the weight carries the "is it on the image" test, because
  // textureSample may not appear in non-uniform control flow.
  let uv = uvAt(U.uv0, U.uv1, vec2f(f32(col) + 0.5, f32(row) + 0.5));
  let ctxRgb = textureSample(ctxImage, ctxSampler, uv).rgb;
  lab = mix(lab, srgbToOklab(ctxRgb), uvWeight(uv, U.imageMix));

  // The rule lines go on AFTER the OKLab conversion, in sRGB. Putting them in OKLab would let the
  // gamut clamp move them, and a mark whose colour depends on what is under it is not a mark.
  let mp = vec2f(f32(col) - U.markerX, f32(row) - U.markerY);
  return vec4f(markerOver(oklabToSrgb(lab), mp, vec2f(U.lineW), U.markerOn), 1.0);
}
`;

/** Chroma radius in OKLab units. 0.11 keeps `a`/`b` inside the sRGB gamut across the whole
 *  lightness range used below, so the shader's clamp is a corner case rather than a routine
 *  distortion — clipping would move hue and lightness, i.e. corrupt exactly the two dimensions
 *  carrying the mode signal. */
const MAX_CHROMA = 0.11;
const BASE_L = 0.62;
const SPAN_L = 0.3;

const UNI_FLOATS = 24; // 96 bytes; the two vec3 UV rows sit at 16-byte alignment from float 16

interface Ctx {
  device: GPUDevice;
  pipeline: GPURenderPipeline;
  format: GPUTextureFormat;
}
let ctxCache: Promise<Ctx> | undefined;

function getCtx(): Promise<Ctx> {
  ctxCache ??= (async () => {
    const device = await getDevice();
    const format = navigator.gpu.getPreferredCanvasFormat();
    const module = await compileShader(device, PAINT_SHADER, "gramModes.paint");
    const pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-strip" },
    });
    return { device, pipeline, format };
  })();
  return ctxCache;
}

const surfaces = new WeakMap<HTMLCanvasElement, GPUCanvasContext>();
let uniBuf: GPUBuffer | undefined;
let chanBuf: GPUBuffer | undefined;
let chanCap = 0;

export interface ModePaintOptions {
  /** Mode-major eigenvectors from `coLocationModes` — `vectors[k*K + a]`. */
  readonly vectors: Float64Array;
  /** How many standard deviations of each mode's own spread saturate the ramp. Larger = flatter. */
  readonly saturate?: number;
  /** Where the wand sample was taken, in raster pixels. Drawn as haloed rule lines so the point a
   *  similarity field is measured *from* stays visible in the map it was picked on. */
  readonly marker?: { readonly col: number; readonly row: number };
  /** Context image to blend under the modes. `uv` maps **raster pixel centres** to image UV. */
  readonly image?: ImageOverlay;
}

/** What the map is showing, for a legend that states the mapping rather than leaving it implicit. */
export interface ModePaintInfo {
  /** Per-mode scale factor actually applied (1 / (saturate·σ_k)). */
  readonly scales: [number, number, number];
  /** Per-mode standard deviation over pixels — `sqrt(λ_k)`, since a projected mode's variance IS
   *  its eigenvalue (an identity `gram.test.ts` pins). */
  readonly sigmas: [number, number, number];
}

/**
 * Paint the leading three co-location modes to a canvas through OKLab.
 *
 * Reads `res.resident` in place — see its lifetime note; call this before the next
 * `gramMatrixGpu`. With fewer than three channels the missing modes are simply zero, which lands
 * on the neutral axis rather than inventing structure.
 */
export async function paintGramModes(canvas: HTMLCanvasElement, res: GramMatrixGpuResult, opts: ModePaintOptions): Promise<ModePaintInfo> {
  const { device, pipeline, format } = await getCtx();
  const K = res.labels.length;
  const { mean, sd, buffer, rowFloats } = res.resident;
  const saturate = opts.saturate ?? 2.5;

  // Interleaved per channel: mean, 1/sd, and the three mode loadings.
  const chan = new Float32Array(K * 5);
  for (let a = 0; a < K; a++) {
    chan[a * 5] = mean[a]!;
    chan[a * 5 + 1] = sd[a]! > 0 ? 1 / sd[a]! : 0;
    for (let k = 0; k < 3; k++) chan[a * 5 + 2 + k] = k < K ? (opts.vectors[k * K + a] ?? 0) : 0;
  }

  // A projected mode's variance over pixels is exactly its eigenvalue, so σ_k = √λ_k — but the
  // eigenvalues are not passed in, and recomputing the projection on the host to measure the spread
  // would defeat the point. vᵀ·corr·v gives the same number from data already here.
  const sigmas: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 3 && k < K; k++) {
    let q = 0;
    for (let a = 0; a < K; a++) {
      for (let b = 0; b < K; b++) q += (opts.vectors[k * K + a] ?? 0) * res.corr[a * K + b]! * (opts.vectors[k * K + b] ?? 0);
    }
    sigmas[k] = Math.sqrt(Math.max(q, 0));
  }
  const scales = sigmas.map((s) => (s > 0 ? 1 / (saturate * s) : 0)) as [number, number, number];

  let ctx = surfaces.get(canvas);
  if (!ctx) {
    ctx = canvas.getContext("webgpu") as GPUCanvasContext;
    ctx.configure({ device, format, alphaMode: "opaque" });
    surfaces.set(canvas, ctx);
  }
  canvas.width = res.width;
  canvas.height = res.height;
  ctx.configure({ device, format, alphaMode: "opaque" });

  // Raster pixel centre -> world. The shader passes (col + 0.5, row + 0.5), and row 0 is the TOP of
  // the bbox, hence the negative Y term.
  const [minX, minY, maxX, maxY] = res.bbox;
  const ov = overlayResources(device, opts.image, [(maxX - minX) / res.width, 0, minX, 0, -(maxY - minY) / res.height, maxY]);
  uniBuf ??= device.createBuffer({ size: UNI_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  if (!chanBuf || chanCap < chan.length) {
    chanCap = Math.max(chan.length, chanCap * 2, 16);
    chanBuf = device.createBuffer({ size: chanCap * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }
  device.queue.writeBuffer(
    uniBuf,
    0,
    // Half-width in RASTER pixels. The canvas is displayed at close to one CSS pixel per raster
    // pixel, so a hair under one keeps the lines thin without letting them alias into dashes.
    new Float32Array([
      res.width,
      res.height,
      rowFloats,
      K,
      scales[0],
      scales[1],
      scales[2],
      BASE_L,
      SPAN_L,
      MAX_CHROMA,
      opts.marker?.col ?? 0,
      opts.marker?.row ?? 0,
      opts.marker ? 1 : 0,
      0.75,
      ov.mix,
      0,
      ov.uv[0]!,
      ov.uv[1]!,
      ov.uv[2]!,
      0,
      ov.uv[3]!,
      ov.uv[4]!,
      ov.uv[5]!,
      0,
    ]),
  );
  device.queue.writeBuffer(chanBuf, 0, chan);

  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniBuf, size: UNI_FLOATS * 4 } },
      { binding: 1, resource: { buffer } },
      { binding: 2, resource: { buffer: chanBuf } },
      { binding: 3, resource: ov.view },
      { binding: 4, resource: ov.sampler },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [
      { view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } },
    ],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.draw(4);
  pass.end();
  device.queue.submit([enc.finish()]);

  return { scales, sigmas };
}

/** The colour a given (mode1, mode2, mode3) triple maps to, in host code — for legends and swatches.
 *  Kept in step with the shader by construction: same constants, same order. */
export function modeSwatch(t: readonly [number, number, number]): Oklab {
  const c = (v: number) => Math.max(-1, Math.min(1, v));
  return [BASE_L + SPAN_L * c(t[0]), MAX_CHROMA * c(t[1]), MAX_CHROMA * c(t[2])];
}
