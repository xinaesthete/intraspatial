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
import type { GramMatrixGpuResult, ResidentRasters } from "./gramMatrix";
import { whiteningMatrix } from "./gramTerrain";
import { IMAGE_OVERLAY_WGSL, type ImageOverlay, overlayResources } from "./imageOverlayWgsl";
import { MARKER_WGSL } from "./markerWgsl";
import { SIMILARITY_WGSL } from "./similarityWgsl";

/** Chroma radius in OKLab units. 0.11 keeps `a`/`b` inside the sRGB gamut across the whole
 *  lightness range used below, so the shader's clamp is a corner case rather than a routine
 *  distortion — clipping would move hue and lightness, i.e. corrupt exactly the two dimensions
 *  carrying the mode signal. */
const MAX_CHROMA = 0.11;
const BASE_L = 0.62;
const SPAN_L = 0.3;

/** The mode look, for any other renderer of the same map (the deck layer): same constants, same map. */
export const MODE_LOOK = { baseL: BASE_L, spanL: SPAN_L, chroma: MAX_CHROMA } as const;

/** OKLab → sRGB in WGSL, the matrices matching `src/color/oklab.ts`. */
export const OKLAB_TO_SRGB_WGSL = /* wgsl */ `
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
`;

const PAINT_SHADER = /* wgsl */ `
struct Uni {
  width: f32, height: f32, rowFloats: f32, K: f32,
  scaleL: f32, scaleA: f32, scaleB: f32, baseL: f32,
  spanL: f32, chroma: f32, markerX: f32, markerY: f32,
  markerOn: f32, lineW: f32, imageMix: f32, m: f32,
  uv0: vec3f, pad1: f32,
  uv1: vec3f, pad2: f32,
  selTol: f32, selOn: f32, pad3: f32, pad4: f32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> rasters: array<f32>;
/** Per-channel mean, sd, and the three mode loadings: 5 floats per channel, interleaved so one
 *  channel's whole contribution is a single contiguous fetch. */
@group(0) @binding(2) var<storage, read> chan: array<f32>;
@group(0) @binding(3) var ctxImage: texture_2d<f32>;
@group(0) @binding(4) var ctxSampler: sampler;
/** K floats of reference z, then m*K of the whitening matrix — see similarityWgsl. */
@group(0) @binding(5) var<storage, read> wand: array<f32>;

fn fetch(a: u32, col: u32, row: u32) -> f32 {
  return rasters[a * u32(U.height) * u32(U.rowFloats) + row * u32(U.rowFloats) + col];
}

${MARKER_WGSL}
${IMAGE_OVERLAY_WGSL}
${SIMILARITY_WGSL}

${OKLAB_TO_SRGB_WGSL}
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
  // Continuous raster coordinates, pixel centres at +0.5. The canvas may be larger than the raster
  // (drawn at screen resolution), so the modes are reconstructed bilinearly between pixel centres —
  // the field is band-limited by the kernel, so this is its proper reconstruction — and the image
  // is sampled at the fragment itself rather than at a raster pixel. At raster size, every fragment
  // lands on a centre and this reduces exactly to the per-pixel lookup.
  let p = in.uv * vec2f(U.width, U.height);
  let q = clamp(p - vec2f(0.5), vec2f(0.0), vec2f(U.width - 1.0, U.height - 1.0));
  let c0 = vec2u(floor(q));
  let c1 = min(c0 + vec2u(1u), vec2u(u32(U.width) - 1u, u32(U.height) - 1u));
  let f = q - vec2f(c0);

  // Mode coordinates and the wand distance — the SAME function the terrain uses, so the outline
  // below cannot enclose a region the terrain colours differently.
  let s = mix(
    mix(sampleWhitened(c0.x, c0.y), sampleWhitened(c1.x, c0.y), f.x),
    mix(sampleWhitened(c0.x, c1.y), sampleWhitened(c1.x, c1.y), f.x),
    f.y,
  );

  // Each mode is scaled by its OWN robust spread: eigenvalues fall off fast, so a shared scale
  // would flatten modes 2-3 to grey.
  let t = clamp(s.xyz * vec3f(U.scaleL, U.scaleA, U.scaleB), vec3f(-1.0), vec3f(1.0));
  var lab = vec3f(U.baseL + U.spanL * t.x, U.chroma * t.y, U.chroma * t.z);

  // Blend the context image in OKLab, before the sRGB conversion — see imageOverlayWgsl. The
  // sample is unconditional and the weight carries the "is it on the image" test, because
  // textureSample may not appear in non-uniform control flow.
  let uv = uvAt(U.uv0, U.uv1, p);
  let ctxRgb = textureSample(ctxImage, ctxSampler, uv).rgb;
  lab = mix(lab, srgbToOklab(ctxRgb), uvWeight(uv, U.imageMix));

  // Annotations go on AFTER the OKLab conversion, in sRGB. Putting them in OKLab would let the
  // gamut clamp move them, and a mark whose colour depends on what is under it is not a mark.
  //
  // Two marks, two meanings, deliberately different colours: the outline is the similarity hue and
  // encloses "what got selected"; the rule lines are white and say "where you sampled". The lines
  // are composited last so the sample point stays legible where it sits on its own boundary.
  let rgb = selectionOver(oklabToSrgb(lab), s.w, U.selTol, U.selOn);

  let mp = q - vec2f(U.markerX, U.markerY);
  return vec4f(markerOver(rgb, mp, vec2f(U.lineW), U.markerOn), 1.0);
}
`;

const UNI_FLOATS = 28; // 112 bytes; the two vec3 UV rows sit at 16-byte alignment from float 16

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
let wandBuf: GPUBuffer | undefined;
let wandCap = 0;

export interface ModePaintOptions {
  /** Mode-major eigenvectors from `coLocationModes` — `vectors[k*K + a]`. */
  readonly vectors: Float64Array;
  /** How many standard deviations of each mode's own spread saturate the ramp. Larger = flatter. */
  readonly saturate?: number;
  /** How the two chroma modes (2 and 3) are weighted against mode 1, in `[0, 1]`.
   *
   *  `1` (default) is fully **whitened**: each mode is scaled by its own σ, so the picture equalises
   *  them and colour distance stands in for the whitened (Mahalanobis) distance — good for *seeing*
   *  faint secondary structure, but it amplifies modes that carry little variance, so a 6%-variance
   *  mode can look as colourful as a 61%-variance one. `0` is **variance-weighted**: the chroma axes
   *  share mode 1's σ, so colour appears only in proportion to `σ_k/σ₁` — faithful to importance,
   *  often near-greyscale when mode 1 dominates. Between, the chroma amplitude is scaled by
   *  `(σ_k/σ₁)^(1−w)`. Mode 1 → lightness is never touched. See docs/cell-stats.md §4. */
  readonly chromaWeight?: number;
  /** Where the wand sample was taken, in raster pixels. Drawn as haloed rule lines so the point a
   *  similarity field is measured *from* stays visible in the map it was picked on. */
  readonly marker?: { readonly col: number; readonly row: number };
  /** Context image to blend under the modes. `uv` maps **raster pixel centres** to image UV. */
  readonly image?: ImageOverlay;
  /** Canvas size in device pixels. Defaults to the raster's; larger draws the image at that
   *  resolution and reconstructs the modes bilinearly. */
  readonly outputSize?: { readonly width: number; readonly height: number };
  /** The wand's standardised channel vector. Present ⇒ the selection boundary is outlined. */
  readonly reference?: Float64Array;
  /** How many leading modes the metric keeps; defaults to 3. See `similarityWgsl`. */
  readonly modesUsed?: number;
  /** Whitened distance at the boundary — the same number the terrain's similarity ramp reaches 0
   *  at, so the outline and the shading are two readings of one setting. */
  readonly tolerance?: number;
  /** Paint with another window's projection instead of this one's — see `ModeBasis`. `vectors`
   *  should then be that window's modes too. */
  readonly basis?: ModeBasis;
}

/**
 * The projection a mode map is painted with, fixed from one window so others can be painted with
 * it: each channel's mean and spread (the standardisation), and `corr`, which sets each mode's scale.
 *
 * By default every window is standardised against itself and gets its own modes, so a colour means
 * "how this place differs from the rest of this window". Painted with a fixed basis, a colour means
 * the same mixture in every window — what a comparison across the slide needs. The rasters must be
 * comparable for that: same channels, same radius, same molecule filter (pixel size does not matter;
 * raster values are kernel sums at pixel centres).
 */
export interface ModeBasis {
  readonly labels: readonly string[];
  readonly mean: ArrayLike<number>;
  readonly sd: ArrayLike<number>;
  readonly corr: ArrayLike<number>;
}

/** What `modeParams` reads of a Gram result. */
export type ModeSource = Pick<GramMatrixGpuResult, "labels" | "corr"> & { readonly resident: Pick<ResidentRasters, "mean" | "sd"> };

/** `res`'s own projection, copied so it outlives the result (whose arrays a later compute may reuse). */
export function modeBasis(res: ModeSource): ModeBasis {
  return {
    labels: [...res.labels],
    mean: Float64Array.from(res.resident.mean),
    sd: Float64Array.from(res.resident.sd),
    corr: Float64Array.from(res.corr),
  };
}

/** What the map is showing, for a legend that states the mapping rather than leaving it implicit. */
export interface ModePaintInfo {
  /** Per-mode scale factor actually applied (1 / (saturate·σ_k)). */
  readonly scales: [number, number, number];
  /** Per-mode standard deviation over pixels — `sqrt(λ_k)`, since a projected mode's variance IS
   *  its eigenvalue (an identity `gram.test.ts` pins). */
  readonly sigmas: [number, number, number];
}

/** What a renderer of the mode map uploads, computed on the host from the result and the options. */
export interface ModeParams extends ModePaintInfo {
  /** 5 floats per channel: mean, 1/sd and the three mode loadings — `chan` in `similarityWgsl`. */
  readonly chan: Float32Array<ArrayBuffer>;
  /** K floats of wand reference z, then the m×K whitening matrix — `wand` in `similarityWgsl`. */
  readonly wand: Float32Array<ArrayBuffer>;
  /** How many modes the wand metric keeps. */
  readonly m: number;
}

/**
 * The host half of the mode map: per-channel standardisation and loadings, per-mode scales, and the
 * wand's whitening. Shared by `paintGramModes` and any other renderer (the deck layer), so the two
 * cannot disagree about what a colour means.
 */
export function modeParams(res: ModeSource, opts: Omit<ModePaintOptions, "marker" | "image" | "outputSize">): ModeParams {
  const K = res.labels.length;
  const basis = opts.basis;
  if (basis && (basis.labels.length !== K || basis.labels.some((l, a) => l !== res.labels[a]))) {
    throw new Error(`mode basis is for channels [${basis.labels.join(", ")}], not [${res.labels.join(", ")}]`);
  }
  const { mean, sd } = basis ?? res.resident;
  const corr = basis?.corr ?? res.corr;
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
      for (let b = 0; b < K; b++) q += (opts.vectors[k * K + a] ?? 0) * corr[a * K + b]! * (opts.vectors[k * K + b] ?? 0);
    }
    sigmas[k] = Math.sqrt(Math.max(q, 0));
  }
  const scales = sigmas.map((s) => (s > 0 ? 1 / (saturate * s) : 0)) as [number, number, number];

  // Chroma weighting: mute modes 2 and 3 towards their variance share. `w=1` leaves them whitened
  // (factor 1); `w=0` scales them by σ_k/σ₁ so colour reflects importance. Folded into the chroma
  // scales here because the flat map's scales drive colour only — the terrain, whose scales also set
  // relief, applies the same factor in its colour path instead. Mode 1 (lightness) is untouched.
  const w = Math.max(0, Math.min(1, opts.chromaWeight ?? 1));
  const chromaFactor = (k: number) => (sigmas[0]! > 0 && sigmas[k]! > 0 ? (sigmas[k]! / sigmas[0]!) ** (1 - w) : 1);
  scales[1] *= chromaFactor(1);
  scales[2] *= chromaFactor(2);

  // The wand buffer, in the layout `similarityWgsl` expects: K floats of reference z, then the m×K
  // whitening matrix. `sigmas` above is √λ for the leading three; the metric needs λ for however
  // many modes it keeps, so `whiteningMatrix` recomputes them from the same vᵀ·corr·v identity.
  const m = Math.max(1, Math.min(opts.modesUsed ?? 3, K, 32));
  const wand = new Float32Array(K + m * K);
  if (opts.reference) {
    const lambda = new Float64Array(K);
    for (let k = 0; k < K; k++) {
      let q = 0;
      for (let a = 0; a < K; a++) {
        for (let b = 0; b < K; b++) q += (opts.vectors[k * K + a] ?? 0) * corr[a * K + b]! * (opts.vectors[k * K + b] ?? 0);
      }
      lambda[k] = Math.max(q, 0);
    }
    const A = whiteningMatrix(opts.vectors, lambda, K, m);
    for (let a = 0; a < K; a++) wand[a] = opts.reference[a] ?? 0;
    for (let i = 0; i < m * K; i++) wand[K + i] = A[i] ?? 0;
  }

  return { chan, wand, m, scales, sigmas };
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
  const { buffer, rowFloats } = res.resident;
  const { chan, wand: wandData, m, scales, sigmas } = modeParams(res, opts);

  let ctx = surfaces.get(canvas);
  if (!ctx) {
    ctx = canvas.getContext("webgpu") as GPUCanvasContext;
    ctx.configure({ device, format, alphaMode: "opaque" });
    surfaces.set(canvas, ctx);
  }
  canvas.width = Math.max(1, Math.round(opts.outputSize?.width ?? res.width));
  canvas.height = Math.max(1, Math.round(opts.outputSize?.height ?? res.height));
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
  if (!wandBuf || wandCap < wandData.length) {
    wandCap = Math.max(wandData.length, wandCap * 2, 16);
    wandBuf = device.createBuffer({ size: wandCap * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
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
      m,
      ov.uv[0]!,
      ov.uv[1]!,
      ov.uv[2]!,
      0,
      ov.uv[3]!,
      ov.uv[4]!,
      ov.uv[5]!,
      0,
      opts.tolerance ?? 1.2,
      opts.reference ? 1 : 0,
      0,
      0,
    ]),
  );
  device.queue.writeBuffer(chanBuf, 0, chan);
  device.queue.writeBuffer(wandBuf, 0, wandData);

  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniBuf, size: UNI_FLOATS * 4 } },
      { binding: 1, resource: { buffer } },
      { binding: 2, resource: { buffer: chanBuf } },
      { binding: 3, resource: ov.view },
      { binding: 4, resource: ov.sampler },
      { binding: 5, resource: { buffer: wandBuf } },
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
