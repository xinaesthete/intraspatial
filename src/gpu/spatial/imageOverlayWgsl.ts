// The context-image blend, shared by the flat mode map and the terrain.
//
// Both views mix the same image against the same mode colour, so the mix has to be one expression
// or the flat map and the terrain would disagree about what "60% image" looks like — and the whole
// point of the pair is that you can move between them without re-reading the picture.
//
// The blend is done in **OKLab**, not sRGB. Mixing sRGB triples is a mix of encoded values, which
// darkens through the midpoint and drags hue; in a perceptually-uniform space the halfway blend
// looks halfway, which is what a slider labelled 0.5 has to mean. It also keeps the mode colours'
// meaning intact under partial blending: distance in the blended field stays proportional to
// distance in the mode field, just compressed.
//
// UV comes from a 2×3 affine the host composes, so each shader does one mad per axis and neither
// has to know anything about the image's own placement. Sampling outside [0,1] is clamped to the
// edge by the sampler and masked here, so a window that extends past the image gets the mode
// colour alone rather than a smear of its border pixels.

export const IMAGE_OVERLAY_WGSL = /* wgsl */ `
/** Apply a 2x3 row-major affine to a point. */
fn uvAt(m0: vec3f, m1: vec3f, p: vec2f) -> vec2f {
  return vec2f(m0.x * p.x + m0.y * p.y + m0.z, m1.x * p.x + m1.y * p.y + m1.z);
}

/** How much to actually blend at this UV: the mix amount on the image, 0 off it.
 *
 *  Returned as a WEIGHT rather than a bool because the caller must sample unconditionally.
 *  textureSample takes implicit derivatives, so WGSL forbids it in non-uniform control flow, and
 *  wrapping the sample in an on-image test is exactly that — it compiles to an invalid shader
 *  module whose only symptom is a pipeline that draws nothing. Sample first, weight second. */
fn uvWeight(uv: vec2f, mix: f32) -> f32 {
  let inside = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
  return select(0.0, mix, inside);
}

fn srgbToLinear1(c: f32) -> f32 {
  return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
}

/** sRGB -> OKLab, the inverse of the oklabToSrgb each shader already carries. Needed because the
 *  image arrives as an 8-bit sRGB texture while the mode colour is generated in OKLab. */
fn srgbToOklab(rgb: vec3f) -> vec3f {
  let lin = vec3f(srgbToLinear1(rgb.r), srgbToLinear1(rgb.g), srgbToLinear1(rgb.b));
  let l = 0.4122214708 * lin.r + 0.5363325363 * lin.g + 0.0514459929 * lin.b;
  let m = 0.2119034982 * lin.r + 0.6806995451 * lin.g + 0.1073969566 * lin.b;
  let s = 0.0883024619 * lin.r + 0.2817188376 * lin.g + 0.6299787005 * lin.b;
  let l_ = pow(max(l, 0.0), 1.0 / 3.0);
  let m_ = pow(max(m, 0.0), 1.0 / 3.0);
  let s_ = pow(max(s, 0.0), 1.0 / 3.0);
  return vec3f(
    0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
    1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
    0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_);
}
`;

/**
 * What a caller hands in to overlay an image.
 *
 * `uvFromWorld` is stated in **world** coordinates — not in whichever local space the consuming
 * shader happens to use — because the caller knows where the image sits in the world and nothing
 * else, while each shader knows its own local→world and nothing else. Each composes the two
 * itself with `composeUv`. Asking the caller to do it would mean exporting the terrain's model
 * space as public API, and any drift between that export and the vertex shader would land the
 * overlay silently off-register.
 */
export interface ImageOverlay {
  readonly texture: GPUTexture;
  /** 2×3 row-major: image UV in [0,1]² from a world XY. */
  readonly uvFromWorld: ArrayLike<number>;
  readonly mix: number;
}

/** Compose two 2×3 row-major affines: `uv ∘ worldFromLocal`. */
export function composeUv(uvFromWorld: ArrayLike<number>, worldFromLocal: ArrayLike<number>): Float32Array {
  const g = (m: ArrayLike<number>, r: number, c: number) => m[r * 3 + c] ?? 0;
  const out = new Float32Array(6);
  for (let r = 0; r < 2; r++) {
    out[r * 3] = g(uvFromWorld, r, 0) * g(worldFromLocal, 0, 0) + g(uvFromWorld, r, 1) * g(worldFromLocal, 1, 0);
    out[r * 3 + 1] = g(uvFromWorld, r, 0) * g(worldFromLocal, 0, 1) + g(uvFromWorld, r, 1) * g(worldFromLocal, 1, 1);
    out[r * 3 + 2] = g(uvFromWorld, r, 0) * g(worldFromLocal, 0, 2) + g(uvFromWorld, r, 1) * g(worldFromLocal, 1, 2) + g(uvFromWorld, r, 2);
  }
  return out;
}

let blank: GPUTexture | undefined;
let sampler: GPUSampler | undefined;

/**
 * Texture view, sampler and uniform values for the overlay — including when there is no image.
 *
 * A bind group must satisfy every binding the shader declares, so "no overlay" cannot simply omit
 * the texture; it binds a 1×1 blank with `mix = 0`, which the shader's own branch then skips. The
 * blank is created once and never destroyed, the pooling rule the rest of this directory follows.
 *
 * The sampler clamps rather than repeats: a window extending past the image would otherwise tile it,
 * and a tiled second copy of the tissue is a far worse failure than a missing edge.
 */
export function overlayResources(device: GPUDevice, image: ImageOverlay | undefined, worldFromLocal: ArrayLike<number>) {
  if (!blank) {
    blank = device.createTexture({
      size: { width: 1, height: 1 },
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: blank }, new Uint8Array([0, 0, 0, 255]), { bytesPerRow: 4 }, { width: 1, height: 1 });
  }
  sampler ??= device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });
  const uv = image ? composeUv(image.uvFromWorld, worldFromLocal) : new Float32Array(6);
  return {
    view: (image?.texture ?? blank).createView(),
    sampler,
    mix: image ? Math.max(0, Math.min(1, image.mix)) : 0,
    uv,
  };
}
