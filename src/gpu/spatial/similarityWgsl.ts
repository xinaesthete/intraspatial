// The wand's whitened distance, and the selection boundary drawn from it — shared by the flat mode
// map and the terrain.
//
// This one MUST be shared rather than merely kept in step. The terrain shades a region by
// similarity and the map draws an outline around it; if the two computed the distance even slightly
// differently, the outline would enclose a region the colour does not agree with, and there would
// be no way to tell from the picture which was wrong. One expression, two consumers.
//
// ## The contract
//
// Unlike `markerWgsl.ts`, this snippet is a *mixin*: it reads bindings and uniform fields the host
// shader declares. Including it requires
//
//   * `fn fetch(a: u32, col: u32, row: u32) -> f32` — the channel raster;
//   * `chan: array<f32>` — 5 floats per channel: mean, 1/sd, three mode loadings;
//   * `wand: array<f32>` — K floats of reference z, then m*K of the whitening matrix, row-major;
//   * `U.K` and `U.m` — channel count and how many modes the metric keeps.
//
// The alternative was passing a dozen arguments through, which buys nothing: any shader that wants
// this already has all four, and naming them in one place is what makes the requirement checkable.
//
// ## Why the distance is whitened
//
// `src/gpu/spatial/gramTerrain.ts` derives it: `d² = Δzᵀ corr⁻¹ Δz = Σ_k (Δy_k)²/λ_k` is
// Mahalanobis distance in channel space, and `A = Λ^{-1/2} Vᵀ` truncated to `m` modes is the matrix
// that computes it. Truncating to 3 makes the distance a distance in exactly the space the OKLab
// map is painted from, so "looks similar" and "is inside the outline" agree by construction.

/** The similarity direction in OKLab's (a, b) plane, and the lightness the selection outline is
 *  drawn at. Chosen away from the mode axes' typical directions so a similarity reading is never
 *  mistaken for a mode reading, and shared so the terrain's shading and the map's outline are
 *  visibly the same colour — one says "how similar", the other says "where the boundary is". */
export const SIM_A = 0.6;
export const SIM_B = -0.8;
export const SELECTION_L = 0.86;

export const SIMILARITY_WGSL = /* wgsl */ `
/** Mode coordinates (xyz) and whitened distance to the wand reference (w), at one raster pixel.
 *  One pass over the channels serves both: the standardised z is needed by each. */
fn sampleWhitened(col: u32, row: u32) -> vec4f {
  let K = u32(U.K);
  let m = u32(U.m);
  var y = vec3f(0.0);
  var u: array<f32, 32>;               // MAX_CHANNELS; m <= K <= 32
  for (var k = 0u; k < m; k = k + 1u) { u[k] = 0.0; }
  for (var a = 0u; a < K; a = a + 1u) {
    let z = (fetch(a, col, row) - chan[a * 5u]) * chan[a * 5u + 1u];
    y = y + z * vec3f(chan[a * 5u + 2u], chan[a * 5u + 3u], chan[a * 5u + 4u]);
    let dz = z - wand[a];
    // u = A·Δz, accumulated channel-major so Δz is computed once for both uses.
    for (var k = 0u; k < m; k = k + 1u) { u[k] = u[k] + wand[K + k * K + a] * dz; }
  }
  var d2 = 0.0;
  for (var k = 0u; k < m; k = k + 1u) { d2 = d2 + u[k] * u[k]; }
  return vec4f(y, sqrt(d2));
}

/** 1 on the boundary of the selected region, the level set d = tol.
 *
 *  The width is taken from the screen-space derivative of d, so the outline is a constant couple of
 *  pixels whether the similarity field is changing fast or slowly. A fixed threshold in d would
 *  draw a hairline through steep gradients and a broad smear through flat ones — which is the same
 *  argument the marker's rule lines make, for the same reason. Fragment-stage only: fwidth has no
 *  meaning in a vertex shader. */
fn selectionEdge(d: f32, tol: f32, on: f32) -> f32 {
  if (on < 0.5) { return 0.0; }
  return step(abs(d - tol), fwidth(d) * 1.2);
}
`;
