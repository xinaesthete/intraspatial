// One sample marker, two shaders.
//
// The wand's sample point has to be visible in the flat mode map AND on the terrain, and the two
// marks must be recognisably the same thing or the viewer has to work out that they are. So the
// mask lives here and both fragment shaders call it, exactly as `kernelWgsl.ts` shares the kernel
// family between `tcmRender` and `gramMatrix` to stop the two drifting apart.
//
// It is a pair of full-span **rule lines** rather than a ring or a pin. A ring has to be found
// before it can be read, and on the terrain it is worse than that: a ring is a locus in the
// surface's XY, so wherever the relief is steep it drapes down a near-vertical face and stops
// reading as a ring at all. Two lines crossing at the sample are found immediately at any zoom, and
// draping them is a feature — each is the surface's profile along one axis through the sample.
//
// It is a *mask* rather than a colour, and it is evaluated at two widths, because the mark has to
// survive an arbitrary background: the mode map is a full-chroma OKLab painting and the terrain is
// the same painting under shading, so any fixed colour is invisible somewhere. Drawing white
// through the narrow mask over black through the wide one gives a haloed mark that reads on both.
//
// Widths are per-axis and in the caller's own units — raster pixels in the flat map, model units on
// the terrain, where they are derived from screen-space derivatives so the lines keep a constant
// weight however the surface is tilted.

export const MARKER_WGSL = /* wgsl */ `
/** Two crossing rule lines through the sample, given mp = position - marker. Returns 1 on a line. */
fn markerMask(mp: vec2f, w: vec2f) -> f32 {
  return max(step(abs(mp.x), w.x), step(abs(mp.y), w.y));
}

/** Composite the haloed lines over a colour already in sRGB. */
fn markerOver(base: vec3f, mp: vec2f, w: vec2f, on: f32) -> vec3f {
  if (on < 0.5) { return base; }
  let halo = markerMask(mp, w * 2.6);
  let core = markerMask(mp, w);
  return mix(mix(base, vec3f(0.02), halo * 0.8), vec3f(1.0), core);
}
`;
