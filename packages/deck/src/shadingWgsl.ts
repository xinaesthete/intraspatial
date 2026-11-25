// WGSL the 3D layers share: the camera's axes, a simple lit material, and a colour ramp.
// `VIEW_FRAME_WGSL` needs deck's `project` module; `RAMP_WGSL` needs the layer to declare
// `palette: array<vec4<f32>>`.

/** The camera's axes in deck's common space. The first two rows of the view-projection matrix are
 *  the screen's right and up there, in any view; which way their cross product points depends on
 *  the view's handedness, so it is settled by asking which way depth decreases. */
export const VIEW_FRAME_WGSL = /* wgsl */ `
struct ViewFrame {
  right: vec3<f32>,
  up: vec3<f32>,
  /** From the scene toward the camera. */
  toward: vec3<f32>,
};

fn view_frame() -> ViewFrame {
  let m = project.viewProjectionMatrix;
  var f: ViewFrame;
  f.right = normalize(vec3<f32>(m[0][0], m[1][0], m[2][0]));
  f.up = normalize(vec3<f32>(m[0][1], m[1][1], m[2][1]));
  let across = cross(f.right, f.up);
  let here = project_common_position_to_clipspace(vec4<f32>(0.0, 0.0, 0.0, 1.0));
  let there = project_common_position_to_clipspace(vec4<f32>(across, 1.0));
  f.toward = select(-across, across, there.z / there.w < here.z / here.w);
  return f;
}

/** A common-space direction in view space: x right, y up, z toward the camera. */
fn to_view(v: vec3<f32>, f: ViewFrame) -> vec3<f32> {
  return vec3<f32>(dot(v, f.right), dot(v, f.up), dot(v, f.toward));
}
`;

/** Colour of a surface point: the base colour lit from the upper left and softly from all round, with a small highlight.
 *  `normal` is in view space (x right, y up, z toward the camera); `shading` 0 gives the flat colour. */
export const MATERIAL_WGSL = /* wgsl */ `
fn material(base: vec3<f32>, normal: vec3<f32>, shading: f32) -> vec3<f32> {
  let light = normalize(vec3<f32>(-0.45, 0.55, 0.7));
  let diffuse = max(dot(normal, light), 0.0);
  let highlight = pow(max(dot(normal, normalize(light + vec3<f32>(0.0, 0.0, 1.0))), 0.0), 40.0);
  // Light from all round, more from above than below, so a side turned from the lamp keeps its form.
  let around = mix(0.16, 0.36, normal.y * 0.5 + 0.5);
  let lit = base * (around + 0.68 * diffuse) + vec3<f32>(0.25 * highlight);
  return mix(base, lit, shading);
}
`;

/** The palette read as a ramp over t in 0..1, interpolated between stops. */
export const RAMP_WGSL = /* wgsl */ `
fn ramp_at(t: f32, stops: u32) -> vec4<f32> {
  if (stops < 2u) {
    return palette[0];
  }
  let s = clamp(t, 0.0, 1.0) * f32(stops - 1u);
  let i = min(u32(floor(s)), stops - 2u);
  return mix(palette[i], palette[i + 1u], s - f32(i));
}
`;
