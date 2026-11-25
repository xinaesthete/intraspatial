// The march `IsoSurfaceLayer` draws with, on its own so a test can run it without deck: the field
// read, interpolation, and the search along one line of sight for where the field crosses the level.
//
// It needs, declared by whoever includes it:
//   var<uniform> iso: a struct with `dims: vec4<f32>` (nx, ny, nz, the level) and
//                     `march: vec4<f32>` (x the most steps);
//   var<storage, read> field: array<f32>, one f32 per voxel, x fastest.
// Positions are box coordinates, 0..1 along each edge; a line of sight is `u0 + du * t`.

export const ISO_MARCH_WGSL = /* wgsl */ `\
fn iso_voxel(x: i32, y: i32, z: i32) -> f32 {
  let n = vec3<i32>(iso.dims.xyz);
  let q = clamp(vec3<i32>(x, y, z), vec3<i32>(0), n - vec3<i32>(1));
  return field[u32(q.x + n.x * (q.y + n.y * q.z))];
}

/** The field at box coordinates u (0..1 along each edge), interpolated between voxel centres. */
fn iso_field(u: vec3<f32>) -> f32 {
  let g = u * iso.dims.xyz - vec3<f32>(0.5);
  let base = floor(g);
  let f = g - base;
  let i = vec3<i32>(base);
  let x00 = mix(iso_voxel(i.x, i.y, i.z), iso_voxel(i.x + 1, i.y, i.z), f.x);
  let x10 = mix(iso_voxel(i.x, i.y + 1, i.z), iso_voxel(i.x + 1, i.y + 1, i.z), f.x);
  let x01 = mix(iso_voxel(i.x, i.y, i.z + 1), iso_voxel(i.x + 1, i.y, i.z + 1), f.x);
  let x11 = mix(iso_voxel(i.x, i.y + 1, i.z + 1), iso_voxel(i.x + 1, i.y + 1, i.z + 1), f.x);
  return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

struct IsoHit {
  hit: bool,
  /** Where along the line of sight. */
  at: f32,
  /** The surface was met from inside: the view starts within it and this is where it is left. */
  exiting: bool,
};

/**
 * The first place on u0 + du * t, for t from enter to leave, where the surface is met.
 *
 * Starting at or above the level, what is met depends on where the start is. On a face of the box
 * (fromNear false), the surface has been cut by the box, and the cut is drawn there, at enter. At
 * the near plane (fromNear true), the view is inside the surface, and what is seen is the inside of
 * its wall: the first place the field falls below the level.
 *
 * Half a voxel at a step. Where a step ends close to the level, the field may have crossed it and come
 * back in between (a line of sight grazing the surface: under the level going in, over it going out),
 * so that stretch is looked at four times as closely: without this the outline breaks into teeth.
 * Closeness is a factor of 2 either side of the level (0.5 going in, 1.5 going out). The crossing is then pinned
 * down by halving.
 */
fn iso_march(u0: vec3<f32>, du: vec3<f32>, enter: f32, leave: f32, fromNear: bool) -> IsoHit {
  let level = iso.dims.w;
  let perStep = 0.5 / max(length(du * iso.dims.xyz), 1e-9);
  let steps = clamp(ceil((leave - enter) / perStep), 1.0, iso.march.x);
  let dt = (leave - enter) / steps;
  var before = iso_field(u0 + du * enter);
  let inside = before >= level;
  let exiting = inside && fromNear;
  var hit = inside && !fromNear;
  // The crossing lies between these: on the starting side at "under", past the crossing at "over".
  var under = enter;
  var over = enter;
  if (!hit) {
    for (var k = 1.0; k <= steps; k = k + 1.0) {
      let t = enter + dt * k;
      let here = iso_field(u0 + du * t);
      if ((here >= level) != exiting) {
        under = t - dt;
        over = t;
        hit = true;
        break;
      }
      // Close under the level going in, or close over it going out.
      let grazing = select(max(before, here) > 0.5 * level, min(before, here) < 1.5 * level, exiting);
      if (grazing) {
        var found = false;
        var foundAt = t;
        for (var j = 1.0; j < 4.0; j = j + 1.0) {
          let tj = t - dt + dt * j * 0.25;
          if (!found && (iso_field(u0 + du * tj) >= level) != exiting) {
            found = true;
            foundAt = tj;
          }
        }
        if (found) {
          under = foundAt - dt * 0.25;
          over = foundAt;
          hit = true;
          break;
        }
      }
      before = here;
    }
  }
  // Only a crossing found by stepping has a stretch to halve; a cut at enter does not.
  if (hit && over > enter) {
    for (var r = 0; r < 6; r = r + 1) {
      let mid = 0.5 * (under + over);
      let past = (iso_field(u0 + du * mid) >= level) != exiting;
      under = select(mid, under, past);
      over = select(over, mid, past);
    }
  }
  var out: IsoHit;
  out.hit = hit;
  out.at = select(0.5 * (under + over), enter, over <= enter);
  out.exiting = exiting;
  return out;
}
`;
