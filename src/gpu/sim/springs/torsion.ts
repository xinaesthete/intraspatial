// Torsion springs: hold the dihedral angle of four consecutive particles a-b-c-d (the twist of
// a-b against c-d about b-c) at a target. A port of FoldSynth's `TorsionSpring.java` (Ben
// Jefferys, extensively reworked by Stephen Todd), which is where CSynth's torsion springs come
// from; CSynth's own shader for them is not in the organic repo.
//
// The four forces are built so the spring only twists:
//   - `fa` and `fd` act along the normals of the planes a-b-c and b-c-d, so they change neither
//     bond lengths nor bond angles;
//   - `fb` and `fc` are then solved from "no net force" and "no net torque".
// Near a straight line the dihedral is ill-defined, so the forces are scaled down there, and
// the drive fades to zero as the error approaches ±π so that it never jumps.
//
// Differences from the Java:
//   - exact `atan2`, where the Java uses a fast approximation good to about 0.07 rad;
//   - the over-strong clamp scales the forces to the limit. The Java line is
//     `Math.sqrt(1 / rr*k*k)`, which by precedence is `sqrt(k² / rr)` and not that;
//   - torque is taken about `b` rather than the origin, which is the same equation and keeps
//     f32 precision far from the origin;
//   - current positions, where the Java uses its integrator's step-average positions.

/** FoldSynth multiplies torsion forces by this to bring them into line with its other springs. */
export const TORSION_SCALE = 10;

/** Dihedral angle of a-b-c-d in (-π, π], or NaN when three of the points are in a line. */
export function dihedral(pos: ArrayLike<number>, a: number, b: number, c: number, d: number, stride = 4): number {
  const out = new Float64Array(13);
  return quad(pos, a, b, c, d, stride, out) ? out[12]! : Number.NaN;
}

/**
 * Geometry shared by the angle and the forces. Fills `out[0..11]` with the unscaled forces on
 * a, b, c, d and `out[12]` with the angle. Returns false when the quad is degenerate or one of
 * its bonds is longer than `maxLenSq` (squared).
 */
function quad(
  pos: ArrayLike<number>,
  a: number,
  b: number,
  c: number,
  d: number,
  stride: number,
  out: Float64Array,
  maxLenSq = Infinity,
): boolean {
  const ax = pos[a * stride]! - pos[b * stride]!;
  const ay = pos[a * stride + 1]! - pos[b * stride + 1]!;
  const az = pos[a * stride + 2]! - pos[b * stride + 2]!;
  const cx = pos[c * stride]! - pos[b * stride]!;
  const cy = pos[c * stride + 1]! - pos[b * stride + 1]!;
  const cz = pos[c * stride + 2]! - pos[b * stride + 2]!;
  const dx = pos[d * stride]! - pos[b * stride]!;
  const dy = pos[d * stride + 1]! - pos[b * stride + 1]!;
  const dz = pos[d * stride + 2]! - pos[b * stride + 2]!;
  // With b at the origin: vab = -a, vbc = c, vcd = d - c.
  const vabx = -ax;
  const vaby = -ay;
  const vabz = -az;
  const vcdx = dx - cx;
  const vcdy = dy - cy;
  const vcdz = dz - cz;
  const bcSq = cx * cx + cy * cy + cz * cz;
  if (ax * ax + ay * ay + az * az > maxLenSq || bcSq > maxLenSq || vcdx * vcdx + vcdy * vcdy + vcdz * vcdz > maxLenSq) return false;
  const lenBC = Math.sqrt(bcSq);
  // xabc = vbc × vab, xbcd = vcd × vbc, each normalised with its length kept.
  let ux = cy * vabz - cz * vaby;
  let uy = cz * vabx - cx * vabz;
  let uz = cx * vaby - cy * vabx;
  let wx = vcdy * cz - vcdz * cy;
  let wy = vcdz * cx - vcdx * cz;
  let wz = vcdx * cy - vcdy * cx;
  const lenABC = Math.hypot(ux, uy, uz);
  const lenBCD = Math.hypot(wx, wy, wz);
  if (!(lenBC > 1e-9 && lenABC > 1e-9 && lenBCD > 1e-9)) return false;
  ux /= lenABC;
  uy /= lenABC;
  uz /= lenABC;
  wx /= lenBCD;
  wy /= lenBCD;
  wz /= lenBCD;
  const sin = ((uy * wz - uz * wy) * cx + (uz * wx - ux * wz) * cy + (ux * wy - uy * wx) * cz) / lenBC;
  out[12] = Math.atan2(sin, ux * wx + uy * wy + uz * wz);

  // Scale down near degenerate, never up.
  const ss = Math.min(1, lenBCD * lenABC);
  const fax = ux * lenBCD * ss;
  const fay = uy * lenBCD * ss;
  const faz = uz * lenBCD * ss;
  const fdx = -wx * lenABC * ss;
  const fdy = -wy * lenABC * ss;
  const fdz = -wz * lenABC * ss;
  // K1 = -(fa + fd); K2 = -(fa × pa + fd × pd); K3 = K1 × pc - K2; fb = (vbcn × K3) / lenBC.
  const k1x = -(fax + fdx);
  const k1y = -(fay + fdy);
  const k1z = -(faz + fdz);
  const k3x = k1y * cz - k1z * cy + (fay * az - faz * ay) + (fdy * dz - fdz * dy);
  const k3y = k1z * cx - k1x * cz + (faz * ax - fax * az) + (fdz * dx - fdx * dz);
  const k3z = k1x * cy - k1y * cx + (fax * ay - fay * ax) + (fdx * dy - fdy * dx);
  const fbx = (cy * k3z - cz * k3y) / bcSq;
  const fby = (cz * k3x - cx * k3z) / bcSq;
  const fbz = (cx * k3y - cy * k3x) / bcSq;
  out.set([fax, fay, faz, fbx, fby, fbz, k1x - fbx, k1y - fby, k1z - fbz, fdx, fdy, fdz]);
  return true;
}

/**
 * Forces of one torsion spring on its four particles, written to `out[0..11]` (a, b, c, d).
 * `strength` is the spring's stiffness times any global factor. Returns false, with `out`
 * untouched or meaningless, when the spring applies no force.
 */
export function torsionForces(
  pos: ArrayLike<number>,
  a: number,
  b: number,
  c: number,
  d: number,
  targetAngle: number,
  strength: number,
  out: Float64Array,
  maxLenSq = Infinity,
  stride = 4,
): boolean {
  if (strength === 0 || !quad(pos, a, b, c, d, stride, out, maxLenSq)) return false;
  let stress = targetAngle - out[12]!;
  if (stress > Math.PI) stress -= 2 * Math.PI;
  if (stress < -Math.PI) stress += 2 * Math.PI;
  // Fade past a quarter turn, so the drive is zero at ±π instead of flipping sign there.
  if (stress < -Math.PI / 2) stress = -Math.PI - stress;
  if (stress > Math.PI / 2) stress = Math.PI - stress;

  let rr = 0;
  for (let k = 0; k < 12; k++) rr += out[k]! * out[k]!;
  const over = rr * strength * strength;
  if (!(over <= 1000)) return false;
  const clamp = over > 1 ? 1 / Math.sqrt(over) : 1;
  const dd = stress * strength * TORSION_SCALE * clamp;
  for (let k = 0; k < 12; k++) out[k] = out[k]! * dd;
  return true;
}
