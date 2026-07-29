// The unit-mass radial kernel family, in WGSL — one definition, shared by every shader that
// splats through it.
//
// `src/spatial/kernels.ts` is the source of truth and this is its device-side mirror. There is no
// way to make a shader import a TypeScript function, so the mirror is unavoidable; what IS
// avoidable is having *two* mirrors drift apart, which is why `tcmRender.ts` and `gramMatrix.ts`
// interpolate this string rather than each carrying their own copy. The constants are derived from
// `GAUSS_TRUNC` at module load so a change there propagates to both.

import { GAUSS_TRUNC } from "../../spatial/kernels";

/** WGSL source declaring `fn kernelAt(d2: f32, r: f32, code: f32) -> f32`, matching
 *  `kernelAt` in `src/spatial/kernels.ts` exactly. `code` is `kernelCode(spec)`: a polynomial
 *  order `>= 0`, or `-1` for the truncated Gaussian. */
export const KERNEL_WGSL = /* wgsl */ `
const PI = 3.14159265358979;
const E_HALF = ${Math.exp((-GAUSS_TRUNC * GAUSS_TRUNC) / 2)};   // exp(-GAUSS_TRUNC^2/2), the truncated Gaussian's lost mass
const T2 = ${(GAUSS_TRUNC * GAUSS_TRUNC).toFixed(1)};

// Unit-mass radial kernels, matching src/spatial/kernels.ts exactly.
//   code >= 0 : polynomial order n, K = (n+1)/(pi r^2) (1 - d^2/r^2)^n
//   code <  0 : gaussian truncated at r = GAUSS_TRUNC * sigma
fn kernelAt(d2: f32, r: f32, code: f32) -> f32 {
  let r2 = r * r;
  let t = d2 / r2;
  if (t >= 1.0) { return 0.0; }
  if (code < 0.0) {
    let s2 = r2 / T2;
    return exp(-d2 / (2.0 * s2)) / (2.0 * PI * s2 * (1.0 - E_HALF));
  }
  return ((code + 1.0) / (PI * r2)) * pow(1.0 - t, code);
}
`;
