// okLab / okLCH for TSL — the shader-side mirror of src/color/oklab.ts, so material colour
// nodes can interpolate perceptually (mix in okLab, ramp hue in okLCH) and output linear-sRGB
// for three's working colour space. Same Ottosson coefficients as the CPU module, which is
// golden-tested; this module is kept honest by matching it line-for-line (a GPU readback
// cross-check is future work — see docs/render-traits-and-expression-dsl.md).
//
// This is a THIN transcription adapter: three's own TSL node types are inconsistent (vec3() →
// VarNode vs mix() → Node; several builtins are typed float-only though componentwise at runtime),
// so the node args here are loosely typed. The nominal, runtime-tagged colour types (`space`
// property) belong on top of this layer, not inside it — see task #27 / ADR-0007.
//
// Colour space note: three treats a material's colorNode value as **linear-sRGB working space**
// and applies the linear→display transform itself. So author perceptual intent in okLab/okLCH,
// convert with oklab*ToLinear here, and let three handle the final encode.
import { cbrt, cos, Fn, sin, vec3 } from "three/tsl";

// biome placeholder: `Tsl` is any TSL vec3-ish node (see header — three's node types don't give a
// single clean name that both VarNode and Node satisfy while carrying the operator methods).
// biome-ignore lint/suspicious/noExplicitAny: three TSL node types are inconsistent; adapter layer.
type Tsl = any;

/** okLab → linear-sRGB. Pure polynomial (cube, no branch) — NaN-safe for out-of-gamut input. */
export const oklabToLinear = Fn(([lab]: [Tsl]) => {
  const L = lab.x;
  const a = lab.y;
  const b = lab.z;
  const l_ = L.add(a.mul(0.3963377774)).add(b.mul(0.2158037573));
  const m_ = L.sub(a.mul(0.1055613458)).sub(b.mul(0.0638541728));
  const s_ = L.sub(a.mul(0.0894841775)).sub(b.mul(1.291485548));
  const l = l_.mul(l_).mul(l_);
  const m = m_.mul(m_).mul(m_);
  const s = s_.mul(s_).mul(s_);
  return vec3(
    l.mul(4.0767416621).sub(m.mul(3.3077115913)).add(s.mul(0.2309699292)),
    l.mul(-1.2684380046).add(m.mul(2.6097574011)).sub(s.mul(0.3413193965)),
    l.mul(-0.0041960863).sub(m.mul(0.7034186147)).add(s.mul(1.707614701)),
  );
});

/** linear-sRGB → okLab (uses cbrt). The documented inverse; not on the trail hot path. */
export const linearToOklab = Fn(([c]: [Tsl]) => {
  const r = c.x;
  const g = c.y;
  const b = c.z;
  const l = r.mul(0.4122214708).add(g.mul(0.5363325363)).add(b.mul(0.0514459929));
  const m = r.mul(0.2119034982).add(g.mul(0.6806995451)).add(b.mul(0.1073969566));
  const s = r.mul(0.0883024619).add(g.mul(0.2817188376)).add(b.mul(0.6299787005));
  const l_ = cbrt(l);
  const m_ = cbrt(m);
  const s_ = cbrt(s);
  return vec3(
    l_.mul(0.2104542553).add(m_.mul(0.793617785)).sub(s_.mul(0.0040720468)),
    l_.mul(1.9779984951).sub(m_.mul(2.428592205)).add(s_.mul(0.4505937099)),
    l_.mul(0.0259040371).add(m_.mul(0.7827717662)).sub(s_.mul(0.808675766)),
  );
});

/** okLCH (L, C, h radians) → okLab. */
export const oklchToOklab = Fn(([lch]: [Tsl]) => {
  const h = lch.z;
  return vec3(lch.x, lch.y.mul(cos(h)), lch.y.mul(sin(h)));
});

/** okLCH → linear-sRGB — the direct path for a scalar → hue ramp in a shader (the channel-bridge
 *  instance-colour work will use this). */
export const oklchToLinear = Fn(([lch]: [Tsl]) => oklabToLinear(oklchToOklab(lch)));
