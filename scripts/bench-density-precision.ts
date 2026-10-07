// What does an f16 density field cost in accuracy, at a range of overlap depths?
//
// The field is a render target the splat blends into additively, so the error is not a single
// rounding — each add rounds, and the error grows with how many contributions land in a texel.
// A tolerance picked by eye would be a guess; this prints the numbers the guess would be made
// from, and `docs/field-precision.md` quotes them.
//
// Not a vitest benchmark, for the reason `bench-scan.ts` gives: run as a plain process.
//
//   pnpm bench:density-precision

import { releaseDevice } from "../src/gpu/device";
import { Graph, pull, registerBuiltinOps } from "../src/gpu/graph";

registerBuiltinOps();

const W = 128;
const H = 128;
const BBOX = [0, 0, 100, 100];

function cloud(n: number, seed: number, spread: number) {
  let a = seed;
  const rnd = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(50 + (rnd() - 0.5) * 100 * spread);
    ys.push(50 + (rnd() - 0.5) * 100 * spread);
  }
  return { xs, ys };
}

async function field(precision: "f32" | "f16", n: number, spread: number, sigma: number) {
  const g = new Graph();
  const { xs, ys } = cloud(n, 0xbeef, spread);
  const h = g.op1("splatDensity", { points: g.points(xs, ys) }, { width: W, height: H, sigma, radiusSigma: 4, bbox: BBOX, precision });
  return (await pull(g, h)).data as Float32Array;
}

async function main() {
  console.log(`Density field ${W}×${H}, σ as shown, f16 vs f32. Relative error is measured only on`);
  console.log("texels above 1e-3 of the peak; mass is the summed field (the integral the statistic uses).\n");
  console.log("n       spread  sigma  peak        max rel err   mean rel err   mass ratio");
  for (const [n, spread, sigma] of [
    [200, 1, 3],
    [400, 1, 3],
    [1600, 0.5, 3],
    [6400, 0.5, 3],
    [6400, 0.25, 6],
    [25000, 0.25, 6],
  ] as const) {
    const [a, b] = await Promise.all([field("f16", n, spread, sigma), field("f32", n, spread, sigma)]);
    let peak = 0;
    for (const v of b) if (v > peak) peak = v;
    const floor = peak * 1e-3;
    let maxRel = 0;
    let relSum = 0;
    let counted = 0;
    let s16 = 0;
    let s32 = 0;
    for (let i = 0; i < b.length; i++) {
      s16 += a[i]!;
      s32 += b[i]!;
      if (b[i]! < floor) continue;
      const rel = Math.abs(a[i]! - b[i]!) / b[i]!;
      maxRel = Math.max(maxRel, rel);
      relSum += rel;
      counted++;
    }
    const f = (x: number, w: number) => x.toPrecision(3).padEnd(w);
    console.log(
      `${String(n).padEnd(8)}${String(spread).padEnd(8)}${String(sigma).padEnd(7)}${f(peak, 12)}${f(maxRel, 14)}${f(relSum / Math.max(counted, 1), 15)}${f(s16 / s32, 10)}`,
    );
  }
  console.log("\nf16 saturates at 65504; a peak approaching that is where the field stops being usable.");
  await releaseDevice();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
