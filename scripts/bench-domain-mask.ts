// What does representing an ROI W as a raster mask cost the cross-PCF? `docs/stats-on-arbitrary-
// domains.md` quotes these tables.
//
// Ground truth comes from domains with closed forms: a rotated rectangle (rotate into its frame,
// then the exact `diskRectArea`) and a disk (lens area). Mask methods, all on the SAME mask
// (pixel-centre rasterisation, random grid phase):
//   centre : h² · #{mask pixels whose centre lies in the annulus}
//   exact  : Σ_{mask pixels} |annulus ∩ pixel| — exact for the pixel union, so its error is the
//            staircase alone
//   cover  : Σ_pixels coverage(p) · |annulus ∩ pixel|, coverage from 8×8 supersampling
//
//   pnpm bench:domain-mask          # end-to-end g (Exp 2) and mis-drawn boundary (Exp 4), ~40 s
//   pnpm bench:domain-mask --all    # also mask area (Exp 3) and per-anchor area error (Exp 1)

import { diskRectArea } from "../src/spatial/edgeCorrection";

const R_EDGES = [0, 10, 20, 30]; // µm, the covid binning: gr20 is [20,30)
const RMAX = 30;

function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Domain {
  name: string;
  inside(x: number, y: number): boolean;
  diskArea(cx: number, cy: number, r: number): number; // exact |disk ∩ W|
  dist(x: number, y: number): number; // distance to boundary (inside points)
  area: number;
  bbox: [number, number, number, number];
}

function rotatedRect(w: number, hgt: number, thetaDeg: number): Domain {
  const t = (thetaDeg * Math.PI) / 180;
  const c = Math.cos(t);
  const s = Math.sin(t);
  const toLocal = (x: number, y: number) => [c * x + s * y, -s * x + c * y] as const;
  const corners = [
    [-w / 2, -hgt / 2],
    [w / 2, -hgt / 2],
    [w / 2, hgt / 2],
    [-w / 2, hgt / 2],
  ].map(([u, v]) => [c * u! - s * v!, s * u! + c * v!]);
  const xs = corners.map((p) => p[0]!);
  const ys = corners.map((p) => p[1]!);
  return {
    name: `rect ${w}×${hgt} @${thetaDeg}°`,
    inside(x, y) {
      const [u, v] = toLocal(x, y);
      return Math.abs(u) <= w / 2 && Math.abs(v) <= hgt / 2;
    },
    diskArea(cx, cy, r) {
      const [u, v] = toLocal(cx, cy);
      return diskRectArea(u, v, r, -w / 2, -hgt / 2, w / 2, hgt / 2);
    },
    dist(x, y) {
      const [u, v] = toLocal(x, y);
      return Math.min(w / 2 - Math.abs(u), hgt / 2 - Math.abs(v));
    },
    area: w * hgt,
    bbox: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
  };
}

function lens(r: number, R: number, d: number): number {
  if (d >= r + R) return 0;
  if (d <= Math.abs(R - r)) return Math.PI * Math.min(r, R) ** 2;
  const a = r * r * Math.acos((d * d + r * r - R * R) / (2 * d * r));
  const b = R * R * Math.acos((d * d + R * R - r * r) / (2 * d * R));
  const k = 0.5 * Math.sqrt((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R));
  return a + b - k;
}

function diskDomain(R: number): Domain {
  return {
    name: `disk R=${R}`,
    inside: (x, y) => x * x + y * y <= R * R,
    diskArea: (cx, cy, r) => (r <= 0 ? 0 : lens(r, R, Math.hypot(cx, cy))),
    dist: (x, y) => R - Math.hypot(x, y),
    area: Math.PI * R * R,
    bbox: [-R, -R, R, R],
  };
}

interface Mask {
  h: number;
  ox: number;
  oy: number;
  cols: number;
  rows: number;
  bin: Uint8Array; // pixel-centre rasterisation
  cov: Float32Array; // 8×8 supersampled coverage
}

function rasterise(d: Domain, h: number, rnd: () => number): Mask {
  const ox = d.bbox[0] - 2 * h - rnd() * h;
  const oy = d.bbox[1] - 2 * h - rnd() * h;
  const cols = Math.ceil((d.bbox[2] - ox) / h) + 2;
  const rows = Math.ceil((d.bbox[3] - oy) / h) + 2;
  const bin = new Uint8Array(cols * rows);
  const cov = new Float32Array(cols * rows);
  const S = 8;
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const x0 = ox + i * h;
      const y0 = oy + j * h;
      const cIn = d.inside(x0 + h / 2, y0 + h / 2);
      bin[j * cols + i] = cIn ? 1 : 0;
      // A centre further than h from ∂W (the half-diagonal is 0.71h) means the pixel is uncut.
      // `dist` never overstates the true distance outside W, so this cannot skip a cut pixel.
      if (Math.abs(d.dist(x0 + h / 2, y0 + h / 2)) > h) {
        cov[j * cols + i] = cIn ? 1 : 0;
        continue;
      }
      let n = 0;
      for (let b = 0; b < S; b++) for (let a = 0; a < S; a++) if (d.inside(x0 + ((a + 0.5) * h) / S, y0 + ((b + 0.5) * h) / S)) n++;
      cov[j * cols + i] = n / (S * S);
    }
  }
  return { h, ox, oy, cols, rows, bin, cov };
}

// Disk ∩ W for the three mask methods, at every bin edge radius.
function maskDiskAreas(m: Mask, cx: number, cy: number) {
  const nE = R_EDGES.length;
  const centre = new Float64Array(nE);
  const exact = new Float64Array(nE);
  const cover = new Float64Array(nE);
  const { h, ox, oy, cols, rows } = m;
  const i0 = Math.max(0, Math.floor((cx - RMAX - ox) / h));
  const i1 = Math.min(cols - 1, Math.floor((cx + RMAX - ox) / h));
  const j0 = Math.max(0, Math.floor((cy - RMAX - oy) / h));
  const j1 = Math.min(rows - 1, Math.floor((cy + RMAX - oy) / h));
  for (let j = j0; j <= j1; j++) {
    for (let i = i0; i <= i1; i++) {
      const k = j * cols + i;
      const b = m.bin[k]!;
      const c = m.cov[k]!;
      if (b === 0 && c === 0) continue;
      const x0 = ox + i * h;
      const y0 = oy + j * h;
      const dc = Math.hypot(x0 + h / 2 - cx, y0 + h / 2 - cy);
      for (let e = 1; e < nE; e++) {
        const r = R_EDGES[e]!;
        if (b && dc < r) centre[e]! += h * h;
        const a = diskRectArea(cx, cy, r, x0, y0, x0 + h, y0 + h);
        if (b) exact[e]! += a;
        cover[e]! += c * a;
      }
    }
  }
  return { centre, exact, cover };
}

function maskInside(m: Mask, x: number, y: number) {
  const i = Math.floor((x - m.ox) / m.h);
  const j = Math.floor((y - m.oy) / m.h);
  return i >= 0 && j >= 0 && i < m.cols && j < m.rows && m.bin[j * m.cols + i] === 1;
}

function sampleIn(d: Domain, rnd: () => number, accept: (x: number, y: number) => boolean) {
  for (;;) {
    const x = d.bbox[0] + rnd() * (d.bbox[2] - d.bbox[0]);
    const y = d.bbox[1] + rnd() * (d.bbox[3] - d.bbox[1]);
    if (d.inside(x, y) && accept(x, y)) return [x, y] as const;
  }
}

const pct = (v: number) => `${(100 * v).toFixed(3)}%`;

// ---- Experiment 1: per-anchor annulus-area error, boundary anchors only -------------------------
function exp1() {
  const domains = [rotatedRect(800, 800, 0), rotatedRect(800, 800, 7), rotatedRect(800, 800, 30), diskDomain(400), diskDomain(100)];
  const hs = [0.5, 1, 2, 4];
  console.log("\n## Exp 1 — relative error of A_r(x) for boundary anchors (dist < 30 µm), 1500 anchors");
  console.log("domain | h µm | bin | centre: mean / rms / max | exact-pixel: mean / rms / max | coverage: mean / rms / max");
  for (const d of domains) {
    for (const h of hs) {
      const rnd = rng(1234 + h * 10);
      const m = rasterise(d, h, rnd);
      const N = 1500;
      const nb = R_EDGES.length - 1;
      const acc = ["centre", "exact", "cover"].map(() => Array.from({ length: nb }, () => ({ s: 0, s2: 0, mx: 0 })));
      for (let n = 0; n < N; n++) {
        const [x, y] = sampleIn(d, rnd, (x, y) => d.dist(x, y) < RMAX);
        const truth = R_EDGES.map((r) => d.diskArea(x, y, r));
        const got = maskDiskAreas(m, x, y);
        [got.centre, got.exact, got.cover].forEach((arr, mi) => {
          for (let k = 0; k < nb; k++) {
            const t = truth[k + 1]! - truth[k]!;
            const g = arr[k + 1]! - arr[k]!;
            const e = (g - t) / t;
            const a = acc[mi]![k]!;
            a.s += e;
            a.s2 += e * e;
            a.mx = Math.max(a.mx, Math.abs(e));
          }
        });
      }
      for (let k = 0; k < nb; k++) {
        const f = (a: { s: number; s2: number; mx: number }) => `${pct(a.s / N)} / ${pct(Math.sqrt(a.s2 / N))} / ${pct(a.mx)}`;
        console.log(`${d.name} | ${h} | [${R_EDGES[k]},${R_EDGES[k + 1]}) | ${f(acc[0]![k]!)} | ${f(acc[1]![k]!)} | ${f(acc[2]![k]!)}`);
      }
    }
  }
}

// ---- Experiment 2: end-to-end g_AB(r) under CSR, non-rectangular domain ------------------------
// Same point set throughout; only the domain model changes. Truth is the exact-domain estimator.
// Reported: g at each bin (mean over seeds), and the ratio to the exact-domain g.
function exp2() {
  console.log("\n## Exp 2 — CSR g_AB per bin; ratio to exact-domain estimator (mean over 10 seeds)");
  const cases: [Domain, number][] = [
    [rotatedRect(805, 805, 30), 1], // ≈ the 648,000 µm² covid ROI, tilted
    [rotatedRect(805, 805, 30), 2],
    [rotatedRect(805, 805, 30), 4],
    [diskDomain(250), 1], // a TMA-core-sized circle
    [diskDomain(250), 4],
  ];
  const rho = 0.002; // per type per µm²
  for (const [d, h] of cases) {
    const nb = R_EDGES.length - 1;
    const models = [
      "exact W",
      "mask exact-pixel",
      "mask centre-count",
      "bbox clip, bbox area",
      "bbox clip, true area",
      "no correction",
    ] as const;
    const sum = models.map(() => new Float64Array(nb));
    const SEEDS = 10;
    for (let s = 0; s < SEEDS; s++) {
      const rnd = rng(99 + s);
      const m = rasterise(d, h, rnd);
      const n = Math.round(rho * d.area);
      const A = Array.from({ length: n }, () => sampleIn(d, rnd, () => true));
      const B = Array.from({ length: n }, () => sampleIn(d, rnd, () => true));
      // Membership: exact-domain models keep everything; mask models keep what the mask says.
      const inM = (p: readonly [number, number]) => maskInside(m, p[0], p[1]);
      const Am = A.filter(inM);
      const Bm = B.filter(inM);
      let maskArea = 0;
      for (let k = 0; k < m.bin.length; k++) maskArea += m.bin[k]! * h * h;
      const [bx0, by0, bx1, by1] = d.bbox;
      const bboxArea = (bx1 - bx0) * (by1 - by0);
      const run = (
        As: (readonly [number, number])[],
        Bs: (readonly [number, number])[],
        area: number,
        areas: (x: number, y: number) => Float64Array,
      ) => {
        const w = new Float64Array(nb);
        for (const [ax, ay] of As) {
          const ar = areas(ax, ay);
          for (const [bx, by] of Bs) {
            const dd = Math.hypot(bx - ax, by - ay);
            if (dd >= RMAX) continue;
            const k = Math.min(nb - 1, Math.floor(dd / 10));
            const ak = ar[k + 1]! - ar[k]!;
            if (ak > 0) w[k]! += 1 / ak;
          }
        }
        const denom = As.length * (Bs.length / area);
        return w.map((v) => v / denom);
      };
      // Interior fast path: >h from ∂W the mask agrees with W, so a disk >RMAX+2h inside is uncut.
      const full = Float64Array.from(R_EDGES.map((r) => Math.PI * r * r));
      const exactAreas = (x: number, y: number) => Float64Array.from(R_EDGES.map((r) => d.diskArea(x, y, r)));
      const results = [
        run(A, B, d.area, exactAreas),
        run(Am, Bm, maskArea, (x, y) => (d.dist(x, y) > RMAX + 2 * h ? full : maskDiskAreas(m, x, y).exact)),
        run(Am, Bm, maskArea, (x, y) => (d.dist(x, y) > RMAX + 2 * h ? full : maskDiskAreas(m, x, y).centre)),
        run(A, B, bboxArea, (x, y) => Float64Array.from(R_EDGES.map((r) => diskRectArea(x, y, r, bx0, by0, bx1, by1)))),
        run(A, B, d.area, (x, y) => Float64Array.from(R_EDGES.map((r) => diskRectArea(x, y, r, bx0, by0, bx1, by1)))),
        run(A, B, d.area, () => Float64Array.from(R_EDGES.map((r) => Math.PI * r * r))),
      ];
      results.forEach((g, mi) => {
        for (let k = 0; k < nb; k++) sum[mi]![k]! += g[k]! / results[0]![k]!;
      });
    }
    console.log(`\n${d.name}, mask h=${h} µm, ρ=${rho}/µm²/type`);
    models.forEach((name, mi) => {
      const row = Array.from(sum[mi]!, (v) => (v / SEEDS).toFixed(4)).join("  ");
      console.log(`  ${name.padEnd(20)} ${row}`);
    });
  }
}

// ---- Experiment 3: |W| from the mask (drives ρ_B) -----------------------------------------------
function exp3() {
  console.log("\n## Exp 3 — mask area vs true area (pixel-centre rule), 50 random grid phases");
  for (const d of [rotatedRect(805, 805, 30), diskDomain(250), diskDomain(60)]) {
    for (const h of [1, 4]) {
      let s = 0;
      let mx = 0;
      for (let t = 0; t < 50; t++) {
        const m = rasterise(d, h, rng(7 + t));
        let a = 0;
        for (let k = 0; k < m.bin.length; k++) a += m.bin[k]!;
        const e = (a * h * h - d.area) / d.area;
        s += e;
        mx = Math.max(mx, Math.abs(e));
      }
      console.log(`${d.name} h=${h}: mean ${pct(s / 50)}  max ${pct(mx)}`);
    }
  }
}

// ---- Experiment 4: sensitivity to a mis-drawn boundary ------------------------------------------
// Cells are CSR in the true W; the estimator is run (exactly) on W grown/shrunk by δ. Membership
// follows the assumed domain, as it would in use.
function exp4() {
  console.log("\n## Exp 4 — g ratio when the domain is drawn δ µm too big (+) or too small (−), 10 seeds");
  const rho = 0.002;
  const mk: [string, (dl: number) => Domain][] = [
    ["rect 805 @30°", (dl) => rotatedRect(805 + 2 * dl, 805 + 2 * dl, 30)],
    ["disk R=250", (dl) => diskDomain(250 + dl)],
  ];
  for (const [name, f] of mk) {
    const truthD = f(0);
    for (const dl of [-5, -2, -1, 1, 2, 5]) {
      const D = f(dl);
      const sum = new Float64Array(3);
      for (let s = 0; s < 10; s++) {
        const rnd = rng(500 + s);
        const n = Math.round(rho * truthD.area);
        const A = Array.from({ length: n }, () => sampleIn(truthD, rnd, () => true));
        const B = Array.from({ length: n }, () => sampleIn(truthD, rnd, () => true));
        const g = (dom: Domain) => {
          const As = A.filter((p) => dom.inside(p[0], p[1]));
          const Bs = B.filter((p) => dom.inside(p[0], p[1]));
          const w = new Float64Array(3);
          for (const [ax, ay] of As) {
            const ar = R_EDGES.map((r) => dom.diskArea(ax, ay, r));
            for (const [bx, by] of Bs) {
              const dd = Math.hypot(bx - ax, by - ay);
              if (dd >= RMAX) continue;
              const k = Math.min(2, Math.floor(dd / 10));
              w[k]! += 1 / (ar[k + 1]! - ar[k]!);
            }
          }
          return w.map((v) => v / (As.length * (Bs.length / dom.area)));
        };
        const t = g(truthD);
        const m = g(D);
        for (let k = 0; k < 3; k++) sum[k]! += m[k]! / t[k]!;
      }
      console.log(`${name} δ=${dl > 0 ? "+" : ""}${dl}: ${Array.from(sum, (v) => (v / 10).toFixed(4)).join("  ")}`);
    }
  }
}

if (process.argv.includes("--all")) {
  exp3();
  exp1();
}
exp2();
exp4();
