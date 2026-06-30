// Source generators — nodes with no inputs that seed a graph with data. These are
// playground-side (data comes from params, not an upstream edge); they call the
// runtime's `Graph.points`/`Graph.grid` so the produced handles are ordinary graph
// resources from then on.
import { Graph } from "../../src/gpu/graph";
import type { GpuField, ShapeKind } from "../../src/gpu/graph";
import type { ParamSpec, Params } from "../../src/gpu/graph";
import { seedGrayScott } from "../../src/gpu/sim/reactionDiffusion";
import { packSwarm, seedSwarm } from "../../src/gpu/sim/danceField";

export interface SourceSpec {
  name: string;
  label: string;
  describe: string;
  outputs: { name: string; kind: ShapeKind }[];
  params: ParamSpec[];
  make(g: Graph, params: Params): Record<string, GpuField>;
}

// A real PRNG (mulberry32): a deterministic stream of good uniforms in [0,1) drawn
// sequentially. Hashing structured per-point seeds (the previous approach) gave
// biased angles and an occasional blow-up — points didn't spread radially around the
// cluster centre. A sequential stream avoids that. Seeded constant so a given set of
// params always yields the same cloud (keeps the source memoisable).
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// One pair of independent standard normals (Box–Muller) from a uniform stream, so a
// "Gaussian cluster" really is a round Gaussian, centred on the cluster.
function gauss2(rng: () => number): [number, number] {
  const u1 = 1 - rng(); // in (0,1], avoids log(0)
  const u2 = rng();
  const r = Math.sqrt(-2 * Math.log(u1));
  return [r * Math.cos(2 * Math.PI * u2), r * Math.sin(2 * Math.PI * u2)];
}

const ringPoints: SourceSpec = {
  name: "ringPoints",
  label: "Ring points",
  describe: "A noisy ring of points — one loop.",
  outputs: [{ name: "points", kind: "points" }],
  params: [
    { name: "n", type: "int", default: 24, min: 4, max: 200 },
    { name: "radius", type: "number", default: 3, min: 0.5, max: 20, step: 0.5 },
    { name: "jitter", type: "number", default: 0.15, min: 0, max: 2, step: 0.05 },
  ],
  make(g, params) {
    const n = params.n as number, r = params.radius as number, j = params.jitter as number;
    const rng = mulberry32(0x9e3779b9);
    const xs: number[] = [], ys: number[] = [];
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      xs.push(Math.cos(a) * r + (rng() - 0.5) * 2 * j);
      ys.push(Math.sin(a) * r + (rng() - 0.5) * 2 * j);
    }
    return { points: g.points(xs, ys) };
  },
};

const blobPoints: SourceSpec = {
  name: "blobPoints",
  label: "Blob clusters",
  describe: "A few Gaussian clusters of points.",
  outputs: [{ name: "points", kind: "points" }],
  params: [
    { name: "perCluster", type: "int", default: 24, min: 4, max: 200 },
    { name: "clusters", type: "int", default: 3, min: 1, max: 8 },
    { name: "spread", type: "number", default: 1.2, min: 0.1, max: 6, step: 0.1, describe: "cluster std dev (world units)" },
  ],
  make(g, params) {
    const per = params.perCluster as number, k = params.clusters as number, s = params.spread as number;
    const rng = mulberry32(0x9e3779b9);
    const xs: number[] = [], ys: number[] = [];
    for (let c = 0; c < k; c++) {
      const cx = Math.cos((c / k) * Math.PI * 2) * 6 + 8;
      const cy = Math.sin((c / k) * Math.PI * 2) * 6 + 8;
      for (let i = 0; i < per; i++) {
        const [gx, gy] = gauss2(rng);
        xs.push(cx + gx * s); // s is the cluster's std dev in world units
        ys.push(cy + gy * s);
      }
    }
    return { points: g.points(xs, ys) };
  },
};

const grayScottSeed: SourceSpec = {
  name: "grayScottSeed",
  label: "Gray–Scott seed",
  describe: "Seed grids (U, V) for reaction–diffusion.",
  outputs: [
    { name: "u", kind: "grid" },
    { name: "v", kind: "grid" },
  ],
  params: [{ name: "size", type: "int", default: 96, min: 16, max: 256 }],
  make(g, params) {
    const size = params.size as number;
    const seed = seedGrayScott(size, size, 0.05);
    return { u: g.grid(seed.u, size, size), v: g.grid(seed.v, size, size) };
  },
};

const noiseGrid: SourceSpec = {
  name: "noiseGrid",
  label: "Noise field",
  describe: "A noise grid — white (per-pixel) or value (smooth lattice) noise.",
  outputs: [{ name: "out", kind: "grid" }],
  params: [
    { name: "width", type: "int", default: 96, min: 8, max: 256 },
    { name: "height", type: "int", default: 96, min: 8, max: 256 },
    { name: "kind", type: "enum", default: "value", options: ["white", "value"] },
    { name: "scale", type: "number", default: 12, min: 2, max: 64, step: 1, describe: "value-noise feature size (px)" },
    { name: "seed", type: "int", default: 1, min: 1, max: 9999 },
    { name: "amp", type: "number", default: 1, min: 0, max: 8, step: 0.1 },
  ],
  make(g, params) {
    const w = params.width as number, h = params.height as number;
    const amp = params.amp as number, seed = params.seed as number;
    const data = new Float32Array(w * h);
    if (params.kind === "value") {
      const scale = Math.max(2, params.scale as number);
      const gw = Math.ceil(w / scale) + 2, gh = Math.ceil(h / scale) + 2;
      const lat = new Float32Array(gw * gh);
      const rng = mulberry32(seed);
      for (let i = 0; i < lat.length; i++) lat[i] = rng() * 2 - 1;
      const sm = (t: number) => t * t * (3 - 2 * t); // smoothstep
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const fx = x / scale, fy = y / scale;
          const x0 = Math.floor(fx), y0 = Math.floor(fy);
          const tx = sm(fx - x0), ty = sm(fy - y0);
          const v00 = lat[y0 * gw + x0]!, v10 = lat[y0 * gw + x0 + 1]!;
          const v01 = lat[(y0 + 1) * gw + x0]!, v11 = lat[(y0 + 1) * gw + x0 + 1]!;
          const a = v00 + (v10 - v00) * tx, b = v01 + (v11 - v01) * tx;
          data[y * w + x] = (a + (b - a) * ty) * amp;
        }
      }
    } else {
      const rng = mulberry32(seed);
      for (let i = 0; i < data.length; i++) data[i] = (rng() * 2 - 1) * amp;
    }
    return { out: g.grid(data, w, h) };
  },
};

const waveGrid: SourceSpec = {
  name: "waveGrid",
  label: "Wave field",
  describe: "A periodic grid — sine grating, radial rings, or checkerboard.",
  outputs: [{ name: "out", kind: "grid" }],
  params: [
    { name: "width", type: "int", default: 96, min: 8, max: 256 },
    { name: "height", type: "int", default: 96, min: 8, max: 256 },
    { name: "kind", type: "enum", default: "sine", options: ["sine", "radial", "checker"] },
    { name: "freq", type: "number", default: 6, min: 0.5, max: 40, step: 0.5, describe: "cycles across the field" },
    { name: "angle", type: "number", default: 0, min: 0, max: 180, step: 5, describe: "grating angle (deg)" },
    { name: "amp", type: "number", default: 1, min: 0, max: 8, step: 0.1 },
  ],
  make(g, params) {
    const w = params.width as number, h = params.height as number;
    const freq = params.freq as number, amp = params.amp as number;
    const ang = ((params.angle as number) * Math.PI) / 180;
    const ca = Math.cos(ang), sa = Math.sin(ang);
    const kind = params.kind as string;
    const data = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const u = x / w - 0.5, v = y / h - 0.5;
        let val: number;
        if (kind === "radial") val = Math.cos(2 * Math.PI * freq * Math.hypot(u, v));
        else if (kind === "checker") val = (Math.floor((x / w) * freq) + Math.floor((y / h) * freq)) % 2 === 0 ? 1 : -1;
        else val = Math.cos(2 * Math.PI * freq * (u * ca + v * sa)); // grating along the angle
        data[y * w + x] = val * amp;
      }
    }
    return { out: g.grid(data, w, h) };
  },
};

const grayScottSeedComplex: SourceSpec = {
  name: "grayScottSeedComplex",
  label: "Gray–Scott seed (complex)",
  describe: "Reaction–diffusion seed as a single complex field (re = U, im = V) — feeds Reaction–diffusion (complex).",
  outputs: [{ name: "state", kind: "grid" }],
  params: [{ name: "size", type: "int", default: 96, min: 16, max: 256 }],
  make(g, params) {
    const size = params.size as number;
    const seed = seedGrayScott(size, size, 0.05);
    const data = new Float32Array(size * size * 2); // interleaved [U, V] = complex (re, im)
    for (let i = 0; i < size * size; i++) {
      data[i * 2] = seed.u[i]!;
      data[i * 2 + 1] = seed.v[i]!;
    }
    return {
      state: g.source({ shape: { kind: "grid", width: size, height: size }, dtype: "f32", element: { kind: "complex" }, data }),
    };
  },
};

// A swarm seed for the dance field: N agents on a jittered spherical shell, carried as
// one vec3 `points` field of 2N rows — rows [0,N) positions, [N,2N) velocities (zero at
// birth) — the packing danceField reads (see src/gpu/sim/danceField.ts). One feedback
// node then carries the whole pos+vel state around the loop.
const danceSwarmSeed: SourceSpec = {
  name: "danceSwarmSeed",
  label: "Dance swarm seed",
  describe: "A cloud of agents (pos ‖ vel) to seed the dance field.",
  outputs: [{ name: "state", kind: "points" }],
  params: [
    { name: "n", type: "int", default: 220, min: 8, max: 800, describe: "agent count" },
    { name: "seed", type: "int", default: 1, min: 1, max: 9999 },
  ],
  make(g, params) {
    const n = params.n as number, seed = params.seed as number;
    const swarm = seedSwarm(n, seed);
    return {
      state: g.source({
        shape: { kind: "points", n: n * 2 },
        dtype: "f32",
        element: { kind: "vec", n: 3 },
        data: packSwarm(swarm),
      }, "danceSwarmSeed"),
    };
  },
};

export const SOURCES: SourceSpec[] = [ringPoints, blobPoints, grayScottSeed, grayScottSeedComplex, noiseGrid, waveGrid, danceSwarmSeed];

const byName = new Map(SOURCES.map((s) => [s.name, s]));
export function getSource(name: string): SourceSpec | undefined {
  return byName.get(name);
}
export function isSource(name: string): boolean {
  return byName.has(name);
}
