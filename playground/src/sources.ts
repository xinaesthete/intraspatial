// Source generators — nodes with no inputs that seed a graph with data. These are
// playground-side (data comes from params, not an upstream edge); they call the
// runtime's `Graph.points`/`Graph.grid` so the produced handles are ordinary graph
// resources from then on.
import { Graph } from "../../src/gpu/graph";
import type { GpuField, ShapeKind } from "../../src/gpu/graph";
import type { ParamSpec, Params } from "../../src/gpu/graph";
import { seedGrayScott } from "../../src/gpu/sim/reactionDiffusion";

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

export const SOURCES: SourceSpec[] = [ringPoints, blobPoints, grayScottSeed];

const byName = new Map(SOURCES.map((s) => [s.name, s]));
export function getSource(name: string): SourceSpec | undefined {
  return byName.get(name);
}
export function isSource(name: string): boolean {
  return byName.has(name);
}
