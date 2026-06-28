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

// A well-mixed integer hash (lowbias32 avalanche) → uniform [0,1). The previous
// `(i*k)>>>0 / 2^32` is a *weak* multiplicative hash: for consecutive seeds the
// outputs differ by a constant, so using i and i+1 for a point's x/y made every
// point land on a diagonal. Avalanching decorrelates successive seeds.
function u01(i: number): number {
  let x = i >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d) >>> 0;
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return x / 4294967296;
}

// A pair of independent standard normals (Box–Muller) for one integer seed — so a
// "Gaussian cluster" really is Gaussian, not a uniform square.
function gauss2(seed: number): [number, number] {
  const u1 = Math.max(1e-9, u01(seed));
  const u2 = u01(seed ^ 0x9e3779b9);
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
    const xs: number[] = [], ys: number[] = [];
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      xs.push(Math.cos(a) * r + (u01(2 * i) - 0.5) * 2 * j);
      ys.push(Math.sin(a) * r + (u01(2 * i + 1) - 0.5) * 2 * j);
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
    const xs: number[] = [], ys: number[] = [];
    let idx = 0;
    for (let c = 0; c < k; c++) {
      const cx = Math.cos((c / k) * Math.PI * 2) * 6 + 8;
      const cy = Math.sin((c / k) * Math.PI * 2) * 6 + 8;
      for (let i = 0; i < per; i++) {
        const [gx, gy] = gauss2(idx * 2654435761 + c * 0x85ebca6b);
        xs.push(cx + gx * s); // s is the cluster's std dev in world units
        ys.push(cy + gy * s);
        idx++;
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
