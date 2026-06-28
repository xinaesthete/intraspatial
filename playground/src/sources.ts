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

function hash(i: number): number {
  return ((i * 2654435761) >>> 0) / 0xffffffff;
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
      xs.push(Math.cos(a) * r + (hash(i * 2) - 0.5) * j);
      ys.push(Math.sin(a) * r + (hash(i * 2 + 1) - 0.5) * j);
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
    { name: "spread", type: "number", default: 1.2, min: 0.1, max: 6, step: 0.1 },
  ],
  make(g, params) {
    const per = params.perCluster as number, k = params.clusters as number, s = params.spread as number;
    const xs: number[] = [], ys: number[] = [];
    let idx = 0;
    for (let c = 0; c < k; c++) {
      const cx = Math.cos((c / k) * Math.PI * 2) * 6 + 8;
      const cy = Math.sin((c / k) * Math.PI * 2) * 6 + 8;
      for (let i = 0; i < per; i++) {
        xs.push(cx + (hash(idx * 2) - 0.5) * s * 4);
        ys.push(cy + (hash(idx * 2 + 1) - 0.5) * s * 4);
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
