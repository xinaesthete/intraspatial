import { describe, it, expect } from "vitest";
import { Graph, pull } from "./index";
import type { PersistenceResult } from "../../spatial/persistence";

// A clean ring of points — one loop, no clusters. The membership-sweep filtration
// (adaptive fuzzy adjacency -> 1-μ distance -> Vietoris–Rips) should recover exactly
// one persistent H1 feature (the loop).
function ring(n: number, r = 3): { xs: number[]; ys: number[] } {
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    xs.push(Math.cos(a) * r);
    ys.push(Math.sin(a) * r);
  }
  return { xs, ys };
}

function persistences(result: PersistenceResult, dim: number): number[] {
  return result.pairs
    .filter((p) => p.dim === dim && Number.isFinite(p.death))
    .map((p) => p.death - p.birth)
    .sort((a, b) => b - a);
}

describe("fuzzy TDA membership-sweep (graph pipeline)", () => {
  it("recovers one loop from a ring (adaptive bandwidth)", async () => {
    const { xs, ys } = ring(16);
    const g = new Graph();
    const pts = g.points(xs, ys);
    const rho = g.op1("kthNeighborDistance", { points: pts }, { k: 4 });
    const mu = g.op1("fuzzyAdjacencyAdaptive", { points: pts, rho }, { scale: 1 });
    const dist = g.op1("membershipToDistance", { membership: mu }, { mode: "oneMinusMu" });
    const diag = g.op1("vietorisRipsPersistence", { distance: dist }, { maxScale: 0 });

    const result = (await pull(g, diag)).payload as PersistenceResult;
    const h1 = persistences(result, 1);
    // eslint-disable-next-line no-console
    console.log("H1 persistences:", h1.map((x) => x.toFixed(3)).join(", "));

    const significant = h1.filter((p) => p > 0.2);
    expect(significant.length).toBe(1); // exactly one robust loop
  });
});
