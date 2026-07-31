import { describe, expect, it } from "vitest";
import { contactNetwork } from "./contactNetwork";
import { mulberry32 } from "./kernelAnalysis";
import type { LabelledCells } from "./pcf";

// The derived columns are pinned against their definitions, and the two primitives against brute
// force. `edges` vs `contacts` is the pair most easily confused — a cell with six B neighbours adds
// six to one and one to the other — so a hand-built layout checks exactly that.

function brute(cells: LabelledCells, radius: number, K: number) {
  const n = cells.xs.length;
  const edges = new Float64Array(K * K);
  const contacts = new Float64Array(K * K);
  for (let i = 0; i < n; i++) {
    const seen = new Set<number>();
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const dx = cells.xs[j]! - cells.xs[i]!;
      const dy = cells.ys[j]! - cells.ys[i]!;
      if (dx * dx + dy * dy >= radius * radius) continue;
      seen.add(cells.typeId[j]!);
      if (j > i) {
        const a = cells.typeId[i]!;
        const b = cells.typeId[j]!;
        edges[a * K + b]! += 1;
        if (a !== b) edges[b * K + a]! += 1;
      }
    }
    for (const b of seen) contacts[cells.typeId[i]! * K + b]! += 1;
  }
  return { edges, contacts };
}

describe("contactNetwork", () => {
  it("separates edge count from contact count", () => {
    // One A at the origin with four B neighbours inside the radius, and a fifth B out of range.
    const cells: LabelledCells = {
      xs: [0, 1, -1, 0, 0, 50],
      ys: [0, 0, 0, 1, -1, 50],
      typeId: [0, 1, 1, 1, 1, 1],
    };
    const r = contactNetwork(cells, { radius: 2, nTypes: 2 });
    expect(r.edges[0 * 2 + 1]).toBe(4); // four A–B edges
    expect(r.contacts[0 * 2 + 1]).toBe(1); // but only one A cell has any B neighbour
    expect(r.meanDegree[0 * 2 + 1]).toBe(4); // 4 edges / 1 A cell
    expect(r.pctContacts[0 * 2 + 1]).toBe(100); // the single A cell is in contact
    // Seen from B: four of the five B cells touch an A, one edge each.
    expect(r.edges[1 * 2 + 0]).toBe(4);
    expect(r.contacts[1 * 2 + 0]).toBe(4);
    expect(r.pctContacts[1 * 2 + 0]).toBeCloseTo(80, 9);
  });

  it("counts each undirected edge once, including within a type", () => {
    const cells: LabelledCells = { xs: [0, 1, 2], ys: [0, 0, 0], typeId: [0, 0, 0] };
    const r = contactNetwork(cells, { radius: 1.5, nTypes: 1 });
    expect(r.edges[0]).toBe(2); // (0,1) and (1,2), not four
    expect(r.totalEdges).toBe(2);
    expect(r.graphMeanDegree).toBeCloseTo((2 * 2) / 3, 9);
  });

  it("agrees with brute force on a random cloud", () => {
    const rnd = mulberry32(17);
    const xs: number[] = [];
    const ys: number[] = [];
    const typeId: number[] = [];
    for (let i = 0; i < 500; i++) {
      xs.push(rnd() * 100);
      ys.push(rnd() * 100);
      typeId.push(Math.floor(rnd() * 4));
    }
    const cells = { xs, ys, typeId };
    const got = contactNetwork(cells, { radius: 7, nTypes: 4 });
    const want = brute(cells, 7, 4);
    expect([...got.edges]).toEqual([...want.edges]);
    expect([...got.contacts]).toEqual([...want.contacts]);
  });

  it("honours the published derived definitions", () => {
    const rnd = mulberry32(19);
    const xs: number[] = [];
    const ys: number[] = [];
    const typeId: number[] = [];
    for (let i = 0; i < 400; i++) {
      xs.push(rnd() * 100);
      ys.push(rnd() * 100);
      typeId.push(Math.floor(rnd() * 3));
    }
    const r = contactNetwork({ xs, ys, typeId }, { radius: 8, nTypes: 3 });
    for (let a = 0; a < 3; a++) {
      let rowSum = 0;
      for (let b = 0; b < 3; b++) rowSum += r.edges[a * 3 + b]!;
      for (let b = 0; b < 3; b++) {
        const at = a * 3 + b;
        expect(r.pctContacts[at]!).toBeCloseTo((100 * r.contacts[at]!) / r.counts[a]!, 9);
        expect(r.meanDegree[at]!).toBeCloseTo(r.edges[at]! / r.counts[a]!, 9);
        expect(r.networkPct[at]!).toBeCloseTo((100 * r.edges[at]!) / rowSum, 9);
      }
    }
  });

  it("keeps edges symmetric and contacts bounded by the population", () => {
    const rnd = mulberry32(23);
    const xs: number[] = [];
    const ys: number[] = [];
    const typeId: number[] = [];
    for (let i = 0; i < 600; i++) {
      xs.push(rnd() * 120);
      ys.push(rnd() * 120);
      typeId.push(Math.floor(rnd() * 5));
    }
    const r = contactNetwork({ xs, ys, typeId }, { radius: 9, nTypes: 5 });
    for (let a = 0; a < 5; a++) {
      for (let b = 0; b < 5; b++) {
        expect(r.edges[a * 5 + b]!).toBe(r.edges[b * 5 + a]!);
        expect(r.contacts[a * 5 + b]!).toBeLessThanOrEqual(r.counts[a]!);
      }
    }
  });

  it("reports NaN network share for a type with no edges, not 0%", () => {
    // Type 1 sits far away from everything: it has no contacts at all, which is not the same
    // statement as "0% of its contacts go to A".
    const cells: LabelledCells = { xs: [0, 1, 900], ys: [0, 0, 900], typeId: [0, 0, 1] };
    const r = contactNetwork(cells, { radius: 3, nTypes: 2 });
    expect(Number.isNaN(r.networkPct[1 * 2 + 0]!)).toBe(true);
    expect(r.networkPct[0 * 2 + 0]!).toBe(100);
  });

  it("rejects a type id outside the declared axis", () => {
    expect(() => contactNetwork({ xs: [0], ys: [0], typeId: [5] }, { radius: 1, nTypes: 2 })).toThrow(/outside/);
  });
});
