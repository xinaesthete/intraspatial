import { describe, expect, it } from "vitest";
import { crossPCF, crossPCFBinMembership, crossPCFMatrix, type LabelledCells } from "./pcf";
import type { CellCloud } from "./tcm";

describe("crossPCF (eq 8, Mode 1)", () => {
  it("bins pairs into the correct annulus with the correct normalisation", () => {
    // One A at the centre of a large ROI (so no edge effect for r ≤ rMax), and 8 B on a ring at
    // radius 11 — every pair must land in bin 5 = [10,12) with an exactly computable g.
    const A: CellCloud = { xs: [100], ys: [100] };
    const bx: number[] = [];
    const by: number[] = [];
    for (let k = 0; k < 8; k++) {
      const th = (k / 8) * 2 * Math.PI;
      bx.push(100 + 11 * Math.cos(th));
      by.push(100 + 11 * Math.sin(th));
    }
    const B: CellCloud = { xs: bx, ys: by };
    const res = crossPCF(A, B, { bbox: [0, 0, 200, 200], rMax: 20, nBins: 10 }); // dr = 2

    expect(res.counts[5]).toBe(8); // all 8 pairs in [10,12)
    expect(res.counts.reduce((s, c, i) => (i === 5 ? s : s + c), 0)).toBe(0); // and nowhere else

    const rhoB = 8 / (200 * 200);
    const annulus = Math.PI * (12 * 12 - 10 * 10);
    expect(res.g[5]!).toBeCloseTo(8 / (1 * rhoB * annulus), 6);
    expect(res.g[0]!).toBe(0);
  });

  it("CSR: g(r) ≈ 1 for uniform A and B (rMax ≪ ROI so edge bias is small)", () => {
    // mulberry32 — a good-quality PRNG; a simple LCG's consecutive values lie on 2D lattice planes,
    // which fabricates spatial structure and breaks the CSR premise.
    let a = 0x9e3779b9;
    const rnd = () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const uniform = (n: number): CellCloud => {
      const xs: number[] = [];
      const ys: number[] = [];
      for (let i = 0; i < n; i++) {
        xs.push(rnd() * 500);
        ys.push(rnd() * 500);
      }
      return { xs, ys };
    };
    const res = crossPCF(uniform(2500), uniform(5000), { bbox: [0, 0, 500, 500], rMax: 15, nBins: 8 });
    const mean = res.g.reduce((a, b) => a + b, 0) / res.g.length;
    expect(mean).toBeGreaterThan(0.85); // slight downward drift is the deferred edge correction
    expect(mean).toBeLessThan(1.15);
  });

  it("clustering: g(small r) ≫ 1 when B tightly surrounds A", () => {
    // A and B are the same tight blob ⇒ strong short-range co-location.
    let s = 99;
    const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < 400; i++) {
      xs.push(250 + (rnd() - 0.5) * 10);
      ys.push(250 + (rnd() - 0.5) * 10);
    }
    const cloud: CellCloud = { xs, ys };
    const res = crossPCF(cloud, cloud, { bbox: [0, 0, 500, 500], rMax: 20, nBins: 10 });
    expect(res.g[0]!).toBeGreaterThan(3); // strongly clustered at the shortest range
  });
});

describe("crossPCFMatrix (N-way, all ordered pairs)", () => {
  it("off-diagonal entries match a single-bin crossPCF for the same pair", () => {
    let a = 12345;
    const rnd = () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    // Three types over a 300² ROI.
    const xs: number[] = [];
    const ys: number[] = [];
    const typeId: number[] = [];
    const clouds: Record<number, CellCloud> = { 1: { xs: [], ys: [] }, 2: { xs: [], ys: [] }, 3: { xs: [], ys: [] } };
    for (const [id, n, cx, cy] of [
      [1, 300, 120, 150],
      [2, 200, 180, 150],
      [3, 150, 150, 220],
    ] as const) {
      for (let i = 0; i < n; i++) {
        const x = cx + (rnd() - 0.5) * 120;
        const y = cy + (rnd() - 0.5) * 120;
        xs.push(x);
        ys.push(y);
        typeId.push(id);
        (clouds[id]!.xs as number[]).push(x);
        (clouds[id]!.ys as number[]).push(y);
      }
    }
    const cells: LabelledCells = { xs, ys, typeId };
    const bbox: [number, number, number, number] = [0, 0, 300, 300];
    const radius = 25;
    const m = crossPCFMatrix(cells, { bbox, radius });
    const N = m.types.length;
    expect(m.types).toEqual([1, 2, 3]);

    // g_{1→2} from the matrix equals a one-bin crossPCF(type1, type2).
    const single = crossPCF(clouds[1]!, clouds[2]!, { bbox, rMax: radius, nBins: 1 });
    const iA = m.types.indexOf(1);
    const iB = m.types.indexOf(2);
    expect(m.g[iA * N + iB]!).toBeCloseTo(single.g[0]!, 9);
  });
});

describe("crossPCFBinMembership", () => {
  const P = { bbox: [0, 0, 200, 200], rMax: 20, nBins: 10 } as const; // dr = 2

  it("marks exactly the bin a pair falls in", () => {
    // One A at the centre, one B at distance 11 → bin 5 = [10, 12) and nothing else, on both sides.
    const A: CellCloud = { xs: [100], ys: [100] };
    const B: CellCloud = { xs: [111], ys: [100] };
    const m = crossPCFBinMembership(A, B, P);
    expect(m.maskA[0]).toBe(1 << 5);
    expect(m.maskB[0]).toBe(1 << 5);
    expect([...m.countA]).toEqual([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
    expect([...m.countB]).toEqual([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
  });

  it("agrees with crossPCF about which bins are occupied — the property the highlight rests on", () => {
    // THE test. The highlight's whole claim is "these are the cells that produced that value", so a
    // bin the curve counted must never come up empty here, and vice versa. Random clouds, so the
    // agreement is not an artefact of a contrived arrangement.
    let seed = 12345;
    const rnd = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    const cloud = (n: number): CellCloud => {
      const xs: number[] = [];
      const ys: number[] = [];
      for (let i = 0; i < n; i++) {
        xs.push(rnd() * 200);
        ys.push(rnd() * 200);
      }
      return { xs, ys };
    };
    const A = cloud(250);
    const B = cloud(300);
    const res = crossPCF(A, B, P);
    const m = crossPCFBinMembership(A, B, P);
    for (let k = 0; k < P.nBins; k++) {
      expect(m.countA[k]! > 0).toBe(res.counts[k]! > 0);
      // A cell cannot be in more pairs than there are pairs, nor fewer than one per occupied bin.
      expect(m.countA[k]!).toBeLessThanOrEqual(res.counts[k]!);
      expect(m.countB[k]!).toBeLessThanOrEqual(res.counts[k]!);
    }
    // Every A cell with any neighbour at all must carry a bit, and no bit may exceed the bin count.
    for (let i = 0; i < A.xs.length; i++) expect(m.maskA[i]! >>> P.nBins).toBe(0);
  });

  it("sets one bit per distinct bin, not one per pair", () => {
    // Three B cells all at radius 11 from the same A: three PAIRS, but one BIN, so one bit — and the
    // A cell must be counted once, not three times. Counting pairs here would make a dense
    // neighbourhood look like many highlighted cells.
    const A: CellCloud = { xs: [100], ys: [100] };
    const B: CellCloud = { xs: [111, 100, 89], ys: [100, 111, 100] };
    const m = crossPCFBinMembership(A, B, P);
    expect(m.maskA[0]).toBe(1 << 5);
    expect(m.countA[5]).toBe(1);
    expect(m.countB[5]).toBe(3); // all three B cells DO participate
  });

  it("excludes pairs at or beyond rMax, exactly as crossPCF does", () => {
    const A: CellCloud = { xs: [100], ys: [100] };
    const B: CellCloud = { xs: [120, 119.5], ys: [100, 100] }; // 20 is out (>= rMax), 19.5 is in
    const m = crossPCFBinMembership(A, B, P);
    expect(m.maskB[0]).toBe(0);
    expect(m.maskB[1]).toBe(1 << 9);
    expect(m.maskA[0]).toBe(1 << 9);
  });

  it("refuses more than 32 bins rather than wrapping the mask", () => {
    // A silently shifted-off bit would light the wrong cells while looking entirely plausible.
    expect(() => crossPCFBinMembership({ xs: [1], ys: [1] }, { xs: [2], ys: [2] }, { ...P, nBins: 33 })).toThrow(/32/);
  });

  it("survives an empty cloud", () => {
    const m = crossPCFBinMembership({ xs: [], ys: [] }, { xs: [1], ys: [1] }, P);
    expect(m.maskA.length).toBe(0);
    expect([...m.countB]).toEqual(new Array(10).fill(0));
  });
});

describe("crossPCFBinMembership — the per-cell invariant the highlight rests on", () => {
  // The bin-level agreement test above says the right BINS are occupied. It does not say the right
  // CELLS are lit, and that is the claim the panel actually makes: "these are the cells that
  // produced g(r) there". A mask built from the wrong cloud, or shifted by one type, would still
  // occupy the right bins. So this checks every cell against a brute-force O(N²) pass.
  const P = { bbox: [0, 0, 300, 300], rMax: 60, nBins: 12 } as const; // dr = 5

  function cloud(n: number, seed: number) {
    let s = seed;
    const rnd = () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x7fffffff;
    };
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < n; i++) {
      xs.push(rnd() * 300);
      ys.push(rnd() * 300);
    }
    return { xs, ys };
  }

  it("lights a cell in bin k exactly when it has a partner at that separation", () => {
    const a = cloud(400, 17);
    const b = cloud(350, 91);
    const m = crossPCFBinMembership(a, b, P);
    const dr = P.rMax / P.nBins;

    // Brute force, both directions.
    const wantA = new Uint32Array(a.xs.length);
    const wantB = new Uint32Array(b.xs.length);
    for (let i = 0; i < a.xs.length; i++) {
      for (let j = 0; j < b.xs.length; j++) {
        const d = Math.hypot(a.xs[i]! - b.xs[j]!, a.ys[i]! - b.ys[j]!);
        if (d >= P.rMax) continue;
        const bit = 1 << Math.min(P.nBins - 1, Math.floor(d / dr));
        wantA[i]! |= bit;
        wantB[j]! |= bit;
      }
    }
    for (let i = 0; i < a.xs.length; i++) expect(m.maskA[i], `A cell ${i}`).toBe(wantA[i]);
    for (let j = 0; j < b.xs.length; j++) expect(m.maskB[j], `B cell ${j}`).toBe(wantB[j]);
  });

  it("holds when the clouds are far apart in index order — the bucket grid is not index-sensitive", () => {
    // Two well-separated blobs, so most candidate pairs are rejected by the grid rather than the
    // distance test. If the 3×3 neighbourhood were ever too small, this is where it shows.
    const a = { xs: [10, 12, 14, 280, 282], ys: [10, 12, 14, 280, 282] };
    const b = { xs: [40, 250], ys: [40, 250] };
    const m = crossPCFBinMembership(a, b, { ...P, rMax: 60, nBins: 12 });
    // A[0] at (10,10) is 42.4 from B[0] at (40,40) → bin 8 = [40,45)
    expect(m.maskA[0]).toBe(1 << 8);
    // A[3] at (280,280) is 42.4 from B[1] at (250,250) → the same bin
    expect(m.maskA[3]).toBe(1 << 8);
    expect(m.maskB[0]! & (1 << 8)).toBeTruthy();
  });
});
