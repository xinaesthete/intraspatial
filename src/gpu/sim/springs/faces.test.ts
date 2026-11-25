import { describe, expect, it } from "vite-plus/test";
import { bondFaces } from "./faces";
import { applyLines, sheetLines } from "./lines";
import { addSpring, createTopology } from "./topology";

/** A chain of `n` particles with backbone bonds only. */
function chain(n: number) {
  const t = createTopology(n, 8);
  for (let i = 0; i + 1 < n; i++) addSpring(t, i, i + 1);
  return t;
}

describe("bondFaces", () => {
  it("finds nothing on a bare chain", () => {
    expect(bondFaces(chain(20))).toHaveLength(0);
  });

  it("fills a triangle once", () => {
    const t = createTopology(3, 4);
    addSpring(t, 0, 1);
    addSpring(t, 1, 2);
    addSpring(t, 0, 2);
    expect(Array.from(bondFaces(t))).toEqual([0, 1, 2]);
  });

  it("fills a four-bond ring with two triangles sharing the diagonal from its lowest particle", () => {
    const t = createTopology(4, 4);
    addSpring(t, 0, 1);
    addSpring(t, 1, 2);
    addSpring(t, 2, 3);
    addSpring(t, 3, 0);
    expect(Array.from(bondFaces(t))).toEqual([0, 1, 2, 0, 2, 3]);
    expect(bondFaces(t, { quads: false })).toHaveLength(0);
  });

  it("leaves a ring with a bond across it to its two triangles", () => {
    const t = createTopology(4, 4);
    addSpring(t, 0, 1);
    addSpring(t, 1, 2);
    addSpring(t, 2, 3);
    addSpring(t, 3, 0);
    addSpring(t, 0, 2);
    expect(bondFaces(t)).toHaveLength(6);
    expect(bondFaces(t, { triangles: false })).toHaveLength(0);
  });

  it("turns a drawn sheet into a ladder of quads: one fewer than the rungs between each pair of strands", () => {
    const t = chain(48);
    const lines = sheetLines(4, 44, 3, 3);
    const made = applyLines(t, lines);
    const faces = bondFaces(t);
    // Rungs per pair of strands: the pairs each line made. Quads between consecutive rungs.
    const rungs = made.pairs.length / lines.length;
    expect(lines).toHaveLength(2);
    expect(rungs).toBeGreaterThan(5);
    expect(faces.length / 6).toBe(lines.length * (rungs - 1));
    expect(Math.max(...faces)).toBeLessThan(48);
  });

  it("can be limited to some bond kinds", () => {
    const t = createTopology(3, 4);
    addSpring(t, 0, 1, 1, 1, 0, 0);
    addSpring(t, 1, 2, 1, 1, 0, 0);
    addSpring(t, 0, 2, 1, 1, 0, 3);
    expect(bondFaces(t, { kinds: 1 << 0 })).toHaveLength(0);
    expect(bondFaces(t, { kinds: (1 << 0) | (1 << 3) })).toHaveLength(3);
  });
});
