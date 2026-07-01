import { describe, expect, it } from "vitest";
import { TrailBuffer } from "./trails";

describe("TrailBuffer", () => {
  it("fills up to capacity and reports frames", () => {
    const t = new TrailBuffer(2, 4);
    expect(t.frames).toBe(0);
    for (let f = 0; f < 6; f++) t.push(new Float32Array([f, 0, 0, -f, 0, 0]));
    expect(t.frames).toBe(4); // capped at capacity
    expect(t.segmentVertexCount()).toBe(2 * (4 - 1) * 2); // n·(frames-1)·2
  });

  it("emits segments along each agent's path, oldest→newest", () => {
    const t = new TrailBuffer(1, 4);
    // agent walks +x: 0,1,2
    t.push(new Float32Array([0, 0, 0]));
    t.push(new Float32Array([1, 0, 0]));
    t.push(new Float32Array([2, 0, 0]));
    const pos = new Float32Array(t.n * (t.frames - 1) * 2 * 3);
    const col = new Float32Array(pos.length);
    const verts = t.fillSegments(pos, col, [1, 1, 1]);
    expect(verts).toBe(2 * 2); // 2 segments, 2 verts each
    // first segment 0→1, second 1→2 (x coords)
    expect([pos[0], pos[3], pos[6], pos[9]]).toEqual([0, 1, 1, 2]);
    // colour fades: tail vertex dimmer than head vertex
    expect(col[0]!).toBeLessThan(col[col.length - 3]!);
  });

  it("returns 0 vertices until it has two frames", () => {
    const t = new TrailBuffer(3, 8);
    expect(t.fillSegments(new Float32Array(0), new Float32Array(0), [1, 1, 1])).toBe(0);
    t.push(new Float32Array(9));
    expect(t.segmentVertexCount()).toBe(0);
  });

  it("reset clears history", () => {
    const t = new TrailBuffer(1, 4);
    t.push(new Float32Array([1, 2, 3]));
    t.push(new Float32Array([4, 5, 6]));
    t.reset();
    expect(t.frames).toBe(0);
    expect(t.segmentVertexCount()).toBe(0);
  });
});
