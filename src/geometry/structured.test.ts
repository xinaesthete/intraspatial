import { describe, expect, it } from "vitest";
import { deg } from "./angle";
import { applyNormal, applyPoint, compose, IDENTITY, mul, rotZ, scaleUniform, translate } from "./placement";
import { branchPlacements, Structured, stackPlacements } from "./structured";
import type { Vec3 } from "./superellipsoid";
import { horn } from "./swept";

const close = (a: Vec3, b: Vec3, p = 10) => {
  expect(a[0]).toBeCloseTo(b[0], p);
  expect(a[1]).toBeCloseTo(b[1], p);
  expect(a[2]).toBeCloseTo(b[2], p);
};

describe("placement algebra", () => {
  it("translate/scale/rotate apply to points", () => {
    close(applyPoint(translate(1, 2, 3), [0, 0, 0]), [1, 2, 3]);
    close(applyPoint(scaleUniform(2), [1, 1, 1]), [2, 2, 2]);
    close(applyPoint(rotZ(Math.PI / 2), [1, 0, 0]), [0, 1, 0]); // +x → +y
  });

  it("composes as apply-right-first (A·B applies B then A)", () => {
    // rotate then translate: point (1,0,0) → rotate to (0,1,0) → translate +x → (1,1,0)
    const m = compose([translate(1, 0, 0), rotZ(Math.PI / 2)]);
    close(applyPoint(m, [1, 0, 0]), [1, 1, 0]);
  });

  it("transforms normals by rotation only (uniform scale cancels)", () => {
    const m = mul(scaleUniform(5), rotZ(Math.PI / 2));
    close(applyNormal(m, [1, 0, 0]), [0, 1, 0]); // still unit, just rotated
  });
});

describe("stack placements", () => {
  it("lifts each copy up the axis by step", () => {
    const ps = stackPlacements(2, 3, { step: 2 }, "deg");
    expect(ps.length).toBe(3);
    close(applyPoint(ps[0] as number[], [0, 0, 0]), [0, 0, 0]);
    close(applyPoint(ps[1] as number[], [0, 0, 0]), [0, 0, 2]);
    close(applyPoint(ps[2] as number[], [0, 0, 0]), [0, 0, 4]);
  });

  it("defaults the step to the base length (contiguous tower)", () => {
    const ps = stackPlacements(3, 2, undefined, "deg");
    close(applyPoint(ps[1] as number[], [0, 0, 0]), [0, 0, 3]);
  });

  it("twists each successive copy", () => {
    const ps = stackPlacements(1, 2, { twist: 90 }, "deg"); // 90° per step
    close(applyPoint(ps[1] as number[], [1, 0, 0]), [0, 1, 1]); // rotated 90° about z, lifted by step=1
  });
});

describe("branch placements", () => {
  it("splays count copies evenly around the axis", () => {
    const ps = branchPlacements(4, { angle: 0 }, "deg"); // no tilt → pure rotation about z
    expect(ps.length).toBe(4);
    close(applyPoint(ps[0] as number[], [1, 0, 0]), [1, 0, 0]);
    close(applyPoint(ps[1] as number[], [1, 0, 0]), [0, 1, 0]); // 90°
    close(applyPoint(ps[2] as number[], [1, 0, 0]), [-1, 0, 0]); // 180°
  });

  it("tilts each branch out from the axis", () => {
    const ps = branchPlacements(1, { angle: deg(90) }, "deg"); // one branch, tilted 90° about x
    close(applyPoint(ps[0] as number[], [0, 0, 1]), [0, -1, 0]); // +z tips to −y
  });
});

describe("Swept.stack / Swept.branch build a Structured", () => {
  it("stack yields a Structured with count instances", () => {
    const g = horn({ radius: 0.5, length: 2 }).stack(4, { step: 2 });
    expect(g).toBeInstanceOf(Structured);
    expect(g.count).toBe(4);
  });

  it("chaining structural ops multiplies instances", () => {
    const g = horn({ radius: 0.5, length: 2 })
      .stack(5)
      .branch(3, { angle: deg(30) });
    expect(g.count).toBe(15);
  });

  it("tessellates into count × base vertices, indices offset per instance", () => {
    const base = horn({ radius: 0.5, length: 2 });
    const bm = base.tessellate({ slices: 4, stacks: 3 });
    const g = base.stack(3, { step: 2 });
    const m = g.tessellate({ slices: 4, stacks: 3 });
    expect(m.vertexCount).toBe(3 * bm.vertexCount);
    expect(m.indices.length).toBe(3 * bm.indices.length);
    expect(Math.max(...m.indices)).toBe(3 * bm.vertexCount - 1);
    for (const i of m.indices) expect(i).toBeLessThan(m.vertexCount);
  });

  it("places instance geometry where the placement says", () => {
    // A stack of 2 with step 2: instance 1's vertices are its base counterpart lifted by z+2.
    const base = horn({ radius: 0.5, length: 2 });
    const bm = base.tessellate({ slices: 4, stacks: 3 });
    const m = base.stack(2, { step: 2 }).tessellate({ slices: 4, stacks: 3 });
    const bv = bm.vertexCount;
    // vertex 0 of instance 1 == vertex 0 of base + (0,0,2)
    close(
      [m.positions[bv * 3] ?? 0, m.positions[bv * 3 + 1] ?? 0, m.positions[bv * 3 + 2] ?? 0],
      [bm.positions[0] ?? 0, bm.positions[1] ?? 0, (bm.positions[2] ?? 0) + 2],
    );
  });

  it("instanceMatrices packs 16 floats per instance", () => {
    const g = horn().stack(3);
    const mats = g.instanceMatrices();
    expect(mats.length).toBe(3 * 16);
    // instance 0 is identity
    expect(Array.from(mats.slice(0, 16))).toEqual(IDENTITY);
  });
});
