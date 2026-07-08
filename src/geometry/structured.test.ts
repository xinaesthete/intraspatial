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

// Continuous / fractional counts: a `count = n + f` emits n full instances plus one *emergent*
// instance carrying fold weight `smoothstep(f)` as extra uniform scale, so a new member grows in
// from a point instead of popping. Integer counts must stay byte-identical to the discrete path.
const smoothstep = (f: number): number => {
  const t = f < 0 ? 0 : f > 1 ? 1 : f;
  return t * t * (3 - 2 * t);
};
const scaleOf = (m: Mat4): number => Math.hypot(m[0] ?? 0, m[1] ?? 0, m[2] ?? 0);
type Mat4 = number[];

describe("fractional stack — the top segment folds in", () => {
  it("integer counts are unchanged (no emergent instance)", () => {
    expect(stackPlacements(2, 3, { step: 2 }, "deg").length).toBe(3);
    // and the full instances are unscaled
    for (const p of stackPlacements(2, 3, { step: 2 }, "deg")) expect(scaleOf(p)).toBeCloseTo(1, 10);
  });

  it("a fractional count emits one extra instance, folded by smoothstep(frac)", () => {
    const ps = stackPlacements(2, 2.5, { step: 2 }, "deg");
    expect(ps.length).toBe(3);
    // full instances 0,1 anchored and unscaled
    close(applyPoint(ps[0] as number[], [1, 0, 0]), [1, 0, 0]);
    close(applyPoint(ps[1] as number[], [1, 0, 0]), [1, 0, 2]);
    // emergent instance 2 sits at its true anchor (z=4) but scaled to smoothstep(0.5)=0.5
    expect(scaleOf(ps[2] as number[])).toBeCloseTo(smoothstep(0.5), 10);
    close(applyPoint(ps[2] as number[], [1, 0, 0]), [0.5, 0, 4]);
  });

  it("just above an integer the emergent instance is near-zero scale (no pop)", () => {
    const ps = stackPlacements(2, 2.001, { step: 2 }, "deg");
    expect(ps.length).toBe(3);
    expect(scaleOf(ps[2] as number[])).toBeCloseTo(smoothstep(0.001), 10);
    expect(scaleOf(ps[2] as number[])).toBeLessThan(1e-4);
  });

  it("compounding scale still applies to the emergent index", () => {
    // scale 0.5 per step: emergent index 2 base scale = 0.5**2, then folded by smoothstep(frac)
    const ps = stackPlacements(1, 2.5, { scale: 0.5 }, "deg");
    expect(scaleOf(ps[2] as number[])).toBeCloseTo(0.5 ** 2 * smoothstep(0.5), 10);
  });
});

describe("fractional branch — the whorl re-spaces as it grows an arm", () => {
  it("integer counts keep even spacing (byte-identical to discrete)", () => {
    const ps = branchPlacements(4, { angle: 0 }, "deg");
    expect(ps.length).toBe(4);
    close(applyPoint(ps[1] as number[], [1, 0, 0]), [0, 1, 0]); // exactly 90°
  });

  it("uses the continuous count as the angular denominator (arms re-space)", () => {
    const c = 2.5;
    const ps = branchPlacements(c, { angle: 0 }, "deg");
    expect(ps.length).toBe(3);
    // full arms 0,1 at phi = 2π·j/c — NOT 2π·j/2
    const phi1 = (2 * Math.PI * 1) / c;
    close(applyPoint(ps[1] as number[], [1, 0, 0]), [Math.cos(phi1), Math.sin(phi1), 0]);
  });

  it("the emergent arm grows from a point at its re-spaced slot", () => {
    const c = 2.5;
    const ps = branchPlacements(c, { angle: 0 }, "deg");
    const w = smoothstep(0.5);
    const phi2 = (2 * Math.PI * 2) / c;
    expect(scaleOf(ps[2] as number[])).toBeCloseTo(w, 10);
    close(applyPoint(ps[2] as number[], [1, 0, 0]), [w * Math.cos(phi2), w * Math.sin(phi2), 0]);
  });

  it("just above an integer, the new arm emerges coincident with arm 0 (no jump)", () => {
    const ps = branchPlacements(3.0005, { angle: 0 }, "deg");
    expect(ps.length).toBe(4);
    const phi3 = (2 * Math.PI * 3) / 3.0005; // ≈ 2π ⇒ near arm 0's direction
    close([Math.cos(phi3), Math.sin(phi3), 0], [1, 0, 0], 2); // within tolerance of +x
    expect(scaleOf(ps[3] as number[])).toBeLessThan(1e-3);
  });
});

describe("Swept.stack / Swept.branch accept fractional counts", () => {
  it("a fractional stack has ceil(count) instances", () => {
    expect(horn({ radius: 0.5, length: 2 }).stack(2.4).count).toBe(3);
    expect(horn({ radius: 0.5, length: 2 }).stack(2.0).count).toBe(2);
  });

  it("counts below 1 clamp to the single leaf", () => {
    expect(horn().stack(0.3).count).toBe(1);
    expect(horn().branch(0.3).count).toBe(1);
  });
});
