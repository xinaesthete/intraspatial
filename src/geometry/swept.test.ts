import { describe, expect, it } from "vitest";
import { deg, rad } from "./angle";
import { linear, ramp } from "./expr";
import type { Vec3 } from "./superellipsoid";
import { catalogue, gridIndices, horn } from "./swept";

const close = (a: Vec3, b: Vec3, p = 10) => {
  expect(a[0]).toBeCloseTo(b[0], p);
  expect(a[1]).toBeCloseTo(b[1], p);
  expect(a[2]).toBeCloseTo(b[2], p);
};

describe("a plain horn is a straight generalized cylinder", () => {
  const g = horn({ radius: 0.5, length: 2 });

  it("places the profile on the sweep axis", () => {
    close(g.position(0, 0), [0.5, 0, 0]);
    close(g.position(1, 0), [0.5, 0, 2]); // tip, θ=0
    close(g.position(0.5, Math.PI / 2), [0, 0.5, 1]); // quarter around, mid-sweep
  });

  it("has an outward radial normal (finite-difference framing)", () => {
    close(g.eval(0.5, 0).normal, [1, 0, 0], 3); // +x face
    close(g.eval(0.5, Math.PI / 2).normal, [0, 1, 0], 3); // +y face
    close(g.eval(0.5, Math.PI).normal, [-1, 0, 0], 3); // −x face
  });

  it("keeps the profile on the circle of the given radius (exponent 1)", () => {
    for (const theta of [0, 0.7, 2.1, 5.5]) {
      const [x, y] = g.position(0.3, theta);
      expect(Math.hypot(x, y)).toBeCloseTo(0.5, 10);
    }
  });
});

describe("profile radius as an {s}-expression", () => {
  it("grows the radius along the sweep", () => {
    const g = horn({ radius: ramp(3), length: 1 }); // radius = 3·s
    expect(Math.hypot(...xy(g.position(0, 0)))).toBeCloseTo(0, 10);
    expect(Math.hypot(...xy(g.position(0.5, 0)))).toBeCloseTo(1.5, 10);
    expect(Math.hypot(...xy(g.position(1, 0)))).toBeCloseTo(3, 10);
  });
});

describe("transform-ops build the stack in chain order", () => {
  it("scale tapers the cross-section", () => {
    const g = horn({ radius: 1, length: 2 }).scale(linear(1, 0)); // full → zero at tip
    expect(Math.hypot(...xy(g.position(0, 0)))).toBeCloseTo(1, 10);
    expect(Math.hypot(...xy(g.position(1, 0)))).toBeCloseTo(0, 10);
  });

  it("twist spins the profile about the travel axis", () => {
    // Constant 90° twist everywhere: the θ=0 profile point rotates onto +y.
    const g = horn({ radius: 1, length: 2 }).twist(90);
    close(g.position(0.5, 0), [0, 1, 1], 6);
  });

  it("twist can vary along s via a ramp", () => {
    const g = horn({ radius: 1, length: 4 }).twist(ramp(360)); // 0 → full turn
    close(g.position(0.25, 0), [0, 1, 1], 6); // quarter turn at s=0.25
    close(g.position(1, 0), [1, 0, 4], 5); // full turn — back to +x
  });

  it("bend swings the sweep axis", () => {
    // 90° bend at the tip rotates the axis from +z into −y (rotation about +x).
    const g = horn({ radius: 1, length: 2 }).bend(ramp(90));
    close(g.position(1, 0), [1, -2, 0], 5);
  });

  it("is immutable — each op returns a new geometry", () => {
    const base = horn();
    const bent = base.bend(10);
    expect(base.stack.length).toBe(0);
    expect(bent.stack.length).toBe(1);
  });
});

describe("angle units", () => {
  it("defaults to degrees (the FormGrow horn convention)", () => {
    const g = horn({ radius: 1, length: 2 }).twist(90);
    close(g.position(0.5, 0), [0, 1, 1], 6);
  });

  it("honours a radians catalogue", () => {
    const g = catalogue({ angleUnit: "rad" })
      .horn({ radius: 1, length: 2 })
      .twist(Math.PI / 2);
    close(g.position(0.5, 0), [0, 1, 1], 6);
  });

  it("a typed Angle bypasses the catalogue default", () => {
    const g = catalogue({ angleUnit: "rad" }).horn({ radius: 1, length: 2 }).twist(deg(90));
    close(g.position(0.5, 0), [0, 1, 1], 6);
    const same = horn({ radius: 1, length: 2 }).twist(rad(Math.PI / 2));
    close(same.position(0.5, 0), [0, 1, 1], 6);
  });
});

describe("breeding surface", () => {
  it("exposes the genes carried by radius and transform expressions", () => {
    const g = horn({ radius: ramp(3, { name: "radius", type: "number", default: 3 }) }).twist(
      ramp(360, { name: "twist", type: "number", default: 360 }),
    );
    expect(g.specs().map((s) => s.name)).toEqual(["radius", "twist"]);
  });
});

describe("GPU param vector (structure/value split)", () => {
  it("orders values as radius-consts, exponent, length, then transform-consts", () => {
    const g = horn({ radius: linear(0.2, 0.6), exponent: 0.8, length: 3 })
      .twist(ramp(360)) // deg default → mul(π/180, mul(360, s)) → consts [π/180, 360]
      .scale(linear(1, 0.5)); // → consts [1, -0.5]
    const pv = Array.from(g.paramVector());
    const want = [0.2, 0.4, 0.8, 3, Math.PI / 180, 360, 1, -0.5];
    expect(pv.length).toBe(want.length);
    // paramVector is a Float32Array (it feeds the GPU), so compare at f32 precision.
    want.forEach((w, i) => {
      expect(pv[i]).toBeCloseTo(w, 5);
    });
  });

  it("same structure horns emit identical WGSL but differ in paramVector", () => {
    const a = horn({ radius: 1, length: 2 }).twist(ramp(360));
    const b = horn({ radius: 1, length: 2 }).twist(ramp(90));
    expect(a.toWgsl()).toBe(b.toWgsl());
    expect(Array.from(a.paramVector())).not.toEqual(Array.from(b.paramVector()));
  });
});

describe("tessellation", () => {
  it("produces a (slices+1)×(stacks+1) vertex grid and quad indices", () => {
    const mesh = horn().tessellate({ slices: 4, stacks: 3 });
    expect(mesh.vertexCount).toBe(5 * 4);
    expect(mesh.positions.length).toBe(mesh.vertexCount * 3);
    expect(mesh.normals.length).toBe(mesh.vertexCount * 3);
    expect(mesh.indices.length).toBe(4 * 3 * 6);
    for (const i of mesh.indices) expect(i).toBeLessThan(mesh.vertexCount);
  });

  it("gridIndices reference only in-range corners", () => {
    const idx = gridIndices(3, 2);
    expect(idx.length).toBe(3 * 2 * 6);
    expect(Math.max(...idx)).toBe((3 + 1) * (2 + 1) - 1);
  });
});

function xy(p: Vec3): [number, number] {
  return [p[0], p[1]];
}
