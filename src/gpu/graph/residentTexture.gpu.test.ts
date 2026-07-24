import { describe, expect, it } from "vitest";
import { splatDensityGpu } from "../spatial/splatDensity";
import { nodeBackend } from "./backend.node";
import { Graph, pull, pullResident, registerBuiltinOps } from "./index";

// Texture-resident values (ADR-0017). `splatDensity` is a render pass, so its output IS a texture;
// carrying it as one means a render→render edge costs nothing, and the texture→buffer copy is paid
// only where a buffer-binding op actually consumes the value.
//
// Kept small: Dawn-on-Node segfaults its teardown past enough GPU work in one process.

const BBOX: [number, number, number, number] = [0, 0, 12, 12];
const SPLAT = { width: 24, height: 24, sigma: 1.2, radiusSigma: 4, bbox: BBOX };

function grid(): { xs: number[]; ys: number[]; packed: Float32Array } {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < 40; i++) {
    xs.push(2 + (i % 7) * 1.3);
    ys.push(2 + Math.floor(i / 7) * 1.4);
  }
  const packed = new Float32Array(2 * xs.length);
  for (let i = 0; i < xs.length; i++) {
    packed[2 * i] = xs[i]!;
    packed[2 * i + 1] = ys[i]!;
  }
  return { xs, ys, packed };
}

function splatGraph() {
  registerBuiltinOps();
  const { xs, ys, packed } = grid();
  const g = new Graph();
  const src = g.source({ shape: { kind: "points", n: xs.length }, dtype: "f32", data: packed }, "pts");
  return { g, xs, ys, density: g.op1("splatDensity", { points: src }, SPLAT) };
}

describe("texture-resident values", () => {
  it("pullResident hands back a TEXTURE, with no buffer and no download", async () => {
    const { g, density } = splatGraph();
    const bridges: string[] = [];
    const v = await pullResident(g, density, { onBridge: (k, d) => bridges.push(d) });

    expect(v.texture, "splatDensity should leave its output in a texture").toBeDefined();
    expect(v.texture?.width).toBe(24);
    expect(v.texture?.height).toBe(24);
    expect(v.buffer, "no buffer should have been leased — nothing asked for one").toBeUndefined();
    expect(v.data, "and nothing should have been downloaded").toBeUndefined();
    // The points still had to be uploaded; what must NOT appear is a download or a de-texture.
    expect(bridges).not.toContain("download");
    expect(bridges).not.toContain("detexture");

    // The caller owns the lease (ADR-0017), exactly as for a resident buffer.
    if (v.texture) nodeBackend.releaseTexture(v.texture);
  });

  it("bridges to a buffer once, and only when something binds one", async () => {
    // `pull` materialises the sink for the host, which is a buffer consumer — so exactly one
    // de-texture happens, then one download, and the numbers match the hand-written host splat.
    const { g, xs, ys, density } = splatGraph();
    const bridges: string[] = [];
    const v = await pull(g, density, { onBridge: (k, d) => bridges.push(d) });

    expect(bridges.filter((b) => b === "detexture")).toHaveLength(1);
    expect(bridges.filter((b) => b === "download")).toHaveLength(1);

    const want = (await splatDensityGpu(xs, ys, SPLAT)).data;
    const got = v.data as Float32Array;
    expect(got.length).toBe(want.length);
    let maxAbs = 0;
    let peak = 0;
    for (let i = 0; i < want.length; i++) {
      peak = Math.max(peak, Math.abs(want[i]!));
      maxAbs = Math.max(maxAbs, Math.abs(got[i]! - want[i]!));
    }
    // Same render, same de-pad arithmetic — only the route differs, so this is exact.
    expect(peak).toBeGreaterThan(0.1);
    expect(maxAbs).toBe(0);
  });
});
