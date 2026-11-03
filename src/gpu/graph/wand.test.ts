// The wand's subgraph against a direct implementation of the rule it composes: flood ∇²(G_σ ∗ d) ≥ 0
// from the seed, open by a disc (erode, dilate with an explicit disc), keep the seed's piece.
import { beforeAll, describe, expect, it } from "vite-plus/test";
import { floodFrom, mixtureDistance, nearestPixel } from "../../spatial/regions";
import { Graph, pull } from "./index";
import { convolveSeparableOp } from "./ops/convolveSeparable";
import { registerWandOps, wandPatch } from "./wand";

beforeAll(() => registerWandOps());

/** Two Gaussian blobs of different mixtures on a background, joined by a thin bridge. */
function scene(width: number, height: number): Float32Array {
  const n = width * height;
  const u = new Float32Array(3 * n);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const a = Math.exp(-((x - 12) ** 2 + (y - 12) ** 2) / 40);
      const b = Math.exp(-((x - 34) ** 2 + (y - 14) ** 2) / 30);
      const bridge = x > 12 && x < 34 && Math.abs(y - 13) < 1.5 ? 0.8 : 0;
      u[i] = 3 * Math.max(a, bridge);
      u[n + i] = 3 * b;
      u[2 * n + i] = 0.3 * Math.sin(x / 3) * Math.cos(y / 4);
    }
  }
  return u;
}

/** The rule, directly: the reference the subgraph must reproduce. */
function direct(u: Float32Array, width: number, height: number, col: number, row: number, sigma: number, rho: number): Float32Array {
  const n = width * height;
  const d = mixtureDistance(u, 3, width, height, col, row);
  const smooth = convolveSeparableOp.cpuGolden!([{ shape: { kind: "grid", width, height }, dtype: "f32", data: d }], {
    kernel: "gaussian",
    sigma,
    radius: Math.ceil(3 * sigma),
  })[0]!.data!;
  const at = (x: number, y: number) => smooth[Math.max(0, Math.min(height - 1, y)) * width + Math.max(0, Math.min(width - 1, x))]!;
  const curving = new Float32Array(n);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      curving[y * width + x] = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4 * at(x, y) >= 0 ? 1 : 0;
  const cell = floodFrom(curving, width, height, col, row, true);
  // Erode and dilate by an explicit disc, the raster's outside counting as outside.
  const R = Math.ceil(rho);
  const disc: [number, number][] = [];
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) if (dx * dx + dy * dy <= rho * rho) disc.push([dx, dy]);
  const inCell = (x: number, y: number) => x >= 0 && y >= 0 && x < width && y < height && cell[y * width + x] === 1;
  const eroded = new Uint8Array(n);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) eroded[y * width + x] = disc.every(([dx, dy]) => inCell(x + dx, y + dy)) ? 1 : 0;
  const opened = new Float32Array(n);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      if (!eroded[y * width + x]) continue;
      for (const [dx, dy] of disc)
        if (x + dx >= 0 && y + dy >= 0 && x + dx < width && y + dy < height) opened[(y + dy) * width + x + dx] = 1;
    }
  return floodFrom(opened, width, height, col, row);
}

async function viaGraph(u: Float32Array, width: number, height: number, col: number, row: number, sigma: number, rho: number) {
  const g = new Graph();
  const src = g.source(
    { shape: { kind: "grid", width, height }, dtype: "f32", data: u, axes: [{ name: "c", type: "channel", length: 3 }] },
    "u",
  );
  const { patch } = wandPatch(g, src, { col, row, sigma, tol: 0, laplacian: true, rho });
  return (await pull(g, patch, { mode: "cpu" })).data!;
}

describe("wandPatch", () => {
  const width = 48;
  const height = 28;
  const u = scene(width, height);
  for (const [col, row, sigma, rho] of [
    [12, 12, 1.5, 2.25],
    [12.4, 11.7, 1, 3],
    [34, 14, 2, 1.5],
    [22, 13, 1, 2.25], // on the bridge: the opening drops it
  ] as const) {
    it(`matches the rule at (${col}, ${row}), σ ${sigma}, ρ ${rho}`, async () => {
      const got = await viaGraph(u, width, height, col, row, sigma, rho);
      const want = direct(u, width, height, col, row, sigma, rho);
      const diff = want.reduce((a, w, i) => a + (w !== got[i] ? 1 : 0), 0);
      expect(diff).toBe(0);
      // A real patch, not two empty rasters agreeing (the bridge seed is the empty case below).
      if (col !== 22) expect(want.reduce((a, v) => a + v, 0)).toBeGreaterThan(20);
    });
  }

  it("is empty when the opening removes the seed's pixel", async () => {
    const got = await viaGraph(u, width, height, 22, 13, 1, 2.25);
    const start = nearestPixel(width, height, 22, 13);
    expect(got[start]).toBe(0);
    expect(got.every((v) => v === 0)).toBe(true);
  });
});
