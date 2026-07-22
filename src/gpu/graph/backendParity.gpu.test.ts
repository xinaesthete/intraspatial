import { beforeAll, describe, expect, it } from "vitest";
import { getDevice } from "../device";
import { adoptDevice } from "../interop/adoptDevice";
import type { GpuBackend } from "./backend";
import { browserBackend } from "./backend.browser";
import { nodeBackend } from "./backend.node";
import { Graph, pullData } from "./index";

// Every `GpuBackend` implementation must be able to run the graph — not just the default one.
//
// WHY THIS EXISTS. Until now every graph test relied on the executor's default
// (`ctx: opts.ctx ?? { backend: nodeBackend }`); no test anywhere passed an explicit `ctx`. So
// the other two implementations were exercised by nothing, and when ADR-0017 added
// `lease`/`release`/`upload`/`poolStats` to the interface, `backend.browser.ts` was missed and
// shipped broken — the playground threw `backend.upload is not a function` the first time it hit
// a resident op. A type error would now catch a *missing* method (the browser backend has since
// moved into `src/`, which the root tsconfig covers), but nothing would catch one that is present
// and wrong. This runs the real thing on each.
//
// The graph below is chosen to touch the whole Tier-2 surface in one pull: a host `grid` source
// forces `upload`, two resident ops force `lease` + interior residency, and the sink forces
// `readbackF32`. An implementation that gets any of those wrong disagrees with `nodeBackend`.
//
// `browserBackend` is testable here because it resolves its device through `src/gpu/device.ts`,
// the same module the Node backend uses — under Node that is Dawn. This does not prove anything
// browser-specific; it proves the backend's own logic, which is where the regression was.

const W = 16;
const H = 16;

function ramp(): Float32Array {
  const g = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) g[y * W + x] = Math.sin(x * 0.4) * Math.cos(y * 0.3) + 1.5;
  }
  return g;
}

/** grid -> convolve -> threshold. Both ops are resident, so the source uploads and the interior
 *  stays on-GPU. */
function chain() {
  const g = new Graph();
  const src = g.grid(ramp(), W, H);
  const smooth = g.op1("convolveSeparable", { grid: src }, { kernel: "gaussian", radius: 2 });
  const mask = g.op1("threshold", { in: smooth }, { thresh: 1.5, soft: true, softness: 8 });
  return { g, mask };
}

/** Largest absolute elementwise difference — one assertion instead of W*H of them. */
function maxDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}

describe("GpuBackend parity", () => {
  let backends: Array<{ name: string; backend: GpuBackend }>;

  beforeAll(async () => {
    const device = await getDevice();
    backends = [
      { name: "node", backend: nodeBackend },
      { name: "adopted", backend: adoptDevice(device, "test-host") },
      { name: "browser", backend: browserBackend },
    ];
  });

  it("implements the whole interface on every backend", () => {
    // Structural guard: catches a method that is absent, or present but not callable, on an
    // object that reached the registry through a cast. Cheap, and it names the culprit.
    const required = ["getDevice", "getRoot", "readbackF32", "lease", "release", "upload", "poolStats"] as const;
    const missing: string[] = [];
    for (const { name, backend } of backends) {
      for (const m of required) {
        if (typeof (backend as unknown as Record<string, unknown>)[m] !== "function") missing.push(`${name}.${m}`);
      }
    }
    expect(missing).toEqual([]);
  });

  it("produces the same result through every backend, and returns every lease", async () => {
    const ref = chain();
    const reference = await pullData(ref.g, ref.mask, { ctx: { backend: nodeBackend } });
    expect(reference.length).toBe(W * H);
    // Guard against comparing two identically-degenerate fields.
    expect(new Set(Array.from(reference)).size).toBeGreaterThan(1);

    // Collected then asserted in one go, so a failure names the offending backend in the diff
    // rather than dissolving into per-element noise.
    const actual: Record<string, { length: number; matchesReference: boolean; leasesOutstanding: number }> = {};
    for (const { name, backend } of backends) {
      const { g, mask } = chain();
      const got = await pullData(g, mask, { ctx: { backend } });
      actual[name] = {
        length: got.length,
        // Same device, same kernels — the backends differ only in device acquisition, pooling
        // and readback, so agreement should be exact rather than approximate.
        matchesReference: maxDiff(got, reference) === 0,
        // A completed `pull` downloads the sink and hands its lease back, so nothing stays
        // checked out. A backend that leaks here grows its pool by one buffer per pull.
        leasesOutstanding: backend.poolStats().live,
      };
    }

    const expected = { length: W * H, matchesReference: true, leasesOutstanding: 0 };
    expect(actual).toEqual({ node: expected, adopted: expected, browser: expected });
  });
});
