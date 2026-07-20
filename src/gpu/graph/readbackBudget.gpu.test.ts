import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { getDevice } from "../device";
import { Graph, pullData } from "./index";
import { installReadbackCounter, measureReadbacks, uninstallReadbackCounter } from "./instrument";

// Enforces `docs/gpu-resource-sync.md` invariant 4 — "Boundary-only transfer: upload at
// sources, download at sinks; interior edges stay on-GPU."
//
// This is the graph-level check the per-op cpuGolden discipline structurally cannot make:
// every op passes its golden whether or not its edge round-tripped to the host, so the
// invariant has been silently violated on every edge. Here we count real downloads
// (mapAsync READ) across chains of increasing length. Under invariant 4 the count is
// CONSTANT in chain length (one download, at the sink). Today it grows — that growth IS
// the Tier-1 violation, measured.
//
// Own file ⇒ own fork (Dawn teardown isolation, ADR-0002/0003).
//
// RUNNING THIS. The table only prints with console intercept disabled — vitest's fork pool
// swallows `console.log`:
//   pnpm exec vitest run -c vitest.gpu.config.ts src/gpu/graph/readbackBudget.gpu.test.ts --disableConsoleIntercept
// It must run under vitest: `tsx` cannot resolve typegpu's internals, so a standalone TS script
// cannot import GPU ops. Vitest (with the unplugin-typegpu plugin) is the only vehicle.

const W = 24;
const H = 24;
const BBOX: [number, number, number, number] = [0, 0, 100, 100];

function makePoints() {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < 40; i++) {
    xs.push(50 + (i % 7) * 0.6);
    ys.push(50 + Math.floor(i / 7) * 0.6);
  }
  return { xs, ys };
}

/** A chain of `n` grid ops after the splat source: splat -> (convolve -> threshold)*. */
function chain(n: number) {
  const { xs, ys } = makePoints();
  const g = new Graph();
  const pts = g.points(xs, ys);
  let f = g.op1("splatDensity", { points: pts }, { width: W, height: H, sigma: 2, bbox: BBOX });
  for (let i = 1; i < n; i++) {
    f =
      i % 2 === 1
        ? g.op1("convolveSeparable", { grid: f }, { kernel: "box", radius: 1 })
        : g.op1("threshold", { in: f }, { thresh: 0.05, soft: false });
  }
  return { g, f };
}

describe("resource-sync invariant 4 — boundary-only transfer", () => {
  beforeAll(async () => {
    await getDevice(); // installs the webgpu globals that expose GPUBuffer
    installReadbackCounter();
  });
  afterAll(() => uninstallReadbackCounter());

  it("counts downloads per pull across chain lengths (baseline for Tier-2)", async () => {
    const rows: Array<{ ops: number; downloads: number; kb: number }> = [];
    for (const n of [1, 2, 3, 4]) {
      const { g, f } = chain(n);
      const { stats } = await measureReadbacks(() => pullData(g, f));
      rows.push({ ops: n, downloads: stats.downloads, kb: Math.round(stats.downloadBytes / 102.4) / 10 });
    }

    const table = rows.map((r) => `  ${r.ops} op(s): ${r.downloads} download(s), ${r.kb} KB`).join("\n");
    console.log(`\ninvariant-4 readback budget (target: 1 download regardless of chain length)\n${table}\n`);

    // Every pull must download at least once — the sink itself.
    for (const r of rows) expect(r.downloads).toBeGreaterThanOrEqual(1);

    // BASELINE RATCHET. Filled from the measured numbers below; tighten as Tier-2 lands.
    // Under invariant 4 this whole array becomes [1, 1, 1, 1].
    expect(rows.map((r) => r.downloads)).toEqual(BASELINE_DOWNLOADS);
  });
});

// Measured on Dawn-on-Node. Downloads SHOULD be constant (1) per invariant 4; the growth
// here is the Tier-1 interior-edge round-trip, quantified.
const BASELINE_DOWNLOADS = [1, 2, 3, 4];
