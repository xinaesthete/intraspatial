import { describe, expect, it } from "vitest";
import { createMemo, Graph, pull } from "./index";

// Memoisation: changing one node's param re-runs only that node and its dependents;
// unchanged upstream outputs are reused (same object reference) across pulls, even
// though the graph is rebuilt from scratch each time. Uses CPU-only `addGrids` so the
// test exercises the executor's content-addressed cache without any GPU work.
function build(topWeight: number) {
  const g = new Graph();
  const a = g.grid(
    Float32Array.from({ length: 16 }, (_, i) => i),
    4,
    4,
  );
  const b = g.grid(
    Float32Array.from({ length: 16 }, (_, i) => 16 - i),
    4,
    4,
  );
  const c = g.op1("addGrids", { a, b }, { wa: 1, wb: 1 }); // shared upstream
  const d = g.op1("addGrids", { a: c, b: a }, { wa: topWeight, wb: 0 }); // param varies
  return { g, c, d };
}

describe("executor memoisation", () => {
  it("reuses the unchanged sub-DAG and recomputes only the changed node", async () => {
    const memo = createMemo();

    const r1 = build(1);
    const seen1 = new Map<string, unknown>();
    await pull(r1.g, r1.d, { cache: memo, onValue: (k, v) => seen1.set(k, v) });
    const c1 = seen1.get(`${r1.c.producer}:out`);
    const d1 = seen1.get(`${r1.d.producer}:out`) as { data: Float32Array };

    const r2 = build(2); // only the top node's param changes
    const seen2 = new Map<string, unknown>();
    await pull(r2.g, r2.d, { cache: memo, onValue: (k, v) => seen2.set(k, v) });
    const c2 = seen2.get(`${r2.c.producer}:out`);
    const d2 = seen2.get(`${r2.d.producer}:out`) as { data: Float32Array };

    expect(c1).toBeDefined();
    expect(c2).toBe(c1); // C reused from the memo — same object, not recomputed
    expect(d2).not.toBe(d1); // D recomputed (its param changed)
    expect(d2.data[5]).not.toBe(d1.data[5]); // and its value actually differs
  });

  it("without a cache, every pull recomputes (no reuse)", async () => {
    const r1 = build(1);
    const seen1 = new Map<string, unknown>();
    await pull(r1.g, r1.d, { onValue: (k, v) => seen1.set(k, v) });
    const r2 = build(1);
    const seen2 = new Map<string, unknown>();
    await pull(r2.g, r2.d, { onValue: (k, v) => seen2.set(k, v) });
    expect(seen2.get(`${r2.c.producer}:out`)).not.toBe(seen1.get(`${r1.c.producer}:out`));
  });
});
