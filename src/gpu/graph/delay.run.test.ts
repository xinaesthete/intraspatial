// Runtime test for the `delay` (z⁻ᵏ) node — the generalisation of `feedback` (z⁻¹) built
// on the FieldRing history primitive. We drive an incrementing scalar (a feedback counter)
// through a depth-3 delay and confirm the output lags by three ticks, then check the
// resident memory the delay holds.
import { beforeAll, describe, expect, it } from "vitest";
import type { FieldValue } from "./handle";
import { advance, createSimState, Graph, registerElementOps, simStateBytes } from "./index";

beforeAll(async () => {
  await registerElementOps(); // addFields
});

const scalar = (v: number): FieldValue => ({ shape: { kind: "scalar" }, dtype: "f32", data: new Float32Array([v]) });

/** Build: counter t = t₋₁ + 1 (feedback), and a depth-`k` delay of t. */
function buildDelay(k: number) {
  const g = new Graph();
  const zero = g.source(scalar(0), "zero");
  const one = g.source(scalar(1), "one");
  const cfb = g.feedback(zero, "counter");
  const t = g.op1("addFields", { a: cfb.state, b: one });
  cfb.close(t);
  const d = g.delay(t, k, "delay");
  d.close(t);
  return { g, out: d.out };
}

describe("delay (z⁻ᵏ)", () => {
  it("lags its input by `depth` ticks (seeded until history accrues)", async () => {
    const { g, out } = buildDelay(3);
    const seq: number[] = [];
    await advance(g, out, {
      steps: 6,
      state: createSimState(),
      mode: "cpu",
      onFrame: (_i, v) => seq.push(v.data?.[0] ?? NaN),
    });
    // counter emits 1,2,3,4,5,6; a depth-3 delay seeds with the live value for the first
    // three ticks, then replays 1,2,3.
    expect(seq).toEqual([1, 2, 3, 1, 2, 3]);
  });

  it("depth 1 reproduces a feedback delay", async () => {
    const { g, out } = buildDelay(1);
    const seq: number[] = [];
    await advance(g, out, { steps: 4, state: createSimState(), mode: "cpu", onFrame: (_i, v) => seq.push(v.data?.[0] ?? NaN) });
    // depth 1: tick0 seeds with live (1), then replays the previous tick: 1,1,2,3
    expect(seq).toEqual([1, 1, 2, 3]);
  });

  it("reports the resident bytes it holds (depth copies of the field)", async () => {
    const { g, out } = buildDelay(3);
    const store = createSimState();
    await advance(g, out, { steps: 5, state: store, mode: "cpu" });
    // feedback holds one scalar (4 B); the depth-3 delay ring holds 3 scalars (12 B).
    expect(simStateBytes(store)).toBe(4 + 12);
  });
});
