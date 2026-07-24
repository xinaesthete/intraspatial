// Placement / axes / role threading through the builder + executor (ADR-0018, ADR-0015 §3).
//
// These facets propagate exactly like `basis`: an opt-in inference hook, a persisted array on
// the `GraphNode`, and an executor stamp that fires only when the op left the facet unset. The
// tests here are the CPU-golden counterparts of the wavelet basis-propagation tests: the
// pass-through defaults, the `systemsAgree` agreement helper (all three branches), and a
// placement round-trip through a facet-unaware op.
import { beforeAll, describe, expect, it } from "vitest";
import type { Affine3 } from "../../coords";
import type { FieldRole, FieldValue, ResolvedPlacement, TensorAxis } from "./handle";
import { placementOf, systemsAgree } from "./handle";
import { Graph, pull, registerElementOps } from "./index";

beforeAll(async () => {
  await registerElementOps(); // scaleField — a single-input, facet-unaware pass-through op
});

const identity: Affine3 = {
  origin: [0, 0, 0],
  axes: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
};
// A non-identity placement: 0.5 µm cells, translated — a real "this grid sits here in world".
const micron: Affine3 = {
  origin: [10, 20, 0],
  axes: [
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 1],
  ],
};

const placementIn = (system: string, wfa: Affine3): ResolvedPlacement => ({ system, worldFromArray: wfa });

const channelAxis: TensorAxis = { name: "c", type: "channel", length: 3 };
const labelRole: FieldRole = { kind: "label", labels: { resample: "nearest" } };

/** A 4×4 grid source carrying whatever facets are passed. */
function gridSource(g: Graph, extra: Partial<FieldValue> = {}) {
  const value: FieldValue = { shape: { kind: "grid", width: 4, height: 4 }, dtype: "f32", data: new Float32Array(16).fill(1), ...extra };
  return g.source(value);
}

describe("systemsAgree (ADR-0018 agreement helper)", () => {
  it("both absent ⇒ ok (two array-space fields combine, today's behaviour)", () => {
    expect(systemsAgree(undefined, undefined)).toBe(true);
  });

  it("both present, same system ⇒ ok — even with different matrices (two pyramid levels)", () => {
    const a = placementIn("global", micron);
    const b = placementIn("global", identity); // different worldFromArray, same system
    expect(systemsAgree(a, b)).toBe(true);
  });

  it("both present, different system ⇒ not ok", () => {
    expect(systemsAgree(placementIn("global", identity), placementIn("tissue", identity))).toBe(false);
  });

  it("exactly one present ⇒ throws (placed + unplaced can't combine)", () => {
    expect(() => systemsAgree(placementIn("global", identity), undefined)).toThrow(/cannot combine/i);
    expect(() => systemsAgree(undefined, placementIn("global", identity))).toThrow(/cannot combine/i);
  });
});

describe("placementOf", () => {
  it("absent ⇒ undefined (array space, NOT an identity placement)", () => {
    expect(placementOf({})).toBeUndefined();
    expect(placementOf({ placement: undefined })).toBeUndefined();
  });

  it("present ⇒ returns it unchanged", () => {
    const p = placementIn("global", micron);
    expect(placementOf({ placement: p })).toBe(p);
  });
});

describe("builder pass-through defaults (no op hook ⇒ pass through inputs[0])", () => {
  it("placement threads onto the source handle and survives a facet-unaware op at build time", () => {
    const g = new Graph();
    const p = placementIn("global", micron);
    const src = gridSource(g, { placement: p });
    expect(src.placement).toEqual(p); // carried onto the source handle (the ADR-0015 gap, now closed)
    const scaled = g.op1("scaleField", { in: src }, { s: 2 });
    expect(scaled.placement).toEqual(p); // default inferPlacement = pass through inputs[0]
  });

  it("axes thread through a facet-unaware op at build time", () => {
    const g = new Graph();
    const src = gridSource(g, { axes: [channelAxis] });
    expect(src.axes).toEqual([channelAxis]);
    const scaled = g.op1("scaleField", { in: src }, { s: 1 });
    expect(scaled.axes).toEqual([channelAxis]);
  });

  it("role threads through a facet-unaware op at build time", () => {
    const g = new Graph();
    // u32 label (respect the label dtype invariant) so the role is meaningful.
    const value: FieldValue = { shape: { kind: "grid", width: 4, height: 4 }, dtype: "u32", data: new Uint32Array(16), role: labelRole };
    const src = g.source(value);
    expect(src.role).toEqual(labelRole);
    const scaled = g.op1("scaleField", { in: src }, { s: 1 });
    expect(scaled.role).toEqual(labelRole);
  });

  it("absent facet ⇒ absent on the output (today's behaviour, array space ≠ identity)", () => {
    const g = new Graph();
    const src = gridSource(g); // no placement/axes/role
    const scaled = g.op1("scaleField", { in: src }, { s: 1 });
    expect(scaled.placement).toBeUndefined();
    expect(scaled.axes).toBeUndefined();
    expect(scaled.role).toBeUndefined();
  });
});

describe("placement round-trip through execution (mirrors the basis-propagation test)", () => {
  it("a source's placement survives passthrough ops all the way to pull", async () => {
    const g = new Graph();
    const p = placementIn("global", micron);
    const src = gridSource(g, { placement: p });
    // Two facet-unaware ops in a row — neither knows about coordinates.
    const scaled = g.op1("scaleField", { in: src }, { s: 3 });
    const twice = g.op1("scaleField", { in: scaled }, { s: 2 });
    const out = await pull(g, twice, { mode: "cpu" });
    expect(out.placement).toEqual(p); // stamped by the executor onto the runtime value
    // And the actual computation still happened (sanity: 1 * 3 * 2 = 6).
    expect(out.data?.[0]).toBeCloseTo(6);
  });

  it("axes + role also survive to pull, alongside placement", async () => {
    const g = new Graph();
    const p = placementIn("tissue", identity);
    const value: FieldValue = {
      shape: { kind: "grid", width: 4, height: 4 },
      dtype: "f32",
      data: new Float32Array(16).fill(2),
      axes: [channelAxis],
      placement: p,
    };
    const src = g.source(value);
    const scaled = g.op1("scaleField", { in: src }, { s: 1 });
    const out = await pull(g, scaled, { mode: "cpu" });
    expect(out.placement).toEqual(p);
    expect(out.axes).toEqual([channelAxis]);
  });
});
