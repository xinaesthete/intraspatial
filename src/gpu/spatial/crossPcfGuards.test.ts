import { describe, expect, it } from "vite-plus/test";
import { crossPCFGpu } from "./crossPcf";

// These guards throw before any GPU work, so they run in the CPU job with no adapter.
describe("crossPCFGpu refuses what it cannot honour", () => {
  const a = { xs: [1, 2], ys: [1, 2] };
  const b = { xs: [2, 3], ys: [2, 3] };
  const p = { bbox: [0, 0, 10, 10] as const, rMax: 5, nBins: 2 };

  it.each(["includeA", "includeB"] as const)("rejects %s rather than ignoring it", async (key) => {
    await expect(crossPCFGpu(a, b, { ...p, [key]: [1, 0] })).rejects.toThrow(/use crossPCF/);
  });

  it("still rejects a domain", async () => {
    const domain = { kind: "rect", bbox: p.bbox } as const;
    await expect(crossPCFGpu(a, b, { ...p, domain })).rejects.toThrow(/domain/);
  });
});
