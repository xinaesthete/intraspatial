import { describe, expect, it } from "vitest";
import { dtypeBytes, fieldBytes, fieldValueBytes, formatBytes, memoryBytes } from "./memory";
import { FieldRing } from "./ringBuffer";
import type { FieldValue } from "./handle";

describe("memory accounting", () => {
  it("computes a field's resident bytes from shape/element/dtype", () => {
    expect(dtypeBytes("f32")).toBe(4);
    // 4×4 grid, scalar
    expect(fieldBytes({ kind: "grid", width: 4, height: 4 })).toBe(16 * 4);
    // 100 points, vec3
    expect(fieldBytes({ kind: "points", n: 100 }, { kind: "vec", n: 3 })).toBe(100 * 3 * 4);
  });

  it("measures a concrete value by its data", () => {
    const v: FieldValue = { shape: { kind: "grid", width: 3, height: 2 }, dtype: "f32", data: new Float32Array(6) };
    expect(fieldValueBytes(v)).toBe(6 * 4);
  });

  it("memoryBytes reads byteLength from a reporter (a FieldRing keeps k copies)", () => {
    const fr = new FieldRing({ kind: "grid", width: 8, height: 8 }, 30); // 30 frames of a 64-cell grid
    expect(memoryBytes(fr)).toBe(64 * 30 * 4);
  });

  it("formats byte counts", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KiB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MiB");
  });
});
