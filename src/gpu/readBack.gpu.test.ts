// readBackBytes shares one staging buffer per key; reads issued together must not trip over it.
import { describe, expect, it } from "vite-plus/test";
import { getDevice, readBackBytes } from "./device";

describe("readBackBytes", () => {
  it("serves concurrent reads on one key, including one that grows the staging buffer", async () => {
    const device = await getDevice();
    const sizes = [64, 64, 4096, 128];
    const srcs = sizes.map((n, k) => {
      const data = Uint32Array.from({ length: n }, (_, i) => k * 100_000 + i);
      const buffer = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, data);
      return { buffer, data };
    });
    const got = await Promise.all(srcs.map((s) => readBackBytes(device, "readBack.test", s.buffer, 0, s.data.byteLength)));
    const wrong = got.filter((b, k) => {
      const v = new Uint32Array(b);
      const want = srcs[k]!.data;
      return v.length !== want.length || v.some((x, i) => x !== want[i]);
    }).length;
    expect(wrong).toBe(0);
  });
});
