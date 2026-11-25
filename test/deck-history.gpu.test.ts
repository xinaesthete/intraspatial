// The history ring behind the deck TraceLayer: snapshots are copied on the device, the ring wraps,
// and `traceSlot` finds the snapshot a given number of steps back.
import { describe, expect, it } from "vite-plus/test";
import { PositionHistory, traceSlot } from "../packages/deck/src/history";
import { getDevice, readBackBytes, writeView } from "../src/gpu/device";

describe("PositionHistory", () => {
  it("keeps the newest snapshots, oldest overwritten, and finds each by age", async () => {
    const device = await getDevice();
    const count = 5;
    const capacity = 3;
    const history = new PositionHistory(device, { count, capacity, label: "test:history" });
    // A source larger than one snapshot, as a simulation's [pos | prev] buffer is.
    const src = device.createBuffer({
      label: "test:history-src",
      size: count * 2 * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const taken = 5;
    for (let step = 0; step < taken; step++) {
      // Every float of snapshot `step` is `step + 1`; the second half of the source is never copied.
      const data = new Float32Array(count * 8).fill(-1);
      data.fill(step + 1, 0, count * 4);
      writeView(device.queue, src, data);
      history.capture(src);
    }
    expect(history.filled).toBe(capacity);
    expect(history.head).toBe(taken % capacity);

    const ring = new Float32Array(await readBackBytes(device, "test:history", history.buffer, 0, capacity * count * 16));
    const slotValue = (slot: number) => {
      const part = ring.subarray(slot * count * 4, (slot + 1) * count * 4);
      // One value throughout a slot: a snapshot is copied whole.
      expect(Math.min(...part)).toBe(Math.max(...part));
      return part[0];
    };
    // Age 0 is the last snapshot taken, then back in time.
    const byAge = [0, 1, 2].map((age) => slotValue(traceSlot(history.head, age, capacity)));
    expect(byAge).toEqual([taken, taken - 1, taken - 2]);

    history.clear();
    expect([history.head, history.filled]).toEqual([0, 0]);
    history.destroy();
    src.destroy();
  });
});
