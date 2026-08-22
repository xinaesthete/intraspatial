import { describe, expect, it } from "vitest";
import { dispatchGrid, exclusiveScanGpu, SCAN_BLOCK, SCAN_WG } from "./prefixSum";

// CPU golden — a serial loop, which is a different algorithm from the GPU's per-thread
// serial pass + workgroup tree + inter-block propagation, not a transcription of it.
function scanCpu(v: Uint32Array): { scan: Uint32Array; total: number } {
  const out = new Uint32Array(v.length);
  let acc = 0;
  for (let i = 0; i < v.length; i++) {
    out[i] = acc >>> 0;
    acc = (acc + v[i]!) >>> 0;
  }
  return { scan: out, total: acc >>> 0 };
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Reduce two arrays to a verdict in one pass. Per-element `expect()` in a GPU test kills
 *  the Dawn fork, so everything is reduced before a single assertion sees it. `nonZero` is
 *  the load-bearing half: every silent GPU failure mode this repo has hit — the 65535
 *  workgroup cap, the ~2s watchdog kill, a 9th storage binding — returns zeroes, and a
 *  scan of zeroes is a perfectly self-consistent scan of zeroes. */
function verdict(got: ArrayLike<number>, want: ArrayLike<number>, tol = 0) {
  let bad = 0;
  let firstBad = -1;
  let maxErr = 0;
  let nonZero = 0;
  for (let i = 0; i < want.length; i++) {
    const err = Math.abs(got[i]! - want[i]!);
    if (err > maxErr) maxErr = err;
    if (err > tol) {
      bad++;
      if (firstBad < 0) firstBad = i;
    }
    if (got[i]! !== 0) nonZero++;
  }
  return { bad, firstBad, maxErr, nonZero, at: firstBad < 0 ? "" : `first at ${firstBad}: got ${got[firstBad]} want ${want[firstBad]}` };
}

describe("exclusiveScanGpu", () => {
  it("scans a hand-worked u32 example", async () => {
    // Worked by hand, not derived from the implementation: the running sum of
    // [3,1,4,1,5,9,2,6] before each position, and 31 after the last.
    const { scan, total } = await exclusiveScanGpu(new Uint32Array([3, 1, 4, 1, 5, 9, 2, 6]));
    expect(Array.from(scan)).toEqual([0, 3, 4, 8, 9, 14, 23, 25]);
    expect(total).toBe(31);
  });

  it("returns an empty scan and a zero total for an empty input", async () => {
    const { scan, total } = await exclusiveScanGpu(new Uint32Array(0));
    expect(Array.from(scan)).toEqual([]);
    expect(total).toBe(0);
  });

  it("scans a single element", async () => {
    const { scan, total } = await exclusiveScanGpu(new Uint32Array([7]));
    expect(Array.from(scan)).toEqual([0]);
    expect(total).toBe(7);
  });

  // The sizes scan implementations actually break at: either side of the workgroup size,
  // either side of the block, and either side of the point where the recursion gains a
  // level. Nothing here is a multiple of anything by accident.
  //   n > 1024      → a second level (block sums must be scanned)
  //   n > 1024*1024 → a third level
  const SIZES = [
    2,
    SCAN_WG - 1,
    SCAN_WG,
    SCAN_WG + 1,
    SCAN_BLOCK - 1,
    SCAN_BLOCK,
    SCAN_BLOCK + 1,
    2 * SCAN_BLOCK + 37,
    100_000,
    SCAN_BLOCK * SCAN_BLOCK, // 1,048,576 — exactly the last size that needs only 2 levels
    2_000_000, // 3 levels: 1954 blocks, then 2, then 1
  ];

  for (const n of SIZES) {
    it(`matches the CPU golden at n=${n}`, async () => {
      const rnd = mulberry32(0x5ca4 + n);
      // Values in [1, 16]: never zero, so a correct scan is strictly increasing and any
      // dropped block shows up as a plateau rather than hiding in the noise.
      const values = Uint32Array.from({ length: n }, () => 1 + Math.floor(rnd() * 16));
      const want = scanCpu(values);
      const { scan, total } = await exclusiveScanGpu(values);

      const v = verdict(scan, want.scan);
      expect(v.bad, `n=${n}: ${v.bad} wrong, ${v.at}`).toBe(0);
      expect(total, `n=${n} total`).toBe(want.total);
      // scan[0] is 0 by definition, every other entry must not be
      expect(v.nonZero, `n=${n} non-zero entries`).toBe(n - 1);
    });
  }

  it("is unchanged when the dispatch is forced into a 2-D grid", async () => {
    // The host-side fold is checked exhaustively in `dispatchGrid` below; this checks the
    // other half — that the shader rebuilds the block index from a 2-D workgroup id
    // correctly. Forcing maxPerDim to 5 turns 21 blocks into a 5x5 grid, the same shape a
    // real 67M-element scan would take, at 1/3000th the memory — and the 4 workgroups past
    // the end also exercise the `block >= numBlocks` guard, without which they would write
    // garbage block sums. A wrong fold here does not error: blocks get scanned twice or not
    // at all and the answer merely comes out wrong.
    const n = 20 * SCAN_BLOCK + 5;
    const rnd = mulberry32(0x2d1d);
    const values = Uint32Array.from({ length: n }, () => 1 + Math.floor(rnd() * 16));
    const want = scanCpu(values);
    const { scan, total } = await exclusiveScanGpu(values, { maxWorkgroupsPerDim: 5 });

    const v = verdict(scan, want.scan);
    expect(v.bad, `${v.bad} wrong, ${v.at}`).toBe(0);
    expect(total).toBe(want.total);
    expect(v.nonZero).toBe(n - 1);
  });

  it("scans f32 values", async () => {
    const n = 3 * SCAN_BLOCK + 11;
    const rnd = mulberry32(0xf10a);
    const values = Float32Array.from({ length: n }, () => 0.25 + rnd());
    const out = new Float64Array(n);
    let acc = 0;
    for (let i = 0; i < n; i++) {
      out[i] = acc;
      acc += values[i]!;
    }
    const { scan, total } = await exclusiveScanGpu(values);

    // Tolerance, not equality: the GPU sums in tree order and the golden serially, so the
    // two round differently. Bound is relative to the largest partial sum (~n * 0.75).
    const tol = acc * 1e-6;
    const v = verdict(scan, out, tol);
    expect(v.bad, `${v.bad} outside ${tol}, ${v.at}`).toBe(0);
    expect(total).toBeCloseTo(acc, 1);
    expect(v.nonZero).toBe(n - 1);
  });
});

describe("dispatchGrid", () => {
  // Pure, so the 65535 crossing is checked exhaustively here rather than by allocating the
  // ~67M elements it takes to reach it through the scan kernel. The kernel rebuilds the
  // block index as `wid.x + wid.y * x`, so the grid must cover every block and `x` must be
  // the real stride.
  it("stays 1-D below the limit", () => {
    expect(dispatchGrid(1, 65535)).toEqual({ x: 1, y: 1 });
    expect(dispatchGrid(65535, 65535)).toEqual({ x: 65535, y: 1 });
  });

  it("folds into rows of exactly maxPerDim past the limit", () => {
    expect(dispatchGrid(65536, 65535)).toEqual({ x: 65535, y: 2 });
    expect(dispatchGrid(78125, 65535)).toEqual({ x: 65535, y: 2 }); // 20M rows / 256
  });

  it("covers every workgroup, and never fewer", () => {
    for (const max of [2, 3, 7, 64]) {
      // max² is the ceiling by construction — both dimensions are capped at max
      for (let wg = 1; wg <= Math.min(200, max * max); wg++) {
        const { x, y } = dispatchGrid(wg, max);
        // enough to cover, both dimensions legal, and no wasted whole row
        expect(x * y >= wg && x <= max && y <= max && (y === 1 || x === max), `wg=${wg} max=${max} → ${x}x${y}`).toBe(true);
      }
    }
  });

  it("refuses a count it cannot fold", () => {
    expect(() => dispatchGrid(101, 10)).toThrow(/cannot be folded/);
  });
});
