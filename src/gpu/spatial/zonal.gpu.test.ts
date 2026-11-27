import { beforeAll, describe, expect, it } from "vite-plus/test";
import { chunkVoxelExtent } from "#src/datasource/multiscale";
import { axisAlignedMultiscale } from "#src/datasource/syntheticLoader";
import { TileCache } from "#src/datasource/tileCache";
import type { ChunkId, Loader, Tile } from "#src/datasource/types";
import { getDevice, readBackBytes } from "#src/gpu/device";
import { domainContains, type PolygonDomain, polygonDomain } from "#src/spatial/domain";
import {
  type ZonalChannelData,
  type ZonalHistogram,
  type ZonalOptions,
  polygonRingBuffers,
  rasterisePolygonsCpu,
  zonalCollectCpu,
  zonalExactPlan,
  zonalExactQuantiles,
  zonalHistogramCpu,
} from "#src/spatial/zonal";
import { denseRasterPlane, residentRasterPlane } from "./histogram";
import { type ZonalTile, ZonalAccumulator, rasterisePolygonsResident, zonalHistogramGpu } from "./zonal";
import { zonalStatsTiled } from "./zonalTiled";

// GPU vs the f64 golden (`zonalHistogramCpu`): counts, CDF, totals and edges EXACTLY, moments to
// f32 accumulation. Small images and one aggregated `expect` per comparison: per-element expects
// and heavy host work are what kill the Dawn fork.

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32;
}

function mismatch(a: ArrayLike<number>, b: ArrayLike<number>, tol = 0): string | undefined {
  if (a.length !== b.length) return `length ${a.length} != ${b.length}`;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = b[i]!;
    if (x === y || (Number.isNaN(x) && Number.isNaN(y))) continue;
    if (!(Math.abs(x - y) <= tol * Math.max(1, Math.abs(y)))) return `index ${i}: got ${x}, want ${y}`;
  }
  return undefined;
}

function diff(gpu: ZonalHistogram, cpu: ZonalHistogram, momentTol = 1e-5): string | undefined {
  return (
    mismatch(gpu.edges, cpu.edges) ??
    mismatch(gpu.counts, cpu.counts) ??
    mismatch(gpu.cdf, cpu.cdf) ??
    mismatch(gpu.totals, cpu.totals) ??
    mismatch(gpu.moments, cpu.moments, momentTol)
  );
}

let device: GPUDevice;
function upload(data: ArrayBufferView, label: string): GPUBuffer {
  const buf = device.createBuffer({
    label,
    size: Math.max(256, Math.ceil(data.byteLength / 256) * 256),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
  return buf;
}

const W = 83;
const H = 61;

/** Labels in 0..R+1 as blobs (with a few stray labels > R), two channels with NaN/Inf/zeros. */
function scene(seed: number, R: number) {
  const rnd = lcg(seed);
  const n = W * H;
  const labels = new Uint32Array(n);
  const a = new Float32Array(n);
  const b = new Float32Array(n);
  const sup = new Float32Array(n);
  const cx = Array.from({ length: R + 2 }, () => rnd() * W);
  const cy = Array.from({ length: R + 2 }, () => rnd() * H);
  for (let i = 0; i < n; i++) {
    const x = i % W;
    const y = Math.floor(i / W);
    let best = 0;
    let bd = 12 ** 2;
    for (let k = 0; k < R + 2; k++) {
      const d = (x - cx[k]!) ** 2 + (y - cy[k]!) ** 2;
      if (d < bd) {
        bd = d;
        best = k + 1;
      }
    }
    labels[i] = best; // R + 1 and R + 2 are stray: ignored
    a[i] = rnd() < 0.15 ? 0 : Math.exp(rnd() * 4 - 2);
    b[i] = (rnd() - 0.3) * 20;
    sup[i] = rnd() < 0.85 ? 1 : rnd() < 0.5 ? 0 : -0;
  }
  a[7] = Number.NaN;
  a[9] = -Infinity;
  b[11] = Infinity;
  return { labels, a, b, sup };
}

/** A tile over the whole image from host arrays. */
function wholeTile(labels: Uint32Array | Float32Array, chans: { values: Float32Array; support?: Float32Array }[]): ZonalTile {
  return {
    labels: { buffer: upload(labels, "zonal-test:labels"), width: W, height: H, dtype: labels instanceof Float32Array ? "f32" : "u32" },
    channels: chans.map((c) => ({
      plane: denseRasterPlane(upload(c.values, "zonal-test:plane"), W, H),
      support: c.support ? { buffer: upload(c.support, "zonal-test:support"), dtype: "f32" as const } : undefined,
    })),
  };
}

describe("zonal GPU", () => {
  beforeAll(async () => {
    device = await getDevice();
  });

  it("rasterises polygons exactly as domainContains, through a frame", async () => {
    const rnd = lcg(3);
    const polys: PolygonDomain[] = [];
    for (let k = 0; k < 9; k++) {
      const cx = rnd() * 200;
      const cy = rnd() * 150;
      const ring: [number, number][] = [];
      const m = 5 + Math.floor(rnd() * 20);
      for (let v = 0; v < m; v++) {
        const t = (v / m) * 2 * Math.PI;
        const r = 10 + rnd() * 30;
        ring.push([cx + r * Math.cos(t), cy + r * Math.sin(t)]);
      }
      polys.push(
        k === 4
          ? polygonDomain([ring, ring.map(([x, y]): [number, number] => [cx + (x - cx) * 0.3, cy + (y - cy) * 0.3])])
          : polygonDomain(ring),
      );
    }
    const frame = { width: 97, height: 71, originX: 13.25, originY: 7.5, scaleX: 2, scaleY: 2 };
    const cpu = rasterisePolygonsCpu(polys, frame);
    const buffer = upload(new Uint32Array(frame.width * frame.height).fill(77), "zonal-test:raster");
    await rasterisePolygonsResident(polygonRingBuffers(polys, frame), { buffer, width: frame.width, height: frame.height }, device);
    const gpu = new Uint32Array(await readBackBytes(device, "zonal-test:rb", buffer, 0, frame.width * frame.height * 4));
    // f32 vertices can flip a centre within ~1e-4 px of an edge; any disagreement must be one.
    const bad: string[] = [];
    for (let q = 0; q < gpu.length; q++) {
      if (gpu[q] === cpu[q]) continue;
      const x = frame.originX + ((q % frame.width) + 0.5) * 2;
      const y = frame.originY + (Math.floor(q / frame.width) + 0.5) * 2;
      const near = polys.some((p) =>
        [-1e-3, 1e-3].some(
          (d) => domainContains(p, x + d, y) !== domainContains(p, x - d, y) || domainContains(p, x, y + d) !== domainContains(p, x, y - d),
        ),
      );
      if (!near) bad.push(`${q}: ${gpu[q]} != ${cpu[q]}`);
    }
    expect(bad.slice(0, 5)).toEqual([]);
    expect(cpu.filter((v) => v > 0).length).toBeGreaterThan(1000);
  });

  it("matches the golden: u32 and f32 labels, support, positive rule, auto and fixed range", async () => {
    const R = 6;
    const { labels, a, b, sup } = scene(1, R);
    const opts: ZonalOptions = { regions: R, bins: 64, channels: [{ valid: "positive" }, { range: [-2, 9] }] };
    const data: ZonalChannelData[] = [{ values: a, support: sup }, { values: b }];
    const cpu = zonalHistogramCpu(labels, data, opts);
    expect(cpu.totals.some((v, i) => i % 8 === 5 && v > 0)).toBe(true); // unsupported pixels exist
    // regions × bins = 384 fits the workgroup-private histogram.
    expect(
      diff(await zonalHistogramGpu(wholeTile(labels, [{ values: a, support: sup }, { values: b }]), opts, device), cpu),
    ).toBeUndefined();
    const f32 = Float32Array.from(labels, (l) => (l === 3 ? 3.7 : l)); // truncated: same regions
    expect(diff(await zonalHistogramGpu(wholeTile(f32, [{ values: a, support: sup }, { values: b }]), opts, device), cpu)).toBeUndefined();
  });

  it("global-atomics path: many regions × bins, past the workgroup cap", async () => {
    const R = 40;
    const { labels, a } = scene(2, R);
    const opts: ZonalOptions = { regions: R, bins: 512, channels: [{}] };
    const cpu = zonalHistogramCpu(labels, [{ values: a }], opts);
    expect(diff(await zonalHistogramGpu(wholeTile(labels, [{ values: a }]), opts, device), cpu)).toBeUndefined();
  });

  it("accumulates tiles (range pass + bin pass) into the whole image's histogram, and the exact pass agrees", async () => {
    const R = 5;
    const { labels, a, b, sup } = scene(4, R);
    const opts: ZonalOptions = { regions: R, bins: 32, channels: [{ valid: "positive" }, {}] };
    const data: ZonalChannelData[] = [{ values: a, support: sup }, { values: b }];
    const cpu = zonalHistogramCpu(labels, data, opts);
    // Four uneven tiles; each its own buffers.
    const tiles: ZonalTile[] = [];
    for (const [x0, y0, x1, y1] of [
      [0, 0, 50, 20],
      [50, 0, W, 20],
      [0, 20, 30, H],
      [30, 20, W, H],
    ] as const) {
      const w = x1 - x0;
      const h = y1 - y0;
      const crop = <T extends Uint32Array | Float32Array>(src: T, make: (n: number) => T): T => {
        const out = make(w * h);
        for (let y = 0; y < h; y++) out.set(src.subarray((y0 + y) * W + x0, (y0 + y) * W + x1), y * w);
        return out;
      };
      const f = (s: Float32Array) => crop(s, (n) => new Float32Array(n));
      tiles.push({
        labels: {
          buffer: upload(
            crop(labels, (n) => new Uint32Array(n)),
            "zonal-test:tl",
          ),
          width: w,
          height: h,
          dtype: "u32",
        },
        channels: [
          {
            plane: denseRasterPlane(upload(f(a), "zonal-test:ta"), w, h),
            support: { buffer: upload(f(sup), "zonal-test:ts"), dtype: "f32" },
          },
          { plane: denseRasterPlane(upload(f(b), "zonal-test:tb"), w, h) },
        ],
      });
    }
    const acc = await ZonalAccumulator.create(opts, device);
    try {
      expect(acc.needsRange).toBe(true);
      for (const t of tiles) acc.rangeTile(t);
      acc.finalizeEdges();
      for (const t of tiles) acc.binTile(t);
      const gpu = await acc.read();
      expect(diff(gpu, cpu, 1e-6)).toBeUndefined();

      const qs = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1];
      const plan = zonalExactPlan(gpu, qs);
      acc.beginCollect(plan);
      for (const t of tiles) acc.collectTile(t);
      const exactGpu = zonalExactQuantiles(gpu, plan, await acc.readCollected());
      const exactCpu = zonalExactQuantiles(cpu, plan, zonalCollectCpu(labels, data, opts, cpu, plan));
      expect(mismatch(exactGpu, exactCpu)).toBeUndefined();
    } finally {
      acc.release();
    }
  });

  it("reads an f16 ResidentRasters plane with padded rows", async () => {
    const R = 4;
    const { labels } = scene(5, R);
    const rowWords = 64; // f16 row of 83 texels = 42 words, padded to 64
    const K = 2;
    const words = new Uint32Array(K * H * rowWords).fill(0x7e007e00); // padding = NaN: must be skipped
    const rnd = lcg(9);
    for (let k = 0; k < K; k++)
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x += 2)
          words[(k * H + y) * rowWords + (x >> 1)] = (Math.floor(rnd() * 0x7bff) | (Math.floor(rnd() * 0x7bff) << 16)) >>> 0;
    const resident = {
      buffer: upload(words, "zonal-test:rasters"),
      rowWords,
      precision: "f16" as const,
      mean: new Float64Array(K),
      sd: new Float64Array(K),
    };
    const f16 = (h: number) => {
      const e = (h >> 10) & 31;
      const m = h & 1023;
      return (h & 0x8000 ? -1 : 1) * (e === 0 ? m * 2 ** -24 : (1 + m / 1024) * 2 ** (e - 15));
    };
    const values = new Float32Array(W * H);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) values[y * W + x] = f16((words[(1 * H + y) * rowWords + (x >> 1)]! >>> (x & 1 ? 16 : 0)) & 0xffff);
    const opts: ZonalOptions = { regions: R, bins: 128, channels: [{}] };
    const tile: ZonalTile = {
      labels: { buffer: upload(labels, "zonal-test:l"), width: W, height: H, dtype: "u32" },
      channels: [{ plane: residentRasterPlane(resident, 1, W, H) }],
    };
    expect(diff(await zonalHistogramGpu(tile, opts, device), zonalHistogramCpu(labels, [{ values }], opts))).toBeUndefined();
  });

  it("tiled driver over a pyramid level: label raster and polygons (overlapping, layered), exact table", async () => {
    const R = 5;
    const { labels, a, b } = scene(6, R);
    const ms = axisAlignedMultiscale({ voxelDims0: [W, H, 1], chunkShape: [32, 16, 1], levelCount: 1 });
    const fromArray = (img: ArrayLike<number>): { ms: typeof ms; loader: Loader<Tile>; cache: TileCache<Tile> } => ({
      ms,
      cache: new TileCache<Tile>({ maxBytes: 1 << 24 }),
      loader: {
        async getChunk(id: ChunkId): Promise<Tile> {
          const [w, h] = chunkVoxelExtent(ms, id);
          const data = new Float32Array(w * h);
          for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[y * w + x] = img[(id.y * 16 + y) * W + id.x * 32 + x]!;
          return { id, dims: [w, h, 1], element: { kind: "scalar" }, dtype: "f32", data };
        },
      },
    });
    const support = Float32Array.from(a, (v) => (v === 0 ? 0 : 1));
    const qs = [0.1, 0.25, 0.5, 0.75, 0.9];
    const res = await zonalStatsTiled(
      {
        level: 0,
        regions: { kind: "labels", source: fromArray(labels), regions: R },
        channels: [
          { source: fromArray(a), support: fromArray(support), name: "a" },
          { source: fromArray(b), name: "b", range: [-5, 5] },
        ],
        bins: 64,
        quantiles: qs,
        exact: true,
      },
      device,
    );
    const opts: ZonalOptions = { regions: R, bins: 64, channels: [{}, { range: [-5, 5] }] };
    const data: ZonalChannelData[] = [{ values: a, support }, { values: b }];
    const cpu = zonalHistogramCpu(labels, data, opts);
    expect(diff(res.histogram, cpu, 1e-6)).toBeUndefined();
    const plan = zonalExactPlan(cpu, qs);
    expect(mismatch(res.exact!, zonalExactQuantiles(cpu, plan, zonalCollectCpu(labels, data, opts, cpu, plan)))).toBeUndefined();
    expect(res.table.columns.get("a_median")!.every((v) => Number.isFinite(v))).toBe(true);

    // Overlapping polygons, in level-0 array space; polygon k's stats = the golden over its own mask.
    const sq = (x: number, y: number, s: number) =>
      polygonDomain([
        [x, y],
        [x + s, y],
        [x + s, y + s],
        [x, y + s],
      ]);
    const polys = [sq(3.2, 2.7, 40), sq(23.1, 12.4, 40), sq(50.5, 30.3, 25.6), sq(10.1, 40.2, 15)];
    const pres = await zonalStatsTiled(
      { level: 0, regions: { kind: "polygons", polygons: polys }, channels: [{ source: fromArray(b), range: [-5, 5] }], bins: 64 },
      device,
    );
    expect(pres.tiles).toBeGreaterThan(1);
    const per = polys.map((p) =>
      zonalHistogramCpu(rasterisePolygonsCpu([p], { width: W, height: H }), [{ values: b }], {
        regions: 1,
        bins: 64,
        channels: [{ range: [-5, 5] }],
      }),
    );
    const counts = new Uint32Array(polys.length * 64);
    per.forEach((h, k) => counts.set(h.counts, k * 64));
    expect(mismatch(pres.histogram.counts, counts)).toBeUndefined();
  });
});
