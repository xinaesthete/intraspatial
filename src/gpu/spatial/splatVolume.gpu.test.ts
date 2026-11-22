import { describe, expect, it } from "vite-plus/test";
import { getDevice, readBackBytes } from "../device";
import { slabLevel, splatVolumeResident, volumeBytes } from "./splatVolume";

describe("splatVolume", () => {
  it("matches a brute-force sum on the host, voxel for voxel, direct and indexed", async () => {
    const device = await getDevice();
    const dims = [9, 7, 5] as const;
    const origin = [-2, -1.5, -1] as const;
    const voxel = 0.5;
    const radius = 1.3;
    const stride = 4;
    const n = 12;
    // A deterministic scatter through and a little beyond the field; the 4th float is ignored.
    const pts = new Float32Array(n * stride);
    for (let i = 0; i < n; i++) pts.set([Math.sin(i * 1.7) * 2.2, Math.cos(i * 2.3) * 1.6, Math.sin(i * 0.9 + 1) * 1.2, 99], i * stride);

    const points = device.createBuffer({
      label: "test:points",
      size: pts.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(points, 0, pts);
    const field = device.createBuffer({
      label: "test:field",
      size: volumeBytes(dims),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    await splatVolumeResident(points, n, field, { dims, origin, voxel, radius, stride, weight: 2, index: false });
    const got = new Float32Array(await readBackBytes(device, "test:field", field, 0, volumeBytes(dims)));
    // The indexed form is the same sum reached another way, points outside the box included.
    const fieldIndexed = device.createBuffer({
      label: "test:field-indexed",
      size: volumeBytes(dims),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    await splatVolumeResident(points, n, fieldIndexed, { dims, origin, voxel, radius, stride, weight: 2, index: true });
    const indexed = new Float32Array(await readBackBytes(device, "test:field-indexed", fieldIndexed, 0, volumeBytes(dims)));
    let apart = 0;
    for (let v = 0; v < got.length; v++) apart = Math.max(apart, Math.abs(got[v]! - indexed[v]!));

    let worst = 0;
    let total = 0;
    for (let z = 0; z < dims[2]; z++) {
      for (let y = 0; y < dims[1]; y++) {
        for (let x = 0; x < dims[0]; x++) {
          const c = [origin[0] + (x + 0.5) * voxel, origin[1] + (y + 0.5) * voxel, origin[2] + (z + 0.5) * voxel];
          let sum = 0;
          for (let i = 0; i < n; i++) {
            const d2 = (pts[i * 4]! - c[0]!) ** 2 + (pts[i * 4 + 1]! - c[1]!) ** 2 + (pts[i * 4 + 2]! - c[2]!) ** 2;
            const q = Math.max(1 - d2 / (radius * radius), 0);
            sum += q * q * 2;
          }
          total += sum;
          worst = Math.max(worst, Math.abs(sum - got[x + dims[0] * (y + dims[1] * z)]!));
        }
      }
    }
    expect(total).toBeGreaterThan(10);
    expect(worst).toBeLessThan(1e-4);
    expect(apart).toBeLessThan(1e-5);
  });

  it("indexes a large cloud: sampled voxels equal the host sum", async () => {
    const device = await getDevice();
    const dims = [64, 64, 32] as const;
    const origin = [0, 0, 0] as const;
    const voxel = 1;
    const radius = 2.5;
    const n = 200_000;
    // A cheap deterministic scatter, denser toward one corner.
    const pts = new Float32Array(n * 3);
    let seed = 12345;
    const next = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    for (let i = 0; i < n; i++) pts.set([next() ** 2 * 64, next() * 64, next() * 32], i * 3);
    const points = device.createBuffer({
      label: "test:cloud",
      size: pts.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(points, 0, pts);
    const field = device.createBuffer({
      label: "test:cloud-field",
      size: volumeBytes(dims),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    await splatVolumeResident(points, n, field, { dims, origin, voxel, radius, stride: 3 });
    const got = new Float32Array(await readBackBytes(device, "test:cloud-field", field, 0, volumeBytes(dims)));

    let worst = 0;
    for (let s = 0; s < 40; s++) {
      const [x, y, z] = [Math.floor(next() * 64), Math.floor(next() * 64), Math.floor(next() * 32)];
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d2 = (pts[i * 3]! - (x + 0.5)) ** 2 + (pts[i * 3 + 1]! - (y + 0.5)) ** 2 + (pts[i * 3 + 2]! - (z + 0.5)) ** 2;
        const q = Math.max(1 - d2 / (radius * radius), 0);
        sum += q * q;
      }
      // Relative: a dense voxel sums a few hundred terms in f32.
      worst = Math.max(worst, Math.abs(sum - got[x + 64 * (y + 64 * z)]!) / Math.max(1, sum));
    }
    expect(worst).toBeLessThan(1e-4);
  });

  it("weights each point by its own value, direct and indexed", async () => {
    const device = await getDevice();
    const dims = [8, 8, 4] as const;
    const o = { dims, origin: [0, 0, 0] as const, voxel: 1, radius: 2, stride: 3 };
    const n = 300;
    const pts = new Float32Array(n * 3);
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      pts.set([(Math.sin(i * 1.3) + 1) * 4, (Math.cos(i * 0.7) + 1) * 4, (Math.sin(i * 2.9) + 1) * 2], i * 3);
      // A third of the points are left out altogether.
      w[i] = i % 3 === 0 ? 0 : 0.25 + (i % 7);
    }
    const make = (label: string, data: Float32Array) => {
      const b = device.createBuffer({ label, size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, data);
      return b;
    };
    const points = make("test:weighted-points", pts);
    const weights = make("test:weights", w);
    const fields: Float32Array[] = [];
    for (const index of [false, true]) {
      const field = device.createBuffer({
        label: "test:weighted-field",
        size: volumeBytes(dims),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      await splatVolumeResident(points, n, field, { ...o, weights, index });
      fields.push(new Float32Array(await readBackBytes(device, "test:weighted-field", field, 0, volumeBytes(dims))));
    }
    let worst = 0;
    let total = 0;
    for (let v = 0; v < fields[0]!.length; v++) {
      const c = [(v % 8) + 0.5, (Math.floor(v / 8) % 8) + 0.5, Math.floor(v / 64) + 0.5];
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d2 = (pts[i * 3]! - c[0]!) ** 2 + (pts[i * 3 + 1]! - c[1]!) ** 2 + (pts[i * 3 + 2]! - c[2]!) ** 2;
        const q = Math.max(1 - d2 / 4, 0);
        sum += q * q * w[i]!;
      }
      total += sum;
      worst = Math.max(worst, Math.abs(sum - fields[0]![v]!) / Math.max(1, sum), Math.abs(sum - fields[1]![v]!) / Math.max(1, sum));
    }
    expect(total).toBeGreaterThan(10);
    expect(worst).toBeLessThan(1e-4);
  });
  it("gives slabLevel's value at the middle of an even slab of points", async () => {
    const device = await getDevice();
    const radius = 3;
    const spacing = 0.25;
    // One voxel, centred on the slab's middle.
    const o = { dims: [1, 1, 1] as const, origin: [-0.5, -0.5, -0.5] as const, voxel: 1, radius, stride: 3 };
    const across = Math.ceil((2 * radius + 2) / spacing);
    const slab = async (thickness: number, layers: number): Promise<number> => {
      const n = across * across * layers;
      const pts = new Float32Array(n * 3);
      let at = 0;
      for (let k = 0; k < layers; k++) {
        // Midpoint layers through the depth, as an even spread of molecules sits.
        const z = layers === 1 ? 0 : -thickness / 2 + ((k + 0.5) * thickness) / layers;
        for (let j = 0; j < across; j++) {
          for (let i = 0; i < across; i++) {
            pts.set([(i + 0.5 - across / 2) * spacing, (j + 0.5 - across / 2) * spacing, z], at);
            at += 3;
          }
        }
      }
      const points = device.createBuffer({
        label: "test:slab",
        size: pts.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(points, 0, pts);
      const field = device.createBuffer({ label: "test:slab-field", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      await splatVolumeResident(points, n, field, { ...o, index: true });
      const got = new Float32Array(await readBackBytes(device, "test:slab-field", field, 0, 4))[0]!;
      points.destroy();
      field.destroy();
      return got / slabLevel(layers / spacing ** 2, radius, thickness);
    };
    // A plane, a slab thinner than the reach, and one deeper than the whole sphere.
    const ratios = [await slab(0, 1), await slab(2, 8), await slab(10, 40)];
    expect(Math.max(...ratios.map((r) => Math.abs(r - 1)))).toBeLessThan(5e-3);
  });
});
