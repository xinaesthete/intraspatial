import { describe, expect, it } from "vite-plus/test";
import { getDevice, readBackBytes, writeView } from "./device";
import { textureToBuffer } from "./graph/textureBridge";
import { adoptDevice } from "./interop/adoptDevice";
import { splatDensityToTexture } from "./spatial/splatDensity";

// Concurrent f16 splats share the f32 accumulator (a scratch texture) and the splat's uniforms.
// They used to await mid-encode, between taking the accumulator and submitting. A concurrent call
// of a different size that landed in that await replaced the slot and destroyed the texture, and
// overwrote the uniforms. Measured before the fix: four calls at alternating sizes, three of them
// came back empty, with Dawn reporting "Destroyed texture … used in a submit".

const XY = new Float32Array([3, 3, 5, 6, 7, 2, 8, 8, 4.5, 7.5]);
const N = XY.length / 2;
const BBOX: [number, number, number, number] = [0, 0, 10, 10];
const SIZES = [16, 64, 16, 64];

describe("concurrent f16 splats", () => {
  it("each match the same splat run alone", async () => {
    const device = await getDevice();
    const backend = adoptDevice(device, "scratchTexture:test");
    const pts = device.createBuffer({ size: XY.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    writeView(device.queue, pts, XY);

    const splat = async (size: number) => {
      const tex = await backend.leaseTexture(size, size, "r16float");
      return { tex, done: splatDensityToTexture(pts, N, tex.texture, { width: size, height: size, sigma: 0.8, bbox: BBOX, device }) };
    };
    const read = async (tex: Awaited<ReturnType<typeof splat>>["tex"]) => {
      const out = device.createBuffer({ size: tex.width * tex.height * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      await textureToBuffer(tex, out, device);
      return new Float32Array(await readBackBytes(device, "scratchTexture:test", out, 0, tex.width * tex.height * 4));
    };

    // One at a time first: the reference for each size.
    const alone: Float32Array[] = [];
    for (const size of SIZES) {
      const s = await splat(size);
      await s.done;
      alone.push(await read(s.tex));
      backend.releaseTexture(s.tex);
    }

    // All at once. Leases are taken first so the splats themselves are what overlap.
    const runs = await Promise.all(SIZES.map(splat));
    await Promise.all(runs.map((r) => r.done));
    let worst = 0;
    let emptiest = Number.POSITIVE_INFINITY;
    for (let i = 0; i < runs.length; i++) {
      const got = await read(runs[i]!.tex);
      const want = alone[i]!;
      let mass = 0;
      for (let t = 0; t < want.length; t++) {
        worst = Math.max(worst, Math.abs(got[t]! - want[t]!));
        mass += got[t]!;
      }
      emptiest = Math.min(emptiest, mass);
      backend.releaseTexture(runs[i]!.tex);
    }
    expect(emptiest, "a concurrent splat came back empty").toBeGreaterThan(0);
    expect(worst).toBe(0);
  });
});
