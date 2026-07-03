import { describe, it, expect } from "vitest";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { getDevice } from "../device";
import { DancerGpuSim, type DancerGpuParams } from "./dancerGpu";
import { seedSwarmBody, tapBlock } from "./body";

const DEFAULTS: DancerGpuParams = {
  constrain: 0.5, cohere: 0.4, cohereRadius: 3, separate: 0.6, separateRadius: 1.4, orbit: 0.35,
  swim: 0, vortex: 0, solenoid: 0, partner: 0.5, partnerOffset: 1, caller: 1, callerGain: 0.09,
  callerSpeed: 0.6, period: 480, callerSeed: 0, timeFactor: 0.2, jerkLimit: 0.05, linDamp: 0.96,
  angDamp: 0.9, speedLimit: 1.2, face: 0.5, maxRadius: 40,
};

/** Column-major model matrix (16 floats) from translation + angle-axis rotation, scale 1 —
 *  the CPU reference for the render-bridge kernel (matches THREE.Matrix4.compose semantics). */
function composeCol(t: [number, number, number], r: [number, number, number]): number[] {
  const theta = Math.hypot(r[0], r[1], r[2]);
  let R: number[][];
  if (theta < 1e-9) {
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  } else {
    const kx = r[0] / theta, ky = r[1] / theta, kz = r[2] / theta;
    const c = Math.cos(theta), s = Math.sin(theta), oc = 1 - c;
    R = [
      [c + kx * kx * oc, kx * ky * oc - kz * s, kx * kz * oc + ky * s],
      [ky * kx * oc + kz * s, c + ky * ky * oc, ky * kz * oc - kx * s],
      [kz * kx * oc - ky * s, kz * ky * oc + kx * s, c + kz * kz * oc],
    ];
  }
  // column-major: [R col0 | R col1 | R col2 | translation]
  return [
    R[0]![0]!, R[1]![0]!, R[2]![0]!, 0,
    R[0]![1]!, R[1]![1]!, R[2]![1]!, 0,
    R[0]![2]!, R[1]![2]!, R[2]![2]!, 0,
    t[0], t[1], t[2], 1,
  ];
}

describe("DancerGpuSim render bridge — instance matrices", () => {
  it("writes column-major model matrices (translation + angle-axis rotation) matching the CPU compose", async () => {
    const device = await getDevice();
    const n = 12;
    const seed = 3;

    const sim = new DancerGpuSim(device, n, seed, DEFAULTS);
    sim.init();

    // a mat4 storage buffer standing in for three's StorageInstancedBufferAttribute
    const root = tgpu.initFromDevice({ device });
    const mtx = root.createBuffer(d.arrayOf(d.f32, 16 * n)).$usage("storage");
    sim.setMatrixTarget(root.unwrap(mtx), 1);
    sim.writeMatrices(); // compose from the seeded pose (no sim step)

    const got = (await root.createBuffer(d.arrayOf(d.f32, 16 * n), root.unwrap(mtx)).read()) as ArrayLike<number>;

    const body = seedSwarmBody(n, seed);
    const pos = tapBlock(body, n, "pos");
    const ang = tapBlock(body, n, "angPos");

    let maxErr = 0;
    for (let i = 0; i < n; i++) {
      const exp = composeCol(
        [pos[i * 3] ?? 0, pos[i * 3 + 1] ?? 0, pos[i * 3 + 2] ?? 0],
        [ang[i * 3] ?? 0, ang[i * 3 + 1] ?? 0, ang[i * 3 + 2] ?? 0],
      );
      for (let e = 0; e < 16; e++) maxErr = Math.max(maxErr, Math.abs((got[i * 16 + e] ?? 0) - (exp[e] ?? 0)));
    }
    expect(maxErr).toBeLessThan(1e-5);
  });
});
