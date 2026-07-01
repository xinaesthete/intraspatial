import { describe, it, expect } from "vitest";
import { getDevice } from "../device";
import { DancerGpuSim, type DancerGpuParams } from "./dancerGpu";
import {
  INTEGRATE_DEFAULTS,
  readBodyState,
  seedSwarmBody,
  tapBlock,
  writeBodyState,
  integrateBody,
  type BodyState,
} from "./body";
import {
  cohereForce,
  constrainForce,
  orbitForce,
  partnerOrbitForce,
  separateForce,
  solenoidForce,
  swimForce,
  vortexForce,
} from "./forces";
import { figureAt, figureTargetVel, partnerIndex } from "./figures";
import { add, readVec3, scale, sub, ZERO3, type Vec3 } from "./vec3";

// The CPU golden — the same step as the app's DancerSim, assembled from the shared
// src/gpu/sim primitives (the app's sim.ts is a docs-site file, not importable here, but it
// is a thin wrapper over exactly these calls).
const DEFAULTS: DancerGpuParams = {
  constrain: 0.5,
  cohere: 0.4,
  cohereRadius: 3,
  separate: 0.6,
  separateRadius: 1.4,
  orbit: 0.35,
  swim: 0,
  vortex: 0,
  solenoid: 0,
  partner: 0.5,
  partnerOffset: 1,
  caller: 1,
  callerGain: 0.09,
  callerSpeed: 0.6,
  period: 480,
  callerSeed: 0,
  timeFactor: 0.2,
  jerkLimit: 0.05,
  linDamp: 0.96,
  angDamp: 0.9,
  speedLimit: 1.2,
  face: 0.5,
  maxRadius: 40,
};

function cpuStep(body: Float32Array, n: number, frame: number, p: DancerGpuParams): Float32Array {
  const posBuf = tapBlock(body, n, "pos");
  const { figure, figureIndex } = figureAt(frame, p.period, p.callerSeed);
  const next = new Float32Array(body.length);
  const integ = {
    ...INTEGRATE_DEFAULTS,
    timeFactor: p.timeFactor,
    jerkLimit: p.jerkLimit,
    linDamp: p.linDamp,
    angDamp: p.angDamp,
    speedLimit: p.speedLimit,
    face: p.face,
    maxRadius: p.maxRadius,
  };
  for (let i = 0; i < n; i++) {
    const s: BodyState = readBodyState(body, i, n);
    let f: Vec3 = ZERO3;
    if (p.constrain > 0) f = add(f, constrainForce(s.pos, p.constrain));
    if (p.cohere > 0) f = add(f, cohereForce(i, posBuf, n, p.cohere, p.cohereRadius));
    if (p.separate > 0) f = add(f, separateForce(i, posBuf, n, p.separate, p.separateRadius));
    if (p.orbit > 0) f = add(f, orbitForce(s.pos, s.vel, p.orbit));
    if (p.swim > 0) f = add(f, swimForce(s.pos, p.swim));
    if (p.vortex > 0) f = add(f, vortexForce(s.pos, p.vortex));
    if (p.solenoid > 0) f = add(f, solenoidForce(s.pos, p.solenoid));
    if (p.partner > 0) {
      const partner = readVec3(posBuf, (i + Math.max(1, Math.round(p.partnerOffset))) % n);
      f = add(f, partnerOrbitForce(s.pos, s.vel, partner, p.partner));
    }
    if (p.caller > 0) {
      const partner = readVec3(posBuf, partnerIndex(i, figureIndex, n));
      const target = figureTargetVel(figure, s.pos, i, partner, p.callerSpeed);
      f = add(f, scale(sub(target, s.vel), p.callerGain * p.caller));
    }
    writeBodyState(next, i, n, integrateBody(s, f, ZERO3, integ));
  }
  return next;
}

const maxAbsDiff = (a: Float32Array, b: Float32Array): number => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs((a[i] ?? 0) - (b[i] ?? 0)));
  return m;
};

// One instance, ONE readback. (Dawn-under-Node segfaults its worker once a handful of `.read()`
// staging buffers accumulate in a run — a known readback/teardown quirk of the Node binding,
// not the kernel; the browser, the real target, has no such issue.) A single step + single
// readBlocks proves everything: the seed carries per-agent orientation, and the first step's
// face-torque makes angVel non-zero and moves angPos — so the full 5-block CPU-golden match
// and the orientation/angular-momentum checks all read from the same snapshot.
describe("DancerGpuSim — GPU kernel vs CPU golden", () => {
  it("matches the CPU golden across all 5 blocks, with orientation + angular momentum", async () => {
    const device = await getDevice();
    // Kept small: Dawn's Node binding segfaults on process-exit teardown once a process does
    // "enough" GPU work (see vitest.gpu.config.ts). The correctness match is n-independent, so
    // a small swarm still exercises every force (incl. the neighbour loop) and the integrator.
    const n = 16;
    const seed = 7;

    const sim = new DancerGpuSim(device, n, seed, DEFAULTS);
    sim.init();
    sim.step();
    const gpu = await sim.readBlocks();

    const cpu = cpuStep(seedSwarmBody(n, seed), n, 0, DEFAULTS);
    // f32 (GPU) vs f64 (CPU) with ~128-term neighbour sums ⇒ a small tolerance. All five
    // rigid-body blocks, so the angular integration is checked, not just the linear.
    expect(maxAbsDiff(gpu.pos, tapBlock(cpu, n, "pos"))).toBeLessThan(2e-3);
    expect(maxAbsDiff(gpu.vel, tapBlock(cpu, n, "vel"))).toBeLessThan(2e-3);
    expect(maxAbsDiff(gpu.accel, tapBlock(cpu, n, "accel"))).toBeLessThan(2e-3);
    expect(maxAbsDiff(gpu.angPos, tapBlock(cpu, n, "angPos"))).toBeLessThan(2e-3);
    expect(maxAbsDiff(gpu.angVel, tapBlock(cpu, n, "angVel"))).toBeLessThan(2e-3);

    // orientation VARIES across agents — not one shared facing (the all-cones-face-one-way bug)
    let minA = Infinity;
    let maxA = -Infinity;
    for (let i = 0; i < n; i++) {
      const a = Math.hypot(gpu.angPos[i * 3] ?? 0, gpu.angPos[i * 3 + 1] ?? 0, gpu.angPos[i * 3 + 2] ?? 0);
      minA = Math.min(minA, a);
      maxA = Math.max(maxA, a);
    }
    expect(maxA - minA).toBeGreaterThan(1e-3);

    // angular momentum is actually present (the face-torque drove angVel non-zero from a zero seed)
    let maxAngVel = 0;
    for (let i = 0; i < gpu.angVel.length; i++) maxAngVel = Math.max(maxAngVel, Math.abs(gpu.angVel[i] ?? 0));
    expect(maxAngVel).toBeGreaterThan(1e-4);

    // finite, bounded, moving
    let finite = true;
    let maxR = 0;
    let maxSpeed = 0;
    for (let i = 0; i < n; i++) {
      const px = gpu.pos[i * 3] ?? 0;
      const py = gpu.pos[i * 3 + 1] ?? 0;
      const pz = gpu.pos[i * 3 + 2] ?? 0;
      if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) finite = false;
      maxR = Math.max(maxR, Math.hypot(px, py, pz));
      maxSpeed = Math.max(maxSpeed, Math.hypot(gpu.vel[i * 3] ?? 0, gpu.vel[i * 3 + 1] ?? 0, gpu.vel[i * 3 + 2] ?? 0));
    }
    expect(finite).toBe(true);
    expect(maxR).toBeLessThan(DEFAULTS.maxRadius + 1);
    expect(maxSpeed).toBeGreaterThan(1e-3);
    expect(maxSpeed).toBeLessThan(DEFAULTS.speedLimit + 1e-2);
  });
});
