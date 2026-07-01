// The dancer simulation for the standalone artefact — a direct CPU loop over the SAME
// force/integrator/figure math the composer's building-block ops wrap (src/gpu/sim/*).
// Running it directly (rather than through the op-graph executor) keeps the Astro bundle
// free of the runtime's GPU-device imports, and the loop is trivially cheap at dancer
// swarm sizes. The `DancerParams` mirror the ops' ParamSpecs, so the Mutator (src/evo)
// breeds one object that drives this and (later) the GPU/TSL port alike.
import {
  BODY_BLOCK_COUNT,
  INTEGRATE_DEFAULTS,
  integrateBody,
  readBodyState,
  seedSwarmBody,
  tapBlock,
  writeBodyState,
} from "../../../../src/gpu/sim/body";
import {
  cohereForce,
  constrainForce,
  orbitForce,
  partnerOrbitForce,
  separateForce,
  solenoidForce,
  swimForce,
  vortexForce,
} from "../../../../src/gpu/sim/forces";
import { figureAt, figureTargetVel, partnerIndex } from "../../../../src/gpu/sim/figures";
import { add, readVec3, scale, sub, ZERO3, type Vec3 } from "../../../../src/gpu/sim/vec3";

export interface DancerParams {
  // force strengths (0 disables)
  constrain: number;
  cohere: number;
  cohereRadius: number;
  separate: number;
  separateRadius: number;
  orbit: number;
  swim: number;
  vortex: number;
  solenoid: number;
  partner: number;
  partnerOffset: number;
  // caller
  caller: number; // tightness (0 disables)
  period: number;
  callerSeed: number;
  callerGain: number;
  callerSpeed: number;
  // integrator
  timeFactor: number;
  jerkLimit: number;
  linDamp: number;
  angDamp: number;
  speedLimit: number;
  face: number;
  maxRadius: number;
}

export const DEFAULT_DANCER_PARAMS: DancerParams = {
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
  period: 480, // frames per figure (~8s at 60fps) — a slower, more deliberate progression
  callerSeed: 0,
  callerGain: 0.09,
  callerSpeed: 0.6,
  timeFactor: 0.2,
  jerkLimit: 0.05,
  linDamp: 0.96,
  angDamp: 0.9,
  speedLimit: 1.2,
  face: 0.5,
  maxRadius: 40,
};

export class DancerSim {
  readonly n: number;
  /** Body field, length 5·N·3 (blocks pos|vel|accel|angPos|angVel). */
  private body: Float32Array;
  params: DancerParams;
  frame = 0;

  constructor(n: number, seed = 1, params: Partial<DancerParams> = {}) {
    this.n = n;
    this.body = seedSwarmBody(n, seed);
    this.params = { ...DEFAULT_DANCER_PARAMS, ...params };
  }

  reset(seed: number): void {
    this.body = seedSwarmBody(this.n, seed);
    this.frame = 0;
  }

  /** Advance one frame. */
  step(): void {
    const { n } = this;
    const p = this.params;
    const posBuf = tapBlock(this.body, n, "pos");
    const velBuf = tapBlock(this.body, n, "vel");
    const { figure, figureIndex } = figureAt(this.frame, p.period, p.callerSeed);
    const next = new Float32Array(this.body.length);
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
      const s = readBodyState(this.body, i, n);
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
    this.body = next;
    this.frame++;
  }

  /** Positions as [x,y,z]×N (a view for the renderer). */
  positions(): Float32Array {
    return tapBlock(this.body, this.n, "pos");
  }

  /** Orientations as angle-axis rotation vectors [x,y,z]×N (renderer converts to quats). */
  orientations(): Float32Array {
    return tapBlock(this.body, this.n, "angPos");
  }

  /** Speeds per agent (for colour-by-motion). */
  speeds(): Float32Array {
    const vel = tapBlock(this.body, this.n, "vel");
    const out = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) out[i] = Math.hypot(vel[i * 3] ?? 0, vel[i * 3 + 1] ?? 0, vel[i * 3 + 2] ?? 0);
    return out;
  }

  currentFigure(): string {
    return figureAt(this.frame, this.params.period, this.params.callerSeed).figure;
  }
}

export { BODY_BLOCK_COUNT };
