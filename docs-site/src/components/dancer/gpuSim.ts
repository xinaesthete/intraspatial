// GPU-resident dancer simulation — three.js WebGPU (TSL) compute. The swarm state lives in
// storage buffers; a compute kernel evaluates the DANCERL force influences + the rigid-body
// integrator on the GPU (the O(N²) neighbour forces are the reason to be here), so large N
// stays interactive. Mirrors the CPU DancerSim math (src/gpu/sim/forces.ts) — the CPU
// version stays the golden reference. Reference: CSynth (GPU force sim + matrix).
//
// Parameters are baked into the kernel as literals and the step kernel is REBUILT when the
// params or the called figure change (both rare — a breed, or a figure transition every few
// seconds). This sidesteps a live-uniform binding quirk in this three.js build (setting a
// compute uniform's `.value` didn't propagate; literals do). Robust NaN idiom throughout:
// never divide by a possibly-zero value (clamp the denominator) and gate with multiplicative
// masks rather than `If` (TSL can lower small `If`s to `select`, which evaluates both arms).
// Baking-gotcha: an `undefined` param baked via `float(x)` emits `NaN.0`, which is not valid
// WGSL and fails pipeline compilation — always bake from a typed, defined source (this is a
// compile-time TS error too; run the typechecker before blaming the shader).
//
// Staging: positions/velocities are read back each frame to feed the existing renderer
// (trails, matrix, hover). A later pass can render straight from the buffers (zero readback).
import type { WebGPURenderer } from "three/webgpu";
import { Fn, If, Loop, float, hash, instanceIndex, instancedArray, vec3 } from "three/tsl";
import { DEFAULT_DANCER_PARAMS, type DancerParams } from "./sim";
import { INTEGRATE_DEFAULTS } from "../../../../src/gpu/sim/body";
import { figureAt, type Figure } from "../../../../src/gpu/sim/figures";

const SHELL = 4.5;

/* TSL is an untyped node DSL; keep the glue loosely typed but contained. */
/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

export class GpuDancerSim {
  readonly n: number;
  params: DancerParams;
  frame = 0;

  private readonly renderer: WebGPURenderer;
  private readonly seed: number;
  private pos!: Node;
  private snap!: Node;
  private vel!: Node;
  private accel!: Node;
  private copyKernel!: Node;
  private stepKernel!: Node;

  private posCPU: Float32Array;
  private velCPU: Float32Array;
  private zeros: Float32Array;
  private stride = 4;
  private ready = false;

  private builtFigure: Figure | null = null;
  private builtParamsKey = "";

  constructor(renderer: WebGPURenderer, n: number, seed = 1, params: Partial<DancerParams> = {}) {
    this.renderer = renderer;
    this.n = n;
    this.seed = seed;
    this.params = { ...DEFAULT_DANCER_PARAMS, ...params };
    this.posCPU = new Float32Array(n * 3);
    this.velCPU = new Float32Array(n * 3);
    this.zeros = new Float32Array(n * 3);
  }

  async init(): Promise<void> {
    const n = this.n;
    this.pos = instancedArray(n, "vec3");
    this.snap = instancedArray(n, "vec3");
    this.vel = instancedArray(n, "vec3");
    this.accel = instancedArray(n, "vec3");
    const S = this.seed >>> 0;

    // seed — a jittered spherical shell + small tangential kick
    const seedKernel = Fn(() => {
      const i = instanceIndex;
      const u1 = hash(i.add(S)).mul(2).sub(1);
      const phi = hash(i.add(S + n)).mul(Math.PI * 2);
      const rr = float(SHELL).mul(float(0.7).add(hash(i.add(S + 2 * n)).mul(0.6)));
      const s = float(1).sub(u1.mul(u1)).max(0).sqrt();
      const px = rr.mul(s).mul(phi.cos());
      const py = rr.mul(u1);
      const pz = rr.mul(s).mul(phi.sin());
      this.pos.element(i).assign(vec3(px, py, pz));
      this.vel.element(i).assign(vec3(pz.mul(-0.03), 0, px.mul(0.03)));
      this.accel.element(i).assign(vec3(0, 0, 0));
    })().compute(n);

    this.copyKernel = Fn(() => {
      this.snap.element(instanceIndex).assign(this.pos.element(instanceIndex));
    })().compute(n);

    this.rebuildStep(figureAt(0, this.params.period, this.params.callerSeed).figure);

    await this.renderer.computeAsync(seedKernel);
    await this.readback();
    this.ready = true;
  }

  private paramsKey(): string {
    const p = this.params;
    return [
      p.constrain, p.cohere, p.cohereRadius, p.separate, p.separateRadius, p.orbit, p.swim,
      p.caller, p.callerGain, p.callerSpeed, p.timeFactor, p.jerkLimit, p.linDamp, p.speedLimit, p.maxRadius,
    ].join(",");
  }

  /** (Re)build the step kernel with the current params + figure baked as literals. All TSL
   *  nodes are constructed INSIDE the Fn (they belong to its build context). */
  private rebuildStep(figure: Figure): void {
    const n = this.n;
    const p = this.params;

    this.stepKernel = Fn(() => {
      const L = (v: number): Node => float(v);
      const UP = vec3(0, 1, 0);
      const spd = L(p.callerSpeed);
      const safe = (u: Node): Node => u.div(u.length().max(1e-4));
      // target velocity for the baked figure (only proven-safe ops)
      const targetVel = (pos: Node, idx: Node): Node => {
        switch (figure) {
          case "gather":
            return safe(pos).mul(L(-p.callerSpeed)); // negative speed baked as a literal
          case "scatter":
            return safe(pos).mul(spd);
          case "mill": {
            const ang = hash(idx.add(7)).mul(Math.PI * 2);
            return vec3(ang.cos(), 0, ang.sin()).mul(L(p.callerSpeed * 0.4));
          }
          default: // swing / grandChain — a global orbit about the y-axis
            return safe(UP.cross(pos)).mul(spd);
        }
      };

      const i = instanceIndex;
      const pos = this.snap.element(i).toVar();
      const v = this.vel.element(i).toVar();
      const a0 = this.accel.element(i);
      const f = vec3(0, 0, 0).toVar();

      // constrain — cubic containment toward origin
      const r = pos.length();
      f.subAssign(pos.mul(L(p.constrain).mul(0.012).mul(float(1).add(r.mul(r).mul(0.04)))));

      // orbit — (p×v)×p (clamped denominator)
      const t = pos.cross(v).cross(pos);
      f.addAssign(t.div(t.length().max(1e-4)).mul(L(p.orbit).mul(0.05)));

      // swim — outward
      f.addAssign(pos.div(r.max(1e-4)).mul(L(p.swim).mul(0.02)));

      // neighbours — separation + cohesion
      const center = vec3(0, 0, 0).toVar();
      const cnt = float(0).toVar();
      const sepR2 = L(p.separateRadius * p.separateRadius).max(1e-4);
      const cohR2 = L(p.cohereRadius * p.cohereRadius);
      Loop(n, ({ i: j }: { i: Node }) => {
        If(j.notEqual(i), () => {
          const q = this.snap.element(j);
          const d = pos.sub(q);
          const d2 = d.dot(d);
          const sepW = float(1).sub(d2.div(sepR2)).max(0); // 0 outside radius
          f.addAssign(d.div(d2.sqrt().max(1e-4)).mul(L(p.separate).mul(0.06)).mul(sepW));
          If(d2.lessThan(cohR2), () => {
            center.addAssign(q);
            cnt.addAssign(1);
          });
        });
      });
      f.addAssign(center.div(cnt.max(1)).sub(pos).mul(L(p.cohere).mul(0.012)).mul(cnt.min(1)));

      // caller — accelerate toward the called figure's state of motion
      if (p.caller > 0) {
        const target = targetVel(pos, i);
        f.addAssign(target.sub(v).mul(L(p.callerGain * p.caller)));
      }

      // integrate — jerk-limited accel, damping, speed cap, containment (clamped denoms)
      const df = f.sub(a0);
      const a = a0.add(df.mul(L(p.jerkLimit).div(df.length().max(1e-6)).min(1))).toVar();
      const tf = L(p.timeFactor * INTEGRATE_DEFAULTS.dt);
      const vv = v.add(a.mul(tf)).mul(L(p.linDamp)).toVar();
      vv.assign(vv.mul(L(p.speedLimit).div(vv.length().max(1e-6)).min(1)));
      const np = pos.add(vv.mul(tf)).toVar();
      np.assign(np.mul(L(p.maxRadius).div(np.length().max(1e-6)).min(1)));

      this.pos.element(i).assign(np);
      this.vel.element(i).assign(vv);
      this.accel.element(i).assign(a);
    })().compute(n);

    this.builtFigure = figure;
    this.builtParamsKey = this.paramsKey();
  }

  private async readback(): Promise<void> {
    const n = this.n;
    const posAB = (await this.renderer.getArrayBufferAsync(this.pos.value)) as ArrayBuffer;
    const velAB = (await this.renderer.getArrayBufferAsync(this.vel.value)) as ArrayBuffer;
    const pf = new Float32Array(posAB);
    const vf = new Float32Array(velAB);
    this.stride = Math.max(3, Math.round(pf.length / n));
    const st = this.stride;
    for (let i = 0; i < n; i++) {
      this.posCPU[i * 3] = pf[i * st] ?? 0;
      this.posCPU[i * 3 + 1] = pf[i * st + 1] ?? 0;
      this.posCPU[i * 3 + 2] = pf[i * st + 2] ?? 0;
      this.velCPU[i * 3] = vf[i * st] ?? 0;
      this.velCPU[i * 3 + 1] = vf[i * st + 1] ?? 0;
      this.velCPU[i * 3 + 2] = vf[i * st + 2] ?? 0;
    }
  }

  async step(): Promise<void> {
    if (!this.ready) return;
    const figure = figureAt(this.frame, this.params.period, this.params.callerSeed).figure;
    if (figure !== this.builtFigure || this.paramsKey() !== this.builtParamsKey) {
      this.rebuildStep(figure);
    }
    await this.renderer.computeAsync(this.copyKernel);
    await this.renderer.computeAsync(this.stepKernel);
    await this.readback();
    this.frame++;
  }

  positions(): Float32Array {
    return this.posCPU;
  }
  orientations(): Float32Array {
    return this.zeros;
  }
  speeds(): Float32Array {
    const n = this.n;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.hypot(this.velCPU[i * 3] ?? 0, this.velCPU[i * 3 + 1] ?? 0, this.velCPU[i * 3 + 2] ?? 0);
    return out;
  }
  currentFigure(): string {
    return figureAt(this.frame, this.params.period, this.params.callerSeed).figure;
  }
}
