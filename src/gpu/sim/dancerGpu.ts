// GPU-resident dancer — the DANCERL force set + rigid-body integrator + Ceilidh caller as
// ONE TypeGPU compute kernel, in the repo's own TGSL idiom (not three.js TSL). This is the
// single GPU source of truth, a faithful port of the CPU golden (forces.ts + body.ts +
// figures.ts) — verified against it in dancerGpu.gpu.test.ts. Because it's plain TypeGPU on
// a GPUDevice, it runs BOTH under Node (Dawn, for the golden test) AND in the browser on a
// device we adopt from three.js (interop/adoptDevice) — so the swarm state lives in the same
// device three renders from, and positions/orientations never leave the GPU (no readback in
// the render path — the fix for the camera-drag stutter).
//
// State is the full rigid body: 5 vec3 buffers [pos | vel | accel | angPos | angVel], plus a
// posSnap the neighbour forces read so every agent sees the same positions within a step
// (the GPU analogue of the CPU's read-old / write-new). Orientation is an angle-axis rotation
// vector (angPos), integrated from angular velocity — the same representation the renderer
// already consumes, so cones actually turn to face their travel and carry angular momentum.
import tgpu, { type StorageFlag, type TgpuBuffer, type UniformFlag } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { INTEGRATE_DEFAULTS, seedSwarmBody, tapBlock } from "./body";
import { figureAt, partnerIndex, type Figure } from "./figures";

const WG = 64;

const FIGURE_CODE: Record<Figure, number> = { swing: 0, grandChain: 1, gather: 2, scatter: 3, mill: 4 };

/** Params the kernel needs — a superset-compatible subset of the app's DancerParams (so the
 *  app passes its params object straight through). `dt` is not among them — it is the fixed
 *  canonical INTEGRATE_DEFAULTS.dt (its absence from DancerParams once baked NaN into the old
 *  TSL kernel; here dt is sourced, not read off the params). */
export interface DancerGpuParams {
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
  caller: number;
  callerGain: number;
  callerSpeed: number;
  period: number;
  callerSeed: number;
  timeFactor: number;
  jerkLimit: number;
  linDamp: number;
  angDamp: number;
  speedLimit: number;
  face: number;
  maxRadius: number;
}

const Params = d.struct({
  n: d.u32,
  figureCode: d.u32,
  callerPartnerStep: d.u32,
  partnerOffsetIdx: d.u32,
  constrain: d.f32,
  cohere: d.f32,
  cohereRadius: d.f32,
  separate: d.f32,
  separateRadius: d.f32,
  orbit: d.f32,
  swim: d.f32,
  vortex: d.f32,
  solenoid: d.f32,
  partnerStrength: d.f32,
  caller: d.f32,
  callerGain: d.f32,
  callerSpeed: d.f32,
  timeFactor: d.f32,
  jerkLimit: d.f32,
  linDamp: d.f32,
  angDamp: d.f32,
  speedLimit: d.f32,
  maxRadius: d.f32,
  face: d.f32,
  dt: d.f32,
});

const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  posSnap: { storage: (n: number) => d.arrayOf(d.vec3f, n), access: "mutable" },
  pos: { storage: (n: number) => d.arrayOf(d.vec3f, n), access: "mutable" },
  vel: { storage: (n: number) => d.arrayOf(d.vec3f, n), access: "mutable" },
  accel: { storage: (n: number) => d.arrayOf(d.vec3f, n), access: "mutable" },
  angPos: { storage: (n: number) => d.arrayOf(d.vec3f, n), access: "mutable" },
  angVel: { storage: (n: number) => d.arrayOf(d.vec3f, n), access: "mutable" },
  // 5·N packing of the five blocks for a SINGLE readback (Dawn-under-Node segfaults once a
  // handful of separate `.read()` staging buffers accumulate — one read sidesteps it).
  snapshot: { storage: (n: number) => d.arrayOf(d.vec3f, n), access: "mutable" },
});

// ── device-function helpers (mirror src/gpu/sim/vec3.ts + quat.ts) ────────────────────────

/** normalize with vec3.ts semantics: |v|<eps ⇒ fallback (default zero). */
const normOr = tgpu.fn([d.vec3f, d.vec3f], d.vec3f)((v, fb) => {
  "use gpu";
  const l = std.length(v);
  return std.select(std.mul(1 / std.max(l, 1e-9), v), fb, l < 1e-9);
});

const safeNorm = tgpu.fn([d.vec3f], d.vec3f)((v) => {
  "use gpu";
  return normOr(v, d.vec3f());
});


/** The forward (+z) axis of orientation angle-axis `r` — Rodrigues-rotate (0,0,1) by r.
 *  Matches forward(fromAxisAngle(normalize(angPos),|angPos|)) in body.ts. */
const rodriguesForward = tgpu.fn([d.vec3f], d.vec3f)((r) => {
  "use gpu";
  const theta = std.length(r);
  const k = std.mul(1 / std.max(theta, 1e-9), r);
  const z = d.vec3f(0, 0, 1);
  const ct = std.cos(theta);
  const st = std.sin(theta);
  const rot = std.add(std.add(std.mul(ct, z), std.mul(st, std.cross(k, z))), std.mul(std.dot(k, z) * (1 - ct), k));
  return std.select(rot, z, theta < 1e-9);
});

/** Move `cur` toward `target` by at most `maxStep` — the C² jerk limiter (body.ts approach). */
const approach = tgpu.fn([d.vec3f, d.vec3f, d.f32], d.vec3f)((cur, target, maxStep) => {
  "use gpu";
  const dd = std.sub(target, cur);
  const dist = std.length(dd);
  const stepped = std.add(cur, std.mul(maxStep / std.max(dist, 1e-9), dd));
  return std.select(target, stepped, dist > maxStep);
});

/** clampLength(v, maxLen) (body.ts / vec3.ts). */
const clampLen = tgpu.fn([d.vec3f, d.f32], d.vec3f)((v, maxLen) => {
  "use gpu";
  const l = std.length(v);
  return std.select(v, std.mul(maxLen / std.max(l, 1e-9), v), l > maxLen);
});

/** Solenoid field direction (unit-ish), fallback (0,-1,0) — forces.ts solenoidForce sans coeff. */
const solenoidDir = tgpu.fn([d.vec3f], d.vec3f)((p) => {
  "use gpu";
  const x = p.x;
  const y = p.y;
  const z = p.z;
  const lxz = std.sqrt(x * x + z * z);
  const R = d.f32(2.5);
  const kk = ((R - lxz) * (R - lxz) + y * y) / ((R + lxz) * (R + lxz) + y * y);
  const stY = (R * (1 + kk)) / (1 - kk + 1e-6) - lxz;
  const fvec = d.vec3f((-y * x) / std.max(lxz, 1e-9), -stY, (-y * z) / std.max(lxz, 1e-9));
  const down = d.vec3f(0, -1, 0);
  return std.select(normOr(fvec, down), down, lxz < 1e-6);
});

/** figureTargetVel — the target *state of motion* per figure (figures.ts). */
const figureTargetVel = tgpu.fn([d.u32, d.vec3f, d.u32, d.vec3f, d.f32], d.vec3f)((code, p, i, partner, speed) => {
  "use gpu";
  const up = d.vec3f(0, 1, 0);
  // swing (0): orbit the couple midpoint
  const mid = std.mul(0.5, std.add(p, partner));
  const rel = std.sub(p, mid);
  const swing = std.mul(speed, safeNorm(std.cross(up, rel)));
  // grandChain (1): counter-rotating by parity
  const sign = std.select(d.f32(-1), d.f32(1), i % d.u32(2) === d.u32(0));
  const gchain = std.mul(speed * sign, safeNorm(std.cross(up, p)));
  // gather (2) / scatter (3)
  const gather = std.mul(-speed, safeNorm(p));
  const scatter = std.mul(speed, safeNorm(p));
  // mill (4): fixed pseudo-random heading per dancer
  const hh = (i + d.u32(1)) * d.u32(2654435761);
  const ang = (d.f32(hh) / 4294967295) * 6.28318530718;
  const mill = std.mul(speed * 0.4, d.vec3f(std.cos(ang), 0, std.sin(ang)));
  return std.select(
    std.select(std.select(std.select(swing, gchain, code === d.u32(1)), gather, code === d.u32(2)), scatter, code === d.u32(3)),
    mill,
    code === d.u32(4),
  );
});

// ── kernels ───────────────────────────────────────────────────────────────────────────────

const copyFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })(({ gid }) => {
    "use gpu";
    const i = gid.x;
    if (i < layout.$.params.n) {
      layout.$.posSnap[i] = d.vec3f(layout.$.pos[i]!);
    }
  })
  .$name("dancerCopy");

/** Pack the five state blocks into `snapshot` as [pos | vel | accel | angPos | angVel], each
 *  [x,y,z]×N — one buffer to read back in a single `.read()`. */
const gatherFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })(({ gid }) => {
    "use gpu";
    const i = gid.x;
    const n = layout.$.params.n;
    if (i < n) {
      layout.$.snapshot[i] = d.vec3f(layout.$.pos[i]!);
      layout.$.snapshot[n + i] = d.vec3f(layout.$.vel[i]!);
      layout.$.snapshot[2 * n + i] = d.vec3f(layout.$.accel[i]!);
      layout.$.snapshot[3 * n + i] = d.vec3f(layout.$.angPos[i]!);
      layout.$.snapshot[4 * n + i] = d.vec3f(layout.$.angVel[i]!);
    }
  })
  .$name("dancerGather");

const stepFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })(({ gid }) => {
    "use gpu";
    const i = gid.x;
    const P = layout.$.params;
    const n = P.n;
    if (i < n) {
      const pSelf = d.vec3f(layout.$.posSnap[i]!);
      const vSelf = d.vec3f(layout.$.vel[i]!);
      const aSelf = d.vec3f(layout.$.accel[i]!);
      const angP = d.vec3f(layout.$.angPos[i]!);
      const angV = d.vec3f(layout.$.angVel[i]!);

      // constrain — cubic containment (power 3 ⇒ d²)
      const dlen = std.length(pSelf);
      const kc = 0.012 * P.constrain * (dlen * dlen);
      const fConstrain = std.select(std.mul(-kc, pSelf), d.vec3f(), dlen < 1e-6);

      // neighbours — cohesion centroid (within cohereRadius) + separation (within separateRadius)
      const cohR2 = P.cohereRadius * P.cohereRadius;
      const sepR2 = P.separateRadius * P.separateRadius;
      let cx = d.f32(0);
      let cy = d.f32(0);
      let cz = d.f32(0);
      let count = d.f32(0);
      let sx = d.f32(0);
      let sy = d.f32(0);
      let sz = d.f32(0);
      for (let j = d.u32(0); j < n; j++) {
        if (j !== i) {
          const q = d.vec3f(layout.$.posSnap[j]!);
          const dx = q.x - pSelf.x;
          const dy = q.y - pSelf.y;
          const dz = q.z - pSelf.z;
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < cohR2) {
            cx = cx + q.x;
            cy = cy + q.y;
            cz = cz + q.z;
            count = count + 1;
          }
          if (d2 < sepR2) {
            if (d2 > 1e-6) {
              const wgt = (1 - d2 / sepR2) / std.sqrt(d2);
              sx = sx - dx * wgt;
              sy = sy - dy * wgt;
              sz = sz - dz * wgt;
            }
          }
        }
      }
      const centroid = d.vec3f(cx / std.max(count, 1), cy / std.max(count, 1), cz / std.max(count, 1));
      const fCohere = std.select(std.mul(0.012 * P.cohere, std.sub(centroid, pSelf)), d.vec3f(), count < 0.5);
      const fSeparate = std.mul(0.06 * P.separate, d.vec3f(sx, sy, sz));

      // field forces
      const fOrbit = std.mul(0.05 * P.orbit, safeNorm(std.cross(std.cross(pSelf, vSelf), pSelf)));
      const fSwim = std.mul(0.02 * P.swim, safeNorm(pSelf));
      const fVortex = std.mul(0.04 * P.vortex, safeNorm(d.vec3f(pSelf.z, 0, -pSelf.x)));
      const fSolenoid = std.mul(0.045 * P.solenoid, solenoidDir(pSelf));

      // partnerOrbit — swing around the couple midpoint with the offset partner
      const partnerPos = d.vec3f(layout.$.posSnap[(i + P.partnerOffsetIdx) % n]!);
      const mid = std.mul(0.5, std.add(pSelf, partnerPos));
      const rel = std.sub(pSelf, mid);
      const partnerA = safeNorm(std.cross(std.cross(rel, vSelf), rel));
      const fPartner = std.select(std.mul(0.05 * P.partnerStrength, partnerA), d.vec3f(), std.length(rel) < 1e-6);

      // caller — accelerate toward the called figure's state of motion
      const callerPartner = d.vec3f(layout.$.posSnap[(i + P.callerPartnerStep) % n]!);
      const target = figureTargetVel(P.figureCode, pSelf, i, callerPartner, P.callerSpeed);
      const fCaller = std.mul(P.callerGain * P.caller, std.sub(target, vSelf));

      const f = std.add(
        std.add(
          std.add(std.add(std.add(std.add(std.add(std.add(fConstrain, fCohere), fSeparate), fOrbit), fSwim), fVortex), fSolenoid),
          fPartner,
        ),
        fCaller,
      );

      // integrate — jerk-limited accel, damping, speed cap, containment
      const tf = P.dt * P.timeFactor;
      const aN = approach(aSelf, f, P.jerkLimit * P.dt);
      const vTmp = std.mul(P.linDamp, std.add(vSelf, std.mul(tf, aN)));
      const vClamped = clampLen(vTmp, P.speedLimit);
      const pTmp = std.add(pSelf, std.mul(tf, vClamped));
      const rr = std.length(pTmp);
      const over = rr > P.maxRadius;
      const pN = std.select(pTmp, std.mul(P.maxRadius / std.max(rr, 1e-9), pTmp), over);
      const vN = std.select(vClamped, std.mul(0.5, vClamped), over);

      // angular — turn to face travel (torque), integrate angVel → angPos
      const spd = std.length(vN);
      const fwd = rodriguesForward(angP);
      const faceTq = std.mul(P.face, std.cross(fwd, std.mul(1 / std.max(spd, 1e-3), vN)));
      const tq = std.select(d.vec3f(), faceTq, spd > 1e-3);
      const angVN = std.mul(P.angDamp, std.add(angV, std.mul(tf, tq)));
      const angPN = std.add(angP, std.mul(tf, angVN));

      layout.$.pos[i] = d.vec3f(pN);
      layout.$.vel[i] = d.vec3f(vN);
      layout.$.accel[i] = d.vec3f(aN);
      layout.$.angPos[i] = d.vec3f(angPN);
      layout.$.angVel[i] = d.vec3f(angVN);
    }
  })
  .$name("dancerStep");

// ── host driver ─────────────────────────────────────────────────────────────────────────

type Root = ReturnType<typeof tgpu.initFromDevice>;
/** A vec3f storage buffer and a Params uniform buffer, typed precisely so `createBindGroup`
 *  accepts them (the loose `ReturnType<createBuffer>` erases to `AnyData`). */
type Vec3Buffer = TgpuBuffer<d.WgslArray<d.Vec3f>> & StorageFlag;
type ParamsBuffer = TgpuBuffer<typeof Params> & UniformFlag;

interface DancerPipe {
  root: Root;
  copyPipeline: GPUComputePipeline;
  stepPipeline: GPUComputePipeline;
  gatherPipeline: GPUComputePipeline;
}

// One TypeGPU root + compiled pipelines PER device (initFromDevice / pipeline compilation is
// once-per-device, like reactionDiffusion's getPipe). Multiple DancerGpuSim instances on the
// same device share this; only the state buffers are per-instance. Keyed weakly so an adopted
// device that goes away is collectable.
const pipeCache = new WeakMap<GPUDevice, DancerPipe>();

function getDancerPipe(device: GPUDevice): DancerPipe {
  const cached = pipeCache.get(device);
  if (cached) return cached;
  const root = tgpu.initFromDevice({ device });
  const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([copyFn, gatherFn, stepFn], { names: "strict" });
  const module = device.createShaderModule({ code });
  const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
  const copyPipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "dancerCopy" } });
  const stepPipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "dancerStep" } });
  const gatherPipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "dancerGather" } });
  const pipe: DancerPipe = { root, copyPipeline, stepPipeline, gatherPipeline };
  pipeCache.set(device, pipe);
  return pipe;
}

function toVec3f(block: Float32Array, n: number): ReturnType<typeof d.vec3f>[] {
  return Array.from({ length: n }, (_, i) => d.vec3f(block[i * 3] ?? 0, block[i * 3 + 1] ?? 0, block[i * 3 + 2] ?? 0));
}

/** Block layout of the readback — the same five vec3 fields as the CPU BodyState, flattened
 *  [x,y,z]×N each. */
export interface DancerBlocks {
  pos: Float32Array;
  vel: Float32Array;
  accel: Float32Array;
  angPos: Float32Array;
  angVel: Float32Array;
}

/** The GPU-resident dancer. Construct with ANY GPUDevice (Dawn under Node, or three.js's
 *  `renderer.backend.device` in the browser); state lives in that device's buffers. The
 *  pipeline is cached per device (getDancerPipe); the state buffers are per instance, so a
 *  browser stage keeps ONE long-lived sim. */
export class DancerGpuSim {
  readonly n: number;
  params: DancerGpuParams;
  frame = 0;

  private readonly device: GPUDevice;
  private readonly seed: number;
  private root!: Root;
  private copyPipeline!: GPUComputePipeline;
  private stepPipeline!: GPUComputePipeline;
  private gatherPipeline!: GPUComputePipeline;
  private paramsBuf!: ParamsBuffer;
  private posSnap!: Vec3Buffer;
  private posBuf!: Vec3Buffer;
  private velBuf!: Vec3Buffer;
  private accelBuf!: Vec3Buffer;
  private angPosBuf!: Vec3Buffer;
  private angVelBuf!: Vec3Buffer;
  private snapshotBuf!: Vec3Buffer;
  private bind!: GPUBindGroup;
  private groups = 1;

  constructor(device: GPUDevice, n: number, seed: number, params: DancerGpuParams) {
    this.device = device;
    this.n = n;
    this.seed = seed;
    this.params = params;
  }

  init(): void {
    const { device, n } = this;
    const pipe = getDancerPipe(device);
    this.root = pipe.root;
    this.copyPipeline = pipe.copyPipeline;
    this.stepPipeline = pipe.stepPipeline;
    this.gatherPipeline = pipe.gatherPipeline;
    const root = this.root;

    const body = seedSwarmBody(n, this.seed);
    const mk = (block: "pos" | "vel" | "accel" | "angPos" | "angVel"): Vec3Buffer =>
      root.createBuffer(d.arrayOf(d.vec3f, n), toVec3f(tapBlock(body, n, block), n)).$usage("storage");
    this.posBuf = mk("pos");
    this.posSnap = root.createBuffer(d.arrayOf(d.vec3f, n), toVec3f(tapBlock(body, n, "pos"), n)).$usage("storage");
    this.velBuf = mk("vel");
    this.accelBuf = mk("accel");
    this.angPosBuf = mk("angPos");
    this.angVelBuf = mk("angVel");
    this.snapshotBuf = root.createBuffer(d.arrayOf(d.vec3f, 5 * n)).$usage("storage");
    this.paramsBuf = root.createBuffer(Params).$usage("uniform");

    this.bind = root.unwrap(
      root.createBindGroup(layout, {
        params: this.paramsBuf,
        posSnap: this.posSnap,
        pos: this.posBuf,
        vel: this.velBuf,
        accel: this.accelBuf,
        angPos: this.angPosBuf,
        angVel: this.angVelBuf,
        snapshot: this.snapshotBuf,
      }),
    );
    this.groups = Math.ceil(n / WG);
    this.writeParams();
  }

  private writeParams(): void {
    const p = this.params;
    const n = this.n;
    const fa = figureAt(this.frame, p.period, p.callerSeed);
    this.paramsBuf.write({
      n,
      figureCode: FIGURE_CODE[fa.figure],
      // partnerIndex(0, figIdx, n) == the per-figure step offset (0 when n≤1)
      callerPartnerStep: partnerIndex(0, fa.figureIndex, n),
      partnerOffsetIdx: n > 0 ? Math.max(1, Math.round(p.partnerOffset)) % n : 0,
      constrain: p.constrain,
      cohere: p.cohere,
      cohereRadius: p.cohereRadius,
      separate: p.separate,
      separateRadius: p.separateRadius,
      orbit: p.orbit,
      swim: p.swim,
      vortex: p.vortex,
      solenoid: p.solenoid,
      partnerStrength: p.partner,
      caller: p.caller,
      callerGain: p.callerGain,
      callerSpeed: p.callerSpeed,
      timeFactor: p.timeFactor,
      jerkLimit: p.jerkLimit,
      linDamp: p.linDamp,
      angDamp: p.angDamp,
      speedLimit: p.speedLimit,
      maxRadius: p.maxRadius,
      face: p.face,
      dt: INTEGRATE_DEFAULTS.dt,
    });
  }

  /** Enqueue one step (copy → force/integrate). No readback — positions/orientations stay on
   *  the GPU for the renderer to read directly. */
  step(): void {
    this.writeParams();
    const { device } = this;
    const enc = device.createCommandEncoder();
    const copy = enc.beginComputePass();
    copy.setPipeline(this.copyPipeline);
    copy.setBindGroup(0, this.bind);
    copy.dispatchWorkgroups(this.groups);
    copy.end();
    const s = enc.beginComputePass();
    s.setPipeline(this.stepPipeline);
    s.setBindGroup(0, this.bind);
    s.dispatchWorkgroups(this.groups);
    s.end();
    device.queue.submit([enc.finish()]);
    this.frame++;
  }

  /** Raw GPUBuffer of current positions — hand to three.js to render from (no readback). */
  positionBuffer(): GPUBuffer {
    return this.root.unwrap(this.posBuf);
  }

  /** Raw GPUBuffer of current orientations (angle-axis vec3). */
  orientationBuffer(): GPUBuffer {
    return this.root.unwrap(this.angPosBuf);
  }

  /** Read all five body blocks back to the host in ONE readback (gather kernel packs them into
   *  a single buffer first) — for the golden test and the occasional low-frequency CPU snapshot
   *  (distance matrix / hover-pick), NOT the render hot path. */
  async readBlocks(): Promise<DancerBlocks> {
    const { device, n } = this;
    const enc = device.createCommandEncoder();
    const g = enc.beginComputePass();
    g.setPipeline(this.gatherPipeline);
    g.setBindGroup(0, this.bind);
    g.dispatchWorkgroups(this.groups);
    g.end();
    device.queue.submit([enc.finish()]);

    // Read via a wrapper over the RAW buffer — the Dawn-on-Node-stable path (a direct `.read()`
    // on a created buffer can segfault the Node binding; see backend.node.ts).
    const raw = this.root.unwrap(this.snapshotBuf);
    const got = (await this.root.createBuffer(d.arrayOf(d.vec3f, 5 * n), raw).read()) as {
      x: number;
      y: number;
      z: number;
    }[];
    const block = (b: number): Float32Array => {
      const out = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const v = got[b * n + i] ?? { x: 0, y: 0, z: 0 };
        out[i * 3] = v.x;
        out[i * 3 + 1] = v.y;
        out[i * 3 + 2] = v.z;
      }
      return out;
    };
    return { pos: block(0), vel: block(1), accel: block(2), angPos: block(3), angVel: block(4) };
  }
}
