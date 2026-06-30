import { describe, expect, it } from "vitest";
import {
  danceStepCpu,
  danceStepsCpu,
  packSwarm,
  seedSwarm,
  swarmXY,
  unpackSwarm,
  type DanceParams,
} from "./danceField";

const allFinite = (a: ArrayLike<number>): boolean => {
  for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i]!)) return false;
  return true;
};
const maxR = (pos: Float32Array): number => {
  let m = 0;
  for (let i = 0; i < pos.length; i += 3) m = Math.max(m, Math.hypot(pos[i]!, pos[i + 1]!, pos[i + 2]!));
  return m;
};

describe("seedSwarm", () => {
  it("is deterministic and packs to 3·n per array", () => {
    const a = seedSwarm(50, 7);
    const b = seedSwarm(50, 7);
    expect(a.pos.length).toBe(150);
    expect(a.vel.length).toBe(150);
    expect(Array.from(a.pos)).toEqual(Array.from(b.pos));
    expect(allFinite(a.pos)).toBe(true);
  });
});

describe("danceStepCpu", () => {
  it("is deterministic and does not mutate its input", () => {
    const s0 = seedSwarm(40, 3);
    const before = Float32Array.from(s0.pos);
    const a = danceStepCpu(s0);
    const b = danceStepCpu(s0);
    expect(Array.from(s0.pos)).toEqual(Array.from(before)); // input untouched
    expect(Array.from(a.pos)).toEqual(Array.from(b.pos));
  });

  it("actually moves the swarm", () => {
    const s0 = seedSwarm(60, 5);
    const s1 = danceStepsCpu(s0, 10);
    let moved = 0;
    for (let i = 0; i < s0.pos.length; i++) if (Math.abs(s1.pos[i]! - s0.pos[i]!) > 1e-4) moved++;
    expect(moved).toBeGreaterThan(0);
  });

  it("stays finite and bounded over a long run, even with every influence on", () => {
    const wild: Partial<DanceParams> = {
      attract: 1, orbit: 1, vortex: 1, solenoid: 1, swim: 1, cohesion: 1, separation: 1,
      attractOn: true, orbitOn: true, vortexOn: true, solenoidOn: true, swimOn: true,
      cohesionOn: true, separationOn: true, speedLimit: 4, damping: 0.99,
    };
    let s = seedSwarm(120, 9);
    for (let i = 0; i < 300; i++) s = danceStepCpu(s, wild);
    expect(allFinite(s.pos)).toBe(true);
    expect(allFinite(s.vel)).toBe(true);
    expect(maxR(s.pos)).toBeLessThanOrEqual(61); // MAX_R backstop (60) + ε
  });

  it("with all influences off, damping brings the swarm to rest", () => {
    const off: Partial<DanceParams> = {
      attractOn: false, orbitOn: false, vortexOn: false, solenoidOn: false,
      swimOn: false, cohesionOn: false, separationOn: false, damping: 0.9,
    };
    let s = seedSwarm(30, 2);
    for (let i = 0; i < 200; i++) s = danceStepCpu(s, off);
    let speed = 0;
    for (let i = 0; i < s.vel.length; i++) speed += Math.abs(s.vel[i]!);
    expect(speed).toBeLessThan(1e-3);
  });
});

describe("field packing", () => {
  it("pack → unpack round-trips through the 2N-row vec3 layout", () => {
    const s = seedSwarm(25, 4);
    const packed = packSwarm(s); // length 6N
    expect(packed.length).toBe(25 * 6);
    const back = unpackSwarm(packed, 25 * 2); // 2N rows
    expect(back.n).toBe(25);
    expect(Array.from(back.pos)).toEqual(Array.from(s.pos));
    expect(Array.from(back.vel)).toEqual(Array.from(s.vel));
  });

  it("swarmXY yields [x,y] pairs (length 2N) for the scatter", () => {
    const s = seedSwarm(10, 1);
    const xy = swarmXY(s);
    expect(xy.length).toBe(20);
    expect(xy[0]).toBe(s.pos[0]);
    expect(xy[1]).toBe(s.pos[1]);
    expect(xy[2]).toBe(s.pos[3]); // second agent's x
  });
});
