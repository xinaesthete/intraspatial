// A scenario running on the device, as a graph (docs/spring-model-as-ops.md). This is what a page
// or a view drives: it advances the scenario's graph with nothing coming back to the host, hands
// out the device buffers a view binds, and brings the state back only when asked.
//
// A graph reads a scenario's settings once, when it is built. So a change of settings is a
// boundary crossing, handled here: the state is brought back, the graph is rebuilt from it with
// the new settings, and the run carries on. The host is never in the loop otherwise.
import type { GpuBackend } from "../../graph/backend";
import { advance, createSimState, disposeSimState, type FieldValue, nodeBackend, type SimState, simStateValue } from "../../graph/index";
import { readSpringState, SPRING_STATE, type SpringGraph, type StatePart, scenarioGraph } from "./graph";
import { clearLeaders } from "./growth";
import { CELL, LEADER, type Scenario } from "./scenarios";

/** Where one part of the state lies on the device: a buffer and the range of it the part
 *  occupies. Pass it as a bind-group resource. In a packed graph several parts share a buffer. */
export interface BufferRange {
  readonly buffer: GPUBuffer;
  /** Bytes from the start of `buffer`. */
  readonly offset: number;
  /** Bytes. */
  readonly size: number;
}

/** The device state of a running scenario, for a view to bind. Valid until the next `advance`. */
export interface SpringSimBuffers {
  readonly n: number;
  /** Bond slots per particle. */
  readonly K: number;
  /** `vec4f` per particle: xyz and density. */
  readonly positions: BufferRange;
  /** `vec4f` per particle: the positions one step ago. */
  readonly previous: BufferRange;
  /** `u32` per particle; 255 is an unused slot. */
  readonly species: BufferRange;
  /** `f32` per particle. */
  readonly radius: BufferRange;
  /** `u32` per bond slot: partner in the low 24 bits, kind in the high 8, `0xFFFFFFFF` empty. */
  readonly link: BufferRange;
  /** `vec4f` per bond slot: rest length, strength, distance power, unused. */
  readonly spring: BufferRange;
}

export interface SpringSim {
  /** Run `steps` steps on the device. Nothing is downloaded. If the scenario's settings have
   *  changed since the graph was built, it is rebuilt first. One `advance` or `sync` at a time. */
  advance(steps: number): Promise<void>;
  /** Bring positions, species, radius, the bond table and the step back into the scenario's host
   *  state: the explicit download, for a readout, a host-drawn view or a test. One `advance` or
   *  `sync` at a time. */
  sync(): Promise<void>;
  /** Say that something the graph was built from has changed on the host in a way this cannot
   *  see for itself (an edit to the topology, say). Settings and rules are noticed without it. */
  invalidate(): void;
  /** The state's device buffers, or undefined before the first `advance`. */
  buffers(): SpringSimBuffers | undefined;
  /** How many times the graph has been built: once, plus once per change of settings. */
  readonly builds: number;
  /** Release the device state. Waits out an in-flight `advance` or `sync` first, and further
   *  calls fail. Safe to call twice. */
  dispose(): Promise<void>;
}

export interface SpringSimOptions {
  /** The graph backend: `browserBackend` in a page. Defaults to the Node one. */
  readonly backend?: GpuBackend;
  /** Keep the state in one buffer and the fixed inputs in another (`SpringGraphOptions.packed`). */
  readonly packed?: boolean;
  /** Do the model's own pair terms in one interpreted pass (`SpringGraphOptions.fusePairs`). */
  readonly fusePairs?: boolean;
  /** Called for every upload or download the executor makes, for tests and diagnostics. */
  readonly onBridge?: (key: string, direction: "download" | "upload" | "detexture") => void;
}

/** What a graph was built from, as text: a change means it must be rebuilt. */
function fingerprint(sc: Scenario): string {
  return JSON.stringify([sc.params, sc.graphRules?.() ?? {}], (_key, value: unknown) =>
    typeof value === "number" && !Number.isFinite(value) ? String(value) : value,
  );
}

/** Run `sc` on the device as a graph. The scenario's host state is the seed, and is what `sync`
 *  writes back to; in between it is stale. */
export function createSpringSim(sc: Scenario, opts: SpringSimOptions = {}): SpringSim {
  const backend = opts.backend ?? nodeBackend;
  const ctx = { backend };
  let built: { graph: SpringGraph; state: SimState; from: string; stepAtBuild: number; ticks: number } | undefined;
  let stale = false;
  let hostCurrent = true;
  let builds = 0;
  let busy = false;
  let disposed = false;
  let idle = Promise.resolve();
  let releaseIdle: (() => void) | undefined;

  const occupy = () => {
    busy = true;
    idle = new Promise((resolve) => {
      releaseIdle = resolve;
    });
  };
  const vacate = () => {
    busy = false;
    releaseIdle?.();
    releaseIdle = undefined;
  };
  const begin = (what: string) => {
    if (disposed) throw new Error("createSpringSim: disposed");
    if (busy) throw new Error(`createSpringSim: ${what} called while another operation is still running`);
    occupy();
  };

  /** The host copy of a built graph. Callers that already hold the lock use this; `sync` is the
   *  public one and takes the lock itself. */
  const writeBack = async () => {
    if (!built || hostCurrent) return;
    await readSpringState(built.state, sc.state, sc.topology, backend);
    // The step is graph state only when a term or a rule reads it, or the graph is packed.
    if (!simStateValue(built.state, SPRING_STATE.step)) sc.state.step = built.stepAtBuild + built.ticks;
    hostCurrent = true;
  };

  const build = async () => {
    if (built) {
      await writeBack();
      disposeSimState(built.state, backend);
    }
    // With leader selection off nothing else would turn existing leaders back into cells.
    const rules = sc.graphRules?.();
    if (rules && !rules.leaders) clearLeaders(sc.state, LEADER, CELL);
    built = {
      graph: scenarioGraph(sc, { packed: opts.packed, fusePairs: opts.fusePairs }),
      state: createSimState(),
      from: fingerprint(sc),
      stepAtBuild: sc.state.step,
      ticks: 0,
    };
    stale = false;
    builds++;
  };

  const range = (v: FieldValue | undefined): BufferRange | undefined =>
    v?.buffer ? { buffer: v.buffer.buffer, offset: v.buffer.offset ?? 0, size: v.buffer.byteLength } : undefined;

  return {
    async advance(steps) {
      begin("advance");
      try {
        if (!built || stale || built.from !== fingerprint(sc)) await build();
        const run = built;
        if (!run || steps <= 0) return;
        await advance(run.graph.graph, run.graph.pos, {
          steps,
          state: run.state,
          ctx,
          materialise: "none",
          onBridge: opts.onBridge,
          // A later tick can throw. Ticks that already committed stay counted, and the host is
          // stale from the first of them, so a following sync downloads instead of no-opping.
          onFrame: () => {
            run.ticks += 1;
            hostCurrent = false;
          },
        });
      } finally {
        vacate();
      }
    },
    async sync() {
      begin("sync");
      try {
        await writeBack();
      } finally {
        vacate();
      }
    },
    invalidate() {
      stale = true;
    },
    buffers() {
      if (!built) return undefined;
      const state = built.state;
      // Packed, every part is in one bundle; unpacked, each has its own key and the bond table its own bundle.
      const packed = simStateValue(state, SPRING_STATE.packed);
      const bonds = simStateValue(state, SPRING_STATE.bonds);
      const part = (name: StatePart, key: string) => range(packed ? packed.parts?.[name] : simStateValue(state, key));
      const positions = part("pos", SPRING_STATE.pos);
      const previous = part("prev", SPRING_STATE.prev);
      const species = part("species", SPRING_STATE.species);
      const radius = part("radius", SPRING_STATE.radius);
      const link = range(packed ? packed.parts?.link : bonds?.parts?.link);
      const spring = range(packed ? packed.parts?.spring : bonds?.parts?.spring);
      if (!positions || !previous || !species || !radius || !link || !spring) return undefined;
      return { n: sc.state.n, K: sc.topology.K, positions, previous, species, radius, link, spring };
    },
    get builds() {
      return builds;
    },
    dispose() {
      disposed = true;
      const finish = () => {
        if (built) disposeSimState(built.state, backend);
        built = undefined;
      };
      if (!busy) {
        finish();
        return Promise.resolve();
      }
      return idle.then(finish);
    },
  };
}
