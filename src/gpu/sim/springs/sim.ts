// A scenario running on the device, as a graph (docs/spring-model-as-ops.md). This is what a page
// or a view drives: it advances the scenario's graph with nothing coming back to the host, hands
// out the device buffers a view binds, and brings the state back only when asked.
//
// A graph reads a scenario's settings once, when it is built. So a change of settings is a
// boundary crossing, handled here: the state is brought back, the graph is rebuilt from it with
// the new settings, and the run carries on. The host is never in the loop otherwise.
import type { GpuBackend } from "../../graph/backend";
import { advance, createSimState, disposeSimState, nodeBackend, type SimState, simStateValue } from "../../graph/index";
import { readSpringState, SPRING_STATE, type SpringGraph, scenarioGraph } from "./graph";
import { clearLeaders } from "./growth";
import { CELL, LEADER, type Scenario } from "./scenarios";

/** The device buffers of a running scenario, for a view to bind. Valid until the next `advance`. */
export interface SpringSimBuffers {
  readonly n: number;
  /** Bond slots per particle. */
  readonly K: number;
  /** `vec4f` per particle: xyz and density. */
  readonly positions: GPUBuffer;
  /** `vec4f` per particle: the positions one step ago. */
  readonly previous: GpuBufferOrNone;
  /** `u32` per particle; 255 is an unused slot. */
  readonly species: GPUBuffer;
  /** `f32` per particle. */
  readonly radius: GPUBuffer;
  /** `u32` per bond slot: partner in the low 24 bits, kind in the high 8, `0xFFFFFFFF` empty. */
  readonly link: GPUBuffer;
  /** `vec4f` per bond slot: rest length, strength, distance power, unused. */
  readonly spring: GPUBuffer;
}
type GpuBufferOrNone = GPUBuffer | undefined;

export interface SpringSim {
  /** Run `steps` steps on the device. Nothing is downloaded. If the scenario's settings have
   *  changed since the graph was built, it is rebuilt first. One call at a time. */
  advance(steps: number): Promise<void>;
  /** Bring positions, species, radius, the bond table and the step back into the scenario's host
   *  state: the explicit download, for a readout, a host-drawn view or a test. */
  sync(): Promise<void>;
  /** Say that something the graph was built from has changed on the host in a way this cannot
   *  see for itself (an edit to the topology, say). Settings and rules are noticed without it. */
  invalidate(): void;
  /** The state's device buffers, or undefined before the first `advance`. */
  buffers(): SpringSimBuffers | undefined;
  /** How many times the graph has been built: once, plus once per change of settings. */
  readonly builds: number;
  /** Release the device state. */
  dispose(): void;
}

export interface SpringSimOptions {
  /** The graph backend: `browserBackend` in a page. Defaults to the Node one. */
  readonly backend?: GpuBackend;
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

  const sync = async () => {
    if (!built || hostCurrent) return;
    await readSpringState(built.state, sc.state, sc.topology, backend);
    // The step is graph state only when a term or a rule reads it.
    if (!simStateValue(built.state, SPRING_STATE.step)) sc.state.step = built.stepAtBuild + built.ticks;
    hostCurrent = true;
  };

  const build = async () => {
    if (built) {
      await sync();
      disposeSimState(built.state, backend);
    }
    // With leader selection off nothing else would turn existing leaders back into cells.
    const rules = sc.graphRules?.();
    if (rules && !rules.leaders && (rules.division || rules.breakRule || rules.formRule)) clearLeaders(sc.state, LEADER, CELL);
    built = { graph: scenarioGraph(sc), state: createSimState(), from: fingerprint(sc), stepAtBuild: sc.state.step, ticks: 0 };
    stale = false;
    builds++;
  };

  const buffer = (state: SimState, key: string) => simStateValue(state, key)?.buffer?.buffer;

  return {
    async advance(steps) {
      if (busy) throw new Error("createSpringSim: advance called while a previous advance is still running");
      busy = true;
      try {
        if (!built || stale || built.from !== fingerprint(sc)) await build();
        const run = built;
        if (!run || steps <= 0) return;
        await advance(run.graph.graph, run.graph.pos, { steps, state: run.state, ctx, materialise: "none", onBridge: opts.onBridge });
        run.ticks += steps;
        hostCurrent = false;
      } finally {
        busy = false;
      }
    },
    sync,
    invalidate() {
      stale = true;
    },
    buffers() {
      if (!built) return undefined;
      const positions = buffer(built.state, SPRING_STATE.pos);
      const species = buffer(built.state, SPRING_STATE.species);
      const radius = buffer(built.state, SPRING_STATE.radius);
      const bonds = simStateValue(built.state, SPRING_STATE.bonds);
      const link = bonds?.parts?.link?.buffer?.buffer;
      const spring = bonds?.parts?.spring?.buffer?.buffer;
      if (!positions || !species || !radius || !link || !spring) return undefined;
      return {
        n: sc.state.n,
        K: sc.topology.K,
        positions,
        previous: buffer(built.state, SPRING_STATE.prev),
        species,
        radius,
        link,
        spring,
      };
    },
    get builds() {
      return builds;
    },
    dispose() {
      if (built) disposeSimState(built.state, backend);
      built = undefined;
    },
  };
}
