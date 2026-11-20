// Runs a spring-model scenario on the animation clock and keeps a history of its readouts.
// The page supplies how to build the scenario and how to draw it; this hook owns the loop.
import { useEffect, useRef, useState } from "react";
import {
  bondReadout,
  CELL,
  createSpringsGpu,
  distanceCorrelation,
  distanceMatrix,
  projectedShape,
  type Scenario,
  type SpringParams,
  type SpringsGpu,
} from "../../../src/gpu/sim/springs/index";

export interface Sample {
  readonly step: number;
  /** Outline of the cells (spheroid). */
  readonly area: number;
  readonly circularity: number;
  readonly cells: number;
  /** Bonds present, and bonds broken so far. */
  readonly bonds: number;
  readonly broken: number;
  /** Stress carried by the bonds along x (relaxation test). */
  readonly stress: number;
  /** Agreement between model and target distances (chromatin). */
  readonly correlation: number;
}

function measure(sc: Scenario): Sample {
  const bonds = bondReadout(sc.state, sc.topology, sc.params);
  const base = { step: sc.state.step, bonds: bonds.count, broken: sc.totals.broken, stress: bonds.virial[0] };
  if (sc.kind === "spheroid") {
    const shape = projectedShape(sc.state, CELL);
    return { ...base, area: shape.area, circularity: shape.circularity, cells: shape.count, correlation: 0 };
  }
  if (sc.kind === "chromatin" && sc.matrices?.dist) {
    const size = sc.matrices.size;
    return {
      ...base,
      area: 0,
      circularity: 0,
      cells: 0,
      correlation: distanceCorrelation(distanceMatrix(sc.state, size), sc.matrices.dist, size),
    };
  }
  return { ...base, area: 0, circularity: 0, cells: 0, correlation: 0 };
}

export interface SimulationOptions {
  /** The scenario to run. A new object restarts the loop and clears the history. */
  readonly scenario: Scenario;
  readonly playing: boolean;
  readonly stepsPerFrame: number;
  /** Run the step on the GPU. Only scenarios without turnover or growth can. */
  readonly gpu: boolean;
  /** Parameter overrides applied every frame, for controls that act on a running model. */
  readonly live: Partial<SpringParams>;
  /** Called once per animation frame with the current scenario. */
  readonly onFrame: (sc: Scenario, time: number) => void;
}

interface History {
  readonly scenario: Scenario;
  readonly samples: readonly Sample[];
}

export function useSimulation(o: SimulationOptions): { samples: readonly Sample[]; error: string | null } {
  const [history, setHistory] = useState<History | null>(null);
  const [error, setError] = useState<string | null>(null);
  // The loop reads the latest options without restarting when they change.
  const latest = useRef(o);
  useEffect(() => {
    latest.current = o;
  });

  const { scenario: sc, gpu: wantGpu } = o;
  useEffect(() => {
    let gpu: SpringsGpu | null = null;
    let busy = false;
    let stopped = false;
    let frame = 0;
    const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));
    if (wantGpu) {
      createSpringsGpu(sc.state.n, sc.topology.K)
        .then((g) => {
          if (stopped) return;
          g.upload(sc.state, sc.topology, sc.matrices);
          gpu = g;
        })
        .catch(report);
    }
    const record = () => {
      const sample = measure(sc);
      setHistory((prev) => ({ scenario: sc, samples: prev?.scenario === sc ? [...prev.samples.slice(-399), sample] : [sample] }));
    };
    const tick = (time: number) => {
      if (stopped) return;
      const now = latest.current;
      sc.params = { ...sc.params, ...now.live };
      if (frame === 0) record();
      if (now.playing && !busy) {
        if (wantGpu) {
          if (gpu) {
            const g = gpu;
            busy = true;
            g.step(sc.params, now.stepsPerFrame);
            g.download(sc.state, sc.topology)
              .catch(report)
              .finally(() => {
                busy = false;
              });
          }
        } else {
          sc.advance(now.stepsPerFrame);
        }
        if (frame % 12 === 11) record();
      }
      frame++;
      now.onFrame(sc, time);
      raf = requestAnimationFrame(tick);
    };
    let raf = requestAnimationFrame(tick);
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
    };
  }, [sc, wantGpu]);

  return { samples: history?.scenario === sc ? history.samples : [], error };
}
