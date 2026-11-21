// Runs a spring-model scenario on the animation clock and keeps a history of its readouts.
// The page supplies how to build the scenario and how to draw it; this hook owns the loop.
import { useEffect, useRef, useState } from "react";
import {
  bondReadout,
  CELL,
  createSpheroidGpuRunner,
  createSpringsGpu,
  distanceCorrelation,
  distanceMatrix,
  LEADER,
  projectedShape,
  type Scenario,
  type SpheroidScenario,
  type SpringParams,
  shape3d,
  shollProfile,
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
  /** 3D spheroid: separate pieces of tissue crossing each shell about the centre, innermost
   *  first, and the most in any shell outside the core. */
  readonly sholl: readonly number[];
  readonly branches: number;
  /** 3D spheroid: furthest reach of the tissue, and the fraction of that sphere it fills. */
  readonly maxRadius: number;
  readonly compactness: number;
}

const NO_3D = { sholl: [] as readonly number[], branches: 0, maxRadius: 0, compactness: 0 };

function measure(sc: Scenario): Sample {
  const bonds = bondReadout(sc.state, sc.topology, sc.params);
  const base = {
    step: sc.state.step,
    bonds: bonds.count,
    broken: sc.totals.broken,
    stress: bonds.virial[0],
    area: 0,
    circularity: 0,
    cells: 0,
    correlation: 0,
    ...NO_3D,
  };
  if (sc.kind === "spheroid" && sc.params.dims === 3) {
    const tissue = [CELL, LEADER];
    const shape = shape3d(sc.state, tissue);
    const sholl = shollProfile(sc.state, tissue, { coreRadius: 4 });
    return {
      ...base,
      cells: shape.count,
      sholl: sholl.pieces,
      branches: sholl.branches,
      maxRadius: shape.maxRadius,
      compactness: shape.compactness,
    };
  }
  if (sc.kind === "spheroid") {
    const shape = projectedShape(sc.state, CELL);
    return { ...base, area: shape.area, circularity: shape.circularity, cells: shape.count };
  }
  if (sc.kind === "chromatin" && sc.matrices?.dist) {
    const size = sc.matrices.size;
    return { ...base, correlation: distanceCorrelation(distanceMatrix(sc.state, size), sc.matrices.dist, size) };
  }
  return base;
}

/** Something that advances a scenario on the device and leaves its host state current. */
interface GpuRunner {
  advance(steps: number): Promise<void>;
}

async function gpuRunner(sc: Scenario): Promise<GpuRunner> {
  if ("rules" in sc) return createSpheroidGpuRunner(sc as SpheroidScenario);
  const gpu = await createSpringsGpu(sc.state.n, sc.topology.K);
  gpu.upload(sc.state, sc.topology, sc.matrices);
  return {
    async advance(steps) {
      gpu.step(sc.params, steps);
      await gpu.download(sc.state, sc.topology);
    },
  };
}

export interface SimulationOptions {
  /** The scenario to run. A new object restarts the loop and clears the history. */
  readonly scenario: Scenario;
  readonly playing: boolean;
  readonly stepsPerFrame: number;
  /** Run on the GPU: the chain setups and the spheroid can. Bond-break events and the
   *  broken-bond total are not reported on this path. */
  readonly gpu: boolean;
  /** Parameter overrides applied every frame, for controls that act on a running model. */
  readonly live: Partial<SpringParams>;
  /** Called once per animation frame with the current scenario. */
  readonly onFrame: (sc: Scenario, time: number) => void;
}

/** The relaxation measurement, kept apart from the rolling chart history so that neither the
 *  baseline nor a crossing already seen is lost when old samples drop off. */
export interface Relaxation {
  /** The first sample at or after step 40: where the measurement starts. */
  readonly baseline?: { readonly step: number; readonly stress: number };
  /** Step of the first sample whose stress was at or below half the baseline. */
  readonly halfStep?: number;
}

interface History extends Relaxation {
  readonly scenario: Scenario;
  readonly samples: readonly Sample[];
}

/** Add a sample to a scenario's history, or start a new history for a new scenario. */
function extend(prev: History | null, scenario: Scenario, sample: Sample): History {
  const same = prev?.scenario === scenario ? prev : null;
  const baseline = same?.baseline ?? (sample.step >= 40 ? { step: sample.step, stress: sample.stress } : undefined);
  const crossed = baseline && sample.step > baseline.step && sample.stress <= baseline.stress / 2;
  return {
    scenario,
    samples: same ? [...same.samples.slice(-399), sample] : [sample],
    baseline,
    halfStep: same?.halfStep ?? (crossed ? sample.step : undefined),
  };
}

export function useSimulation(o: SimulationOptions): { samples: readonly Sample[]; error: string | null; relaxation: Relaxation } {
  const [history, setHistory] = useState<History | null>(null);
  // Keyed by scenario, so an error from a run that has been replaced is not shown on the next.
  const [error, setError] = useState<{ scenario: Scenario; message: string } | null>(null);
  // The loop reads the latest options without restarting when they change.
  const latest = useRef(o);
  useEffect(() => {
    latest.current = o;
  });

  const { scenario: sc, gpu: wantGpu } = o;
  useEffect(() => {
    let gpu: GpuRunner | null = null;
    let busy = false;
    let stopped = false;
    let frame = 0;
    let advances = 0;
    const report = (e: unknown) => {
      if (!stopped) setError({ scenario: sc, message: e instanceof Error ? e.message : String(e) });
    };
    if (wantGpu) {
      gpuRunner(sc)
        .then((g) => {
          if (!stopped) gpu = g;
        })
        .catch(report);
    }
    const record = () => {
      const sample = measure(sc);
      setHistory((prev) => extend(prev, sc, sample));
    };
    const tick = (time: number) => {
      if (stopped) return;
      const now = latest.current;
      sc.params = { ...sc.params, ...now.live };
      if (frame === 0) record();
      if (now.playing && !busy) {
        if (wantGpu) {
          if (gpu) {
            busy = true;
            gpu
              .advance(now.stepsPerFrame)
              .then(() => {
                // A device advance can span several frames, so samples are counted in advances.
                if (!stopped && ++advances % 4 === 0) record();
              })
              .catch((e: unknown) => {
                // Stop submitting to a device that has just failed; the error stays on screen.
                gpu = null;
                report(e);
              })
              .finally(() => {
                busy = false;
              });
          }
        } else {
          sc.advance(now.stepsPerFrame);
          if (++advances % 12 === 0) record();
        }
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

  const current = history?.scenario === sc ? history : null;
  return {
    samples: current?.samples ?? [],
    error: error?.scenario === sc ? error.message : null,
    relaxation: { baseline: current?.baseline, halfStep: current?.halfStep },
  };
}
