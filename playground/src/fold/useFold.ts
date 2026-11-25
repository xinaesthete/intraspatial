// Runs a chain on the GPU and keeps what the FoldSynth views need beside it: a ring of past
// positions (the history trace) and the surface its bonds enclose.
//
// The bond table is the host's: nothing on the device changes it for a chain. So an edit to the
// lines is applied to the host's table and sent down (`uploadTopology`) in the same frame, with no
// download and no wait. Positions are the device's: nothing is read back to draw. A small copy of
// them comes back a few times a second for the camera only (picking a bead, following the chain).
import { PositionHistory } from "@intraspatial/deck";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  applyLines,
  bondFaces,
  chainScenario,
  createSpringsGpu,
  type LineSpringOptions,
  type LineSprings,
  type MatrixLine,
  type SpringParams,
  type SpringsGpu,
} from "../../../src/gpu/sim/springs/index";

export interface FoldOptions {
  /** Deck's device; null until it exists. */
  readonly device: GPUDevice | null;
  /** Particles in the chain. */
  readonly n: number;
  /** A new value starts a fresh chain. */
  readonly seed: number;
  readonly lines: readonly MatrixLine[];
  readonly springs: LineSpringOptions;
  readonly playing: boolean;
  readonly stepsPerFrame: number;
  /** Snapshots the history trace keeps. */
  readonly traceLength: number;
  /** Frames between snapshots. */
  readonly traceEvery: number;
  /** Parameter overrides applied every frame. */
  readonly live: Partial<SpringParams>;
}

export interface Fold {
  /** The running chain's device object, for layers to bind. */
  readonly gpu: SpringsGpu | null;
  readonly history: PositionHistory | null;
  /** Triangles over the particles, from the loops the bonds close. */
  readonly faces: Uint32Array | null;
  /** The latest host copy of the positions (`[x, y, z, density]` per particle), a few frames old at
   *  most; null before the first. For the camera, not for drawing. */
  readonly positions: () => Float32Array | null;
  /** Springs the lines asked for that did not fit a particle's bond slots. */
  readonly skipped: number;
  readonly step: number;
  readonly error: string | null;
}

/** Frames between host copies of the positions (each is one small download). */
const POSITIONS_EVERY = 6;

interface Readout {
  readonly key: object;
  readonly gpu: SpringsGpu;
  readonly faces: Uint32Array;
  readonly skipped: number;
}

export function useFold(o: FoldOptions): Fold {
  const { device, n, seed, traceLength } = o;
  const latest = useRef(o);
  useEffect(() => {
    latest.current = o;
  });
  const [readout, setReadout] = useState<Readout | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState(0);
  const hostPositions = useRef<Float32Array | null>(null);
  const positions = useCallback(() => hostPositions.current, []);

  const history = useMemo(
    () => (device ? new PositionHistory(device, { count: n, capacity: traceLength, label: "fold:history" }) : null),
    [device, n, traceLength],
  );
  // Freed a little after it is replaced: a frame already on its way may still be drawing it.
  useEffect(() => {
    if (!history) return;
    return () => void setTimeout(() => history.destroy(), 1000);
  }, [history]);
  const historyRef = useRef(history);
  useEffect(() => {
    historyRef.current = history;
  }, [history]);

  // One run per device, length and seed.
  const key = useMemo(() => ({ device, n, seed }), [device, n, seed]);
  useEffect(() => {
    const { device, n, seed } = key;
    if (!device) return;
    let stopped = false;
    let busy = false;
    let raf = 0;
    let frame = 0;
    const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));
    const sc = chainScenario({ n, seed });
    let applied: { lines: readonly MatrixLine[]; springs: LineSpringOptions; made: LineSprings } | undefined;

    createSpringsGpu(n, sc.topology.K, { device })
      .then((gpu) => {
        if (stopped) return;
        gpu.upload(sc.state, sc.topology);
        historyRef.current?.clear();
        hostPositions.current = null;
        /** Make the model's springs match the lines: on the host, then down to the device. */
        const applyEdits = () => {
          const now = latest.current;
          applied = { lines: now.lines, springs: now.springs, made: applyLines(sc.topology, now.lines, now.springs, applied?.made) };
          gpu.uploadTopology(sc.state, sc.topology);
          setReadout({ key, gpu, faces: bondFaces(sc.topology), skipped: applied.made.skipped });
        };
        const tick = () => {
          if (stopped) return;
          const now = latest.current;
          if (applied?.lines !== now.lines || applied.springs !== now.springs) applyEdits();
          if (now.playing) {
            sc.params = { ...sc.params, ...now.live };
            gpu.step(sc.params, now.stepsPerFrame);
            if (frame % Math.max(1, now.traceEvery) === 0) historyRef.current?.capture(gpu.positions());
          }
          if (!busy && frame % POSITIONS_EVERY === 0) {
            busy = true;
            // Copied on the device at this call, so it is the step just taken, not a later one.
            gpu
              .snapshotPositions()
              .then((snapshot) => {
                if (!stopped) hostPositions.current = snapshot;
              })
              .catch(report)
              .finally(() => {
                busy = false;
              });
          }
          if (frame % 15 === 0) setStep(gpu.stepIndex);
          frame++;
          raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
      })
      .catch(report);
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
    };
  }, [key]);

  const current = readout?.key === key ? readout : null;
  return {
    gpu: current?.gpu ?? null,
    history,
    faces: current?.faces ?? null,
    positions,
    skipped: current?.skipped ?? 0,
    step,
    error,
  };
}
