// The spring model, live: CSynth's dynamics with bond turnover, adhesion and growth.
// Three setups share one page: a spheroid of cells in a matrix (Elosegui-Artola et al. 2023),
// the stress-relaxation test that characterises that matrix, and CSynth's own problem of
// folding a chain to match a distance or contact map.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  applyLines,
  chainRadius,
  chainScenario,
  DEFAULT_LINE_SPRINGS,
  helixLine,
  type LineSprings,
  type MatrixLine,
  sheetLines,
  chromatinScenario,
  DEFAULT_SPHEROID,
  distanceMatrix,
  type PhasePoint,
  type PhaseSetting,
  type RelaxationScenario,
  relaxationScenario,
  type Scenario,
  type SpheroidConfig,
  type SpheroidScenario,
  type SpringParams,
  spheroidDefaults,
  spheroidScenario,
} from "../../../src/gpu/sim/springs/index";
import { MathTex } from "../Math";
import { BarChart, LineChart, PhaseGrid } from "./Charts";
import { MapEditor } from "./MapEditor";
import { drawBall, drawChainAesthetic, drawChainAnalytic, drawEvents, drawMatrix, drawSheetAesthetic, drawSheetAnalytic } from "./draw";
import type { PhaseMessage } from "./phaseWorker";
import { useSimulation } from "./useSimulation";

type Kind = "spheroid" | "ball" | "relaxation" | "distances" | "contacts" | "draw";
type Look = "analytic" | "aesthetic";

const KINDS: readonly { id: Kind; label: string; blurb: string }[] = [
  {
    id: "spheroid",
    label: "Spheroid in a matrix",
    blurb:
      "A ball of dividing, moving cells grows inside a gel. The gel is a network of bonds. If the bonds never break the gel is elastic and holds the cells in; if they break under load and re-form, the gel creeps and the cells push out into it.",
  },
  {
    id: "ball",
    label: "Branching in 3D",
    blurb:
      "A ball of cells in a block of gel, on the GPU. A few cells on the surface become leaders (red): each pulls outward, holds on to the cells behind it, and stops new leaders forming nearby. A leader can cut into a gel that would hold back the tissue as a whole, and cells divide into the space behind it. That makes a strand, and new leaders on its flanks make branches.",
  },
  {
    id: "relaxation",
    label: "Stress relaxation test",
    blurb:
      "A strip of gel is stretched and held. This is the measurement used to say how viscoelastic a real gel is: how long the stress takes to fall to half. An elastic strip holds its stress for ever.",
  },
  {
    id: "distances",
    label: "Chain from a distance map",
    blurb:
      "CSynth's problem. A tangled chain is given the distance every pair of its points should be apart, and nothing else. Springs between all pairs pull it into the shape those distances came from.",
  },
  {
    id: "contacts",
    label: "Chain from a contact map",
    blurb:
      "The same, from contact strengths, as with Hi-C data. Each pair attracts in proportion to its contact strength and everything repels everything; the two balance at the distance the contact implies.",
  },
  {
    id: "draw",
    label: "Draw a fold",
    blurb:
      "FoldSynth's tool. The square beside the chain has a cell for every pair of particles. Draw a line across it and every pair on the line is tied with a spring. A line running alongside the diagonal winds a helix; a line running across it pairs two stretches of chain into a sheet.",
  },
];

/** The spheroid settings that are numbers, which is what a slider or a phase axis can vary. */
type NumericSetting = PhaseSetting;

/** Spheroid settings the page exposes. `infinite` adds an off switch that sets the value to ∞. */
const SPHEROID_CONTROLS: readonly {
  key: NumericSetting;
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
  infinite?: string;
  values: readonly number[];
}[] = [
  {
    key: "yieldStrain",
    label: "Yield strain",
    hint: "how far a matrix bond stretches before it breaks",
    min: 0.02,
    max: 0.6,
    step: 0.01,
    infinite: "no strain-based breaking",
    values: [0.05, 0.1, 0.2, 0.3, 0.45, Number.POSITIVE_INFINITY],
  },
  {
    key: "stiffness",
    label: "Stiffness",
    hint: "strength of a matrix bond",
    min: 0.01,
    max: 0.8,
    step: 0.01,
    values: [0.03, 0.1, 0.2, 0.3, 0.5],
  },
  {
    key: "reformProb",
    label: "Healing rate",
    hint: "chance per step that broken matrix re-bonds",
    min: 0,
    max: 0.02,
    step: 0.0001,
    values: [0, 0.0002, 0.0005, 0.002, 0.01, 0.1],
  },
  {
    key: "bondLifetime",
    label: "Bond lifetime",
    hint: "bonds also break at random after about this many steps",
    min: 20,
    max: 3000,
    step: 10,
    infinite: "no random breaking",
    values: [100, 300, 1000, 3000, Number.POSITIVE_INFINITY],
  },
  {
    key: "matrixRepulsion",
    label: "Matrix incompressibility",
    hint: "how hard the matrix resists being squeezed",
    min: 0.01,
    max: 0.5,
    step: 0.01,
    values: [0.02, 0.05, 0.1, 0.2, 0.4],
  },
  {
    key: "matrixMobility",
    label: "Matrix mobility",
    hint: "inverse of the matrix's viscosity relative to the cells",
    min: 0.1,
    max: 3,
    step: 0.05,
    values: [0.25, 0.5, 1, 2],
  },
  {
    key: "motility",
    label: "Cell motility",
    hint: "random force on each cell",
    min: 0,
    max: 0.2,
    step: 0.005,
    values: [0, 0.02, 0.05, 0.08, 0.12],
  },
  {
    key: "cellAdhesion",
    label: "Cell–cell adhesion",
    hint: "how strongly touching cells hold together",
    min: 0,
    max: 0.05,
    step: 0.001,
    values: [0, 0.01, 0.02, 0.035],
  },
  {
    key: "ligand",
    label: "Cell–matrix adhesion",
    hint: "strength of bonds between cells and matrix",
    min: 0,
    max: 0.2,
    step: 0.005,
    values: [0, 0.02, 0.06, 0.15],
  },
  {
    key: "divisionTime",
    label: "Division time",
    hint: "steps between divisions of an uncrowded cell",
    min: 50,
    max: 2000,
    step: 10,
    infinite: "no division",
    values: [100, 200, 400, 800, Number.POSITIVE_INFINITY],
  },
];

/** Structure of the 3D run, and the two regimes found by searching. */
const BALL_STRUCTURE: SpheroidConfig = { dims: 3, cellRadius: 2.5, matrixRadius: 11, maxCells: 3000, matrixDisorder: 0.2 };
const BALL_BASE: SpheroidConfig = {
  stiffness: 0.3,
  yieldStrain: 0.3,
  reformProb: 0.0005,
  matrixRepulsion: 0.05,
  motility: 0.02,
  adhesionRange: 0.25,
  protrusionRange: 2,
  leaderGrip: 10,
  leaderSpacing: 4,
  leaderTime: 60,
  divisionTime: 120,
  ligand: 0.02,
};
const COHESIVE: SpheroidConfig = { cellAdhesion: 0.1, cellRepulsion: 0.5 };
const BALL_PRESETS: readonly { label: string; hint: string; config: SpheroidConfig }[] = [
  {
    label: "Branching",
    hint: "cells hold together: thick arms that stay attached, each led by a tip cell",
    config: { ...BALL_BASE, ...COHESIVE, leaderThrust: 0.6 },
  },
  {
    label: "Dense branching",
    hint: "stronger leaders: more arms, and a few leaders tear free",
    config: { ...BALL_BASE, ...COHESIVE, leaderThrust: 0.8 },
  },
  {
    label: "Breaking away",
    hint: "weaker cohesion: thin arms that pinch off and travel as clusters",
    config: { ...BALL_BASE, leaderThrust: 0.6, cellAdhesion: 0.06, cellRepulsion: 0.3, leaderTime: 200, divisionTime: 200 },
  },
  {
    label: "No leaders",
    hint: "the same gel with nothing to cut it: the ball stays a ball",
    config: { ...BALL_BASE, ...COHESIVE, leaderThrust: 0 },
  },
  {
    label: "Elastic gel",
    hint: "leaders, but a gel that never yields",
    config: { ...BALL_BASE, ...COHESIVE, leaderThrust: 0.6, yieldStrain: Number.POSITIVE_INFINITY },
  },
];
const BALL_CONTROLS: readonly {
  key: NumericSetting;
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
  infinite?: string;
}[] = [
  {
    key: "leaderThrust",
    label: "Leader pull",
    hint: "force a leader cell pulls outward with; 0 is no leaders",
    min: 0,
    max: 1.2,
    step: 0.05,
  },
  {
    key: "leaderSpacing",
    label: "Leader spacing",
    hint: "no new leader forms this close to another: how far apart branches start",
    min: 2,
    max: 10,
    step: 0.5,
  },
  { key: "leaderTime", label: "Leader delay", hint: "steps before an exposed cell becomes a leader", min: 20, max: 600, step: 10 },
  {
    key: "leaderGrip",
    label: "Leader grip",
    hint: "how much harder a leader holds its followers than cells hold each other",
    min: 1,
    max: 15,
    step: 0.5,
  },
  { key: "cellAdhesion", label: "Cell–cell adhesion", hint: "how strongly touching cells hold together", min: 0, max: 0.14, step: 0.005 },
  {
    key: "cellRepulsion",
    label: "Cell stiffness",
    hint: "how hard cells resist being squeezed; must outweigh adhesion",
    min: 0.2,
    max: 0.7,
    step: 0.02,
  },
  { key: "stiffness", label: "Gel stiffness", hint: "strength of a gel bond", min: 0.02, max: 0.6, step: 0.01 },
  {
    key: "yieldStrain",
    label: "Gel yield strain",
    hint: "how far a gel bond stretches before it breaks",
    min: 0.05,
    max: 0.6,
    step: 0.01,
    infinite: "never breaks (elastic)",
  },
  { key: "reformProb", label: "Gel healing rate", hint: "chance per step that broken gel re-bonds", min: 0, max: 0.02, step: 0.0001 },
  {
    key: "divisionTime",
    label: "Division time",
    hint: "steps between divisions of an uncrowded cell",
    min: 40,
    max: 800,
    step: 10,
    infinite: "no division",
  },
  { key: "motility", label: "Cell motility", hint: "random force on each cell", min: 0, max: 0.1, step: 0.005 },
];

const PHASE_BASE: SpheroidConfig = { cellRadius: 3, matrixRadius: 13, maxCells: 400 };

function Slider(props: {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  infinite?: string;
  onChange: (v: number) => void;
}) {
  const finite = Number.isFinite(props.value);
  return (
    <div className="slider" title={props.hint}>
      <label>
        <span>
          {props.label}: <b>{finite ? props.value : "∞"}</b>
        </span>
        <input
          type="range"
          min={props.min}
          max={props.max}
          step={props.step}
          disabled={!finite}
          value={finite ? props.value : props.max}
          onChange={(e) => props.onChange(Number(e.target.value))}
        />
      </label>
      {props.infinite && (
        <label className="check">
          <input
            type="checkbox"
            checked={!finite}
            onChange={(e) => props.onChange(e.target.checked ? Number.POSITIVE_INFINITY : (props.min + props.max) / 2)}
          />
          {props.infinite}
        </label>
      )}
    </div>
  );
}

function Explain(props: { title: string; children: React.ReactNode; math?: string }) {
  return (
    <details className="explain">
      <summary>{props.title}</summary>
      <div className="explain-body">{props.children}</div>
      {props.math && (
        <div className="explain-math">
          <MathTex tex={props.math} />
        </div>
      )}
    </details>
  );
}

export function SpringsDemo() {
  const [kind, setKind] = useState<Kind>("spheroid");
  const [look, setLook] = useState<Look>("analytic");
  const [playing, setPlaying] = useState(true);
  const [stepsPerFrame, setStepsPerFrame] = useState(4);
  const [nonce, setNonce] = useState(0);
  const [spheroid, setSpheroid] = useState<SpheroidConfig>({});
  const [relax, setRelax] = useState({ bondLifetime: 200, yieldStrain: Number.POSITIVE_INFINITY });
  const [chainN, setChainN] = useState(240);
  const [gpu, setGpu] = useState(false);
  const [live, setLive] = useState<Partial<SpringParams>>({});
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelMapRef = useRef<HTMLCanvasElement>(null);
  const targetMapRef = useRef<HTMLCanvasElement>(null);

  const [drawN, setDrawN] = useState(64);
  const chain = kind === "distances" || kind === "contacts";
  const [ball, setBall] = useState<SpheroidConfig>(BALL_PRESETS[0]?.config ?? BALL_BASE);
  const useGpu = kind === "ball" || (gpu && chain);
  // `nonce` seeds the run, so Restart gives a fresh one. Only the settings that decide which
  // particles exist are dependencies here; the rest are applied to the running scenario below.
  const { cellRadius, matrixRadius, maxCells } = spheroid;
  const scenario = useMemo((): Scenario | SpheroidScenario | RelaxationScenario => {
    const seed = 1 + nonce;
    if (kind === "spheroid") return spheroidScenario({ cellRadius, matrixRadius, maxCells, seed });
    if (kind === "ball") return spheroidScenario({ ...BALL_STRUCTURE, seed });
    if (kind === "relaxation") return relaxationScenario({ seed });
    if (kind === "draw") return chainScenario({ n: drawN, seed });
    return chromatinScenario({ n: chainN, mode: kind, seed });
  }, [kind, cellRadius, matrixRadius, maxCells, chainN, drawN, nonce]);
  // Declared before the simulation hook, so a new scenario has its settings before its first step.
  useEffect(() => {
    if (!("configure" in scenario)) return;
    if (scenario.kind === "relaxation") scenario.configure(relax);
    // Defaults first, so a preset or a slider that leaves a setting out resets it, and the
    // model never keeps a value the controls no longer show.
    else if (scenario.params.dims === 3) scenario.configure({ ...spheroidDefaults(3), ...ball });
    else scenario.configure({ ...spheroidDefaults(2), ...spheroid });
  }, [scenario, spheroid, relax, ball]);

  // Lines drawn on the pair matrix, kept across restarts and re-applied to each new chain.
  const [lines, setLines] = useState<MatrixLine[]>(() => [helixLine(6, 30, 4)]);
  const [lineRest, setLineRest] = useState(DEFAULT_LINE_SPRINGS.rest);
  const [twistAngle, setTwistAngle] = useState(DEFAULT_LINE_SPRINGS.torsionAngle);
  const [twistStrength, setTwistStrength] = useState(DEFAULT_LINE_SPRINGS.torsionStrength);
  const applied = useRef<{ scenario: Scenario; springs: LineSprings } | null>(null);
  useEffect(() => {
    if (scenario.kind !== "chain") return;
    const previous = applied.current?.scenario === scenario ? applied.current.springs : undefined;
    const options = { ...DEFAULT_LINE_SPRINGS, rest: lineRest, torsionAngle: twistAngle, torsionStrength: twistStrength };
    applied.current = { scenario, springs: applyLines(scenario.topology, lines, options, previous) };
  }, [scenario, lines, lineRest, twistAngle, twistStrength]);

  const extent =
    kind === "spheroid"
      ? (spheroid.matrixRadius ?? DEFAULT_SPHEROID.matrixRadius) + 1
      : kind === "ball"
        ? (BALL_STRUCTURE.matrixRadius ?? 10) * 0.95
        : kind === "relaxation"
          ? 18
          : kind === "draw"
            ? chainRadius(drawN) * 1.6
            : (chainN / 14) * 1.7;
  const onFrame = useCallback(
    (sc: Scenario, time: number) => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (!canvas || !ctx) return;
      const view = { width: canvas.width, height: canvas.height, extent, yaw: time * 0.0003 };
      if (sc.kind === "chromatin" || sc.kind === "chain") (look === "analytic" ? drawChainAnalytic : drawChainAesthetic)(ctx, sc, view);
      else if (sc.params.dims === 3) drawBall(ctx, sc, view, look === "aesthetic");
      else {
        (look === "analytic" ? drawSheetAnalytic : drawSheetAesthetic)(ctx, sc, view);
        drawEvents(ctx, sc.takeEvents(), view);
      }
    },
    [look, extent],
  );

  const { samples, error, relaxation } = useSimulation({ scenario, playing, stepsPerFrame, gpu: useGpu, live: chain ? live : {}, onFrame });

  // Distance maps for the chain scenarios: the target once, the model whenever a sample lands.
  const size = scenario.matrices?.size ?? 0;
  const target = scenario.matrices?.dist;
  const latestStep = samples.at(-1)?.step;
  // Recomputed whenever a sample lands; before the first there is nothing to show.
  const drawDistances = useMemo(
    () => (scenario.kind === "chain" && latestStep !== undefined ? distanceMatrix(scenario.state, scenario.state.n) : null),
    [scenario, latestStep],
  );
  // Both maps share the target's colour scale.
  const targetMax = useMemo(() => (target ? target.reduce((a, b) => Math.max(a, b), 0) : 0), [target]);
  useEffect(() => {
    const t = targetMapRef.current?.getContext("2d");
    if (target && size > 0 && t) drawMatrix(t, target, size, targetMax);
  }, [target, size, targetMax]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: `latestStep` is the redraw trigger
  useEffect(() => {
    const m = modelMapRef.current?.getContext("2d");
    if (target && size > 0 && m) drawMatrix(m, distanceMatrix(scenario.state, size), size, targetMax);
  }, [scenario, target, size, targetMax, latestStep]);

  // Phase diagram, run in a worker.
  const [xKey, setXKey] = useState<NumericSetting>("yieldStrain");
  const [yKey, setYKey] = useState<NumericSetting>("motility");
  const [phaseSteps, setPhaseSteps] = useState(3000);
  const [points, setPoints] = useState<PhasePoint[]>([]);
  const [running, setRunning] = useState(false);
  const [phaseError, setPhaseError] = useState<string | null>(null);
  // The axes and shared settings of the run on screen. The selectors only configure the next one.
  const [phaseRun, setPhaseRun] = useState<{
    xKey: NumericSetting;
    yKey: NumericSetting;
    xLabel: string;
    yLabel: string;
    xs: readonly number[];
    ys: readonly number[];
    base: SpheroidConfig;
  } | null>(null);
  const worker = useRef<Worker | null>(null);
  const xAxis = SPHEROID_CONTROLS.find((c) => c.key === xKey) ?? SPHEROID_CONTROLS[0];
  const yAxis = SPHEROID_CONTROLS.find((c) => c.key === yKey) ?? SPHEROID_CONTROLS[0];
  const stopPhase = useCallback(() => {
    worker.current?.terminate();
    worker.current = null;
    setRunning(false);
  }, []);
  useEffect(() => stopPhase, [stopPhase]);
  const runPhase = () => {
    if (!xAxis || !yAxis || xKey === yKey) return;
    stopPhase();
    const base = { ...spheroid, ...PHASE_BASE };
    const steps = Math.max(1, Math.round(phaseSteps) || 1);
    setPoints([]);
    setPhaseError(null);
    setPhaseRun({ xKey, yKey, xLabel: xAxis.label, yLabel: yAxis.label, xs: xAxis.values, ys: yAxis.values, base });
    setRunning(true);
    let w: Worker;
    try {
      w = new Worker(new URL("./phaseWorker.ts", import.meta.url), { type: "module" });
    } catch (e) {
      setPhaseError(e instanceof Error ? e.message : String(e));
      setRunning(false);
      return;
    }
    worker.current = w;
    w.onerror = (e) => {
      setPhaseError(e.message || "The phase diagram worker failed.");
      stopPhase();
    };
    w.onmessage = (e: MessageEvent<PhaseMessage>) => {
      if (e.data.type === "point") {
        const point = e.data.point;
        setPoints((prev) => [...prev, point]);
      } else stopPhase();
    };
    w.postMessage({
      x: { key: xKey, values: xAxis.values },
      y: { key: yKey, values: yAxis.values },
      base,
      steps,
    });
  };
  const pick = (q: PhasePoint) => {
    if (!phaseRun) return;
    // The settings the square was measured with, not whatever the sliders say now.
    setSpheroid({ ...phaseRun.base, [phaseRun.xKey]: q.x, [phaseRun.yKey]: q.y });
    setKind("spheroid");
    setNonce((v) => v + 1);
  };

  const xs = useMemo(() => samples.map((s) => s.step), [samples]);
  const blurb = KINDS.find((k) => k.id === kind)?.blurb;
  const stress0 = relaxation.baseline?.stress ?? samples[0]?.stress ?? 1;
  const halfAt = kind === "relaxation" ? relaxation.halfStep : undefined;

  return (
    <div>
      <h2 className="page">Springs: bonds that break, cells that stick</h2>
      <p className="page">
        A port of the spring dynamics from{" "}
        <a href="https://csynth.github.io/csynth/" target="_blank" rel="noreferrer">
          CSynth
        </a>{" "}
        (Todd et al., Bioinformatics 2021), with three additions: bonds that break and re-form, adhesion between particles, and cells that
        divide. Everything on this page runs the same step function.
      </p>
      <p className="page prototype">
        <strong>First prototype.</strong> This page is a starting point for trying the model out, not a finished tool. The views are drawn
        on a 2D canvas for now, the settings are in the model's own arbitrary units, and the results have not been compared with{" "}
        <a href="https://doi.org/10.1038/s41563-022-01400-4" target="_blank" rel="noreferrer">
          the paper the model follows
        </a>{" "}
        (Elosegui-Artola et al., Nature Materials 2023). Treat what you see as a sketch of the behaviour.
      </p>
      <div className="layout">
        <div className="controls">
          <label>
            Setup
            <select
              value={kind}
              onChange={(e) => {
                const next = e.target.value as Kind;
                // The 3D setup runs on the GPU and allows far more steps a frame than the CPU can keep up with.
                if (next !== "ball") setStepsPerFrame((v) => Math.min(v, 20));
                setKind(next);
              }}
            >
              {KINDS.map((k) => (
                <option key={k.id} value={k.id}>
                  {k.label}
                </option>
              ))}
            </select>
          </label>
          <p className="hint">{blurb}</p>
          <div className="row">
            <button type="button" onClick={() => setPlaying((v) => !v)}>
              {playing ? "Pause" : "Play"}
            </button>
            <button type="button" onClick={() => setNonce((v) => v + 1)}>
              Restart
            </button>
            <button type="button" onClick={() => setLook(look === "analytic" ? "aesthetic" : "analytic")}>
              View: {look}
            </button>
          </div>
          <Slider
            label="Steps per frame"
            hint="simulation speed"
            value={stepsPerFrame}
            min={1}
            max={kind === "ball" ? 100 : 20}
            step={1}
            onChange={setStepsPerFrame}
          />

          {kind === "spheroid" && (
            <>
              <div className="row">
                <button type="button" onClick={() => setSpheroid({ yieldStrain: Number.POSITIVE_INFINITY })}>
                  Elastic gel
                </button>
                <button type="button" onClick={() => setSpheroid({})}>
                  Viscoelastic gel
                </button>
              </div>
              {SPHEROID_CONTROLS.map((c) => (
                <Slider
                  key={c.key}
                  label={c.label}
                  hint={c.hint}
                  min={c.min}
                  max={c.max}
                  step={c.step}
                  infinite={c.infinite}
                  value={spheroid[c.key] ?? DEFAULT_SPHEROID[c.key]}
                  onChange={(v) => setSpheroid({ ...spheroid, [c.key]: v })}
                />
              ))}
              <p className="hint">Settings act on the running tissue. Press Restart for a fresh one.</p>
            </>
          )}
          {kind === "ball" && (
            <>
              <div className="row presets">
                {BALL_PRESETS.map((preset) => (
                  <button type="button" key={preset.label} title={preset.hint} onClick={() => setBall(preset.config)}>
                    {preset.label}
                  </button>
                ))}
              </div>
              {BALL_CONTROLS.map((c) => (
                <Slider
                  key={c.key}
                  label={c.label}
                  hint={c.hint}
                  min={c.min}
                  max={c.max}
                  step={c.step}
                  infinite={c.infinite}
                  value={ball[c.key] ?? DEFAULT_SPHEROID[c.key]}
                  onChange={(v) => setBall({ ...ball, [c.key]: v })}
                />
              ))}
              <label className="check">
                <input
                  type="checkbox"
                  checked={ball.yieldInCompression === false}
                  onChange={(e) => setBall({ ...ball, yieldInCompression: !e.target.checked })}
                />
                Gel only fails where it is pulled apart
              </label>
              <p className="hint">
                Settings act on the running tissue. Arms take a few thousand steps to appear; raise steps per frame to get there sooner.
              </p>
            </>
          )}
          {kind === "relaxation" && (
            <>
              <Slider
                label="Bond lifetime"
                hint="mean steps before a bond lets go"
                value={relax.bondLifetime}
                min={20}
                max={1000}
                step={10}
                infinite="no random breaking"
                onChange={(v) => setRelax({ ...relax, bondLifetime: v })}
              />
              <Slider
                label="Yield strain"
                hint="strain past which a bond breaks"
                value={relax.yieldStrain}
                min={0.02}
                max={0.4}
                step={0.01}
                infinite="never"
                onChange={(v) => setRelax({ ...relax, yieldStrain: v })}
              />
            </>
          )}
          {kind === "draw" && (
            <>
              <Slider label="Chain length" hint="particles in the chain" value={drawN} min={24} max={120} step={4} onChange={setDrawN} />
              <div className="row">
                <button
                  type="button"
                  onClick={() =>
                    setLines([
                      ...lines,
                      helixLine(4 + ((lines.length * 9) % Math.max(1, drawN - 30)), 28 + ((lines.length * 9) % Math.max(1, drawN - 30)), 4),
                    ])
                  }
                >
                  Add helix
                </button>
                <button type="button" onClick={() => setLines([...lines, ...sheetLines(4, Math.min(drawN - 4, 44), 3, 3)])}>
                  Add sheet
                </button>
                <button type="button" onClick={() => setLines([])}>
                  Clear
                </button>
              </div>
              <Slider
                label="Spring length"
                hint="how far apart a tied pair settles, in particle diameters"
                value={lineRest}
                min={0.8}
                max={3}
                step={0.05}
                onChange={setLineRest}
              />
              <Slider
                label="Twist angle"
                hint="target twist, in radians, where a line ties near neighbours"
                value={twistAngle}
                min={-3.1}
                max={3.1}
                step={0.05}
                onChange={setTwistAngle}
              />
              <Slider
                label="Twist strength"
                hint="0 leaves twist free"
                value={twistStrength}
                min={0}
                max={0.05}
                step={0.001}
                onChange={setTwistStrength}
              />
              <p className="hint">
                Drag on the square to draw a line. Drag an end or the middle to move it; hold Ctrl to stop it snapping to a diagonal.
                Double-click or right-click a line to remove it.
              </p>
            </>
          )}
          {chain && (
            <>
              <Slider
                label="Chain length"
                hint="particles in the chain"
                value={chainN}
                min={60}
                max={gpu ? 1500 : 500}
                step={20}
                onChange={setChainN}
              />
              <label className="check">
                <input
                  type="checkbox"
                  checked={gpu}
                  onChange={(e) => {
                    // A fresh chain, within the limit of whichever side now runs it: nothing the
                    // device was still doing can land in a model the CPU has taken over.
                    if (!e.target.checked) setChainN((v) => Math.min(v, 500));
                    setGpu(e.target.checked);
                    setNonce((v) => v + 1);
                  }}
                />
                Run the step on the GPU
              </label>
              <Slider
                label="Damping"
                hint="fraction of velocity kept each step"
                value={live.damp ?? 0.9}
                min={0}
                max={0.99}
                step={0.01}
                onChange={(v) => setLive({ ...live, damp: v })}
              />
              <Slider
                label="Noise"
                hint="random kick each step; shakes the chain out of knots"
                value={live.noiseforce ?? 0}
                min={0}
                max={0.5}
                step={0.01}
                onChange={(v) => setLive({ ...live, noiseforce: v })}
              />
            </>
          )}
          {error && <div className="status error">{error}</div>}
        </div>

        <div className="views">
          <div className="stage">
            <canvas ref={canvasRef} width={720} height={720} className="sim-canvas" />
            <div className="side">
              {kind === "spheroid" && (
                <>
                  <LineChart
                    title="Tissue area"
                    xs={xs}
                    series={[{ label: "area", colour: "#2dd4bf", values: samples.map((s) => s.area) }]}
                  />
                  <LineChart
                    title="Circularity (1 = round disc)"
                    xs={xs}
                    yMin={0}
                    yMax={1}
                    series={[{ label: "4πA/P²", colour: "#e879f9", values: samples.map((s) => s.circularity) }]}
                  />
                  <LineChart title="Cells" xs={xs} series={[{ label: "cells", colour: "#fbbf24", values: samples.map((s) => s.cells) }]} />
                  <LineChart
                    title="Bonds broken so far"
                    xs={xs}
                    series={[{ label: "broken", colour: "#f87171", values: samples.map((s) => s.broken) }]}
                  />
                </>
              )}
              {kind === "ball" && (
                <>
                  <BarChart
                    title="Pieces of tissue crossing each shell (Sholl profile)"
                    values={samples.at(-1)?.sholl ?? []}
                    xLabel="distance from centre →"
                  />
                  <LineChart
                    title="Branches: most pieces in any outer shell"
                    xs={xs}
                    yMin={0}
                    series={[{ label: "branches", colour: "#e879f9", values: samples.map((q) => q.branches) }]}
                  />
                  <LineChart
                    title="Reach of the tissue from its centre"
                    xs={xs}
                    yMin={0}
                    series={[{ label: "radius", colour: "#2dd4bf", values: samples.map((q) => q.maxRadius) }]}
                  />
                  <LineChart
                    title="Cells"
                    xs={xs}
                    yMin={0}
                    series={[{ label: "cells", colour: "#fbbf24", values: samples.map((q) => q.cells) }]}
                  />
                  <p className="hint">
                    The bars cut space into shells around the tissue's centre and count the separate pieces of tissue in each. A ball gives
                    one piece per shell. Arms give several pieces in the shells they pass through, shown in pink. Leaders are red in the
                    picture; colour otherwise shows distance from the centre.
                  </p>
                </>
              )}
              {kind === "relaxation" && (
                <>
                  <LineChart
                    title="Stress carried by the strip"
                    xs={xs}
                    yMin={0}
                    series={[{ label: "stress", colour: "#f87171", values: samples.map((s) => s.stress) }]}
                    guides={[{ y: stress0 / 2, label: "half" }]}
                  />
                  <LineChart
                    title="Bonds"
                    xs={xs}
                    yMin={0}
                    series={[{ label: "bonds", colour: "#94a3b8", values: samples.map((s) => s.bonds) }]}
                  />
                  <p className="hint">
                    {Number.isFinite(relax.bondLifetime)
                      ? `Predicted half-time ${(relax.bondLifetime * Math.LN2).toFixed(0)} steps (lifetime × ln 2). ${halfAt ? `Measured: stress halved by step ${halfAt}, counted from step ${relaxation.baseline?.step ?? 40}.` : "Measured: not yet halved."}`
                      : "No random breaking: any relaxation here comes from bonds passing their yield strain."}
                  </p>
                </>
              )}
              {kind === "draw" && (
                <>
                  <MapEditor size={drawN} lines={lines} onChange={setLines} distances={drawDistances} />
                  <p className="hint">
                    Each white square is a tied pair; the picture is mirrored across the diagonal because a pair is the same either way
                    round. Behind the lines, dark means the two particles are close now.
                  </p>
                </>
              )}
              {chain && (
                <>
                  <LineChart
                    title="Match to target (1 = exact)"
                    xs={xs}
                    yMin={-0.2}
                    yMax={1}
                    series={[{ label: "correlation", colour: "#2dd4bf", values: samples.map((s) => s.correlation) }]}
                  />
                  <div className="maps">
                    <figure>
                      <canvas ref={targetMapRef} width={size || 1} height={size || 1} className="map" />
                      <figcaption>target distances</figcaption>
                    </figure>
                    <figure>
                      <canvas ref={modelMapRef} width={size || 1} height={size || 1} className="map" />
                      <figcaption>model distances</figcaption>
                    </figure>
                  </div>
                </>
              )}
            </div>
          </div>

          {look === "analytic" && (kind === "spheroid" || kind === "relaxation") && (
            <div className="legend">
              <span className="legend-row">
                <span className="swatch" style={{ background: "#2dd4bf" }} /> cell (whiter = more crowded)
              </span>
              <span className="legend-row">
                <span className="swatch" style={{ background: "#64748b" }} /> matrix particle
              </span>
              <span className="legend-row">
                <span className="swatch" style={{ background: "rgb(255,60,140)" }} /> stretched bond
              </span>
              <span className="legend-row">
                <span className="swatch" style={{ background: "rgb(120,60,255)" }} /> compressed bond
              </span>
              <span className="legend-row">
                <span className="swatch" style={{ background: "#facc15" }} /> cell–matrix bond
              </span>
              <span className="legend-row">○ pinned</span>
            </div>
          )}

          <h3>Phase diagram</h3>
          <p className="hint">
            Runs the spheroid once for every pair of values and colours each square by how round the tissue ended up. The number is how many
            times its area grew. Click a square to load that run above.
          </p>
          <div className="row phase-controls">
            <label>
              across
              <select value={xKey} onChange={(e) => setXKey(e.target.value as NumericSetting)}>
                {SPHEROID_CONTROLS.map((c) => (
                  <option key={c.key} value={c.key}>
                    {c.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              up
              <select value={yKey} onChange={(e) => setYKey(e.target.value as NumericSetting)}>
                {SPHEROID_CONTROLS.map((c) => (
                  <option key={c.key} value={c.key}>
                    {c.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              steps per run
              <input type="number" value={phaseSteps} min={500} step={500} onChange={(e) => setPhaseSteps(Number(e.target.value))} />
            </label>
            <button type="button" disabled={!running && xKey === yKey} onClick={running ? stopPhase : runPhase}>
              {running ? "Stop" : "Run"}
            </button>
          </div>
          {xKey === yKey && <p className="hint">Choose two different settings: one across and one up.</p>}
          {phaseError && <div className="status error">{phaseError}</div>}
          {phaseRun && (points.length > 0 || running) && (
            <>
              <PhaseGrid
                xLabel={phaseRun.xLabel}
                yLabel={phaseRun.yLabel}
                xs={phaseRun.xs}
                ys={phaseRun.ys}
                points={points}
                onPick={pick}
              />
              <div className="status">
                {points.length} of {phaseRun.xs.length * phaseRun.ys.length} runs done{running ? "…" : "."} Teal is round; magenta has
                broken symmetry.
              </div>
            </>
          )}

          <div className="explainers">
            <Explain title="What is a bond, here?">
              Every particle has a short list of the particles it is tied to, with a rest length and a strength for each tie. A tie is
              written in both particles' lists, so each particle can work out the pull on itself by reading only its own list. That is what
              lets a graphics card update every particle at once.
            </Explain>
            <Explain
              title="Why does breaking bonds make a material flow?"
              math={"\\sigma(t) = \\sigma_0\\, e^{-t/\\tau}, \\qquad t_{1/2} = \\tau \\ln 2"}
            >
              A stretched bond stores stress. If it lets go and a new bond forms at whatever length the gap now is, the new bond is relaxed,
              so that bit of stress is gone for good. A network whose bonds each last about τ steps loses its stress at that rate. The
              stress relaxation setup measures exactly this, and the half-time it reports is the number experimenters quote for a real gel.
            </Explain>
            <Explain title="How is roundness measured?" math={"C = \\frac{4\\pi A}{P^2}"}>
              The cells are blurred into a single blob and its outline is traced. Circularity compares the area A inside the outline with
              the area a circle of the same perimeter P would have. A disc scores 1; a shape with arms has far more edge for its area and
              scores low.
            </Explain>
            <Explain title="What does the phase diagram show?">
              The paper's model has three ratios that decide the outcome: how fast the matrix relaxes compared with how fast cells act, how
              fast cells divide, and how viscous the tissue is compared with the matrix. The controls here are the ingredients of those
              ratios. Yield strain, healing rate and bond lifetime set the matrix's relaxation; division time sets proliferation; matrix
              mobility sets the viscosity ratio.
            </Explain>
            <Explain title="How close is this to the paper?">
              It reproduces the main split: an elastic matrix keeps the tissue small and round, and a yielding one lets it grow about twice
              as large. How far the outline breaks symmetry varies from run to run, so press Restart a few times before reading much into
              one shape. It is not a quantitative reproduction. The force laws here are this project's own, not the authors', and the arms
              are lobes rather than long branches.
            </Explain>
          </div>
        </div>
      </div>
    </div>
  );
}
