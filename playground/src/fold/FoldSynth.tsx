// FoldSynth: fold a chain by drawing on its pair matrix, and watch it in 3D. A line parallel to the
// matrix's diagonal winds a helix; a line across it pairs two stretches of chain into a sheet.
// After FoldSynth by Peter Todd and Stephen Todd (matrix lines, torsion springs), on the spring
// model this repo ports from CSynth.
//
// The 3D view is deck.gl on WebGPU, drawn from the simulation's own buffers with
// `@intraspatial/deck` layers: the chain as an extruded tube, the surface its bonds enclose, and
// the path each particle has taken.
import { type Layer, OrbitView, type OrbitViewState } from "@deck.gl/core";
import DeckGL from "@deck.gl/react";
import {
  anchorUnder,
  BondLayer,
  type Color,
  cartoonProfile,
  hueRamp,
  IsoSurfaceLayer,
  nearestPoint,
  ParticleLayer,
  PivotOrbitController,
  type PivotPicker,
  SurfaceLayer,
  TraceLayer,
  TubeLayer,
} from "@intraspatial/deck";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ATTR_FLOATS,
  ATTR_RADIUS,
  chainRadius,
  DEFAULT_LINE_SPRINGS,
  helixLine,
  type MatrixLine,
  SLOT_WORDS,
  sheetLines,
} from "../../../src/gpu/sim/springs/index";
import { GLOBAL_DEVICE_PROPS, useGlobalDevice } from "../deck/useGlobalDevice";
import { useGpuFrameCost } from "../deck/useGpuFrameCost";
import { MatrixEditor } from "./MatrixEditor";
import { lineStructures } from "./structures";
import { useMetaballs } from "./useMetaballs";
import { useFold } from "./useFold";

/** Which way the history trace moves as it ages: a fixed direction in the model's space, or away from the viewer. */
type RecedeAlong = "back" | "down" | "side" | "camera";
const RECEDE_AXIS: Record<Exclude<RecedeAlong, "camera">, readonly [number, number, number]> = {
  back: [0, 0, -1],
  down: [0, -1, 0],
  side: [1, 0, 0],
};

const VIEW = new OrbitView({ id: "orbit", orbitAxis: "Y", fovy: 35 });
/** Colour along the chain, first particle to last: the same ramp on the tube, surface and trace. */
const CHAIN_RAMP = hueRamp(16);
const BEAD: readonly Color[] = [[226, 232, 240, 255]];
const FOLLOWED: readonly Color[] = [[250, 204, 21, 255]];
const PIVOT: readonly Color[] = [[34, 211, 238, 255]];
/** By bond kind: the backbone (0) is the tube, so only the ties the lines made (3) are drawn. */
const TIES: readonly Color[] = [
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [248, 250, 252, 170],
];

/** A helix and a three-strand sheet on a chain of `n`, as a starting point. */
function starterLines(n: number): MatrixLine[] {
  const third = Math.floor(n / 3);
  return [helixLine(2, third, 4), ...sheetLines(third + 4, n - 3, 3, 3)];
}

function Range(props: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label title={props.hint}>
      {props.label}: {props.value}
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </label>
  );
}

function Check(props: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="check">
      <input type="checkbox" checked={props.checked} onChange={(e) => props.onChange(e.target.checked)} />
      {props.label}
    </label>
  );
}

export function FoldSynth() {
  const { device, status } = useGlobalDevice();
  const cost = useGpuFrameCost(device);
  const [n, setN] = useState(64);
  const [seed, setSeed] = useState(1);
  const [playing, setPlaying] = useState(true);
  const [stepsPerFrame, setStepsPerFrame] = useState(4);
  const [lines, setLines] = useState<MatrixLine[]>(() => starterLines(64));
  const [rest, setRest] = useState(DEFAULT_LINE_SPRINGS.rest);
  const [twistAngle, setTwistAngle] = useState(DEFAULT_LINE_SPRINGS.torsionAngle);
  const [twistStrength, setTwistStrength] = useState(DEFAULT_LINE_SPRINGS.torsionStrength);
  const [noise, setNoise] = useState(0.02);
  const [show, setShow] = useState({ tube: true, surface: true, trace: true, ties: false, beads: false, blobs: false });
  const [tubeRadius, setTubeRadius] = useState(0.3);
  const [chainAs, setChainAs] = useState<"tube" | "cartoon">("cartoon");
  const [ribbon, setRibbon] = useState(2);
  const [structureColours, setStructureColours] = useState(true);
  const [blobReach, setBlobReach] = useState(1.6);
  const [blobLevel, setBlobLevel] = useState(0.5);
  const [blobDetail, setBlobDetail] = useState(96);
  const [traceLength, setTraceLength] = useState(128);
  const [traceMode, setTraceMode] = useState<"sheet" | "paths">("sheet");
  const [traceOpacity, setTraceOpacity] = useState(0.6);
  const [recede, setRecede] = useState(0.1);
  const [recedeAlong, setRecedeAlong] = useState<RecedeAlong>("back");

  const springs = useMemo(
    () => ({ ...DEFAULT_LINE_SPRINGS, rest, torsionAngle: twistAngle, torsionStrength: twistStrength }),
    [rest, twistAngle, twistStrength],
  );
  const live = useMemo(() => ({ noiseforce: noise }), [noise]);
  const fold = useFold({ device, n, seed, lines, springs, playing, stepsPerFrame, traceLength, traceEvery: 2, live });

  // The camera is the page's to move, so that it can follow the chain.
  const startView = useMemo(
    // 2^zoom pixels per model unit: the starting spiral, with room to fold, into about 500 pixels.
    (): OrbitViewState => ({ target: [0, 0, 0], rotationX: 20, rotationOrbit: 30, zoom: Math.log2(250 / (chainRadius(n) * 1.6)) }),
    [n],
  );
  const [view, setView] = useState<{ for: OrbitViewState; now: OrbitViewState }>({ for: startView, now: startView });
  if (view.for !== startView) setView({ for: startView, now: startView });
  const viewState = view.for === startView ? view.now : startView;

  // What the camera follows: nothing, the whole chain (its centre), or one bead (picked by a click).
  const [follow, setFollow] = useState<"none" | "chain" | "bead">("chain");
  const [bead, setBead] = useState<number | null>(null);
  /** The bead a turn in progress pivots on, when nothing is followed: shown while the drag lasts. */
  const [pivotBead, setPivotBead] = useState<number | null>(null);
  const dragging = useRef(false);
  const { positions } = fold;
  /** Where the followed thing is now: the bead, or the chain's centre. Null when nothing is followed. */
  const followed = useCallback((): [number, number, number] | null => {
    const p = positions();
    if (!p || follow === "none" || (follow === "bead" && bead === null)) return null;
    if (follow === "bead" && bead !== null) return [p[bead * 4] ?? 0, p[bead * 4 + 1] ?? 0, p[bead * 4 + 2] ?? 0];
    const count = p.length / 4;
    const centre: [number, number, number] = [0, 0, 0];
    for (let i = 0; i < count; i++) for (let k = 0; k < 3; k++) centre[k]! += p[i * 4 + k]! / count;
    return centre;
  }, [positions, follow, bead]);
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      // Not while the pointer is down: the gesture has the camera until it is let go.
      const goal = dragging.current ? null : followed();
      if (goal) {
        setView((v) => {
          const [x, y, z] = v.now.target;
          // Ease toward it: the host's copy of the positions arrives a few times a second.
          const next: [number, number, number] = [x + (goal[0] - x) * 0.12, y + (goal[1] - y) * 0.12, z + (goal[2] - z) * 0.12];
          const moved = Math.hypot(next[0] - x, next[1] - y, next[2] - z);
          return moved < 1e-3 ? v : { ...v, now: { ...v.now, target: next } };
        });
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [followed]);

  // Turning: about the followed thing when there is one, else about the bead under the pointer.
  const getPivot = useCallback<PivotPicker>(
    (pos, viewport) => {
      const tracked = followed();
      if (tracked) return tracked;
      const p = positions();
      const hit = p ? nearestPoint(p, pos, viewport, { maxPixels: 30 }) : null;
      setPivotBead(hit?.index ?? null);
      return hit?.position ?? null;
    },
    [positions, followed],
  );
  // Zoom and pan hold on to the bead under the pointer, at its depth, when nothing is followed.
  const getAnchor = useCallback<PivotPicker>(
    (pos, viewport) => {
      const p = positions();
      return p && follow === "none" ? anchorUnder(p, pos, viewport, { maxPixels: 30 }) : null;
    },
    [positions, follow],
  );
  const controller = useMemo(
    // While following, zoom keeps the followed thing in the middle instead of chasing the pointer.
    () => ({ type: PivotOrbitController, getPivot, getAnchor, zoomAround: follow === "none" ? ("pointer" as const) : ("center" as const) }),
    [getPivot, getAnchor, follow],
  );
  const onViewStateChange = useCallback(
    ({ viewState: next, interactionState }: { viewState: unknown; interactionState: { isPanning?: boolean } }) => {
      // Panning is the user taking the camera somewhere else: stop following.
      if (interactionState.isPanning) setFollow("none");
      setView((v) => ({ ...v, now: next as OrbitViewState }));
    },
    [],
  );
  // The pointer going down and coming up: follow waits while it is down, and the pivot mark goes when it is up.
  const onInteractionStateChange = useCallback((state: { isDragging?: boolean }) => {
    dragging.current = state.isDragging === true;
    if (!dragging.current) setPivotBead(null);
  }, []);
  const onClick = useCallback(
    (info: { x: number; y: number; viewport?: { project(xyz: number[]): number[] } }) => {
      const p = positions();
      const hit = p && info.viewport ? nearestPoint(p, [info.x, info.y], info.viewport, { maxPixels: 30 }) : null;
      if (!hit) return;
      setBead(hit.index);
      setFollow("bead");
    },
    [positions],
  );

  // The cartoon: what each line is building, as a section for the tube. Null is the round tube.
  const profile = useMemo(
    () =>
      chainAs === "cartoon"
        ? cartoonProfile(n, lineStructures(lines, n), {
            broad: ribbon,
            ...(structureColours ? {} : { helixColor: null, sheetColor: null }),
          })
        : null,
    [chainAs, lines, n, ribbon, structureColours],
  );

  const { gpu, history, faces } = fold;
  const blobs = useMetaballs({
    device,
    source: gpu,
    hostPositions: fold.positions,
    enabled: show.blobs,
    resolution: blobDetail,
    radius: blobReach,
  });
  const layers = useMemo(() => {
    if (!gpu) return [];
    const positions = () => gpu.positions();
    const out: Layer[] = [];
    if (show.surface)
      out.push(new SurfaceLayer({ id: "surface", positions, indices: faces, count: gpu.n, colors: CHAIN_RAMP, opacity: 0.9 }));
    if (show.tube) {
      out.push(
        new TubeLayer({
          id: "tube",
          positions,
          count: gpu.n,
          radius: tubeRadius,
          colors: CHAIN_RAMP,
          // Only once the chain is the profile's length: a new chain's first frames may still be the old one's.
          profile: profile && gpu.n === n ? profile : null,
          sides: 16,
        }),
      );
    }
    if (show.blobs && blobs) {
      out.push(new IsoSurfaceLayer({ id: "blobs", ...blobs, threshold: blobLevel, color: [125, 211, 252, 255] }));
    }
    if (show.beads) {
      out.push(
        new ParticleLayer({
          id: "beads",
          positions,
          count: gpu.n,
          attributes: gpu.attrs,
          attributeStride: ATTR_FLOATS,
          radiusOffset: ATTR_RADIUS,
          radiusScale: 0.8,
          colors: BEAD,
          // No category is read (offset 0 is not one): every bead takes the one colour.
          categoryOffset: 0,
          deadCategory: -1,
        }),
      );
    }
    if (show.ties) {
      out.push(
        new BondLayer({
          id: "ties",
          positions,
          slots: gpu.slots,
          count: gpu.n,
          slotsPerParticle: gpu.K,
          slotWords: SLOT_WORDS,
          colors: TIES,
        }),
      );
    }
    if (show.trace && history) {
      out.push(
        new TraceLayer({
          id: "trace",
          history,
          mode: traceMode,
          colors: CHAIN_RAMP,
          opacity: traceOpacity,
          ...(recedeAlong === "camera"
            ? { recede }
            : {
                drift: [
                  RECEDE_AXIS[recedeAlong][0] * recede,
                  RECEDE_AXIS[recedeAlong][1] * recede,
                  RECEDE_AXIS[recedeAlong][2] * recede,
                ] as const,
              }),
          // A see-through sheet must not hide what is behind it, so it leaves the depth buffer alone.
          parameters: { depthWriteEnabled: traceOpacity >= 1 },
        }),
      );
    }
    // Markers, drawn over everything. The followed bead wears a yellow ring; the bead a turn pivots
    // on, when nothing is followed, shows a small cyan disc for as long as the drag lasts.
    const marker = { positions, count: 1, shading: 0, parameters: { depthCompare: "always" as const } };
    if (follow === "bead" && bead !== null && bead < gpu.n) {
      out.push(
        new ParticleLayer({ ...marker, id: "followed", first: bead, style: "ring", radius: 0.9, radiusMinPixels: 9, colors: FOLLOWED }),
      );
    }
    if (pivotBead !== null && pivotBead < gpu.n) {
      out.push(
        new ParticleLayer({ ...marker, id: "pivot", first: pivotBead, style: "disc", radius: 0.22, radiusMinPixels: 4, colors: PIVOT }),
      );
    }
    return out;
  }, [
    gpu,
    history,
    faces,
    show,
    blobs,
    blobLevel,
    tubeRadius,
    profile,
    n,
    traceMode,
    traceOpacity,
    recede,
    recedeAlong,
    follow,
    bead,
    pivotBead,
  ]);

  const restart = (length = n) => {
    setN(length);
    setSeed((s) => s + 1);
  };
  const toggle = (key: keyof typeof show) => (v: boolean) => setShow({ ...show, [key]: v });
  const at = (lines.length * 9) % Math.max(1, n - 30);

  return (
    <>
      <h2 className="page">FoldSynth: fold a chain by drawing on its matrix</h2>
      <p className="page">
        A chain of beads, joined in a row like a protein's backbone. The square has one cell for every pair of beads, and a line drawn
        across it ties each pair it crosses with a spring. A line running alongside the diagonal ties each bead to one a few places further
        on, which winds a helix. A line running across the diagonal pairs two stretches of the chain side by side, which makes a sheet.
      </p>
      <p className="page prototype">
        <strong>First prototype.</strong> A sketch of FoldSynth's drawing tools on this repo's spring model, to try the 3D views. The beads
        are not amino acids and the springs are not real forces between them: nothing here predicts how a protein folds.
      </p>
      <div className="layout">
        <div className="controls">
          <div className="row">
            <button type="button" onClick={() => setPlaying((v) => !v)}>
              {playing ? "Pause" : "Play"}
            </button>
            <button type="button" onClick={() => restart()}>
              Restart
            </button>
          </div>
          <MatrixEditor device={device} gpu={fold.gpu} lines={lines} onChange={setLines} />
          <p className="hint">
            Drag on the square to draw a line. Drag an end or the middle to move it; hold Ctrl to stop it snapping to a diagonal.
            Double-click or right-click a line to remove it. The picture behind is how far apart each pair of beads is now (dark is near);
            pale cells are pairs tied by a spring.
          </p>
          <div className="row">
            <button type="button" onClick={() => setLines([...lines, helixLine(4 + at, 28 + at, 4)])}>
              Add helix
            </button>
            <button type="button" onClick={() => setLines([...lines, ...sheetLines(4, Math.min(n - 4, 44), 3, 3)])}>
              Add sheet
            </button>
            <button type="button" onClick={() => setLines([])}>
              Clear
            </button>
          </div>
          <Range
            label="Chain length"
            hint="beads in the chain; changing it starts a fresh chain"
            value={n}
            min={32}
            max={256}
            step={8}
            onChange={(v) => {
              setLines(starterLines(v));
              restart(v);
            }}
          />
          <Range label="Steps per frame" value={stepsPerFrame} min={1} max={40} step={1} onChange={setStepsPerFrame} />
          <Range
            label="Spring length"
            hint="how far apart a tied pair settles, in bead diameters"
            value={rest}
            min={0.8}
            max={3}
            step={0.05}
            onChange={setRest}
          />
          <Range
            label="Twist angle"
            hint="target twist, in radians, where a line ties near neighbours"
            value={twistAngle}
            min={-3.1}
            max={3.1}
            step={0.05}
            onChange={setTwistAngle}
          />
          <Range
            label="Twist strength"
            hint="0 leaves twist free"
            value={twistStrength}
            min={0}
            max={0.05}
            step={0.001}
            onChange={setTwistStrength}
          />
          <Range label="Jiggle" hint="random force on every bead" value={noise} min={0} max={0.3} step={0.005} onChange={setNoise} />
          {fold.skipped > 0 && <div className="status error">{fold.skipped} ties were left out: a bead can hold only so many springs.</div>}
          {fold.error && <div className="status error">{fold.error}</div>}
        </div>

        <div className="views">
          <div className="deck-canvas">
            <DeckGL
              views={VIEW}
              viewState={viewState}
              onViewStateChange={onViewStateChange}
              onInteractionStateChange={onInteractionStateChange}
              controller={controller}
              onClick={onClick}
              layers={layers}
              deviceProps={GLOBAL_DEVICE_PROPS}
              onAfterRender={cost.onAfterRender}
              // The buffers change under the layers every step, which deck cannot see: draw every frame.
              _animate
            />
            <div className="view-status">
              {status} · step {fold.step.toLocaleString()} · {cost.ms === null ? "–" : cost.ms.toFixed(1)} ms of GPU work per frame
            </div>
          </div>
          <div className="row toggles">
            <Check label="Tube" checked={show.tube} onChange={toggle("tube")} />
            <Check label="Surface" checked={show.surface} onChange={toggle("surface")} />
            <Check label="History trace" checked={show.trace} onChange={toggle("trace")} />
            <Check label="Ties" checked={show.ties} onChange={toggle("ties")} />
            <Check label="Beads" checked={show.beads} onChange={toggle("beads")} />
            <Check label="Blobs" checked={show.blobs} onChange={toggle("blobs")} />
            <label>
              Camera follows
              <select value={follow} onChange={(e) => setFollow(e.target.value as typeof follow)}>
                <option value="none">nothing</option>
                <option value="chain">the whole chain</option>
                <option value="bead">{bead === null ? "a bead (click one)" : `bead ${bead}`}</option>
              </select>
            </label>
          </div>
          <p className="hint">
            Drag to turn the chain. While the camera follows something, it turns about that: a followed bead wears a{" "}
            <span style={{ color: "#facc15" }}>yellow ring</span>. With nothing followed it turns about the bead under the pointer, marked
            with a <span style={{ color: "#22d3ee" }}>cyan dot</span> while you drag. Click a bead to follow it. Shift-drag to move the view
            sideways, which stops the following. Scroll to zoom.
          </p>
          <div className="row toggles">
            <Range
              label="Blob reach"
              hint="how far each bead's influence spreads, in bead diameters"
              value={blobReach}
              min={0.8}
              max={4}
              step={0.1}
              onChange={setBlobReach}
            />
            <Range
              label="Blob level"
              hint="the skin is drawn where the beads' summed influence equals this; lower is fatter"
              value={blobLevel}
              min={0.05}
              max={2}
              step={0.05}
              onChange={setBlobLevel}
            />
            <Range
              label="Blob detail"
              hint="voxels along each edge of the box the skin is worked out in"
              value={blobDetail}
              min={32}
              max={160}
              step={16}
              onChange={setBlobDetail}
            />
            <label>
              Chain drawn as
              <select value={chainAs} onChange={(e) => setChainAs(e.target.value === "tube" ? "tube" : "cartoon")}>
                <option value="cartoon">a cartoon: ribbons and arrows</option>
                <option value="tube">a round tube</option>
              </select>
            </label>
            <Range label="Tube radius" value={tubeRadius} min={0.05} max={0.6} step={0.01} onChange={setTubeRadius} />
            <Range
              label="Ribbon width"
              hint="how wide a helix ribbon or sheet arrow is, in tube radii (cartoon only)"
              value={ribbon}
              min={1}
              max={6}
              step={0.1}
              onChange={setRibbon}
            />
            <Check label="Colour helices and sheets" checked={structureColours} onChange={setStructureColours} />
            <Range
              label="Trace length"
              hint="how many past moments the trace keeps"
              value={traceLength}
              min={16}
              max={320}
              step={16}
              onChange={setTraceLength}
            />
            <Range label="Trace opacity" value={traceOpacity} min={0.05} max={1} step={0.05} onChange={setTraceOpacity} />
            <Range
              label="Trace recedes"
              hint="how far each older moment of the trace is moved along, in bead diameters"
              value={recede}
              min={0}
              max={0.5}
              step={0.01}
              onChange={setRecede}
            />
            <label>
              It recedes
              <select value={recedeAlong} onChange={(e) => setRecedeAlong(e.target.value as RecedeAlong)}>
                <option value="back">backward in the model's space</option>
                <option value="down">downward in the model's space</option>
                <option value="side">sideways in the model's space</option>
                <option value="camera">away from the viewer</option>
              </select>
            </label>
            <label>
              Trace drawn as
              <select value={traceMode} onChange={(e) => setTraceMode(e.target.value === "paths" ? "paths" : "sheet")}>
                <option value="sheet">a sheet swept by the chain</option>
                <option value="paths">a line per bead</option>
              </select>
            </label>
          </div>
          <dl className="explain">
            <dt>Tube</dt>
            <dd>
              The chain itself, as a smooth tube through the beads in order. Colour runs from one end of the chain to the other. As a
              cartoon, the way proteins are usually drawn, a stretch a line is winding into a helix becomes a broad ribbon (pink) facing out
              from the helix, and a stretch a line is pairing into a sheet becomes a flat arrow (yellow) lying in the sheet and pointing
              along the chain. The cartoon shows what the lines ask for, straight away, whether or not the chain has folded that way yet;
              when you change a line the shapes ease from the old to the new.
            </dd>
            <dt>Surface</dt>
            <dd>
              Wherever springs close a small loop of three or four beads, the loop is filled in. Two neighbouring strands of a sheet are
              joined by a ladder of such loops, so a sheet shows as a sheet; a helix shows as a band winding between its turns.
            </dd>
            <dt>History trace</dt>
            <dd>
              The surface the chain has swept through as it moved. One edge is the chain now; going across the sheet is going back in time,
              and it fades as it gets older. With "recedes" above zero each older moment is also moved along a fixed direction in the
              model's space, so the past streams away from the chain instead of piling up on it, and turns with the model when you orbit. A
              stretch that has settled leaves no sheet; one that is still folding leaves a wide one.
            </dd>
            <dt>Blobs</dt>
            <dd>
              Metaballs: every bead spreads a soft influence around itself, strongest at the bead and gone by "reach". The influences are
              added up at each point of a grid of small cubes around the chain, and a skin is drawn wherever the total equals "level". Beads
              close together merge into one body; the same drawing will show a 3D density of measured points.
            </dd>
            <dt>Ties</dt>
            <dd>The springs the lines made, as thin white lines.</dd>
          </dl>
        </div>
      </div>
    </>
  );
}
