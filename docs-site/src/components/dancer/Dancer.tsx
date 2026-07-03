// The standalone dancer artefact — a live 3D Ceilidh of force-fields (three.js WebGPU),
// with a fullscreen stage, the distance-matrix lens, and the cultural framing. The sim is
// the direct CPU loop in ./sim (the same math as the composer's building-block ops); the
// renderer is three.js on its WebGPU backend. `client:only="react"` — everything here is
// browser-only.
import { useEffect, useRef, useState } from "react";
import { OnePole } from "../../../../src/gpu/graph/onePole";
import { DancerGpuSim } from "../../../../src/gpu/sim/dancerGpu";
import { BreedingStrip } from "./BreedingStrip";
import { drawDistanceMatrix, matrixCell } from "./matrix";
import { createDancerRenderer, type DancerRenderer } from "./renderer";
import { type DancerParams, DancerSim, DEFAULT_DANCER_PARAMS } from "./sim";

const AGENT_OPTIONS = [180, 600, 1200, 2400, 4800] as const;
const DEFAULT_AGENTS = 180;

// Genome ↔ vector, for one-pole smoothing of TraitSpace transitions: adopting a specimen eases the
// whole genome (motion + look) toward the new point instead of snapping. τ in frames (~0.4s @60fps).
const PARAM_KEYS = Object.keys(DEFAULT_DANCER_PARAMS) as (keyof DancerParams)[];
const PARAM_TAU = 24;
const paramsToVec = (p: DancerParams, out: Float32Array): Float32Array => {
  PARAM_KEYS.forEach((key, k) => {
    out[k] = p[key];
  });
  return out;
};
const vecToParams = (vec: Float32Array, into: DancerParams): void => {
  PARAM_KEYS.forEach((key, k) => {
    into[key] = vec[k] ?? into[key];
  });
};
// The swarm runs GPU-resident on our own TypeGPU kernel (src/gpu/sim/dancerGpu), adopting
// three.js's WebGPU device so our compute writes three's instanceMatrix buffer directly — the
// render reads pose (position + orientation) off the GPU with no per-frame readback (smooth
// under camera drag; cones face their travel). A throttled snapshot feeds the CPU-side panels.
// If GPU init throws, we transparently fall back to the (golden) CPU sim. Set false for CPU.
const USE_GPU = true;

export default function Dancer() {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const matrixRef = useRef<HTMLCanvasElement | null>(null);
  const [figure, setFigure] = useState("…");
  const [status, setStatus] = useState<"init" | "running" | "unsupported" | "error">("init");
  const [showAbout, setShowAbout] = useState(true);
  const [showBreed, setShowBreed] = useState(false);
  const [showMatrix, setShowMatrix] = useState(false); // hidden by default (more interesting for e.g. protein folding)
  const [paused, setPaused] = useState(false); // freeze the Ceilidh figure progression
  const [agents, setAgents] = useState<number>(DEFAULT_AGENTS); // whole pipeline rebuilds on change
  const [breedDevice, setBreedDevice] = useState<GPUDevice | null>(null); // shared with the breeding cells
  const simRef = useRef<DancerSim | DancerGpuSim | null>(null);
  const paramsRef = useRef<DancerParams>(DEFAULT_DANCER_PARAMS); // full genome incl. render traits (for the stage renderer)
  // cross-link state, read by the render loop each frame (refs → no re-render on hover)
  const hoverAgentRef = useRef<number | null>(null); // hovered dancer (3D) → matrix row
  const hoverCellRef = useRef<[number, number] | null>(null); // hovered matrix cell → 3D pair
  const showMatrixRef = useRef(false); // mirror for the render loop (avoids re-subscribing the effect)

  useEffect(() => {
    const canvas = canvasRef.current;
    const stage = stageRef.current;
    if (!canvas || !stage) return;
    if (typeof navigator === "undefined" || !("gpu" in navigator)) {
      setStatus("unsupported");
      return;
    }

    let renderer: DancerRenderer | null = null;
    let raf = 0;
    let disposed = false;
    let frame = 0;
    const sim: DancerSim = new DancerSim(agents, 1); // CPU golden (fallback + peripherals)
    let gpu: DancerGpuSim | null = null;
    simRef.current = sim;

    const resize = () => {
      if (!renderer) return;
      const r = stage.getBoundingClientRect();
      renderer.resize(Math.max(1, r.width), Math.max(1, r.height));
    };

    const cleanups: Array<() => void> = [];

    createDancerRenderer(canvas, agents)
      .then(async (r) => {
        if (disposed) {
          r.dispose();
          return;
        }
        renderer = r;
        resize();
        // publish the stage's device so the breeding cells can share it (one device, many canvases)
        setBreedDevice((r.renderer.backend as unknown as { device?: GPUDevice }).device ?? null);

        if (USE_GPU) {
          try {
            const device = (r.renderer.backend as unknown as { device?: GPUDevice }).device;
            if (!device) throw new Error("no WebGPU device on renderer backend");
            const g = new DancerGpuSim(device, agents, 1, sim.params);
            // three owns the per-instance state buffers; the sim computes its pos/angPos/vel/angVel
            // INTO them, so the render builds the model transform + traits straight off the GPU in the
            // shader (no readback, no matrix/channel bridge kernels).
            const stateBufs = await r.gpuStateBuffers();
            if (!stateBufs) throw new Error("no render state buffers");
            g.init(stateBufs);
            // share three's trail-history GPUBuffer so our compute appends into the ring three renders
            // the trails from (GPU-resident; no CPU snapshot feeding the trail geometry).
            const trailBuf = await r.gpuTrailBuffer();
            if (trailBuf) g.setTrailTarget(trailBuf, r.trailCapacity());
            if (disposed) return;
            gpu = g;
            simRef.current = g;
          } catch (err) {
            console.warn("dancer: GPU sim unavailable, using CPU", err);
            r.setCpuMode(true);
          }
        }
        r.setRenderTraits(paramsRef.current); // seed the colour/size mapping from the current genome
        setStatus("running");

        // hover a dancer (3D) → remember it (highlights its matrix row)
        const onStageMove = (e: PointerEvent) => {
          if (!renderer) return;
          const rect = canvas.getBoundingClientRect();
          const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
          const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
          hoverAgentRef.current = renderer.pick(x, y);
        };
        const onStageLeave = () => {
          hoverAgentRef.current = null;
        };
        canvas.addEventListener("pointermove", onStageMove);
        canvas.addEventListener("pointerleave", onStageLeave);
        cleanups.push(() => {
          canvas.removeEventListener("pointermove", onStageMove);
          canvas.removeEventListener("pointerleave", onStageLeave);
        });

        // hover a matrix cell → remember the pair (highlights it + a line in 3D)
        const mcanvas = matrixRef.current;
        if (mcanvas) {
          const onMatrixMove = (e: PointerEvent) => {
            hoverCellRef.current = matrixCell(mcanvas, e.clientX, e.clientY, sim.n);
          };
          const onMatrixLeave = () => {
            hoverCellRef.current = null;
          };
          mcanvas.addEventListener("pointermove", onMatrixMove);
          mcanvas.addEventListener("pointerleave", onMatrixLeave);
          cleanups.push(() => {
            mcanvas.removeEventListener("pointermove", onMatrixMove);
            mcanvas.removeEventListener("pointerleave", onMatrixLeave);
          });
        }

        // GPU path: step() enqueues the sim + writes three's instanceMatrix (no readback) —
        // the render reads pose straight off the GPU, so camera motion never stalls. A
        // low-frequency snapshot feeds the CPU-side panels (colour, trails, matrix, pick).
        const zeros = new Float32Array(agents * 3);
        let snapPos: Float32Array = new Float32Array(agents * 3);
        let snapSpeed: Float32Array = new Float32Array(agents);
        let snapping = false;
        const speedsFrom = (vel: Float32Array): Float32Array => {
          const out = new Float32Array(agents);
          for (let i = 0; i < agents; i++) out[i] = Math.hypot(vel[i * 3] ?? 0, vel[i * 3 + 1] ?? 0, vel[i * 3 + 2] ?? 0);
          return out;
        };

        // Smooth TraitSpace transitions: adopting a specimen eases the whole genome (sim forces +
        // render traits) toward it with a one-pole lag rather than snapping. `liveParams` is the eased
        // genome; the active sim reads it BY REFERENCE, so easing it morphs the dance, and the
        // renderer's colour/size mapping tracks the same eased values.
        const liveParams: DancerParams = { ...paramsRef.current };
        const smoother = new OnePole(PARAM_KEYS.length, { tau: PARAM_TAU, initial: paramsToVec(paramsRef.current, new Float32Array(PARAM_KEYS.length)) });
        const targetVec = new Float32Array(PARAM_KEYS.length);
        sim.params = liveParams;
        if (gpu) gpu.params = liveParams;

        const loop = () => {
          if (disposed || !renderer) return;
          const ha = hoverAgentRef.current;
          const hc = hoverCellRef.current;
          const highlightAgents: number[] = [];
          if (ha !== null) highlightAgents.push(ha);
          if (hc) highlightAgents.push(hc[0], hc[1]);
          // ease the genome toward the adopted target, then drive sim + render from the eased values
          paramsToVec(paramsRef.current, targetVec);
          vecToParams(smoother.push(targetVec), liveParams);
          renderer.setRenderTraits(liveParams);

          if (gpu) {
            gpu.step(); // advance + write instanceMatrix + append trail ring on the GPU (no readback)
            renderer.setTrailHead(gpu.trailHead()); // point the trail shader at the newest slot
            // The snapshot readback (for CPU-side trails/colour/matrix) mapAsync-syncs the GPU;
            // suspend it during camera interaction so it can't contend with the drag render (the
            // periodic pause). Trails/colour briefly freeze while dragging — acceptable until
            // they move onto the GPU.
            if (frame % 12 === 0 && !snapping && !renderer.isInteracting()) {
              snapping = true;
              gpu
                .readBlocks()
                .then((b) => {
                  snapPos = b.pos;
                  snapSpeed = speedsFrom(b.vel);
                })
                .catch(() => {})
                .finally(() => {
                  snapping = false;
                });
            }
            renderer.update(snapPos, zeros, snapSpeed, { agents: highlightAgents, pair: hc });
            renderer.render();
            if (matrixRef.current && showMatrixRef.current) drawDistanceMatrix(matrixRef.current, snapPos, gpu.n, { row: ha, pair: hc });
            if (frame % 15 === 0) setFigure(gpu.currentFigure());
          } else {
            sim.step();
            const p = sim.positions();
            renderer.update(p, sim.orientations(), sim.speeds(), { agents: highlightAgents, pair: hc });
            renderer.render();
            if (matrixRef.current && showMatrixRef.current) drawDistanceMatrix(matrixRef.current, p, sim.n, { row: ha, pair: hc });
            if (frame % 15 === 0) setFigure(sim.currentFigure());
          }
          frame++;
          raf = requestAnimationFrame(loop);
        };
        raf = requestAnimationFrame(loop);
      })
      .catch((e) => {
        console.error("dancer renderer failed", e);
        setStatus("error");
      });

    const ro = new ResizeObserver(resize);
    ro.observe(stage);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      for (const c of cleanups) c();
      setBreedDevice(null); // the shared device goes away with the renderer
      renderer?.dispose();
    };
  }, [agents]); // rebuild the whole pipeline (renderer + sim + buffers) when the agent count changes

  useEffect(() => {
    showMatrixRef.current = showMatrix;
  }, [showMatrix]);

  useEffect(() => {
    simRef.current?.pauseFigures(paused);
  }, [paused]);

  const toggleFullscreen = () => {
    const stage = stageRef.current;
    if (!stage) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void stage.requestFullscreen();
  };

  return (
    <div className="dancer-root">
      <div className="dancer-stage" ref={stageRef}>
        <canvas ref={canvasRef} className="dancer-canvas" />

        <div className="dancer-hud">
          <div className="dancer-figure">
            <span className="dancer-figure-label">figure</span> {figure}
          </div>
          <div className="dancer-hud-right">
            <label className="dancer-fs" title="number of dancers — rebuilds the simulation">
              ⦿{" "}
              <select
                value={agents}
                onChange={(e) => setAgents(Number(e.target.value))}
                style={{ background: "transparent", border: "none", color: "inherit", font: "inherit", cursor: "pointer" }}
              >
                {AGENT_OPTIONS.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <button type="button" className="dancer-fs" onClick={() => setPaused((v) => !v)} aria-pressed={paused} title="hold the current Ceilidh figure (motion keeps running)">
              {paused ? "▶ resume figures" : "⏸ hold figure"}
            </button>
            <button type="button" className="dancer-fs" onClick={() => setShowMatrix((v) => !v)} aria-pressed={showMatrix} title="pairwise distance matrix">
              ▦ {showMatrix ? "hide matrix" : "matrix"}
            </button>
            <button
              type="button"
              className="dancer-fs"
              onClick={() => {
                const next = !showBreed;
                setShowBreed(next);
                if (next) setShowAbout(false);
              }}
              aria-pressed={showBreed}
            >
              ⚘ {showBreed ? "hide breeder" : "breed"}
            </button>
            <button type="button" className="dancer-fs" onClick={toggleFullscreen} aria-label="fullscreen">⤢ fullscreen</button>
          </div>
        </div>

        {showBreed && (
          <BreedingStrip
            device={breedDevice}
            onAdopt={(p: DancerParams) => {
              paramsRef.current = p; // set the TARGET genome; the loop eases the live sim + render toward it
            }}
          />
        )}

        {/* kept mounted (ref stable for the hover cross-link); just hidden + not drawn when off */}
        <div
          className="dancer-matrix"
          style={{ display: showMatrix ? undefined : "none" }}
          title="pairwise distance matrix — couples are off-diagonal hot pairs"
        >
          <canvas ref={matrixRef} className="dancer-matrix-canvas" width={96} height={96} />
          <div className="dancer-matrix-cap">distance matrix</div>
        </div>

        {status === "unsupported" && (
          <div className="dancer-overlay">This piece needs WebGPU. Try a recent Chrome, Edge, or Safari.</div>
        )}
        {status === "error" && <div className="dancer-overlay">The 3D stage failed to start (see console).</div>}

        {showAbout && (
          <aside className="dancer-about">
            <button type="button" className="dancer-about-close" onClick={() => setShowAbout(false)} aria-label="close">×</button>
            <h1>Folk-Algorithm based Dance Simulation</h1>
            <p>
              After <em>DANCERL</em> (Andy Lomas, IBM, 1992) — the force-field motion controller for
              William Latham's SIGGRAPH film, written in the ESME language designed by Stephen Todd and colleagues at IBM Winchester.
              <br />
              Experiment in vibe-coding based on a knowledge passed down through the generations, and a reflection on the current
              xeitgeist. I always loved the way my Dad describes how this algorithm behaves; like a very chaotic Ceilidh where no-one knows
              what they're supposed to be doing, and all frantically try to move on to the correct movement as the caller calls it.
              <br />
              As of this writing, the actual simulation isn't yet doing anything I consider very interesting, but hopefully will at some point.
            </p>
            <p className="dancer-about-note">
              A 2026 reconstruction by Claude Code, directed by Peter Todd. Drag to orbit · ⤢ for fullscreen.
            </p>
          </aside>
        )}
        {!showAbout && (
          <button type="button" className="dancer-about-open" onClick={() => setShowAbout(true)}>about</button>
        )}
      </div>
    </div>
  );
}
