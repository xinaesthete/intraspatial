// The standalone dancer artefact — a live 3D Ceilidh of force-fields (three.js WebGPU),
// with a fullscreen stage, the distance-matrix lens, and the cultural framing. The sim is
// the direct CPU loop in ./sim (the same math as the composer's building-block ops); the
// renderer is three.js on its WebGPU backend. `client:only="react"` — everything here is
// browser-only.
import { useEffect, useRef, useState } from "react";
import { DancerSim, type DancerParams } from "./sim";
import { DancerGpuSim } from "../../../../src/gpu/sim/dancerGpu";
import { createDancerRenderer, type DancerRenderer } from "./renderer";
import { drawDistanceMatrix, matrixCell } from "./matrix";
import { BreedingStrip } from "./BreedingStrip";

const AGENTS = 180;
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
  const simRef = useRef<DancerSim | DancerGpuSim | null>(null);
  // cross-link state, read by the render loop each frame (refs → no re-render on hover)
  const hoverAgentRef = useRef<number | null>(null); // hovered dancer (3D) → matrix row
  const hoverCellRef = useRef<[number, number] | null>(null); // hovered matrix cell → 3D pair

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
    const sim: DancerSim = new DancerSim(AGENTS, 1); // CPU golden (fallback + peripherals)
    let gpu: DancerGpuSim | null = null;
    simRef.current = sim;

    const resize = () => {
      if (!renderer) return;
      const r = stage.getBoundingClientRect();
      renderer.resize(Math.max(1, r.width), Math.max(1, r.height));
    };

    const cleanups: Array<() => void> = [];

    createDancerRenderer(canvas, AGENTS)
      .then(async (r) => {
        if (disposed) {
          r.dispose();
          return;
        }
        renderer = r;
        resize();

        if (USE_GPU) {
          try {
            const device = (r.renderer.backend as unknown as { device?: GPUDevice }).device;
            if (!device) throw new Error("no WebGPU device on renderer backend");
            const g = new DancerGpuSim(device, AGENTS, 1, sim.params);
            g.init();
            // share three's instanceMatrix GPUBuffer so the render reads pose off the GPU
            const mtxBuf = await r.gpuInstanceMatrixBuffer();
            if (!mtxBuf) throw new Error("no instanceMatrix buffer");
            g.setMatrixTarget(mtxBuf, 1);
            g.writeMatrices(); // seed pose into the matrices before first frame
            r.setGpuMatrix(true);
            if (disposed) return;
            gpu = g;
            simRef.current = g;
          } catch (err) {
            console.warn("dancer: GPU sim unavailable, using CPU", err);
          }
        }
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
        const zeros = new Float32Array(AGENTS * 3);
        let snapPos: Float32Array = new Float32Array(AGENTS * 3);
        let snapSpeed: Float32Array = new Float32Array(AGENTS);
        let snapping = false;
        const speedsFrom = (vel: Float32Array): Float32Array => {
          const out = new Float32Array(AGENTS);
          for (let i = 0; i < AGENTS; i++) out[i] = Math.hypot(vel[i * 3] ?? 0, vel[i * 3 + 1] ?? 0, vel[i * 3 + 2] ?? 0);
          return out;
        };

        const loop = () => {
          if (disposed || !renderer) return;
          const ha = hoverAgentRef.current;
          const hc = hoverCellRef.current;
          const agents: number[] = [];
          if (ha !== null) agents.push(ha);
          if (hc) agents.push(hc[0], hc[1]);

          if (gpu) {
            gpu.step(); // advance + write instanceMatrix on the GPU (no readback)
            if (frame % 12 === 0 && !snapping) {
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
            renderer.update(snapPos, zeros, snapSpeed, { agents, pair: hc });
            renderer.render();
            if (matrixRef.current) drawDistanceMatrix(matrixRef.current, snapPos, gpu.n, { row: ha, pair: hc });
            if (frame % 15 === 0) setFigure(gpu.currentFigure());
          } else {
            sim.step();
            const p = sim.positions();
            renderer.update(p, sim.orientations(), sim.speeds(), { agents, pair: hc });
            renderer.render();
            if (matrixRef.current) drawDistanceMatrix(matrixRef.current, p, sim.n, { row: ha, pair: hc });
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
      renderer?.dispose();
    };
  }, []);

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
            <button
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
            <button className="dancer-fs" onClick={toggleFullscreen} aria-label="fullscreen">⤢ fullscreen</button>
          </div>
        </div>

        {showBreed && (
          <BreedingStrip
            onAdopt={(p: DancerParams) => {
              if (simRef.current) simRef.current.params = p;
            }}
          />
        )}

        <div className="dancer-matrix" title="pairwise distance matrix — couples are off-diagonal hot pairs">
          <canvas ref={matrixRef} className="dancer-matrix-canvas" width={96} height={96} />
          <div className="dancer-matrix-cap">distance matrix</div>
        </div>

        {status === "unsupported" && (
          <div className="dancer-overlay">This piece needs WebGPU. Try a recent Chrome, Edge, or Safari Technology Preview.</div>
        )}
        {status === "error" && <div className="dancer-overlay">The 3D stage failed to start (see console).</div>}

        {showAbout && (
          <aside className="dancer-about">
            <button className="dancer-about-close" onClick={() => setShowAbout(false)} aria-label="close">×</button>
            <h1>A Ceilidh of force-fields</h1>
            <p>
              After <em>DANCERL</em> (Andy Lomas, IBM, 1992) — the force-field motion controller for
              William Latham's SIGGRAPH film, written in the ESME/Mutator language Stephen Todd and
              Latham built. No dancer follows a keyframe: motion <em>emerges</em> from a superposition
              of named influences, and at each call the dancers scramble not to a position but to a
              shared <em>state of motion</em> — couples swinging, then advancing through partners.
            </p>
            <p className="dancer-about-note">
              A 2026 reconstruction of the algorithm, not the surface. Drag to orbit · ⤢ for fullscreen.
            </p>
          </aside>
        )}
        {!showAbout && (
          <button className="dancer-about-open" onClick={() => setShowAbout(true)}>about</button>
        )}
      </div>
    </div>
  );
}
