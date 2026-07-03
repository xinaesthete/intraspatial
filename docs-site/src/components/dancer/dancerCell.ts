// A single breeding-strip cell: a small, non-interactive dancer view sharing ONE central GPUDevice
// with the stage and the other cells (the webgpu multipleCanvases pattern — three's WebGPURenderer
// takes an existing `device`, so N canvases cost one device, not N). Each cell runs its own GPU
// mini-sim writing its own state buffers, which the same shader renders with the specimen's bred
// render traits — so you breed on how a swarm both MOVES and LOOKS. Cheap: 48 agents, tiny canvas.
//
// The renderer + its state/trail buffers are built ONCE; breeding swaps the specimen via setParams
// (a fresh mini-sim seeded into the same buffers), so the canvas/context is never torn down.
import { DancerGpuSim } from "../../../../src/gpu/sim/dancerGpu";
import { createDancerRenderer } from "./renderer";
import type { DancerParams } from "./sim";

export interface DancerCell {
  /** Swap the dancer shown here (new specimen): reseeds a mini-sim into the same buffers + traits. */
  setParams(params: DancerParams, seed: number): void;
  /** Advance + draw one frame (no-op after dispose, or before the first setParams). */
  step(): void;
  dispose(): void;
}

/** Build an (empty) cell on the shared `device`. Call setParams to populate it. Never throws for a
 *  sim failure — a broken cell just renders empty rather than breaking the strip. */
export async function createDancerCell(canvas: HTMLCanvasElement, n: number, device: GPUDevice): Promise<DancerCell> {
  const r = await createDancerRenderer(canvas, n, { device, interactive: false });
  const stateBufs = await r.gpuStateBuffers();
  const trailBuf = await r.gpuTrailBuffer();
  let disposed = false;
  let sim: DancerGpuSim | null = null;

  return {
    setParams(params: DancerParams, seed: number): void {
      if (disposed || !stateBufs) return;
      try {
        const g = new DancerGpuSim(device, n, seed, params);
        g.init(stateBufs); // reseeds the shared state buffers to this specimen's initial swarm
        if (trailBuf) g.setTrailTarget(trailBuf, r.trailCapacity());
        r.setRenderTraits(params);
        sim = g;
      } catch (err) {
        console.warn("dancer cell: GPU sim unavailable", err);
      }
    },
    step(): void {
      if (disposed) return;
      if (sim) {
        sim.step();
        r.setTrailHead(sim.trailHead());
      }
      r.render();
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      r.dispose();
    },
  };
}
