// The Mutator breeding strip — Todd & Latham aesthetic selection over the dance. Each cell
// is a *live* mini-simulation (top-down), so you breed by watching how a swarm *behaves*,
// not how a frame looks. Click a cell to send that dance to the main stage; Breed makes a
// new generation of mutations of your selection; Cross marries two selections. Lineage and
// steering could layer on later. Uses src/evo directly (the generic Mutator).
import { useEffect, useRef, useState } from "react";
import { breed, marry, mulberry32, paramsToSpecimen, type Specimen } from "../../../../src/evo";
import { DancerSim, DEFAULT_DANCER_PARAMS, type DancerParams } from "./sim";
import { DANCER_TRAIT_SPACE, specimenToDancerParams } from "./traits";

const COUNT = 6;
const MINI_AGENTS = 48;

function initialGeneration(seed: number): Specimen[] {
  // start from the default dance, then a generation of mutations around it
  const progenitor = paramsToSpecimen(DANCER_TRAIT_SPACE, DEFAULT_DANCER_PARAMS as unknown as Record<string, unknown>, seed);
  return breed(DANCER_TRAIT_SPACE, [progenitor], COUNT, { rate: 0.35, rng: mulberry32(seed), keepElite: true });
}

function drawSwarm2D(canvas: HTMLCanvasElement, positions: Float32Array, n: number): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const s = Math.min(w, h) / 22; // ~±11 world units in view
  const cx = w / 2, cy = h / 2;
  ctx.fillStyle = "#7fb0ff";
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3] ?? 0;
    const y = positions[i * 3 + 1] ?? 0;
    ctx.fillRect(cx + x * s - 1, cy - y * s - 1, 2, 2);
  }
}

export function BreedingStrip({ onAdopt }: { onAdopt: (p: DancerParams) => void }) {
  const seedRef = useRef(1);
  const [generation, setGeneration] = useState<Specimen[]>(() => initialGeneration(seedRef.current++));
  const [selected, setSelected] = useState<number[]>([0]);
  const canvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const simsRef = useRef<DancerSim[]>([]);

  // (Re)build the mini-sims whenever the generation changes.
  useEffect(() => {
    simsRef.current = generation.map((sp, i) => new DancerSim(MINI_AGENTS, 1 + i, specimenToDancerParams(sp)));
  }, [generation]);

  // One shared rAF loop steps + draws all cells.
  useEffect(() => {
    let raf = 0;
    const loop = () => {
      const sims = simsRef.current;
      for (let i = 0; i < sims.length; i++) {
        const sim = sims[i];
        const canvas = canvasRefs.current[i];
        if (!sim) continue;
        sim.step();
        if (canvas) drawSwarm2D(canvas, sim.positions(), sim.n);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  const select = (i: number, additive: boolean) => {
    setSelected((prev) => {
      if (additive) return prev.includes(i) ? prev.filter((x) => x !== i) : [...prev, i].slice(-2);
      return [i];
    });
    const sp = generation[i];
    if (sp) onAdopt(specimenToDancerParams(sp));
  };

  const parents = (): Specimen[] => selected.map((i) => generation[i]).filter((s): s is Specimen => !!s);

  const doBreed = () => {
    const rng = mulberry32(seedRef.current++);
    const next = breed(DANCER_TRAIT_SPACE, parents(), COUNT, { rate: 0.28, rng, keepElite: true });
    setGeneration(next);
    setSelected([0]);
  };

  const doCross = () => {
    const ps = parents();
    if (ps.length < 2) return;
    const rng = mulberry32(seedRef.current++);
    const a = ps[0];
    const b = ps[1];
    if (!a || !b) return;
    const next = [a, ...Array.from({ length: COUNT - 1 }, () => marry(DANCER_TRAIT_SPACE, a, b, rng))];
    setGeneration(next);
    setSelected([0]);
  };

  const doRandom = () => {
    setGeneration(initialGeneration(seedRef.current++ * 2654435761));
    setSelected([0]);
  };

  return (
    <div className="breed-strip">
      <div className="breed-cells">
        {generation.map((_, i) => (
          <button
            key={i}
            className={"breed-cell" + (selected.includes(i) ? " sel" : "")}
            onClick={(e) => select(i, e.shiftKey)}
            title="click to send this dance to the stage · shift-click to pick a second parent"
          >
            <canvas
              ref={(el) => {
                canvasRefs.current[i] = el;
              }}
              width={96}
              height={96}
            />
          </button>
        ))}
      </div>
      <div className="breed-controls">
        <button onClick={doBreed} title="a new generation of mutations of your selection">Breed ⚘</button>
        <button onClick={doCross} disabled={selected.length < 2} title="marry two selected dances (shift-click to pick two)">Cross ⚭</button>
        <button onClick={doRandom} title="a fresh random generation">Shuffle ⤳</button>
        <span className="breed-hint">select on how they <em>move</em></span>
      </div>
    </div>
  );
}
