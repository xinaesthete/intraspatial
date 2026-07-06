// The Mutator breeding strip — Todd & Latham aesthetic selection over the dance. Each cell is a
// *live* 3D mini-simulation (the same GPU renderer as the stage, sharing one device), so you breed
// by watching how a swarm both *moves* and *looks* — behaviour AND the bred render traits (colour,
// size). Click a cell to send that dance to the main stage; Breed makes a new generation of
// mutations of your selection; Cross marries two selections. Uses src/evo (the generic Mutator).
import { useCallback, useEffect, useRef, useState } from "react";
import { breed, marry, mulberry32, paramsToSpecimen, type Specimen } from "../../../../src/evo";
import { createDancerCell, type DancerCell } from "./dancerCell";
import { type DancerParams, DEFAULT_DANCER_PARAMS } from "./sim";
import { DANCER_TRAIT_SPACE, specimenToDancerParams } from "./traits";

const COUNT = 6;
const MINI_AGENTS = 48;

function initialGeneration(seed: number): Specimen[] {
  // start from the default dance, then a generation of mutations around it
  const progenitor = paramsToSpecimen(DANCER_TRAIT_SPACE, DEFAULT_DANCER_PARAMS as unknown as Record<string, unknown>, seed);
  return breed(DANCER_TRAIT_SPACE, [progenitor], COUNT, { rate: 0.35, rng: mulberry32(seed), keepElite: true });
}

export function BreedingStrip({ device, onAdopt }: { device: GPUDevice | null; onAdopt: (p: DancerParams) => void }) {
  const seedRef = useRef(1);
  const [generation, setGeneration] = useState<Specimen[]>(() => initialGeneration(seedRef.current++));
  const [selected, setSelected] = useState<number[]>([0]);
  const canvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const cellsRef = useRef<DancerCell[]>([]);
  const generationRef = useRef(generation);
  generationRef.current = generation; // mirror latest generation for the async build/apply

  // Push the current generation's specimens (params + render traits) into the built cells.
  const applyGeneration = useCallback((): void => {
    const cells = cellsRef.current;
    const gen = generationRef.current;
    for (let i = 0; i < cells.length; i++) {
      const sp = gen[i];
      if (sp) cells[i].setParams(specimenToDancerParams(sp), i + 1);
    }
  }, []);

  // Build the cells ONCE per device (each shares the stage's GPUDevice, its own canvas). Breeding
  // then only swaps specimens (setParams) — no renderer/canvas teardown.
  useEffect(() => {
    if (!device) return;
    let cancelled = false;
    const built: DancerCell[] = [];
    (async () => {
      for (let i = 0; i < COUNT; i++) {
        const canvas = canvasRefs.current[i];
        if (!canvas) continue;
        const cell = await createDancerCell(canvas, MINI_AGENTS, device).catch(() => null);
        if (cancelled) {
          cell?.dispose();
          return;
        }
        if (cell) {
          built.push(cell);
          cellsRef.current = built;
        }
      }
      applyGeneration();
    })();
    return () => {
      cancelled = true;
      for (const c of built) c.dispose();
      cellsRef.current = [];
    };
  }, [device, applyGeneration]);

  // Re-seed the cells whenever the generation changes.
  // biome-ignore lint/correctness/useExhaustiveDependencies: applyGeneration reads refs; keyed on generation.
  useEffect(() => {
    applyGeneration();
  }, [generation]);

  // One shared rAF loop steps + draws all ready cells.
  useEffect(() => {
    let raf = 0;
    const loop = () => {
      for (const c of cellsRef.current) c.step();
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
            type="button"
            // biome-ignore lint/suspicious/noArrayIndexKey: we should give Specimen something that can be used as key
            key={i}
            className={`breed-cell${selected.includes(i) ? " sel" : ""}`}
            onClick={(e) => select(i, e.shiftKey)}
            title="click to send this dance to the stage · shift-click to pick a second parent"
          >
            <canvas
              ref={(el) => {
                canvasRefs.current[i] = el;
              }}
              width={104}
              height={104}
            />
          </button>
        ))}
      </div>
      <div className="breed-controls">
        <button type="button" onClick={doBreed} title="a new generation of mutations of your selection">
          Breed ⚘
        </button>
        <button type="button" onClick={doCross} disabled={selected.length < 2} title="marry two selected dances (shift-click to pick two)">
          Cross ⚭
        </button>
        <button type="button" onClick={doRandom} title="a fresh random generation">
          Shuffle ⤳
        </button>
        <span className="breed-hint">
          select on how they <em>move</em>
        </span>
      </div>
    </div>
  );
}
