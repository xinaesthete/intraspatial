// Cross-PCF inside the painted mask (docs/stats-on-arbitrary-domains.md): every pair of cell types,
// measured over the mask as an arbitrarily shaped domain, beside the same pairs over the whole
// analysis window. The mask's area sets ρ_B and its outline clips each cell's annuli, so the two
// read the same scale: g = 1 is chance inside each region.
//
// Measured on demand, in a worker (maskStatsWorker.ts): a pan or a new stroke would otherwise recount
// every pair, and on the host the count froze the page for seconds. A run whose mask, channels,
// window or rMax change under it is terminated rather than left to report stale numbers.

import { useEffect, useRef, useState } from "react";
import type { PcfMatrixBinnedResult } from "../../../src/spatial/pcf";
import { domainArea } from "../../../src/spatial/domain";
import type { ChannelCloud } from "../../../src/spatial/gram";
import { MatrixView } from "../modes/MatrixView";
import { Explain } from "../transcript/Explain";
import type { MaskStatsRequest, MaskStatsResponse } from "./maskStatsWorker";
import { maskToDomain, type WorldMask } from "./worldMask";

const BINS = 8;

interface Props {
  /** One unweighted channel per cell type, in the mask's frame. */
  readonly channels: readonly ChannelCloud[] | undefined;
  readonly mask: WorldMask | undefined;
  /** The analysis window, to compare against. */
  readonly window: readonly [number, number, number, number] | undefined;
  /** Largest radius measured, in the frame's units. */
  readonly rMax: number;
  /** Frame units → µm, for the labels. */
  readonly umPerUnit: number;
  readonly unit: string;
}

interface Measured extends Inputs {
  readonly inMask: PcfMatrixBinnedResult;
  readonly inWindow: PcfMatrixBinnedResult;
  readonly area: number;
  readonly ms: number;
}

/** Every channel's cells in one list, typed by channel index, as arrays the worker can take over. */
function labelled(channels: readonly ChannelCloud[]): Pick<MaskStatsRequest, "xs" | "ys" | "typeId"> {
  const n = channels.reduce((s, c) => s + c.xs.length, 0);
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  const typeId = new Int32Array(n);
  let at = 0;
  channels.forEach((c, t) => {
    for (let i = 0; i < c.xs.length; i++, at++) {
      xs[at] = c.xs[i]!;
      ys[at] = c.ys[i]!;
      typeId[at] = t;
    }
  });
  return { xs, ys, typeId };
}

/** The inputs a run was started from, to tell when it has gone stale. */
interface Inputs {
  readonly mask: WorldMask;
  readonly channels: readonly ChannelCloud[];
  readonly rMax: number;
  readonly window: readonly [number, number, number, number];
}

const sameInputs = (a: Inputs, b: Partial<Inputs>) =>
  a.mask === b.mask && a.channels === b.channels && a.rMax === b.rMax && !!b.window && b.window.every((v, i) => v === a.window[i]);

export function MaskStats({ channels, mask, window, rMax, umPerUnit, unit }: Props) {
  const [measured, setMeasured] = useState<Measured>();
  const [bin, setBin] = useState(BINS / 2 - 1);
  const [error, setError] = useState<string>();

  const weighted = channels?.some((c) => c.weights);
  const ready = channels && !weighted && mask && mask.keys.size > 0 && window && rMax > 0;
  const stale = measured && !sameInputs(measured, { mask, channels, rMax, window });

  // The run in flight: the inputs it was started from, and its worker. Inputs that move under it
  // end it, since its numbers would describe a mask or window no longer shown; the check runs during
  // render, and the effect below then terminates the worker.
  const [running, setRunning] = useState<Inputs>();
  if (running && !sameInputs(running, { mask, channels, rMax, window })) setRunning(undefined);
  const busy = running !== undefined;
  const worker = useRef<Worker>(undefined);
  useEffect(() => {
    if (running) return;
    worker.current?.terminate();
    worker.current = undefined;
  }, [running]);
  useEffect(() => () => worker.current?.terminate(), []);

  const measure = () => {
    if (!ready) return;
    setError(undefined);
    const t0 = performance.now();
    let domain;
    try {
      domain = maskToDomain(mask)!;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return;
    }
    const area = domainArea(domain);
    const inputs: Inputs = { mask, channels, rMax, window };
    const cells = labelled(channels);
    const req: MaskStatsRequest = { ...cells, nTypes: channels.length, domain, window, rMax, nBins: BINS };
    worker.current?.terminate();
    const w = new Worker(new URL("./maskStatsWorker.ts", import.meta.url), { type: "module" });
    worker.current = w;
    setRunning(inputs);
    w.onmessage = (e: MessageEvent<MaskStatsResponse>) => {
      if (worker.current !== w) return; // stopped or superseded
      const res = e.data;
      if (res.ok) setMeasured({ ...inputs, inMask: res.inMask, inWindow: res.inWindow, area, ms: performance.now() - t0 });
      else setError(res.error);
      setRunning(undefined);
    };
    // Failures inside the count come back as messages, so an error event here is the worker itself:
    // with no message, its script never loaded (in development, the dev server is not reachable).
    w.onerror = (e) => {
      if (worker.current !== w) return;
      setError(e.message ? `measurement worker: ${e.message}` : "the measurement worker could not load its script — reload the page");
      setRunning(undefined);
    };
    // The cells and the mask raster were built for this run alone, so hand them over rather than copy.
    w.postMessage(req, [cells.xs.buffer, cells.ys.buffer, cells.typeId.buffer, domain.data.buffer]);
  };

  const m = measured;
  const K = m?.channels.length ?? 0;
  const at = (res: PcfMatrixBinnedResult, a: number, b: number) => res.g[a * K * BINS + b * BINS + bin] ?? 0;
  const pairsAt = (res: PcfMatrixBinnedResult, a: number, b: number) => res.pairs[a * K * BINS + b * BINS + bin] ?? 0;
  // A pair with no cells of either type in the mask has no g, not g = 0. Nor does a type with one
  // cell against itself: self-pairs are excluded, so it has no possible pair. A populated pair with
  // no neighbours in this bin is a real g = 0 and stays.
  const empty = (a: number, b: number) => !m?.inMask.counts[a] || !m.inMask.counts[b] || (a === b && m.inMask.counts[a]! < 2);
  const values = m
    ? Float64Array.from({ length: K * K }, (_, i) =>
        empty(Math.floor(i / K), i % K) ? Number.NaN : at(m.inMask, Math.floor(i / K), i % K),
      )
    : undefined;
  const um = (world: number) => (world * umPerUnit).toPrecision(3);
  const dr = m ? m.rMax / BINS : 0;
  const inside = m ? m.inMask.counts.reduce((s, n) => s + n, 0) : 0;

  return (
    <section>
      <h3>Which cell types are together inside the mask</h3>
      <p className="row">
        <button type="button" disabled={!ready || busy} onClick={measure}>
          {busy ? "measuring…" : "measure in mask"}
        </button>
        <span className="hint">
          {weighted
            ? "needs cell types: choose an obs column (expression channels are not point types)"
            : !mask || mask.keys.size === 0
              ? "paint a mask with the wand or brush first"
              : stale
                ? "the mask, channels or window changed — measure again"
                : ""}
          {error && <span className="error"> {error}</span>}
        </span>
      </p>
      {m && values && (
        <>
          <label>
            Distance {um(bin * dr)}–{um((bin + 1) * dr)} {unit}
            <input type="range" min={0} max={BINS - 1} value={bin} onChange={(e) => setBin(Number(e.target.value))} />
          </label>
          <p className="hint">
            Blank: a type with no cells in the mask, or one cell against itself. {inside} cells in the mask ({m.inMask.outside} outside) ·
            area {(m.area * umPerUnit * umPerUnit).toPrecision(3)} {unit}² · {m.ms.toFixed(0)} ms
          </p>
          <MatrixView
            labels={m.channels.map((c) => c.label)}
            values={values}
            scale="log2"
            describe={(a, b) => {
              const A = m.channels[a]?.label;
              const B = m.channels[b]?.label;
              const nA = m.inMask.counts[a] ?? 0;
              const nB = m.inMask.counts[b] ?? 0;
              if (!nA || !nB) return `${A} → ${B}: no ${!nA ? A : B} cells in the mask`;
              if (a === b && nA < 2) return `${A} → ${A}: one cell in the mask, so no pairs`;
              const gi = at(m.inMask, a, b);
              const gw = at(m.inWindow, a, b);
              return `${A} → ${B}: g = ${gi.toFixed(2)} in the mask (${pairsAt(m.inMask, a, b)} pairs) · ${gw.toFixed(2)} over the window (${pairsAt(m.inWindow, a, b)} pairs)`;
            }}
          />
          <p className="hint raised">
            ⚠ If wand patches drew this mask, it was chosen by the mixture of these same cell types, so g inside it is not a fair estimate:
            pairs that built a mask read 14–23% low on independent clustered types. A mask painted only with the brush has no such bias.
          </p>
        </>
      )}
      <Explain title="What does the mask measurement show?">
        Each cell is how often one type (row) has another (column) at the chosen distance, compared with chance <i>inside the mask</i>: 1
        means as often as if the column's cells were scattered at random over the mask, above 1 more often, below 1 less. The mask's own
        area and outline are used, so cells near its edge are not undercounted. Hover a cell to compare with the same pair over the whole
        analysis window. A wand mask is picked by how similar the mixture of cells looks, so the cells it is then measured on helped draw
        it; that makes g inside it biased, usually low. Painting past the tissue's edge also lowers it, by counting empty ground as
        observed.
      </Explain>
    </section>
  );
}
