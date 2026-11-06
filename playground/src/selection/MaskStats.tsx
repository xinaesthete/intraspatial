// Cross-PCF inside the painted mask (docs/stats-on-arbitrary-domains.md): every pair of cell types,
// measured over the mask as an arbitrarily shaped domain, beside the same pairs over the whole
// analysis window. The mask's area sets ρ_B and its outline clips each cell's annuli, so the two
// read the same scale: g = 1 is chance inside each region.
//
// Measured on demand, on the host: a pan or a new stroke would otherwise recount every pair.

import { useState } from "react";
import { crossPCFMatrixBinned, type LabelledCells, type PcfMatrixBinnedResult } from "../../../src/spatial/pcf";
import { domainArea, rectDomain } from "../../../src/spatial/domain";
import type { ChannelCloud } from "../../../src/spatial/gram";
import { MatrixView } from "../modes/MatrixView";
import { Explain } from "../transcript/Explain";
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

interface Measured {
  readonly mask: WorldMask;
  readonly channels: readonly ChannelCloud[];
  readonly rMax: number;
  readonly inMask: PcfMatrixBinnedResult;
  readonly inWindow: PcfMatrixBinnedResult;
  readonly area: number;
  readonly ms: number;
}

function labelled(channels: readonly ChannelCloud[]): LabelledCells {
  const xs: number[] = [];
  const ys: number[] = [];
  const typeId: number[] = [];
  channels.forEach((c, t) => {
    for (let i = 0; i < c.xs.length; i++) {
      xs.push(c.xs[i]!);
      ys.push(c.ys[i]!);
      typeId.push(t);
    }
  });
  return { xs, ys, typeId };
}

export function MaskStats({ channels, mask, window, rMax, umPerUnit, unit }: Props) {
  const [measured, setMeasured] = useState<Measured>();
  const [bin, setBin] = useState(BINS / 2 - 1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();

  const weighted = channels?.some((c) => c.weights);
  const ready = channels && !weighted && mask && mask.keys.size > 0 && window && rMax > 0;
  const stale = measured && (measured.mask !== mask || measured.channels !== channels || measured.rMax !== rMax);

  const measure = () => {
    if (!ready) return;
    setBusy(true);
    setError(undefined);
    // Yield first, so "measuring…" paints before the host is busy.
    setTimeout(() => {
      try {
        const t0 = performance.now();
        const domain = maskToDomain(mask)!;
        const cells = labelled(channels);
        const common = { bbox: window, rMax, nBins: BINS, nTypes: channels.length, edgeCorrected: true };
        const inMask = crossPCFMatrixBinned(cells, { ...common, domain });
        const inWindow = crossPCFMatrixBinned(cells, { ...common, domain: rectDomain(window) });
        setMeasured({ mask, channels, rMax, inMask, inWindow, area: domainArea(domain), ms: performance.now() - t0 });
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
    }, 0);
  };

  const m = measured;
  const K = m?.channels.length ?? 0;
  const at = (res: PcfMatrixBinnedResult, a: number, b: number) => res.g[a * K * BINS + b * BINS + bin] ?? 0;
  const pairsAt = (res: PcfMatrixBinnedResult, a: number, b: number) => res.pairs[a * K * BINS + b * BINS + bin] ?? 0;
  // A pair with no cells of either type in the mask has no g, not g = 0.
  const empty = (a: number, b: number) => !m?.inMask.counts[a] || !m.inMask.counts[b];
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
                ? "the mask or channels changed — measure again"
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
            Blank: a type with no cells in the mask. {inside} cells in the mask ({m.inMask.outside} outside) · area{" "}
            {(m.area * umPerUnit * umPerUnit).toPrecision(3)} {unit}² · {m.ms.toFixed(0)} ms
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
