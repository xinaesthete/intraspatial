// What the wand sampled, in two readings: the channels furthest from typical there (each with its
// raw density, so the σ count is anchored to a real quantity), and the pixel's coordinates in the
// mode basis — which is literally its colour on the map.

import type { CoLocationModes } from "../../../src/spatial/gram";
import { modeColour } from "./modeColour";
import type { WandSample } from "./useWand";

/** A smoothed density for a rare gene is genuinely ~1e-3, and a fixed 2 dp would print it as "0.00". */
const fmtDensity = (v: number) => (v === 0 ? "0" : Math.abs(v) >= 0.01 ? v.toFixed(2) : v.toExponential(1));

export function WandReadout({ wand, modes }: { wand: WandSample; modes: CoLocationModes }) {
  const { labels, vectors } = modes;
  const K = labels.length;
  const top = labels
    .map((label, a) => ({ label, z: wand.z[a] ?? 0, dens: wand.raw[a] ?? 0 }))
    .sort((p, q) => Math.abs(q.z) - Math.abs(p.z))
    .slice(0, 4);
  const proj = Array.from({ length: Math.min(3, K) }, (_, k) => {
    let y = 0;
    for (let a = 0; a < K; a++) y += (wand.z[a] ?? 0) * (vectors[k * K + a] ?? 0);
    return y;
  });
  return (
    <div className="wand-readout">
      <b>
        wand @ px {wand.col}, {wand.row}
      </b>{" "}
      —{" "}
      {top.map((p, i) => (
        <span key={p.label}>
          {p.z >= 0 ? "+" : "−"}
          {p.label} {Math.abs(p.z).toFixed(1)}σ <span className="dim">(ρ {fmtDensity(p.dens)})</span>
          {i < top.length - 1 ? " · " : ""}
        </span>
      ))}
      <br />
      <span className="dim">in the mode basis</span>{" "}
      {proj.map((y, k) => (
        <span key={k} style={{ color: modeColour(k) }}>
          mode {k + 1} {y >= 0 ? "+" : "−"}
          {Math.abs(y).toFixed(2)}
          {k < proj.length - 1 ? " · " : ""}
        </span>
      ))}
    </div>
  );
}
