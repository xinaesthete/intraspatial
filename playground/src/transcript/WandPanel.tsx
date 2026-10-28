// Controls and readout for `useWand`: set how close counts as "similar", and see what the
// picked place is made of.

import { oklabToSrgb } from "../../../src/color/oklab";
import { cssRgb } from "../../../src/color/ramps";
import { similaritySwatch } from "../../../src/gpu/spatial/gramTerrain";
import { Explain } from "./Explain";
import type { Wand } from "./useWand";

/** A smoothed density for a sparse gene is genuinely ~1e-3; fixed decimals would print it as 0. */
const density = (v: number) => (v === 0 ? "0" : Math.abs(v) >= 0.01 ? v.toFixed(2) : v.toExponential(1));

export function WandControls({ wand }: { wand: Wand }) {
  return (
    <div className="picker">
      <div className="picker-head">
        Wand: click the map to sample a place; shift-drag to move the sample (the map does not pan while you do)
      </div>
      <label>
        How close counts as similar: {wand.tolerance.toFixed(2)}
        <input
          type="range"
          min={0.25}
          max={5}
          step={0.05}
          value={wand.tolerance}
          onChange={(e) => wand.setTolerance(Number(e.target.value))}
        />
      </label>
      <label className="check">
        <input type="checkbox" checked={wand.connected} onChange={(e) => wand.setConnected(e.target.checked)} />
        Only the patch the sample is in (off: every similar place)
      </label>
      <label>
        Measured over
        <select value={wand.modesUsed} onChange={(e) => wand.setModesUsed(Number(e.target.value))}>
          <option value={3}>the 3 modes the colours show</option>
          <option value={0}>every mode (full Mahalanobis)</option>
        </select>
      </label>
      {wand.place && (
        <button type="button" onClick={wand.clear}>
          clear the wand
        </button>
      )}
    </div>
  );
}

export function WandReadout({ wand, labels }: { wand: Wand; labels: readonly string[] }) {
  const s = wand.sample;
  if (wand.error) return <p className="hint error">wand: {wand.error.message}</p>;
  if (!s) {
    return (
      <p className="hint">
        {wand.place ? "The wand's place is outside this window." : "Click the map to outline places like the one you pick."}
      </p>
    );
  }
  // The channels furthest from typical here, signed, with the density each was standardised from.
  const top = labels
    .map((label, a) => ({ label, z: s.z[a] ?? 0, raw: s.raw[a] ?? 0 }))
    .sort((p, q) => Math.abs(q.z) - Math.abs(p.z))
    .slice(0, 4);
  return (
    <p className="status">
      <b style={{ color: cssRgb(oklabToSrgb(similaritySwatch(1))) }}>wand</b> at map pixel {s.col.toFixed(1)}, {s.row.toFixed(1)} —{" "}
      {top.map((t, i) => (
        <span key={t.label}>
          {i > 0 && " · "}
          {t.z >= 0 ? "+" : "−"}
          {t.label} {Math.abs(t.z).toFixed(1)}σ <span className="hint">(ρ {density(t.raw)})</span>
        </span>
      ))}
      . Outlined: everywhere within {wand.tolerance.toFixed(2)} of it.
    </p>
  );
}

export function WandExplain() {
  return (
    <Explain
      title="What does the wand outline?"
      math={String.raw`d(x) = \big\lVert \Lambda^{-1/2} V^\top \big(z(x) - z_{\text{wand}}\big) \big\rVert, \qquad \text{outlined where } d(x) = \text{tolerance}`}
    >
      Pick a place, and every other place is scored by how different its mixture of channels is from the one you picked. Differences are
      counted in each mode's own typical spread, so a channel pair that always rises together is not counted twice, and a mode that barely
      varies counts a small change as a big one. A distance of 1 means "about one typical spread away"; the outline is drawn where the
      distance reaches the tolerance, so a larger tolerance takes in more of the tissue. Measured over the 3 modes, it is distance in
      exactly the colours you see; over every mode, it also counts the differences the colours leave out. "Only the patch the sample is in"
      keeps just the outlined area you can reach from the sample without leaving it, stepping between neighbouring map pixels; similar
      places elsewhere are left unmarked. Above, z(x) is each channel's standardised value at x and V, Λ are the modes and their variances.
    </Explain>
  );
}
