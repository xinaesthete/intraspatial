// Controls and explanation for `useWand`: how close counts as "similar", over which modes, and
// whether only the sample's own patch is outlined. `WandReadout` shows what the sample is made of.

import { Explain } from "../transcript/Explain";
import type { Wand } from "./useWand";

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

export function WandExplain() {
  return (
    <Explain
      title="What does the wand outline?"
      math={String.raw`d(x) = \big\lVert \Lambda^{-1/2} V^\top \big(z(x) - z_{\text{wand}}\big) \big\rVert, \qquad \text{outlined where } d(x) = \text{tolerance}`}
    >
      Click the map to sample the mixture of channels at one place; shift-drag to move the sample and watch the outline follow. The readout
      over the map lists the channels furthest from typical there, and the place's position in the modes, which is its colour. Every other
      place is scored by how different its mixture of channels is from the one you picked. Differences are counted in each mode's own
      typical spread, so a channel pair that always rises together is not counted twice, and a mode that barely varies counts a small change
      as a big one. A distance of 1 means "about one typical spread away"; the outline is drawn where the distance reaches the tolerance, so
      a larger tolerance takes in more of the tissue. Measured over the 3 modes, it is distance in exactly the colours you see; over every
      mode, it also counts the differences the colours leave out. "Only the patch the sample is in" keeps just the outlined area you can
      reach from the sample without leaving it, stepping between neighbouring map pixels; similar places elsewhere are left unmarked. Above,
      z(x) is each channel's standardised value at x and V, Λ are the modes and their variances.
    </Explain>
  );
}
