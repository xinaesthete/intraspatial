// PROTOTYPE: the boundaries and wand controls: presets, the tool hint, the mask, and the fine-tune
// settings (docs/wand-metric-and-regions.md).

import type { CoLocationModes } from "../../../src/spatial/gram";
import { Explain } from "../transcript/Explain";
import { maskArea, type WorldMask } from "../selection/worldMask";
import type { BoundaryField } from "./boundaryStrength";
import type { BoundaryNow } from "./useBoundaryField";
import { WAND_PRESETS, type WandPresetName, type WandSettingsState } from "./useWandSettings";
import type { Patch } from "./wandPatch";

// Only the numbers the panel reads. React's dev build diffs changed props into the Performance panel
// and walks a typed array element by element: a patch's mask or the field's raster cost seconds a click.
/** What the panel shows of the boundary field. */
export type BoundarySummary = Pick<BoundaryNow, "sigma2" | "q"> & Pick<BoundaryField, "quantiles">;
/** What the panel shows of the wand's patch. */
export type PatchSummary = Pick<Patch, "count" | "total" | "dEdge" | "gated" | "moved" | "scales">;

export function boundarySummary(b: BoundaryNow | undefined): BoundarySummary | undefined {
  return b && { sigma2: b.sigma2, q: b.q, quantiles: b.field.quantiles };
}

export function patchSummary(p: Patch | undefined): PatchSummary | undefined {
  return p && { count: p.count, total: p.total, dEdge: p.dEdge, gated: p.gated, moved: p.moved, scales: p.scales };
}

const PRESET_NAMES = Object.keys(WAND_PRESETS) as WandPresetName[];

interface Props {
  readonly show: boolean;
  readonly onShow: (on: boolean) => void;
  readonly wand: WandSettingsState;
  /** θ in effect: the chosen one, or the window's 90th percentile. */
  readonly theta: number;
  readonly bounds: BoundarySummary | undefined;
  /** The window's channel count, the most main modes there can be. */
  readonly channels: number | undefined;
  readonly modes: CoLocationModes | undefined;
  /** The broad scale's radius and map, once computed, when judging at two scales. */
  readonly broad?: { radius: number; raster: { width: number; height: number } } | undefined;
  /** Why the broad scale could not be computed, shown in place of "computing…". */
  readonly broadError?: string | undefined;
  /** Whether the page can compute a broader scale; without it the option is hidden. */
  readonly twoScale?: boolean;
  /** What the map's channels mix, for the heading: "genes", "cells". */
  readonly mixtures: string;
  readonly patch: PatchSummary | undefined;
  readonly brushR: number;
  readonly mask: WorldMask | undefined;
  readonly onClear: () => void;
  readonly smoothEdges: boolean;
  readonly onSmoothEdges: (on: boolean) => void;
}

export function BoundaryPanel({
  show,
  onShow,
  wand,
  theta,
  bounds,
  channels,
  modes,
  broad,
  broadError,
  patch,
  brushR,
  mask,
  onClear,
  smoothEdges,
  onSmoothEdges,
  mixtures,
  twoScale = false,
}: Props) {
  const { settings: s, set, preset, applyPreset } = wand;
  return (
    <div className="picker">
      <div className="picker-head">Boundaries (prototype): where the mixture of {mixtures} changes fastest</div>
      <label className="check">
        <input type="checkbox" checked={show} onChange={(e) => onShow(e.target.checked)} />
        Show boundaries (the wand, top left of the map, selects by them)
      </label>
      <label>
        Settings
        <select
          value={preset ?? "custom"}
          onChange={(e) => {
            if (e.target.value in WAND_PRESETS) applyPreset(e.target.value as WandPresetName);
          }}
        >
          {PRESET_NAMES.map((name) => (
            <option key={name} value={name}>
              {WAND_PRESETS[name].label}
            </option>
          ))}
          {!preset && <option value="custom">Custom (fine-tuned below)</option>}
        </select>
      </label>
      <p className="hint">{preset ? WAND_PRESETS[preset].hint : "Your own settings: pick a preset to go back."}</p>
      <p className="hint">
        Pick a tool on the map. Wand (W): hover to preview the patch (cyan), click or drag to paint. Brush (B): drag to paint a disc. Shift:
        add · Alt: subtract what matches · Cmd/Ctrl+Alt: remove the piece · Shift+Alt: intersect · Space: pan · H: back to the hand ·
        Cmd/Ctrl+scroll or <kbd>[</kbd> <kbd>]</kbd>: size (wand σ {s.dSmooth.toFixed(2)}r, brush radius {Math.round(brushR)} µm).
        {mask && mask.keys.size > 0 ? ` Mask: ${(maskArea(mask) / 1e6).toFixed(3)} mm² (amber).` : ""}
      </p>
      {mask && mask.keys.size > 0 && (
        <button type="button" onClick={onClear}>
          clear the mask
        </button>
      )}
      <details>
        <summary>Fine-tune</summary>
        <label className="check">
          <input type="checkbox" checked={s.graded} onChange={(e) => set("graded", e.target.checked)} />
          Graded necks (a part joined by a neck near the opening's width fades with the neck; a press paints what is at least half shown.
          Off: the binary opening, whose parts appear and vanish whole)
        </label>
        <label className="check">
          <input type="checkbox" checked={smoothEdges} onChange={(e) => onSmoothEdges(e.target.checked)} />
          Smooth edges (outlines from a smoothed distance field, quadratic filter, antialiased; off: the pixel staircase, bilinear, as
          before)
        </label>
        <label className="check">
          <input type="checkbox" checked={s.lines} onChange={(e) => set("lines", e.target.checked)} />
          As thin lines (off: every place above θ)
        </label>
        <label>
          Bound the clicked cell by
          <select
            value={s.rule}
            onChange={(e) =>
              set("rule", e.target.value === "ridges" ? "ridges" : e.target.value === "threshold" ? "threshold" : "laplacian")
            }
          >
            <option value="laplacian">where the difference from the click stops speeding up (∇²d = 0)</option>
            <option value="threshold">a plain tolerance on the difference (d ≤ tol)</option>
            <option value="ridges">the boundary lines above θ</option>
          </select>
        </label>
        <label>
          Smooth the difference first (σ, in r): {s.dSmooth.toFixed(2)}
          <input type="range" min={0} max={3} step={0.1} value={s.dSmooth} onChange={(e) => set("dSmooth", Number(e.target.value))} />
        </label>
        <label>
          …and no further than this difference (tolerance): {s.tol > 0 ? s.tol.toFixed(2) : "off"}
          <input type="range" min={0} max={6} step={0.1} value={s.tol} onChange={(e) => set("tol", Number(e.target.value))} />
        </label>
        <label>
          Drop parts narrower than (in r): {s.open > 0 ? s.open.toFixed(2) : "off"}
          <input type="range" min={0} max={4} step={0.1} value={s.open} onChange={(e) => set("open", Number(e.target.value))} />
        </label>
        {twoScale && (
          <label className="check">
            <input type="checkbox" checked={s.twoScale} onChange={(e) => set("twoScale", e.target.checked)} />
            Judge the patch at a broader neighbourhood too
            {s.twoScale &&
              (broad
                ? ` (r ${broad.radius} µm*, map ${broad.raster.width}×${broad.raster.height})`
                : broadError
                  ? ` (failed: ${broadError})`
                  : " (computing…)")}
          </label>
        )}
        {twoScale && s.twoScale && (
          <>
            <label>
              Broader neighbourhood: {s.broad.toFixed(1)}× r
              <input type="range" min={1.5} max={6} step={0.5} value={s.broad} onChange={(e) => set("broad", Number(e.target.value))} />
            </label>
            <label>
              How far the edge may sit outside it (in r): {s.slack.toFixed(2)}
              <input type="range" min={0} max={4} step={0.25} value={s.slack} onChange={(e) => set("slack", Number(e.target.value))} />
            </label>
          </>
        )}
        <label>
          Main modes (q): {Math.min(s.q, channels ?? s.q)}
          {channels !== undefined && s.q >= channels ? " = every mode" : ""}
          <input type="range" min={1} max={channels ?? 32} step={1} value={s.q} onChange={(e) => set("q", Number(e.target.value))} />
        </label>
        <label>
          Count what the main modes leave out (w): {s.w.toFixed(2)}
          <input type="range" min={0} max={1} step={0.05} value={s.w} onChange={(e) => set("w", Number(e.target.value))} />
        </label>
        <label>
          Smooth what they leave out first (σ, in r): {s.residualSmooth.toFixed(2)}
          <input
            type="range"
            min={0}
            max={3}
            step={0.1}
            value={s.residualSmooth}
            onChange={(e) => set("residualSmooth", Number(e.target.value))}
          />
        </label>
        <label>
          Boundary strength (θ): {theta.toPrecision(3)}
          {s.theta === undefined ? " (this window's 90th percentile)" : ""}
          <input
            type="range"
            min={0}
            max={bounds ? bounds.quantiles.max : 10}
            step={(bounds?.quantiles.max ?? 10) / 200}
            value={theta}
            onChange={(e) => set("theta", Number(e.target.value))}
          />
        </label>
        {s.theta !== undefined && (
          <button type="button" onClick={() => set("theta", undefined)}>
            follow the 90th percentile
          </button>
        )}
      </details>
      {bounds && (
        <p className="hint">
          Strength per radius r: median {bounds.quantiles.p50.toPrecision(3)} · 90% {bounds.quantiles.p90.toPrecision(3)} · 99%{" "}
          {bounds.quantiles.p99.toPrecision(3)} · max {bounds.quantiles.max.toPrecision(3)}
          {patch
            ? ` · cell: ${patch.count.toLocaleString()} of ${patch.total.toLocaleString()} map pixels (${((100 * patch.count) / patch.total).toFixed(1)}%)${patch.dEdge !== undefined ? `, d at its edge ≈ ${patch.dEdge.toFixed(2)}${patch.gated ? `, ${(100 * patch.gated).toFixed(0)}% of the edge set by the tolerance` : ""}${patch.moved ? ", the pointer is on a branch narrower than the opening: nothing to pick" : ""}${patch.scales ? ` (broad cell ${(100 * patch.scales.broad).toFixed(1)}%, narrow ${(100 * patch.scales.narrow).toFixed(1)}%)` : ""}` : ""}`
            : ""}
          {channels !== undefined && bounds.q < channels
            ? ` · noise floor σ² ${bounds.sigma2.toPrecision(3)} (λ₁ ${modes?.values[0]?.toPrecision(3)})`
            : ""}
        </p>
      )}
    </div>
  );
}

/** What the boundaries show. `mixtures` names what is mixed: "genes", "cells". */
export function BoundariesExplain({ mixtures }: { mixtures: string }) {
  return (
    <Explain
      title="What do the boundaries show? (prototype)"
      math={String.raw`u = A z,\quad A_k = \begin{cases} v_k^\top/\sqrt{\lambda_k} & k \le q \\ \sqrt{w}\, v_k^\top/\sigma & k > q \end{cases},\quad \sigma^2 = \overline{\lambda_{k>q}},\qquad \text{strength} = r\,\sqrt{\lambda_{\max}\!\big(\textstyle\sum_k \nabla u_k \nabla u_k^\top\big)}`}
    >
      How fast the mixture of {mixtures} changes at each place, in the direction it changes fastest, measured the way the wand measures
      difference. 1 means "one typical spread of difference across one radius". The <b>main modes</b> count in their own spread. The rest
      are counted at the noise level, scaled by <b>w</b>: at 0 they are ignored (only what the colours show), at 1 they count as much as
      noise does. Setting q to every mode is the old "full Mahalanobis" metric, where the faintest modes count most. Amber marks places at
      or above θ, where a wand patch bounded by edges would stop. White shows weaker change.
    </Explain>
  );
}
