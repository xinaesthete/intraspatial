// Controls, status and legend for the niche map (useNicheMap.ts).

import { type NicheMap, type NicheOptions, nicheColour } from "./useNicheMap";

const BIN_SIDES = [20, 40, 80, 160];
const pct = (f: number) => `${+(f * 100).toPrecision(2)}%`;

interface Props {
  readonly opts: NicheOptions;
  readonly onChange: (o: NicheOptions) => void;
  readonly opacity: number;
  readonly onOpacity: (v: number) => void;
  readonly map: NicheMap;
  readonly labels: readonly string[];
}

export function NichePanel({ opts, onChange, opacity, onOpacity, map, labels }: Props) {
  const set = (patch: Partial<NicheOptions>) => onChange({ ...opts, ...patch });
  return (
    <fieldset className="channel-mode">
      <legend>Niche map</legend>
      <label className="check">
        <input type="checkbox" checked={opts.enabled} onChange={(e) => set({ enabled: e.target.checked })} />
        Label each bin by its mix of these channels (e-means, fitted once to the whole element)
      </label>
      {opts.enabled && (
        <>
          <label>
            Niches: {opts.k}
            <input type="range" min={3} max={16} step={1} value={opts.k} onChange={(e) => set({ k: Number(e.target.value) })} />
          </label>
          <label>
            Bin side (µm*, rounded to the element's quadtree)
            <select value={opts.binSide} onChange={(e) => set({ binSide: Number(e.target.value) })}>
              {BIN_SIDES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label>
            Opacity: {opacity.toFixed(2)}
            {/* Squared: the useful range, where the image still shows through, is the low end. */}
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={Math.sqrt(opacity)}
              onChange={(e) => onOpacity(Number(e.target.value) ** 2)}
            />
          </label>
          <NicheNote map={map} labels={labels} minSampled={opts.minSampled} />
        </>
      )}
    </fieldset>
  );
}

function NicheNote({ map, labels, minSampled }: { map: NicheMap; labels: readonly string[]; minSampled: number }) {
  const { fit, tiles } = map;
  const err = fit.sampleError ?? fit.error ?? tiles.error;
  if (err) return <p className="hint error">{err.message}</p>;
  if (labels.length < 2) return <p className="hint">A niche is a mix of channels: pick at least two.</p>;
  if (fit.sampleProgress)
    return (
      <p className="hint">
        Reading a sample of the whole element: tile {fit.sampleProgress.done} of {fit.sampleProgress.total}…
      </p>
    );
  const f = fit.value;
  if (fit.loading || !f) return <p className="hint">Fitting the niches (weighted e-means)…</p>;
  const { summary: s, sample, plan } = f;
  const d = f.d;
  const total = s.mass.reduce((a, b) => a + b, 0);
  return (
    <>
      <p className="hint">
        Fitted once, to {sample.molecules.toLocaleString()} molecules of the whole element (
        {sample.fraction >= 1
          ? "every one"
          : sample.perGene
            ? `each gene a random sample of ${pct(sample.fraction)} or more, rare ones whole`
            : `a ${pct(sample.fraction)} random sample`}
        ) in {sample.kept.toLocaleString()} bins of {+plan.bins.tileSize.toPrecision(3)} µm* (of {sample.bins.toLocaleString()} with any
        channel molecule), each described by the mix in the 3 × 3 bins around it — kept when that holds {minSampled}+ molecules read — and
        weighted by the molecules it stands for{f.qvMin > 0 ? `; molecules below qv ${f.qvMin} are left out, as for the modes` : ""}. Best
        of 4 k-means++ starts, batch {s.batch}, {(s.ms / 1000).toFixed(1)} s. The view is labelled tile by tile against that table, from one
        fixed level, so a bin keeps its niche wherever you pan or zoom.{" "}
        {tiles.progress ? `Labelling tiles: ${tiles.progress.done} of ${tiles.progress.total}…` : `${tiles.tiles.length} tiles labelled.`}
      </p>
      <div className="legend">
        {Array.from({ length: f.k }, (_, j) => {
          // A centroid is a mean of Hellinger rows: its squares, renormalised, are the niche's mix.
          const sq = Array.from({ length: d }, (_, c) => f.centroids[j * d + c]! ** 2);
          const norm = sq.reduce((a, b) => a + b, 0) || 1;
          const top = sq
            .map((v, c) => ({ label: labels[c] ?? `#${c}`, p: v / norm }))
            .sort((a, b) => b.p - a.p)
            .filter((x, i) => i === 0 || x.p >= 0.1)
            .slice(0, 3)
            .map((x) => `${x.label} ${pct(x.p)}`)
            .join(", ");
          const [r, g, b] = nicheColour(j);
          return (
            <div key={j} className="legend-row">
              <span className="swatch" style={{ background: `rgb(${r},${g},${b})` }} />N{j + 1} · {pct((s.mass[j] ?? 0) / (total || 1))} of
              molecules · {top}
            </div>
          );
        })}
      </div>
    </>
  );
}
