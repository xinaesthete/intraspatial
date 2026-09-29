// The co-location mode map (mode 1 → lightness, modes 2–3 → hue) and a legend that says which
// channels push each mode which way.

import { useEffect, useRef, useState } from "react";
import { oklabToSrgb } from "../../../src/color/oklab";
import { cssRgb } from "../../../src/color/ramps";
import { modeSwatch, paintGramModes } from "../../../src/gpu/spatial/gramModes";
import type { ImageOverlay } from "../../../src/gpu/spatial/imageOverlayWgsl";
import { isLatest, onGpu, type TranscriptGram } from "./useTranscriptGram";

interface Props {
  readonly gram: TranscriptGram;
  readonly image?: ImageOverlay;
  readonly saturate: number;
  readonly chromaWeight: number;
}

export function ModeMap({ gram, image, saturate, chromaWeight }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState<string>();
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    void onGpu(async () => {
      if (!isLatest(gram.generation)) return; // its rasters have been overwritten
      await paintGramModes(c, gram.res, { vectors: gram.modes.vectors, saturate, chromaWeight, image });
    }).then(
      () => setError(undefined),
      (e: unknown) => setError(String(e)),
    );
  }, [gram, image, saturate, chromaWeight]);

  return (
    <div>
      {/* The raster's row 0 is the window's top in world y (maxY); flip so the map reads y-down,
          like the image and the overview. */}
      <canvas ref={canvas} className="modemap" style={{ aspectRatio: `${gram.raster.width} / ${gram.raster.height}` }} />
      {error && <p className="error">{error}</p>}
    </div>
  );
}

const swatch = (k: number, sign: 1 | -1): string =>
  cssRgb(oklabToSrgb(modeSwatch([k === 0 ? sign : 0, k === 1 ? sign : 0, k === 2 ? sign : 0])));

export function ModeLegend({ gram }: { gram: TranscriptGram }) {
  const { vectors, explained, labels } = gram.modes;
  const K = labels.length;
  const roles = ["brightness", "hue (green ↔ red)", "hue (blue ↔ yellow)"];
  return (
    <div className="legend">
      {[0, 1, 2]
        .filter((k) => k < K)
        .map((k) => {
          const load = labels.map((label, a) => ({ label, v: vectors[k * K + a] ?? 0 })).sort((p, q) => q.v - p.v);
          const top = (sign: 1 | -1) =>
            (sign > 0 ? load : [...load].reverse())
              .filter((l) => l.v * sign > 0.15)
              .slice(0, 4)
              .map((l) => l.label)
              .join(", ") || "(none)";
          return (
            <div key={k} className="legend-mode">
              <div className="legend-head">
                Mode {k + 1} · {roles[k]} · {((explained[k] ?? 0) * 100).toFixed(0)}% of the variation
              </div>
              <div className="legend-row">
                <span className="swatch" style={{ background: swatch(k, 1) }} /> more: {top(1)}
              </div>
              <div className="legend-row">
                <span className="swatch" style={{ background: swatch(k, -1) }} /> more: {top(-1)}
              </div>
            </div>
          );
        })}
    </div>
  );
}
