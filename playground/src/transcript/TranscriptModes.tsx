// Transcript co-location modes: the Gram form over transcript density, for a window you can drag
// across the slide (docs/cell-stats.md §4, §12; ADR-0008 points amendment).

import { useEffect, useMemo, useRef, useState } from "react";
import { type Rect, selectPointsTiles, tileRect } from "../../../src/datasource/points";
import { composeUv, type ImageOverlay } from "../../../src/gpu/spatial/imageOverlayWgsl";
import { listImageElements, loadContextImage, uvFromWorld } from "../datasource/imageContext";
import { listPointsElements } from "../datasource/pointsTileLoader";
import { useAsync, useSettled } from "../hooks/useAsync";
import { devicePixelRatio, useElementSize } from "../hooks/useElementSize";
import { usePointsSource, usePointsTiles, useSpatialData } from "../hooks/useSpatialData";
import { ChannelPicker } from "./ChannelPicker";
import { buildChannels, NEGATIVE_CONTROLS, type Selection, STARTER_SETS } from "./channelModel";
import { Explain } from "./Explain";
import { MatrixView } from "./MatrixView";
import { ModeLegend, ModeMap } from "./ModeMap";
import { onGpu, useTranscriptGram } from "./useTranscriptGram";
import { type Overview, WindowPicker } from "./WindowPicker";

const DEFAULT_STORE = "http://localhost:8080/xenium_2.q0.001.htj2k.index-permutations.zarr/";
const DEFAULT_SELECTION: Selection = { sets: [...STARTER_SETS.map((s) => s.name), NEGATIVE_CONTROLS], genes: [] };

/** Inverse of a 2×3 row-major affine, or null when singular. */
function invert2x3(m: ArrayLike<number>): number[] | null {
  const [a = 0, b = 0, c = 0, d = 0, e = 0, f = 0] = Array.from(m);
  const det = a * e - b * d;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-30) return null;
  return [e / det, -b / det, (b * f - c * e) / det, -d / det, a / det, (c * d - a * f) / det];
}

function centred(extent: Rect, w: number, h: number, around?: Rect): Rect {
  const cw = Math.min(w, extent.maxX - extent.minX);
  const ch = Math.min(h, extent.maxY - extent.minY);
  const cx = around ? (around.minX + around.maxX) / 2 : (extent.minX + extent.maxX) / 2;
  const cy = around ? (around.minY + around.maxY) / 2 : (extent.minY + extent.maxY) / 2;
  const minX = Math.min(Math.max(cx - cw / 2, extent.minX), extent.maxX - cw);
  const minY = Math.min(Math.max(cy - ch / 2, extent.minY), extent.maxY - ch);
  return { minX, minY, maxX: minX + cw, maxY: minY + ch };
}

export function TranscriptModes() {
  const [storeUrl, setStoreUrl] = useState(DEFAULT_STORE);
  const [storeDraft, setStoreDraft] = useState(DEFAULT_STORE);
  const [pointsName, setPointsName] = useState<string>();
  const [imageName, setImageName] = useState<string>();
  const [radius, setRadius] = useState(50);
  const [qvMin, setQvMin] = useState(20);
  const [rasterSide, setRasterSide] = useState(0);
  const [size, setSize] = useState({ w: 3000, h: 1200 });
  const [win, setWin] = useState<Rect>();
  const [selection, setSelection] = useState(DEFAULT_SELECTION);
  const [saturate, setSaturate] = useState(2.5);
  const [chromaWeight, setChromaWeight] = useState(0.5);
  const [mix, setMix] = useState(0.35);

  const sd = useSpatialData(storeUrl);
  const pointsNames = useMemo(() => (sd.value ? listPointsElements(sd.value) : []), [sd.value]);
  const pointsEl = pointsName ?? pointsNames.find((n) => /morton/.test(n) && !/feature_then/.test(n)) ?? pointsNames[0];
  const source = usePointsSource(sd.value, pointsEl, { columns: ["qv"] });
  const imageNames = useAsync(() => (sd.value ? listImageElements(sd.value) : undefined), [sd.value]);
  const imageEl = imageName ?? imageNames.value?.find((n) => /he/i.test(n)) ?? imageNames.value?.[0];

  const extent = source.value?.grid.bounds;
  // Memoised: a fresh object each render would read as a moved window and restart every load.
  // biome-ignore lint/correctness/useExhaustiveDependencies: size changes set `win` directly
  const initial = useMemo(() => (extent ? centred(extent, size.w, size.h) : undefined), [extent]);
  const window = win ?? initial;
  const settled = useSettled(window, 300);
  const tiles = usePointsTiles(source.value, settled, radius);
  const built = useMemo(
    () => (source.value ? buildChannels(selection, source.value.features) : { channels: [], missing: {} }),
    [selection, source.value],
  );
  const gram = useTranscriptGram(tiles.value, built.channels, { radius, qvMin, rasterSide });

  // Both views span the full width of this column; their canvases are sized to it in device pixels.
  const viewsRef = useRef<HTMLElement>(null);
  const cssWidth = useElementSize(viewsRef).width;
  const devWidth = Math.round(cssWidth * devicePixelRatio());
  const space = source.value?.space;
  const worldFromLocal = useMemo(
    () =>
      space ? ([space.affine.a, space.affine.c, space.affine.tx, space.affine.b, space.affine.d, space.affine.ty] as const) : undefined,
    [space],
  );

  // Overview: the whole slide, at the level that fills the overview's width (a power-of-two budget,
  // so resizing the window does not reload it). Only its pixels are kept.
  const overviewBudget = devWidth > 0 ? 2 ** Math.ceil(Math.log2(devWidth)) : 0;
  const overview = useAsync(async () => {
    if (!sd.value || !imageEl || !worldFromLocal || !overviewBudget) return undefined;
    const img = await loadContextImage(sd.value, imageEl, { maxSide: overviewBudget, keepPixels: true });
    img.texture.destroy(); // never drawn on the GPU
    const uvWorld = uvFromWorld(img);
    const localFromUv = uvWorld && invert2x3(composeUv(uvWorld, worldFromLocal));
    if (!localFromUv || !img.pixels) return { note: "The image carries no usable transform, so it cannot be aligned." };
    const bitmap = await createImageBitmap(new ImageData(new Uint8ClampedArray(img.pixels), img.width, img.height));
    const [p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0] = localFromUv;
    // pixel → uv is a scale by the image size; fold it in.
    const view: Overview = { bitmap, localFromPixel: [p0 / img.width, p1 / img.height, p2, p3 / img.width, p4 / img.height, p5] };
    return { view, label: img.label };
  }, [sd.value, imageEl, worldFromLocal, overviewBudget]);

  const grid = source.value?.grid;
  const needTiles = useMemo(
    () => (grid && window ? selectPointsTiles(grid, window, radius).chunks.map((c) => tileRect(grid, c.id.x, c.id.y)) : []),
    [grid, window, radius],
  );

  const g = gram.value;

  // Mode map: only the analysed window, at the level that matches the map's on-screen resolution.
  const mapWindow = g?.window;
  const mapPx =
    mapWindow && devWidth > 0
      ? Math.ceil((devWidth * Math.max(1, (mapWindow.maxY - mapWindow.minY) / (mapWindow.maxX - mapWindow.minX))) / 128) * 128
      : 0;
  const mapImage = useAsync(async () => {
    if (!sd.value || !imageEl || !worldFromLocal || !mapWindow || !mapPx) return undefined;
    const [a, c, tx, b, d, ty] = worldFromLocal;
    const toWorld = (x: number, y: number) => [a * x + c * y + tx, b * x + d * y + ty] as const;
    const w = mapWindow;
    const world = [toWorld(w.minX, w.minY), toWorld(w.maxX, w.minY), toWorld(w.maxX, w.maxY), toWorld(w.minX, w.maxY)];
    const img = await loadContextImage(sd.value, imageEl, { region: { world, screenPx: mapPx } });
    const uvWorld = uvFromWorld(img);
    return uvWorld ? { texture: img.texture, uvLocal: composeUv(uvWorld, worldFromLocal), label: img.label } : undefined;
  }, [sd.value, imageEl, worldFromLocal, mapWindow, mapPx]);
  // Release a replaced region texture — through the GPU queue, so paints already queued keep it.
  useEffect(() => {
    const t = mapImage.value?.texture;
    return () => {
      if (t) void onGpu(async () => t.destroy());
    };
  }, [mapImage.value]);
  const overlay = useMemo<ImageOverlay | undefined>(() => {
    const v = mapImage.value;
    return v && mix > 0 ? { texture: v.texture, uvFromWorld: v.uvLocal, mix } : undefined;
  }, [mapImage.value, mix]);
  const error = sd.error ?? source.error ?? tiles.error ?? gram.error;
  const busy = tiles.progress ? `loading tiles ${tiles.progress.done}/${tiles.progress.total}…` : gram.loading ? "computing…" : "";

  return (
    <div className="page">
      <h2 className="page">Where genes are found together — transcript co-location modes</h2>
      <p className="page">
        Every dot in a Xenium experiment is one detected RNA molecule of a known gene. Instead of first assigning molecules to cells, this
        page looks at where each gene's molecules sit, smoothed over a neighbourhood, and asks which genes turn up in the same places. Drag
        the box to choose the region; the colours then show which mixtures of genes dominate where.
      </p>

      <div className="layout">
        <aside className="controls">
          <label>
            Store
            <span className="row">
              <input value={storeDraft} onChange={(e) => setStoreDraft(e.target.value)} />
              <button type="button" onClick={() => setStoreUrl(storeDraft.trim())}>
                open
              </button>
            </span>
          </label>
          <label>
            Transcripts
            <select value={pointsEl ?? ""} onChange={(e) => setPointsName(e.target.value)}>
              {pointsNames.map((n) => (
                <option key={n}>{n}</option>
              ))}
            </select>
          </label>
          <label>
            Image
            <select value={imageEl ?? ""} onChange={(e) => setImageName(e.target.value)}>
              {(imageNames.value ?? []).map((n) => (
                <option key={n}>{n}</option>
              ))}
            </select>
          </label>
          <label>
            Neighbourhood radius r: {radius} µm*
            <input type="range" min={10} max={300} step={5} value={radius} onChange={(e) => setRadius(Number(e.target.value))} />
          </label>
          <label>
            Map resolution (long side, 0 = from r)
            <input
              type="number"
              min={0}
              max={2048}
              step={16}
              value={rasterSide}
              onChange={(e) => setRasterSide(Math.max(0, Number(e.target.value) || 0))}
            />
          </label>
          <label>
            Minimum molecule quality (qv)
            <input type="number" min={0} max={40} value={qvMin} onChange={(e) => setQvMin(Math.max(0, Number(e.target.value) || 0))} />
          </label>
          <label>
            Window size (µm*)
            <span className="row">
              <input
                type="number"
                min={100}
                step={100}
                value={size.w}
                onChange={(e) => {
                  const w = Math.max(100, Number(e.target.value) || 100);
                  setSize({ ...size, w });
                  if (extent) setWin(centred(extent, w, size.h, window));
                }}
              />
              ×
              <input
                type="number"
                min={100}
                step={100}
                value={size.h}
                onChange={(e) => {
                  const h = Math.max(100, Number(e.target.value) || 100);
                  setSize({ ...size, h });
                  if (extent) setWin(centred(extent, size.w, h, window));
                }}
              />
            </span>
          </label>
          <label>
            Colour strength
            <input type="range" min={0.5} max={6} step={0.1} value={saturate} onChange={(e) => setSaturate(Number(e.target.value))} />
          </label>
          <label>
            Hue weight (0 = by importance, 1 = equalised)
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={chromaWeight}
              onChange={(e) => setChromaWeight(Number(e.target.value))}
            />
          </label>
          <label>
            Image underneath
            <input type="range" min={0} max={1} step={0.05} value={mix} onChange={(e) => setMix(Number(e.target.value))} />
          </label>
          {source.value && (
            <ChannelPicker features={source.value.features} selection={selection} onChange={setSelection} missing={built.missing} />
          )}
          <p className="hint">µm*: the store does not declare a unit; Xenium writes transcript coordinates in micrometres.</p>
        </aside>

        <main className="views" ref={viewsRef}>
          {extent && window && (
            <WindowPicker
              extent={extent}
              window={window}
              onChange={setWin}
              overview={overview.value && "view" in overview.value ? overview.value.view : undefined}
              pixelWidth={devWidth}
              tiles={needTiles}
            />
          )}
          <p className="status">
            {error ? <span className="error">{error.message}</span> : busy}
            {!error && !busy && g && (
              <>
                {g.modes.labels.length} channels · {g.stats.perChannel.reduce((s, n) => s + n, 0).toLocaleString()} molecules in them ·{" "}
                {g.stats.belowMinimum.toLocaleString()} below qv {qvMin} · map {g.raster.width}×{g.raster.height} ·{" "}
                {tiles.value?.fetched ?? 0} tiles fetched, {(tiles.value?.tiles.length ?? 0) - (tiles.value?.fetched ?? 0)} from cache ·
                channels {g.ms.channels.toFixed(0)} ms · Gram {g.ms.gram.toFixed(0)} ms
              </>
            )}
            {overview.value && "note" in overview.value && <span className="hint"> · {overview.value.note}</span>}
          </p>
          <p className="hint">
            {overview.value && "label" in overview.value && `Overview image: ${overview.value.label}. `}
            {mapImage.value && `Map image: ${mapImage.value.label}.`}
            {mapImage.error && <span className="error"> Map image: {mapImage.error.message}</span>}
          </p>
          {g && (
            <>
              <ModeMap gram={g} image={overlay} saturate={saturate} chromaWeight={chromaWeight} pixelWidth={devWidth} />
              <ModeLegend gram={g} />
              <h3>How often each pair turns up together, compared with chance</h3>
              <MatrixView gram={g} />
            </>
          )}

          <section className="explainers">
            <Explain title="What is a channel?">
              A channel is one gene, or a group of genes counted together — "T cells" adds up the molecules of several T-cell marker genes.
              A gene can only be in one channel at a time. <b>Negative controls</b> are probes that match no gene: they show what random
              background looks like, so a real pattern should look different from them.
            </Explain>
            <Explain
              title="What does the matrix show?"
              math={String.raw`g_{ab} = \frac{|A|\,\big(C_{ab} - S_{ab}\big)}{W_a W_b}, \qquad C = M M^\top`}
            >
              For each pair of channels, <b>g</b> compares how often their molecules lie within about 2r of each other with what you would
              expect if they were scattered at random over the window. 1 means "no more than chance", 2 means twice as often, below 1 means
              they keep apart. Molecules all sit inside tissue, so almost every pair comes out above 1 — compare pairs with each other, and
              with the negative controls, rather than with 1. In the formula, M holds each channel's smoothed map, W its total, |A| the
              window area and S the self-pairs, which are removed.
            </Explain>
            <Explain
              title="What do the colours mean?"
              math={String.raw`\mathrm{corr} = V \Lambda V^\top, \qquad \text{colour}(x) = \mathrm{OKLab}\big(z(x)\!\cdot\! v_1,\ z(x)\!\cdot\! v_2,\ z(x)\!\cdot\! v_3\big)`}
            >
              The channels' maps are summarised by a few <b>modes</b>: patterns of channels that rise and fall together across the window.
              Mode 1 sets brightness and modes 2 and 3 set hue, so places with similar colours have similar mixtures of genes. The legend
              says which channels push each mode each way, and how much of the variation it carries.
            </Explain>
            <Explain title="Why does the radius set the resolution?">
              r is the neighbourhood each molecule is smoothed over. Anything smaller than r is blurred away, so the map is computed at
              about three pixels per radius — a finer grid would cost more without showing more. A large r shows tissue-scale structure; a
              small r gets closer to single cells, at more cost.
            </Explain>
            <Explain title="What happens when I move the window?">
              Molecules are fetched in tiles (the grey grid) and kept, so moving the box mostly re-uses what is already loaded; the
              statistics are recomputed once you stop. Each mode's sign is kept consistent between windows so colours do not flip for no
              reason — but if two modes swap order, the colours change, and that change is real.
            </Explain>
            <Explain title="What is the quality filter?">
              Every molecule has a quality score, qv. At 20, the chance that it was read as the wrong gene is about 1 in 100. Molecules
              below the threshold are dropped, as in Xenium's own per-cell counts.
            </Explain>
          </section>
        </main>
      </div>
    </div>
  );
}
