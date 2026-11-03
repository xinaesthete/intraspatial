// Sidebar controls every mode-map page has: which store, which image under the map, and how the
// map is painted. Each comes with the hook that holds its state, so a page keeps only what is its own.

import { useState } from "react";

/** How the mode map is painted: what `GramModesLayer` takes as `saturate`, `chromaWeight`, `opacity`. */
export function useMapStyle(initial: { saturate?: number; chromaWeight?: number; opacity?: number } = {}) {
  const [saturate, setSaturate] = useState(initial.saturate ?? 2.5);
  const [chromaWeight, setChromaWeight] = useState(initial.chromaWeight ?? 1);
  const [opacity, setOpacity] = useState(initial.opacity ?? 0.8);
  return { saturate, setSaturate, chromaWeight, setChromaWeight, opacity, setOpacity };
}

export type MapStyle = ReturnType<typeof useMapStyle>;

export function MapStyleControls({ style }: { style: MapStyle }) {
  return (
    <>
      <label>
        Colour strength (saturate at ±{style.saturate.toFixed(1)}σ)
        <input
          type="range"
          min={0.5}
          max={6}
          step={0.1}
          value={style.saturate}
          onChange={(e) => style.setSaturate(Number(e.target.value))}
        />
      </label>
      <label>
        Hue weight (0 = by importance, 1 = equalised)
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={style.chromaWeight}
          onChange={(e) => style.setChromaWeight(Number(e.target.value))}
        />
      </label>
      <label>
        Map over image
        <input type="range" min={0} max={1} step={0.05} value={style.opacity} onChange={(e) => style.setOpacity(Number(e.target.value))} />
      </label>
    </>
  );
}

/** A store URL box; `onOpen` gets the trimmed URL when the button is pressed. */
export function StoreInput({ initial, onOpen }: { initial: string; onOpen: (url: string) => void }) {
  const [draft, setDraft] = useState(initial);
  return (
    <label>
      Store
      <span className="row">
        <input value={draft} onChange={(e) => setDraft(e.target.value)} />
        <button type="button" onClick={() => onOpen(draft.trim())}>
          open
        </button>
      </span>
    </label>
  );
}

/** `choice` for "no image". `undefined` means "pick one". */
const NO_IMAGE = "";

/**
 * The image under the map: the one chosen, or `pick(names)` (the first by default) until one is, or
 * none. `reset` goes back to picking, for a new store or coordinate system.
 */
export function useImageChoice(names: readonly string[], pick: (names: readonly string[]) => string | undefined = (n) => n[0]) {
  const [choice, setChoice] = useState<string>();
  const image = choice === NO_IMAGE ? undefined : (choice ?? pick(names));
  return { names, image, choose: setChoice, reset: () => setChoice(undefined) };
}

export type ImageChoice = ReturnType<typeof useImageChoice>;

export function ImagePicker({ images }: { images: ImageChoice }) {
  return (
    <label>
      Image
      <select value={images.image ?? NO_IMAGE} onChange={(e) => images.choose(e.target.value)}>
        {images.names.map((n) => (
          <option key={n}>{n}</option>
        ))}
        <option value={NO_IMAGE}>(none)</option>
      </select>
    </label>
  );
}
