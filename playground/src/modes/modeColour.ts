// The OKLab axis colour of mode k (0, 1, 2) — one key shared by the map, the legend, the scree and
// the loadings, so a mode's bar, its row and its colour on screen read as the same thing.

import { oklabToSrgb } from "../../../src/color/oklab";
import { cssRgb } from "../../../src/color/ramps";
import { modeSwatch } from "../../../src/gpu/spatial/gramModes";

export const modeColour = (k: number, sign: 1 | -1 = 1): string =>
  cssRgb(oklabToSrgb(modeSwatch([k === 0 ? sign : 0, k === 1 ? sign : 0, k === 2 ? sign : 0])));
