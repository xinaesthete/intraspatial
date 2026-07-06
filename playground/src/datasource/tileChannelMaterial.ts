// Tile channel-composite material (ADR-0010 colormap/contrast; ADR-0009 horizon note).
// An OWNED TSL NodeMaterial that replaces the throwaway MeshBasicMaterial: it composites up
// to 4 raw channel planes (packed R,G,B,A in the tile texture) into colour on the GPU, each
// channel with a live { color, contrastLimits, visible } — the viv/MDV channel model, so an
// sd.js/MDV image renders the same and their channel state maps onto us 1:1.
//
// The blend across channels is a PARAMETER, not hardwired (additive today, max as a second
// mode) — the seam for the future raster-blend / layers-panel direction (ADR-0009 horizon),
// built without committing that architecture. `positionNode` is left untouched: the seam where
// psychogeo-style height-field geometry LOD lands later.
//
// Compositing is on the GPU from raw channel samples, so moving the contrast/colour sliders
// re-composites resident tiles live — no refetch, no re-decode. The windowing is vectorised
// across the 4 channels (one vec4 op), which also keeps the TSL types tractable (ADR-0009 §1).
import * as THREE from "three";
import { clamp, Fn, float, max, mix, step, texture, uniform, vec3, vec4 } from "three/tsl";
import { MeshBasicNodeMaterial } from "three/webgpu";

export const MAX_CHANNELS = 4; // packed into one RGBA tile texture

export type BlendMode = "additive" | "max";

/** One channel's live render settings — viv-shaped. `contrastLimits` are in the tile's
 *  normalised [0,1] sample space (the loader normalises by dtype). */
export interface ChannelSettings {
  readonly label: string;
  color: [number, number, number];
  contrastLimits: [number, number];
  visible: boolean;
}

/**
 * Holds the shared uniform nodes for all tile meshes of one image and mints per-mesh
 * materials that reference them. Because the uniforms are shared node objects, `update()`
 * changes every resident tile at once — the live-slider path.
 */
export class ChannelComposite {
  private uColor0 = uniform(vec3(1));
  private uColor1 = uniform(vec3(1));
  private uColor2 = uniform(vec3(1));
  private uColor3 = uniform(vec3(1));
  private uLo = uniform(new THREE.Vector4(0, 0, 0, 0)); // per-channel contrast min
  private uHi = uniform(new THREE.Vector4(1, 1, 1, 1)); // per-channel contrast max
  private uVis = uniform(new THREE.Vector4(0, 0, 0, 0)); // per-channel visibility 0/1
  private uBlend = uniform(0); // 0 = additive, 1 = max

  constructor(settings: readonly ChannelSettings[], blend: BlendMode = "additive") {
    this.update(settings, blend);
  }

  /** Push channel settings into the shared uniforms — affects all live tiles immediately. */
  update(settings: readonly ChannelSettings[], blend?: BlendMode): void {
    const cols = [this.uColor0, this.uColor1, this.uColor2, this.uColor3];
    const lo = new THREE.Vector4(0, 0, 0, 0);
    const hi = new THREE.Vector4(1, 1, 1, 1);
    const vis = new THREE.Vector4(0, 0, 0, 0);
    const comp = ["x", "y", "z", "w"] as const;
    for (let i = 0; i < MAX_CHANNELS; i++) {
      const s = settings[i];
      const uc = cols[i];
      const key = comp[i];
      if (uc) uc.value = new THREE.Vector3(...(s ? s.color : [0, 0, 0]));
      if (key) {
        lo[key] = s ? s.contrastLimits[0] : 0;
        hi[key] = s ? s.contrastLimits[1] : 1;
        vis[key] = s?.visible ? 1 : 0;
      }
    }
    this.uLo.value = lo;
    this.uHi.value = hi;
    this.uVis.value = vis;
    if (blend) this.uBlend.value = blend === "max" ? 1 : 0;
  }

  /** A material for one tile: samples its own texture, composites via the shared uniforms. */
  makeMaterial = (tex: THREE.Texture): THREE.Material => {
    const mat = new MeshBasicNodeMaterial();
    mat.side = THREE.DoubleSide;
    const build = Fn(() => {
      const s = texture(tex); // vec4: .x..w are the 4 raw channel planes
      // windowed intensity per channel (vec4), then gated by visibility
      const w = clamp(s.sub(this.uLo).div(max(this.uHi.sub(this.uLo), vec4(1e-4))), 0, 1).mul(this.uVis);
      const c0 = vec3(this.uColor0).mul(w.x);
      const c1 = vec3(this.uColor1).mul(w.y);
      const c2 = vec3(this.uColor2).mul(w.z);
      const c3 = vec3(this.uColor3).mul(w.w);
      const additive = c0.add(c1).add(c2).add(c3);
      const maxed = max(max(c0, c1), max(c2, c3));
      const rgb = mix(additive, maxed, step(float(0.5), this.uBlend));
      return vec4(clamp(rgb, 0, 1), 1);
    });
    // biome-ignore lint/suspicious/noExplicitAny: TSL node output → colorNode (ADR-0009 §1 friction)
    mat.colorNode = build() as any;
    return mat;
  };
}

// A default fluorescence palette (DAPI-blue first), used when omero carries no channel colours.
const FLUOR_PALETTE: [number, number, number][] = [
  [0.2, 0.5, 1.0], // blue
  [0.2, 1.0, 0.4], // green
  [1.0, 0.35, 0.1], // orange-red
  [1.0, 0.2, 0.8], // magenta
];
const RGB_COLORS: [number, number, number][] = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

/** Default per-channel colours for an image: r/g/b images get RGB (so additive = the original
 *  picture); everything else gets a distinct fluorescence palette. */
export function defaultChannelColors(labels: readonly string[]): [number, number, number][] {
  const isRGB = labels.length === 3 && labels.every((l, i) => l.toLowerCase() === "rgb"[i]);
  const palette = isRGB ? RGB_COLORS : FLUOR_PALETTE;
  return labels.map((_, i) => palette[i % palette.length] ?? [1, 1, 1]);
}
