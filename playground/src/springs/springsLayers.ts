// The spring model as deck layers: `@intraspatial/deck`'s ParticleLayer and BondLayer bound to a
// `SpringsGpu`'s own buffers. Nothing is read back: the layers ask for the position buffer at every
// draw, because the step swaps two buffers.
import type { Layer, LayerProps } from "@deck.gl/core";
import { BondLayer, type Color, ParticleLayer } from "@intraspatial/deck";
import { oklchToSrgbMapped, rgbBytes } from "../../../src/color/index";
import { ATTR_FLOATS, ATTR_RADIUS, ATTR_SPECIES, DEAD, SLOT_WORDS, type SpringSim } from "../../../src/gpu/sim/springs/index";

const deg = (d: number) => (d * Math.PI) / 180;
const HIDDEN: Color = [0, 0, 0, 0];
/** By species: cell, matrix, leader (scenarios.ts). */
const CELL_COLOUR: Color = rgbBytes(oklchToSrgbMapped([0.78, 0.12, deg(190)]));
const MATRIX_COLOUR: Color = [...rgbBytes(oklchToSrgbMapped([0.62, 0.03, deg(260)])), 255];
const LEADER_COLOUR: Color = rgbBytes(oklchToSrgbMapped([0.68, 0.2, deg(28)]));
/** By bond kind: plain, matrix, ligand. */
const BOND_COLOURS: readonly Color[] = [
  [226, 232, 240, 150],
  [148, 163, 184, 70],
  [251, 191, 36, 200],
];

/** A 4×4 matrix, column-major, as deck's `modelMatrix` takes it. */
export type Matrix16 = NonNullable<LayerProps["modelMatrix"]>;

/**
 * What a view needs of a running particle model, and nothing more: the GPU buffers to bind and
 * their sizes. Views take this, not the object that happens to own the buffers. There are two
 * sources, which lay a particle's species and radius out differently:
 *   - `SpringsGpu` packs them into one attribute buffer (`attrs`, with `attrLayout`);
 *   - a model run as graph ops (`createSpringSim`, through `simBuffers`) keeps each in a buffer of
 *     its own (`species`, `radii`), and its bond table is one word per slot.
 */
export interface SimBuffers {
  /** `vec4f` per particle (xyz, density), the current positions first. May change every step: ask per frame. */
  positions(): GPUBuffer | null;
  readonly n: number;
  readonly K: number;
  /** Per-particle attributes, `attrLayout.stride` f32 each. */
  readonly attrs?: GPUBuffer;
  /** Where the species and radius sit in `attrs`. Default: `SpringsGpu`'s layout. */
  readonly attrLayout?: { readonly stride: number; readonly species: number; readonly radius: number };
  /** One u32 per particle, when there is no `attrs`. Asked per frame. */
  readonly species?: () => GPUBuffer | null;
  /** One f32 per particle, when there is no `attrs`. Asked per frame. */
  readonly radii?: () => GPUBuffer | null;
  /** `K` slots per particle: word 0 of a slot is partner | kind << 24, all ones when empty. */
  readonly slots: GPUBuffer | (() => GPUBuffer | null);
  /** u32 per slot. Default: `SpringsGpu`'s 4. */
  readonly slotWords?: number;
}

/** A view's own copy of a graph-run model's state, and how to refresh it. */
export interface SimSnapshot {
  readonly buffers: SimBuffers;
  /** Copy the model's current state across, on the device. Call when no advance is running. */
  refresh(): void;
  destroy(): void;
}

/**
 * A graph-run model's state as a view takes it: copies the view owns, refreshed on the device
 * (no download) after each advance. The model's own buffers cannot be bound directly: while an
 * advance is running they are being rewritten or handed to other ops, and a frame drawn then
 * shows nothing or nonsense. The copy is small: a few tens of bytes per particle and 4 per bond slot.
 */
export function simSnapshot(device: GPUDevice, sim: SpringSim, n: number, K: number): SimSnapshot {
  const make = (label: string, size: number) =>
    device.createBuffer({
      label: `springs:view-${label}`,
      size: Math.max(16, size),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  const own = {
    positions: make("positions", n * 16),
    species: make("species", n * 4),
    radius: make("radius", n * 4),
    link: make("link", n * K * 4),
  };
  let filled = false;
  const ready = <T>(v: T) => (filled ? v : null);
  return {
    buffers: {
      n,
      K,
      positions: () => ready(own.positions),
      species: () => ready(own.species),
      radii: () => ready(own.radius),
      slots: () => ready(own.link),
      slotWords: 1,
    },
    refresh() {
      const now = sim.buffers();
      if (!now) return;
      const enc = device.createCommandEncoder({ label: "springs:view-copy" });
      for (const part of ["positions", "species", "radius", "link"] as const) {
        const from = now[part];
        enc.copyBufferToBuffer(from.buffer, from.offset, own[part], 0, Math.min(from.size, own[part].size));
      }
      device.queue.submit([enc.finish()]);
      filled = true;
    },
    destroy() {
      for (const b of Object.values(own)) b.destroy();
    },
  };
}

export interface SpringsLayerOptions {
  /** Prefix for the layer ids, so two sets can share a Deck. */
  readonly id?: string;
  /** Draw the gel: its particles as small beads and its bonds as lines. */
  readonly matrix?: boolean;
  /** Lit spheres (1) or flat discs (0). */
  readonly shading?: number;
  /** Smallest drawn cell, in pixels. */
  readonly radiusMinPixels?: number;
  /** Model units to the view's units, when the model is drawn over data: a 4×4 matrix, column-major. */
  readonly modelMatrix?: Matrix16;
  /** The scale in `modelMatrix`, which deck applies to positions but not to sizes. */
  readonly scale?: number;
  /** Draw over whatever is there, ignoring depth: for a flat overlay on a map. */
  readonly onTop?: boolean;
}

/** Layers for one running spring model. Cells first, then (optionally) the gel. */
export function springsLayers(gpu: SimBuffers, o: SpringsLayerOptions = {}): Layer[] {
  const layout = gpu.attrLayout ?? { stride: ATTR_FLOATS, species: ATTR_SPECIES, radius: ATTR_RADIUS };
  const id = o.id ?? "springs";
  const scale = o.scale ?? 1;
  const placed = {
    ...(o.modelMatrix ? { modelMatrix: o.modelMatrix } : {}),
    ...(o.onTop ? { parameters: { depthCompare: "always" as const } } : {}),
  };
  // Species and radius: from the one attribute buffer, or from a buffer each.
  const fields = gpu.attrs
    ? {
        attributes: gpu.attrs,
        attributeStride: layout.stride,
        categoryOffset: layout.species,
        categoryFormat: "f32" as const,
        radiusOffset: layout.radius,
      }
    : { attributes: gpu.species, attributeStride: 1, categoryOffset: 0, categoryFormat: "u32" as const, radii: gpu.radii };
  const shared = {
    positions: () => gpu.positions(),
    count: gpu.n,
    ...fields,
    deadCategory: DEAD,
    shading: o.shading ?? 1,
    ...placed,
  };
  const layers: Layer[] = [
    new ParticleLayer({
      ...shared,
      id: `${id}-cells`,
      colors: [CELL_COLOUR, HIDDEN, LEADER_COLOUR, HIDDEN],
      radiusScale: scale,
      radiusMinPixels: o.radiusMinPixels ?? 0,
    }),
  ];
  if (o.matrix) {
    layers.push(
      new ParticleLayer({ ...shared, id: `${id}-matrix`, colors: [HIDDEN, MATRIX_COLOUR, HIDDEN, HIDDEN], radiusScale: 0.3 * scale }),
      new BondLayer({
        id: `${id}-bonds`,
        positions: () => gpu.positions(),
        slots: gpu.slots,
        count: gpu.n,
        slotsPerParticle: gpu.K,
        slotWords: gpu.slotWords ?? SLOT_WORDS,
        colors: BOND_COLOURS,
        ...placed,
      }),
    );
  }
  return layers;
}
