// The vocabulary of the view-driven datasource (ADR-0008; glossary in CONTEXT.md).
// Milestone 0 keeps these deliberately independent of the op-graph's 2-D-only
// `Shape`: a Tile carries its own 3-D `dims` so a plane (dims[2] === 1) and a
// volume are one type. The op-graph `Tileset` shape is wired in Milestone 1.

import type { Dtype, ElementType } from "../gpu/graph/handle";
import type { Affine3 } from "./math";

export type { Dtype, ElementType } from "../gpu/graph/handle";

/** A `Multiscale` handle: cheap metadata for an addressable pyramid, no samples.
 *  Level L downsamples every axis by `2^L`; chunks hold a constant voxel count per
 *  level (`chunkShape`). `worldFromArray` places level-0 voxels into world. */
export interface Multiscale {
  /** Full-resolution voxel dimensions `[W, H, D]`; `D === 1` for a plane. */
  readonly voxelDims0: readonly [number, number, number];
  /** Voxels per chunk along each axis, constant across levels. */
  readonly chunkShape: readonly [number, number, number];
  /** Number of pyramid levels (level 0 = full resolution). */
  readonly levelCount: number;
  /** Array space (level-0 voxel units) → world. */
  readonly worldFromArray: Affine3;
  readonly element: ElementType;
  readonly dtype: Dtype;
}

/** Names one chunk of one Level: the atom of I/O and caching. */
export interface ChunkId {
  readonly level: number;
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

/** A resolved Chunk — decoded samples of one `(level, x, y, z)`. `dims` is the
 *  chunk's actual voxel extent (smaller than `chunkShape` at pyramid borders);
 *  `data` is lane-major interleaved, length `dims[0]·dims[1]·dims[2]·lanes`. */
export interface Tile {
  readonly id: ChunkId;
  readonly dims: readonly [number, number, number];
  readonly element: ElementType;
  readonly dtype: Dtype;
  readonly data: Float32Array;
}

/** The impure seam Resolve calls (deck.gl-`getTileData`-shaped). */
export interface Loader {
  getChunk(id: ChunkId): Promise<Tile>;
}

/** One entry of a Selection: which chunk, plus the geometry that justified it —
 *  `nearestDepth` for load-priority ordering, `approxBytes` for the budget HUD. */
export interface SelectedChunk {
  readonly id: ChunkId;
  /** Optical-axis depth of the chunk's nearest point (world units). */
  readonly nearestDepth: number;
  /** Decoded (uncompressed) byte size of this chunk's samples. */
  readonly approxBytes: number;
}

/** First-class data on an edge: what the current Camera (or region selector) wants.
 *  Names the chunks; does not fetch. */
export interface Selection {
  readonly chunks: readonly SelectedChunk[];
  /** Sum of `approxBytes` — the working-set estimate the Resource ceiling bounds. */
  readonly totalApproxBytes: number;
  /** Count of selected chunks per level (index = level), for the HUD. */
  readonly countByLevel: readonly number[];
}

/** A fallible result (Rust-flavoured), used where the Resource ceiling can force a
 *  give-up after degrade-to-fit (ADR-0008 §5). */
export type Result<T, E = string> = { readonly ok: true; readonly value: T } | { readonly ok: false; readonly error: E };

const DtypeSizes: Record<Dtype, number> = {
  f32: 4,
  u32: 4,
  i32: 4,
} as const;
/** Bytes one sample of `(dtype, element)` occupies once decoded. */
export function bytesPerSample(dtype: Dtype, element: ElementType): number {
  const lanes = element.kind === "complex" ? 2 : element.kind === "vec" ? element.n : element.kind === "quaternion" ? 4 : 1;
  return DtypeSizes[dtype] * lanes; // f32/i32/u32 are all 4 bytes, but in future we should support others
}
