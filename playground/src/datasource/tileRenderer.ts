// The textured-tile render backend (ADR-0008 slice 2): turns a Selection into the real
// image on the plane, at each chunk's chosen level. Resolve is realised here — miss →
// loader.getChunk → upload to a GPU texture → cache (the working-set TileCache, tier 3);
// the decoded CPU Tile is dropped after upload. Meshes for out-of-view chunks are
// removed but their textures stay cached (LRU) for cheap return. Plane only for now
// (volume raymarch is a later slice).
import * as THREE from "three";
import {
  applyAffine,
  type ChunkId,
  chunkArrayBox,
  chunkKey,
  type Loader,
  type MemoryReporting,
  type Multiscale,
  type SelectedChunk,
  type Selection,
  type Tile,
  TileCache,
  type Vec3,
} from "../../../src/datasource";
import { greyscaleMaterial } from "./tileChannelMaterial";

/** Per-level depth separation (in array-z / voxel units): each coarser level sits this much farther
 *  along the plane normal, so a finer tile wins the depth test over a coexisting coarse fallback
 *  without z-fighting. Sub-voxel, so no visible parallax; big enough to clear depth-buffer precision
 *  when zoomed in (where it matters). Fixed direction for now — a camera-relative (view-dependent)
 *  bias is a follow-up (see docs/datasource-renderer-plan.md). */
const LEVEL_Z_BIAS = 0.1;

/** Lanes a tile carries (a scalar tile is one lane). */
const tileLanes = (tile: Tile): number => (tile.element.kind === "vec" ? tile.element.n : 1);

/** GPU texture format for `lanes` channel planes. WebGPU has no 3-channel texture, so 3 lanes
 *  pack into RGBA (the 4th unused). `comps` is the stored components-per-texel. */
function texFormat(lanes: number): { format: THREE.PixelFormat; comps: number } {
  if (lanes <= 1) return { format: THREE.RedFormat, comps: 1 };
  if (lanes === 2) return { format: THREE.RGFormat, comps: 2 };
  return { format: THREE.RGBAFormat, comps: 4 };
}

/** Decoded (uncompressed) VRAM bytes of a tile's texture — half-float (2 B) × components. */
const textureBytes = (tile: Tile): number => tile.dims[0] * tile.dims[1] * texFormat(tileLanes(tile)).comps * 2;

export class TileRenderer implements MemoryReporting {
  readonly group = new THREE.Group();
  private cache: TileCache<THREE.Texture>;
  private meshes = new Map<string, THREE.Mesh>();
  private meshIds = new Map<string, ChunkId>(); // chunk id per resident mesh (for coverage pruning)
  private loading = new Set<string>();
  private desired = new Set<string>();
  private selChunks: readonly SelectedChunk[] = []; // the current Selection, for coverage pruning

  /** Optional material factory: given a tile texture, return the mesh material. When absent,
   *  a plain greyscale/RGB `MeshBasicMaterial` (synthetic path). The SpatialData path passes the
   *  owned channel-composite material (ADR-0010) — the tile texture then carries RAW channel
   *  planes and the material does colour/window/composite on the GPU. */
  private makeMaterial?: (tex: THREE.Texture) => THREE.Material;
  private blending: THREE.Blending = THREE.NormalBlending; // layer blend mode, applied to every tile material
  private depthTest = true; // overlays turn this off so they composite over the base regardless of z
  private depthWrite = true;
  private renderOrderVal = 0; // draw order = layer order (later layers on top)

  constructor(
    private ms: Multiscale,
    private loader: Loader,
    opts: { maxBytes?: number; makeMaterial?: (tex: THREE.Texture) => THREE.Material } = {},
  ) {
    this.makeMaterial = opts.makeMaterial;
    this.cache = new TileCache<THREE.Texture>({ maxBytes: opts.maxBytes ?? 256 * 1024 * 1024, dispose: (t) => t.dispose() });
  }

  /** Resident texture VRAM (MemoryReporting) — the actual working set held on the GPU. */
  get byteLength(): number {
    return this.cache.byteLength;
  }
  /** Chunks currently being fetched/decoded — lets the host refresh the memory readout as
   *  tiles stream in (resident bytes grow between camera moves). */
  get loadingCount(): number {
    return this.loading.size;
  }

  /** Ensure textured meshes for the current Selection; async, non-blocking. Rather than drop
   *  out-of-selection tiles immediately (which flashes the dark background while the new tiles
   *  decode), we KEEP them as a fallback and only prune one once every newly-desired tile
   *  covering its region has loaded — so a moving camera shows coarser detail, never holes. */
  update(selection: Selection): void {
    this.selChunks = selection.chunks;
    this.desired = new Set(selection.chunks.map((c) => chunkKey(c.id)));
    for (const sc of selection.chunks) {
      const k = chunkKey(sc.id);
      if (this.meshes.has(k)) continue;
      const tex = this.cache.get(k);
      if (tex) {
        this.addMesh(k, sc.id, tex);
        continue;
      }
      if (this.loading.has(k)) continue;
      this.loading.add(k);
      void this.load(k, sc.id);
    }
    this.prune();
  }

  /** Set the layer blend mode for this image's tiles (Normal = over, Additive = sum). Applies to
   *  every resident tile material and is remembered for tiles that stream in later. */
  setBlending(blending: THREE.Blending): void {
    this.blending = blending;
    this.forEachMaterial((m) => {
      m.blending = blending;
    });
  }

  /** Depth behaviour for this image's tiles. The base layer keeps depth on (intra-image LOD: finer
   *  tiles win via LEVEL_Z_BIAS); overlay layers turn depthTest off so they composite over the base
   *  regardless of z (the layers model — one shared depth buffer can't isolate per-image LOD AND
   *  blend across images without render-to-texture). */
  setDepth(depthTest: boolean, depthWrite: boolean): void {
    this.depthTest = depthTest;
    this.depthWrite = depthWrite;
    this.forEachMaterial((m) => {
      m.depthTest = depthTest;
      m.depthWrite = depthWrite;
    });
  }

  /** Draw order for this image's tiles (higher = later = on top), so layers composite in order. */
  setRenderOrder(order: number): void {
    this.renderOrderVal = order;
    for (const mesh of this.meshes.values()) mesh.renderOrder = order;
  }

  private forEachMaterial(fn: (m: THREE.Material) => void): void {
    for (const mesh of this.meshes.values()) {
      const m = mesh.material;
      if (Array.isArray(m)) for (const mm of m) fn(mm);
      else fn(m);
    }
  }

  private async load(k: string, id: Tile["id"]): Promise<void> {
    try {
      const tile = await this.loader.getChunk(id);
      const tex = this.makeTexture(tile);
      this.cache.set(k, tex, textureBytes(tile));
      if (this.desired.has(k) && !this.meshes.has(k)) this.addMesh(k, id, tex);
    } finally {
      this.loading.delete(k);
      this.prune(); // a freshly-resident tile may now cover (and release) a fallback
    }
  }

  /** Drop a retained fallback tile once no still-loading desired tile overlaps its region — i.e.
   *  it is either fully covered by resident desired tiles or entirely out of the new selection. */
  private prune(): void {
    for (const [k, id] of [...this.meshIds]) {
      if (this.desired.has(k)) continue; // desired tiles are never fallbacks
      const box = chunkArrayBox(this.ms, id);
      const stillNeeded = this.selChunks.some((sc) => {
        if (this.meshes.has(chunkKey(sc.id))) return false; // that desired tile is already up
        return overlaps2D(box, chunkArrayBox(this.ms, sc.id));
      });
      if (!stillNeeded) this.removeMesh(k);
    }
  }

  private makeTexture(tile: Tile): THREE.Texture {
    const [ex, ey] = tile.dims;
    const lanes = tileLanes(tile);
    const { format, comps } = texFormat(lanes);
    // Store the decoded samples as HALF-FLOAT, not quantised to 8-bit: 8-bit banded the narrow
    // uint16 fluorescence window. Planes are packed into exactly `comps` channels (no greyscale
    // replication) — a scalar tile is one Red channel, the material/greyscale shader reads .r.
    // (Precision is fixed at fp16 for now; making it configurable — fp32 exact ↔ 8-bit for VRAM —
    // is a future VRAM/quality knob.) This texture is raw DATA (NoColorSpace); the material colours it.
    const half = new Uint16Array(ex * ey * comps);
    const toHalf = THREE.DataUtils.toHalfFloat;
    for (let i = 0; i < ex * ey; i++) {
      for (let c = 0; c < comps; c++) half[i * comps + c] = toHalf(c < lanes ? (tile.data[i * lanes + c] ?? 0) : 0);
    }
    const tex = new THREE.DataTexture(half, ex, ey, format, THREE.HalfFloatType);
    // nb in viv and elsewhere there is an assumption that Nearest is more appropriate;
    // I disagree; I don't think aliasing is a more faithful representation of a signal...
    // but ultimately, this should be tweakable.
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.colorSpace = THREE.NoColorSpace;
    tex.needsUpdate = true;
    return tex;
  }

  private addMesh(k: string, id: Tile["id"], tex: THREE.Texture): void {
    const [lo, hi] = chunkArrayBox(this.ms, id);
    const a = this.ms.worldFromArray;
    // Sit coarser levels a hair farther along the plane normal (+array-z), so where a coarse
    // fallback and a finer tile briefly coexist the finer one wins the depth test — no z-fighting —
    // while every tile still depth-tests AND writes depth against the rest of the scene.
    // (polygonOffset is the view-independent tool for this but three's WebGPU backend ignores it.)
    // The bias assumes the viewer is on the plane's front side, as an image viewer is; from the far
    // side it degrades to showing the coarser tile, not a mosaic. Per-region masking is the real fix.
    const z = 0.5 + id.level * LEVEL_Z_BIAS;
    const p = (x: number, y: number): Vec3 => applyAffine(a, [x, y, z]);
    const c00 = p(lo[0], lo[1]),
      c10 = p(hi[0], lo[1]),
      c11 = p(hi[0], hi[1]),
      c01 = p(lo[0], hi[1]);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute([...c00, ...c10, ...c11, ...c00, ...c11, ...c01], 3));
    geo.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1], 2));
    const material = this.makeMaterial ? this.makeMaterial(tex) : greyscaleMaterial(tex);
    material.blending = this.blending;
    material.depthTest = this.depthTest;
    material.depthWrite = this.depthWrite;
    const mesh = new THREE.Mesh(geo, material);
    mesh.renderOrder = this.renderOrderVal;
    this.meshes.set(k, mesh);
    this.meshIds.set(k, id);
    this.group.add(mesh);
  }

  private removeMesh(k: string): void {
    const mesh = this.meshes.get(k);
    if (!mesh) return;
    this.group.remove(mesh);
    mesh.geometry.dispose();
    const m = mesh.material;
    if (Array.isArray(m))
      m.forEach((mm) => {
        mm.dispose();
      });
    else m.dispose();
    this.meshes.delete(k);
    this.meshIds.delete(k);
  }

  dispose(): void {
    for (const k of [...this.meshes.keys()]) this.removeMesh(k);
    this.cache.clear();
  }
}

/** Do two array-space chunk boxes overlap in the plane (x,y)? Touching edges don't count. */
function overlaps2D(a: [Vec3, Vec3], b: [Vec3, Vec3]): boolean {
  return a[0][0] < b[1][0] && a[1][0] > b[0][0] && a[0][1] < b[1][1] && a[1][1] > b[0][1];
}
