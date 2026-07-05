// The textured-tile render backend (ADR-0008 slice 2): turns a Selection into the real
// image on the plane, at each chunk's chosen level. Resolve is realised here — miss →
// loader.getChunk → upload to a GPU texture → cache (the working-set TileCache, tier 3);
// the decoded CPU Tile is dropped after upload. Meshes for out-of-view chunks are
// removed but their textures stay cached (LRU) for cheap return. Plane only for now
// (volume raymarch is a later slice).
import * as THREE from "three";
import {
  applyAffine,
  chunkArrayBox,
  chunkKey,
  type Loader,
  type Multiscale,
  type Selection,
  type Tile,
  TileCache,
  type Vec3,
} from "../../../src/datasource";

export class TileRenderer {
  readonly group = new THREE.Group();
  private cache: TileCache<THREE.Texture>;
  private meshes = new Map<string, THREE.Mesh>();
  private loading = new Set<string>();
  private desired = new Set<string>();

  constructor(
    private ms: Multiscale,
    private loader: Loader,
    maxBytes = 256 * 1024 * 1024,
  ) {
    this.cache = new TileCache<THREE.Texture>({ maxBytes, dispose: (t) => t.dispose() });
  }

  /** Ensure textured meshes for the current Selection; async, non-blocking. */
  update(selection: Selection): void {
    this.desired = new Set(selection.chunks.map((c) => chunkKey(c.id)));
    for (const k of [...this.meshes.keys()]) {
      if (!this.desired.has(k)) this.removeMesh(k);
    }
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
  }

  private async load(k: string, id: Tile["id"]): Promise<void> {
    try {
      const tile = await this.loader.getChunk(id);
      const tex = this.makeTexture(tile);
      this.cache.set(k, tex, tile.dims[0] * tile.dims[1] * 4);
      if (this.desired.has(k) && !this.meshes.has(k)) this.addMesh(k, id, tex);
    } finally {
      this.loading.delete(k);
    }
  }

  private makeTexture(tile: Tile): THREE.Texture {
    const [ex, ey] = tile.dims;
    const rgba = new Uint8Array(ex * ey * 4);
    for (let i = 0; i < ex * ey; i++) {
      const g = Math.max(0, Math.min(255, Math.round((tile.data[i] ?? 0) * 255)));
      rgba[i * 4] = g;
      rgba[i * 4 + 1] = g;
      rgba[i * 4 + 2] = g;
      rgba[i * 4 + 3] = 255;
    }
    const tex = new THREE.DataTexture(rgba, ex, ey);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    return tex;
  }

  private addMesh(k: string, id: Tile["id"], tex: THREE.Texture): void {
    const [lo, hi] = chunkArrayBox(this.ms, id);
    const a = this.ms.worldFromArray;
    const z = 0.5;
    const p = (x: number, y: number): Vec3 => applyAffine(a, [x, y, z]);
    const c00 = p(lo[0], lo[1]),
      c10 = p(hi[0], lo[1]),
      c11 = p(hi[0], hi[1]),
      c01 = p(lo[0], hi[1]);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute([...c00, ...c10, ...c11, ...c00, ...c11, ...c01], 3));
    geo.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1], 2));
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide }));
    this.meshes.set(k, mesh);
    this.group.add(mesh);
  }

  private removeMesh(k: string): void {
    const mesh = this.meshes.get(k);
    if (!mesh) return;
    this.group.remove(mesh);
    mesh.geometry.dispose();
    const m = mesh.material;
    if (Array.isArray(m)) m.forEach((mm) => mm.dispose());
    else m.dispose();
    this.meshes.delete(k);
  }

  dispose(): void {
    for (const k of [...this.meshes.keys()]) this.removeMesh(k);
    this.cache.clear();
  }
}
