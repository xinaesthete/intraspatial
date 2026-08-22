// The NAIVE volume renderer — the baseline to compare the brick-atlas + page-table
// `VolumeRenderer` against. One draw call per selected chunk: each chunk is its own box
// with its own 3D texture, raymarched in its own local space, composited back-to-front by
// three's transparent sort. No atlas, no page table, no per-chunk-LOD stitching in a single
// pass — just N passes with overdraw. Simpler to reason about; the cost is the overdraw and
// the draw-call count. (Depth-read/write parity with the brick-page version is intentionally
// omitted here — this is a rendering-technique A/B, not a feature-complete twin.)
import * as THREE from "three";
import {
  cameraPosition,
  clamp,
  Fn,
  float,
  Loop,
  max,
  min,
  mix,
  normalize,
  oneMinus,
  positionWorld,
  texture3D,
  uniform,
  vec3,
  vec4,
} from "three/tsl";
import { MeshBasicNodeMaterial } from "three/webgpu";
import {
  chunkArrayBox,
  chunkKey,
  hostSamples,
  type Loader,
  type MemoryReporting,
  type Multiscale,
  type ResidentTexture,
  type Selection,
  type Tile,
  worldFromArrayOf,
} from "../../../src/datasource";
import { destroyTileTexture } from "../../../src/gpu/tiles/assemble";

const STEPS = 48; // per chunk — each box is small, so fewer steps than the whole-volume march

interface ChunkMesh {
  mesh: THREE.Mesh;
  tex: THREE.Data3DTexture | THREE.ExternalTexture;
  /** Present when the brick came back device-resident: the GPUTexture WE own and must free.
   *  three never destroys an `ExternalTexture`'s source (WebGPUTextureUtils skips it explicitly),
   *  so `tex.dispose()` alone would leak the whole brick's VRAM on every eviction. */
  owned?: ResidentTexture;
  bytes: number;
  use: number;
}

export class NaiveVolumeRenderer implements MemoryReporting {
  readonly group = new THREE.Group();
  private ms: Multiscale;
  private loader: Loader;
  private uCmin = uniform(0.15);
  private uCmax = uniform(1.0);
  private uGamma = uniform(1.0);
  private uSolid = uniform(0.55);
  private resident = new Map<string, ChunkMesh>();
  private loading = new Set<string>();
  private useClock = 0;
  private M: THREE.Matrix4; // base array→world (no gizmo tracking in the naive baseline)

  constructor(ms: Multiscale, loader: Loader) {
    this.ms = ms;
    this.loader = loader;
    const { axes, origin } = worldFromArrayOf(ms);
    const [ax0, ax1, ax2] = axes;
    this.M = new THREE.Matrix4()
      .makeBasis(
        new THREE.Vector3(ax0[0], ax0[1], ax0[2]),
        new THREE.Vector3(ax1[0], ax1[1], ax1[2]),
        new THREE.Vector3(ax2[0], ax2[1], ax2[2]),
      )
      .setPosition(origin[0], origin[1], origin[2]);
  }

  /** Resident VRAM (MemoryReporting): one 3D texture per resident chunk (no shared atlas). */
  get byteLength(): number {
    let n = 0;
    for (const r of this.resident.values()) n += r.bytes;
    return n;
  }

  setTransfer(cmin: number, cmax: number, gamma: number): void {
    this.uCmin.value = cmin;
    this.uCmax.value = Math.max(cmin + 1e-3, cmax);
    this.uGamma.value = gamma;
  }
  setSolid(threshold: number): void {
    this.uSolid.value = threshold;
  }
  // Depth-read/write are not modelled in the naive baseline (see the file header).
  setDepthRead(_on: boolean): void {}
  setDepthWrite(_on: boolean): void {}

  update(sel: Selection): void {
    const desired = new Set(sel.chunks.map((c) => chunkKey(c.id)));
    for (const c of sel.chunks) {
      const k = chunkKey(c.id);
      const r = this.resident.get(k);
      if (r) {
        r.use = ++this.useClock;
        continue;
      }
      if (!this.loading.has(k)) {
        this.loading.add(k);
        void this.load(k, c.id);
      }
    }
    // Drop chunk meshes no longer selected (naive: no LRU pool, just release immediately).
    for (const [k, r] of [...this.resident]) {
      if (desired.has(k)) continue;
      this.group.remove(r.mesh);
      r.mesh.geometry.dispose();
      (r.mesh.material as THREE.Material).dispose();
      r.tex.dispose();
      if (r.owned) destroyTileTexture(r.owned);
      this.resident.delete(k);
    }
  }

  private async load(k: string, id: Tile["id"]): Promise<void> {
    try {
      const tile = await this.loader.getChunk(id);
      const [ex, ey, ez] = tile.dims;
      const entry: Omit<ChunkMesh, "mesh"> = tile.texture
        ? // Device-resident: the samples were assembled on the renderer's own device and are
          // already in their final format. Nothing here touches them — this is the whole point.
          { tex: externalTexture3D(tile.texture), owned: tile.texture, bytes: ex * ey * ez * 2, use: ++this.useClock }
        : // Host path, unchanged: quantise to R8 and upload.
          (() => {
            const src = hostSamples(tile);
            const data = new Uint8Array(ex * ey * ez);
            for (let i = 0; i < data.length; i++) data[i] = Math.round((src[i] ?? 0) * 255);
            return { tex: makeR8Texture(data, ex, ey, ez), bytes: data.byteLength, use: ++this.useClock };
          })();
      const mesh = this.makeChunkMesh(id, entry.tex);
      this.group.add(mesh);
      this.resident.set(k, { ...entry, mesh });
    } finally {
      this.loading.delete(k);
    }
  }

  /** One box for one chunk, at its own array box under the base affine, with a per-chunk
   *  raymarch that samples that chunk's texture in [0,1] chunk space. */
  private makeChunkMesh(id: Tile["id"], tex: THREE.Data3DTexture | THREE.ExternalTexture): THREE.Mesh {
    const [lo, hi] = chunkArrayBox(this.ms, id);
    const ext: [number, number, number] = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    const centre: [number, number, number] = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    // Mesh local space is the centred chunk box; its matrix places it in world via the affine.
    const meshMatrix = this.M.clone().multiply(new THREE.Matrix4().makeTranslation(centre[0], centre[1], centre[2]));
    // world → [0,1] chunk-texture coords = T(0.5) · diag(1/ext) · meshMatrix⁻¹.
    const worldToChunk = new THREE.Matrix4()
      .makeTranslation(0.5, 0.5, 0.5)
      .multiply(new THREE.Matrix4().makeScale(1 / ext[0], 1 / ext[1], 1 / ext[2]))
      .multiply(meshMatrix.clone().invert());
    const uW2C = uniform(worldToChunk);

    const mat = new MeshBasicNodeMaterial();
    mat.transparent = true;
    mat.depthWrite = false;
    mat.depthTest = false;
    mat.side = THREE.BackSide;

    mat.colorNode = Fn(() => {
      const camW = cameraPosition;
      const dirW = normalize(positionWorld.sub(camW));
      const camC = uW2C.mul(vec4(camW, 1.0)).xyz;
      const dirC = uW2C.mul(vec4(dirW, 0.0)).xyz;
      const invD = vec3(1.0).div(dirC);
      const tA = vec3(0.0).sub(camC).mul(invD);
      const tB = vec3(1.0).sub(camC).mul(invD);
      const tmin = min(tA, tB);
      const tmax = max(tA, tB);
      const tEnter = max(max(tmin.x, tmin.y), tmin.z).max(0.0);
      const tExit = min(min(tmax.x, tmax.y), tmax.z);
      const stepLen = tExit.sub(tEnter).div(float(STEPS));
      const col = vec3(0.0).toVar();
      const alpha = float(0.0).toVar();
      Loop(STEPS, ({ i }) => {
        const t = tEnter.add(stepLen.mul(float(i).add(0.5)));
        const p = camC.add(dirC.mul(t));
        const density = texture3D(tex, p).r;
        const dm = clamp(density.sub(this.uCmin).div(this.uCmax.sub(this.uCmin)), 0.0, 1.0).pow(this.uGamma);
        const a = dm.mul(0.12).mul(oneMinus(alpha));
        const sc = mix(vec3(0.1, 0.03, 0.18), vec3(1.0, 0.86, 0.55), dm);
        col.addAssign(sc.mul(a));
        alpha.addAssign(a);
      });
      return vec4(col, alpha);
    })() as ReturnType<typeof vec4>;

    const geo = new THREE.BoxGeometry(ext[0], ext[1], ext[2]);
    const mesh = new THREE.Mesh(geo, mat);
    meshMatrix.decompose(mesh.position, mesh.quaternion, mesh.scale);
    return mesh;
  }

  dispose(): void {
    for (const r of this.resident.values()) {
      r.mesh.geometry.dispose();
      (r.mesh.material as THREE.Material).dispose();
      r.tex.dispose();
    }
    this.resident.clear();
  }
}

/** Wrap a device-resident brick so three can sample it, with the two adaptations the spike found
 *  necessary (docs/gpu-resident-loader.md §8a; the spike page itself was removed once this path exercised both on real data):
 *
 *  - `isData3DTexture` is REQUIRED. three picks the bind-group view dimension from that flag alone,
 *    so without it a 3-D source binds as `texture_2d`, the generated WGSL fails to compile, and the
 *    brick renders BLACK with nothing thrown — indistinguishable from a chunk that failed to load.
 *  - the wrap modes must be set explicitly, or three warns `Unsupported texture wrap type
 *    "undefined"` once per texture.
 *
 *  (`texture.image` is deliberately NOT set: the spike showed three never reads it for an external
 *  texture, and a fake one would only invite someone to trust it.) */
function externalTexture3D(res: ResidentTexture): THREE.ExternalTexture {
  const t = new THREE.ExternalTexture(res.texture);
  (t as unknown as { isData3DTexture: boolean }).isData3DTexture = true;
  t.colorSpace = THREE.NoColorSpace;
  t.minFilter = THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  (t as unknown as { wrapR: THREE.Wrapping }).wrapR = THREE.ClampToEdgeWrapping;
  return t;
}

function makeR8Texture(data: Uint8Array, w: number, h: number, d: number): THREE.Data3DTexture {
  const t = new THREE.Data3DTexture(data, w, h, d);
  t.format = THREE.RedFormat;
  t.type = THREE.UnsignedByteType;
  t.minFilter = THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.wrapR = t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  t.needsUpdate = true;
  return t;
}
