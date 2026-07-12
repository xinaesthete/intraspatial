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
  type Loader,
  type MemoryReporting,
  type Multiscale,
  type Selection,
  type Tile,
} from "../../../src/datasource";

const STEPS = 48; // per chunk — each box is small, so fewer steps than the whole-volume march

interface ChunkMesh {
  mesh: THREE.Mesh;
  tex: THREE.Data3DTexture;
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
    const { axes, origin } = ms.worldFromArray;
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
    for (const r of this.resident.values()) n += (r.tex.image.data as ArrayBufferView | undefined)?.byteLength ?? 0;
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
      this.resident.delete(k);
    }
  }

  private async load(k: string, id: Tile["id"]): Promise<void> {
    try {
      const tile = await this.loader.getChunk(id);
      const [ex, ey, ez] = tile.dims;
      const data = new Uint8Array(ex * ey * ez);
      for (let i = 0; i < data.length; i++) data[i] = Math.round((tile.data[i] ?? 0) * 255);
      const tex = makeR8Texture(data, ex, ey, ez);
      const mesh = this.makeChunkMesh(id, tex);
      this.group.add(mesh);
      this.resident.set(k, { mesh, tex, use: ++this.useClock });
    } finally {
      this.loading.delete(k);
    }
  }

  /** One box for one chunk, at its own array box under the base affine, with a per-chunk
   *  raymarch that samples that chunk's texture in [0,1] chunk space. */
  private makeChunkMesh(id: Tile["id"], tex: THREE.Data3DTexture): THREE.Mesh {
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
