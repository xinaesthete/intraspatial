// Volume raymarch (ADR-0008 slice 2c): assemble the in-view volume at the level the
// select camera demands into one 3D texture, and raymarch it in a world-space box with
// a TSL shader (max-intensity projection — order-independent, so no sorting). The
// assembled level is capped so it stays interactive; it re-assembles only when the
// demanded level changes, so orbiting is cheap. Fetches go through the Loader like
// everything else.
import * as THREE from "three";
import { cameraPosition, float, Fn, Loop, max, min, mix, normalize, positionWorld, texture3D, uniform, vec3, vec4 } from "three/tsl";
import { MeshBasicNodeMaterial } from "three/webgpu";
import { chunkCounts, levelVoxelDims, worldAabbOfArrayBox, type Loader, type Multiscale, type Selection, type Vec3 } from "../../../src/datasource";

const DIM_CAP = 128; // largest assembled edge; keeps the 3D texture interactive
const STEPS = 96;

interface Volume {
  tex: THREE.Data3DTexture;
  boxMin: Vec3;
  boxSize: Vec3;
}

async function assembleVolume(ms: Multiscale, loader: Loader, level: number): Promise<Volume> {
  const [W, H, D] = levelVoxelDims(ms, level);
  const counts = chunkCounts(ms, level);
  const data = new Uint8Array(W * H * D);
  const jobs: Promise<void>[] = [];
  for (let cz = 0; cz < counts[2]; cz++) {
    for (let cy = 0; cy < counts[1]; cy++) {
      for (let cx = 0; cx < counts[0]; cx++) {
        jobs.push(
          loader.getChunk({ level, x: cx, y: cy, z: cz }).then((tile) => {
            const [ex, ey, ez] = tile.dims;
            const bx = cx * ms.chunkShape[0], by = cy * ms.chunkShape[1], bz = cz * ms.chunkShape[2];
            let o = 0;
            for (let k = 0; k < ez; k++) {
              for (let j = 0; j < ey; j++) {
                const row = ((bz + k) * H + (by + j)) * W + bx;
                for (let i = 0; i < ex; i++) data[row + i] = Math.round((tile.data[o++] ?? 0) * 255);
              }
            }
          }),
        );
      }
    }
  }
  await Promise.all(jobs);

  const tex = new THREE.Data3DTexture(data, W, H, D);
  tex.format = THREE.RedFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapR = tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;

  const b = worldAabbOfArrayBox(ms.worldFromArray, [0, 0, 0], [ms.voxelDims0[0], ms.voxelDims0[1], ms.voxelDims0[2]]);
  return { tex, boxMin: b.min, boxSize: [b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]] };
}

function volumeMaterial(vol: Volume): THREE.Material {
  const bMin = uniform(new THREE.Vector3(vol.boxMin[0], vol.boxMin[1], vol.boxMin[2]));
  const bSize = uniform(new THREE.Vector3(vol.boxSize[0], vol.boxSize[1], vol.boxSize[2]));
  const mat = new MeshBasicNodeMaterial();
  mat.transparent = true;
  mat.depthWrite = false;
  mat.side = THREE.BackSide;
  mat.colorNode = Fn(() => {
    const camW = cameraPosition;
    const dir = normalize(positionWorld.sub(camW));
    const invDir = vec3(1.0).div(dir);
    const t0 = bMin.sub(camW).mul(invDir);
    const t1 = bMin.add(bSize).sub(camW).mul(invDir);
    const tmin = min(t0, t1);
    const tmax = max(t0, t1);
    const tEnter = max(max(tmin.x, tmin.y), tmin.z).max(0.0);
    const tExit = min(min(tmax.x, tmax.y), tmax.z);
    const stepLen = tExit.sub(tEnter).div(float(STEPS));
    const acc = float(0).toVar();
    Loop(STEPS, ({ i }) => {
      const t = tEnter.add(stepLen.mul(float(i).add(0.5)));
      const p = camW.add(dir.mul(t)).sub(bMin).div(bSize); // → [0,1]
      acc.assign(max(acc, texture3D(vol.tex, p).r));
    });
    const col = mix(vec3(0.06, 0.02, 0.12), vec3(1.0, 0.85, 0.55), acc);
    return vec4(col, acc);
  })();
  return mat;
}

export class VolumeRenderer {
  readonly group = new THREE.Group();
  private floorLevel: number;
  private currentLevel = -1;
  private gen = 0;
  private mesh: THREE.Mesh | null = null;

  constructor(private ms: Multiscale, private loader: Loader) {
    let L = 0;
    while (L < ms.levelCount - 1 && Math.max(...levelVoxelDims(ms, L)) > DIM_CAP) L++;
    this.floorLevel = L;
  }

  /** The level to assemble: the finest the Selection asked for, capped so it stays interactive. */
  private chooseLevel(sel: Selection): number {
    let m = this.ms.levelCount - 1;
    for (const c of sel.chunks) m = Math.min(m, c.id.level);
    return Math.max(this.floorLevel, m);
  }

  update(sel: Selection): void {
    const level = this.chooseLevel(sel);
    if (level === this.currentLevel) return;
    this.currentLevel = level;
    const gen = ++this.gen;
    void assembleVolume(this.ms, this.loader, level).then((vol) => {
      if (gen !== this.gen) { vol.tex.dispose(); return; }
      this.setMesh(vol);
    });
  }

  private setMesh(vol: Volume): void {
    this.clearMesh();
    const geo = new THREE.BoxGeometry(vol.boxSize[0], vol.boxSize[1], vol.boxSize[2]);
    const mesh = new THREE.Mesh(geo, volumeMaterial(vol));
    mesh.position.set(vol.boxMin[0] + vol.boxSize[0] / 2, vol.boxMin[1] + vol.boxSize[1] / 2, vol.boxMin[2] + vol.boxSize[2] / 2);
    mesh.userData.tex = vol.tex;
    this.mesh = mesh;
    this.group.add(mesh);
  }

  private clearMesh(): void {
    if (!this.mesh) return;
    this.group.remove(this.mesh);
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
    (this.mesh.userData.tex as THREE.Data3DTexture | undefined)?.dispose();
    this.mesh = null;
  }

  dispose(): void {
    this.clearMesh();
  }
}
