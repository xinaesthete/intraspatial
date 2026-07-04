// Volume raymarch with PER-CHUNK LOD (ADR-0008), via a brick atlas + page table —
// the standard single-pass GPU volume-LOD technique:
//   • resident chunk bricks are packed into one 3D texture ATLAS (slots of B³);
//   • a low-res PAGE TABLE (one texel per finest-level chunk cell) maps a volume
//     position → the atlas slot + level of the brick covering it;
//   • one raymarch over the whole volume box: each step does a page-table lookup,
//     then samples that brick in the atlas. Mixed levels compose in a single pass
//     (cheap, no overdraw), and depth-write (later) is trivial (one fragment/pixel).
// Bricks are uploaded into atlas slots with renderer.copyTextureToTexture (partial
// 3D upload); the page table is tiny and re-uploaded whole per selection change.
import * as THREE from "three";
import { Break, cameraFar, cameraNear, cameraPosition, cameraViewMatrix, clamp, Discard, exp2, float, Fn, If, Loop, max, min, mix, normalize, oneMinus, perspectiveDepthToViewZ, positionWorld, step, struct, texture3D, uniform, vec3, vec4, viewportDepthTexture, viewZToPerspectiveDepth } from "three/tsl";
import { MeshBasicNodeMaterial } from "three/webgpu";
import { chunkArrayBox, chunkCounts, chunkKey, levelVoxelDims, worldAabbOfArrayBox, type Loader, type Multiscale, type Selection, type Tile, type Vec3 } from "../../../src/datasource";

const STEPS = 128;
// One raymarch, two outputs: the compositated colour AND the solid-surface clip depth.
// colorNode + depthNode build into the same fragment function, so sharing one struct-
// returning node emits the loop ONCE (vs. a full march per output).
const RayResult = struct({ color: "vec4", depth: "float" });

interface Resident { slot: [number, number, number]; level: number; brick: THREE.Data3DTexture; use: number; }

export class VolumeRenderer {
  readonly group = new THREE.Group();
  private renderer: WebGPURendererLike;
  private ms: Multiscale;
  private loader: Loader;
  private B: number; // brick edge (chunk voxels)
  private slotsPerAxis: number;
  private pageDims: [number, number, number];
  private atlas: THREE.Data3DTexture;
  private pageTable: THREE.Data3DTexture;
  private pageData: Uint8Array;
  private free: number[] = [];
  private resident = new Map<string, Resident>();
  private loading = new Set<string>();
  private useClock = 0;
  private desired = new Set<string>();
  private mesh: THREE.Mesh;
  private uCmin = uniform(0.15);
  private uCmax = uniform(1.0);
  private uGamma = uniform(1.0);
  private uSolid = uniform(0.55);
  private uDepthAware = uniform(1); // 1 = respect scene depth (cull), 0 = draw as a flat overlay
  private mat: MeshBasicNodeMaterial | null = null;

  constructor(ms: Multiscale, loader: Loader, renderer: WebGPURendererLike) {
    this.ms = ms;
    this.loader = loader;
    this.renderer = renderer;
    this.B = ms.chunkShape[0];
    this.pageDims = [
      Math.ceil(ms.voxelDims0[0] / ms.chunkShape[0]),
      Math.ceil(ms.voxelDims0[1] / ms.chunkShape[1]),
      Math.ceil(ms.voxelDims0[2] / ms.chunkShape[2]),
    ];
    // Atlas big enough to hold a generous working set of bricks.
    this.slotsPerAxis = 8; // 8³ = 512 slots
    const atlasEdge = this.slotsPerAxis * this.B;
    this.atlas = makeR8Texture(new Uint8Array(atlasEdge * atlasEdge * atlasEdge), atlasEdge, atlasEdge, atlasEdge, THREE.LinearFilter);
    for (let i = this.slotsPerAxis ** 3 - 1; i >= 0; i--) this.free.push(i);

    const [pw, ph, pd] = this.pageDims;
    this.pageData = new Uint8Array(pw * ph * pd * 4);
    this.pageTable = makeRGBA8Texture(this.pageData, pw, ph, pd);

    this.mesh = this.makeMesh();
    this.group.add(this.mesh);
  }

  setTransfer(cmin: number, cmax: number, gamma: number): void {
    this.uCmin.value = cmin;
    this.uCmax.value = Math.max(cmin + 1e-3, cmax);
    this.uGamma.value = gamma;
  }

  /** Reads scene depth: the ray stops at opaque geometry, so the volume is *occluded by*
   *  the image plane. Off ⇒ the volume ignores scene depth (flat overlay). */
  setDepthRead(on: boolean): void {
    this.uDepthAware.value = on ? 1 : 0;
  }

  /** Writes the solid-surface depth to the z-buffer, so the volume *occludes* opaque
   *  geometry drawn against it. */
  setDepthWrite(on: boolean): void {
    if (this.mat) this.mat.depthWrite = on;
  }

  setSolid(threshold: number): void {
    this.uSolid.value = threshold;
  }

  update(sel: Selection): void {
    this.desired = new Set(sel.chunks.map((c) => chunkKey(c.id)));
    for (const c of sel.chunks) {
      const k = chunkKey(c.id);
      const r = this.resident.get(k);
      if (r) { r.use = ++this.useClock; continue; }
      if (!this.loading.has(k)) { this.loading.add(k); void this.load(k, c.id); }
    }
    this.rebuildPageTable();
  }

  private async load(k: string, id: Tile["id"]): Promise<void> {
    try {
      const tile = await this.loader.getChunk(id);
      const slot = this.allocSlot();
      if (slot === null) return; // atlas full (shouldn't happen at demo scale)
      const brick = makeR8Texture(this.brickData(tile), this.B, this.B, this.B, THREE.LinearFilter);
      brick.needsUpdate = true;
      this.renderer.copyTextureToTexture(
        brick, this.atlas,
        new THREE.Box3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(this.B, this.B, this.B)),
        new THREE.Vector3(slot[0] * this.B, slot[1] * this.B, slot[2] * this.B),
      );
      this.resident.set(k, { slot, level: id.level, brick, use: ++this.useClock });
      if (this.desired.has(k)) this.rebuildPageTable();
    } finally {
      this.loading.delete(k);
    }
  }

  /** Nearest-upsample a tile (dims ≤ B) into a full B³ brick so the page-table UV
   *  mapping is uniform regardless of the chunk's own resolution. */
  private brickData(tile: Tile): Uint8Array {
    const B = this.B;
    const [ex, ey, ez] = tile.dims;
    const out = new Uint8Array(B * B * B);
    for (let z = 0; z < B; z++) {
      const sz = Math.min(ez - 1, (z * ez / B) | 0);
      for (let y = 0; y < B; y++) {
        const sy = Math.min(ey - 1, (y * ey / B) | 0);
        for (let x = 0; x < B; x++) {
          const sx = Math.min(ex - 1, (x * ex / B) | 0);
          out[(z * B + y) * B + x] = Math.round((tile.data[(sz * ey + sy) * ex + sx] ?? 0) * 255);
        }
      }
    }
    return out;
  }

  private allocSlot(): [number, number, number] | null {
    let idx = this.free.pop();
    if (idx === undefined) {
      // Evict LRU resident not in the desired set.
      let lruKey: string | null = null, lruUse = Infinity;
      for (const [k, r] of this.resident) {
        if (this.desired.has(k)) continue;
        if (r.use < lruUse) { lruUse = r.use; lruKey = k; }
      }
      if (lruKey === null) return null;
      const r = this.resident.get(lruKey);
      if (!r) return null;
      r.brick.dispose();
      this.resident.delete(lruKey);
      idx = (r.slot[2] * this.slotsPerAxis + r.slot[1]) * this.slotsPerAxis + r.slot[0];
    }
    const s = this.slotsPerAxis;
    return [idx % s, ((idx / s) | 0) % s, (idx / (s * s)) | 0];
  }

  private rebuildPageTable(): void {
    this.pageData.fill(0);
    const [pw, ph] = this.pageDims;
    for (const k of this.desired) {
      const r = this.resident.get(k);
      if (!r) continue;
      const span = 2 ** r.level; // finest cells this brick covers per axis
      // Parse the chunk coords back from the key: "level:x:y:z".
      const [, x, y, z] = k.split(":").map(Number);
      const fx = (x ?? 0) * span, fy = (y ?? 0) * span, fz = (z ?? 0) * span;
      for (let dz = 0; dz < span; dz++) {
        for (let dy = 0; dy < span; dy++) {
          for (let dx = 0; dx < span; dx++) {
            const cx = fx + dx, cy = fy + dy, cz = fz + dz;
            if (cx >= pw || cy >= ph || cz >= this.pageDims[2]) continue;
            const o = ((cz * ph + cy) * pw + cx) * 4;
            this.pageData[o] = r.slot[0];
            this.pageData[o + 1] = r.slot[1];
            this.pageData[o + 2] = r.slot[2];
            this.pageData[o + 3] = r.level + 1; // 0 = empty
          }
        }
      }
    }
    this.pageTable.needsUpdate = true;
  }

  private makeMesh(): THREE.Mesh {
    const b = worldAabbOfArrayBox(this.ms.worldFromArray, [0, 0, 0], [this.ms.voxelDims0[0], this.ms.voxelDims0[1], this.ms.voxelDims0[2]]);
    const size: Vec3 = [b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]];
    const bMin = uniform(new THREE.Vector3(b.min[0], b.min[1], b.min[2]));
    const bSize = uniform(new THREE.Vector3(size[0], size[1], size[2]));
    const pageDimsU = uniform(new THREE.Vector3(this.pageDims[0], this.pageDims[1], this.pageDims[2]));
    const slots = uniform(this.slotsPerAxis);
    const invB = uniform(0.5 / this.B);
    const atlasTex = this.atlas, pageTex = this.pageTable;

    const mat = new MeshBasicNodeMaterial();
    mat.transparent = true;
    mat.depthWrite = false; // toggled by setWriteDepth
    mat.depthTest = false; // occlusion is handled inside the ray (depth-aware), not by z-test
    mat.side = THREE.BackSide;

    // Sample density (0 where empty) at volume-space point p, via page table → atlas.
    const sampleAt = Fn(([p]) => {
      const pt = texture3D(pageTex, p);
      const mask = pt.a.mul(255.0).min(1.0); // 1 if occupied, 0 if empty
      const level = pt.a.mul(255.0).sub(1.0).max(0.0);
      const cells = exp2(level);
      const fo = p.mul(pageDimsU).floor().div(cells).floor().mul(cells);
      const local = clamp(p.mul(pageDimsU).sub(fo).div(cells), invB, float(1.0).sub(invB));
      const slot = pt.rgb.mul(255.0).add(0.5).floor();
      return texture3D(atlasTex, slot.add(local).div(slots)).r.mul(mask);
    });
    // Transfer function: contrast window + gamma → normalised intensity.
    const tf = (d: ReturnType<typeof float>) => clamp(d.sub(this.uCmin).div(this.uCmax.sub(this.uCmin)), 0.0, 1.0).pow(this.uGamma);
    const enter = (camW: ReturnType<typeof cameraPosition.sub>, dir: ReturnType<typeof normalize>) => {
      const inv = vec3(1.0).div(dir);
      const t0 = bMin.sub(camW).mul(inv), t1 = bMin.add(bSize).sub(camW).mul(inv);
      const tmin = min(t0, t1), tmax = max(t0, t1);
      const tEnter = max(max(tmin.x, tmin.y), tmin.z).max(0.0);
      const tExit = min(min(tmax.x, tmax.y), tmax.z);
      return { tEnter, tExit };
    };

    // ONE march feeding both outputs. Emission-absorption compositing accumulates
    // colour/alpha; the first sample crossing uSolid records the clip depth for z-write.
    const march = Fn(() => {
      const camW = cameraPosition;
      const dir = normalize(positionWorld.sub(camW));
      const { tEnter, tExit } = enter(camW, dir);
      // Occlusion bail (gated by uDepthAware): clamp the exit to the opaque scene surface
      // along this ray. Solving viewZ(t) = sceneViewZ for t — view-z is affine in t — both
      // stops the ray at the image plane AND packs all STEPS into the *visible* span,
      // instead of masking occluded samples one-by-one while still marching them.
      const sceneViewZ = perspectiveDepthToViewZ(viewportDepthTexture(), cameraNear, cameraFar);
      const a0 = cameraViewMatrix.mul(vec4(camW, 1.0)).z;
      const ad = cameraViewMatrix.mul(vec4(dir, 0.0)).z;
      const tSurface = sceneViewZ.sub(a0).div(ad);
      const tExitEff = mix(tExit, min(tExit, max(tEnter, tSurface)), this.uDepthAware);
      const stepLen = tExitEff.sub(tEnter).div(float(STEPS));

      const col = vec3(0.0).toVar();
      const alpha = float(0.0).toVar();
      const found = float(0.0).toVar();
      const outDepth = float(1.0).toVar();
      const dmPrev = float(0.0).toVar(); // density at the previous step, for iso-interpolation
      const tPrev = tEnter.toVar();
      Loop(STEPS, ({ i }) => {
        const t = tEnter.add(stepLen.mul(float(i).add(0.5)));
        const pWorld = camW.add(dir.mul(t));
        const dm = tf(sampleAt(pWorld.sub(bMin).div(bSize)));
        const a = dm.mul(0.1).mul(oneMinus(alpha));
        const sc = mix(vec3(0.10, 0.03, 0.18), vec3(1.0, 0.86, 0.55), dm);
        col.addAssign(sc.mul(a));
        alpha.addAssign(a);
        // First crossing of uSolid → the surface depth. Interpolate the exact iso-position
        // between the last two samples instead of snapping to the step centre, so the
        // written depth is smooth rather than banded to stepLen (which makes anything
        // depth-tested against it — the probe rod — stair-step at the occlusion edge).
        const crossing = step(this.uSolid, dm).mul(oneMinus(found));
        const frac = clamp(this.uSolid.sub(dmPrev).div(max(dm.sub(dmPrev), float(1e-4))), 0.0, 1.0);
        const pHit = camW.add(dir.mul(mix(tPrev, t, frac)));
        const clipD = viewZToPerspectiveDepth(cameraViewMatrix.mul(vec4(pHit, 1.0)).z, cameraNear, cameraFar);
        outDepth.assign(mix(outDepth, clipD, crossing));
        found.assign(max(found, crossing));
        dmPrev.assign(dm);
        tPrev.assign(t);
        // Early-out once the ray is effectively opaque (the solid depth is already fixed).
        If(alpha.greaterThan(0.995), () => { Break(); });
      });
      // Empty ray (no visible density AND no solid): discard so we neither tint the pixel
      // nor stamp far-depth over whatever opaque geometry sits behind the volume box —
      // depthTest is off, so an unconditional write would clobber the image plane's depth.
      Discard(alpha.lessThan(0.003).and(found.equal(0.0)));
      return RayResult(vec4(col, alpha), outDepth);
    })();

    mat.colorNode = march.get("color");
    mat.depthNode = march.get("depth");

    this.mat = mat;

    const geo = new THREE.BoxGeometry(size[0], size[1], size[2]);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(b.min[0] + size[0] / 2, b.min[1] + size[1] / 2, b.min[2] + size[2] / 2);
    return mesh;
  }

  dispose(): void {
    for (const r of this.resident.values()) r.brick.dispose();
    this.resident.clear();
    this.atlas.dispose();
    this.pageTable.dispose();
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}

interface WebGPURendererLike {
  copyTextureToTexture(src: THREE.Texture, dst: THREE.Texture, srcRegion: THREE.Box3, dstPosition: THREE.Vector3): void;
}

function makeR8Texture(data: Uint8Array, w: number, h: number, d: number, filter: THREE.MagnificationTextureFilter): THREE.Data3DTexture {
  const t = new THREE.Data3DTexture(data, w, h, d);
  t.format = THREE.RedFormat;
  t.type = THREE.UnsignedByteType;
  t.minFilter = filter;
  t.magFilter = filter;
  t.wrapR = t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  t.needsUpdate = true;
  return t;
}

function makeRGBA8Texture(data: Uint8Array, w: number, h: number, d: number): THREE.Data3DTexture {
  const t = new THREE.Data3DTexture(data, w, h, d);
  t.format = THREE.RGBAFormat;
  t.type = THREE.UnsignedByteType;
  t.minFilter = THREE.NearestFilter;
  t.magFilter = THREE.NearestFilter;
  t.wrapR = t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  t.needsUpdate = true;
  return t;
}
