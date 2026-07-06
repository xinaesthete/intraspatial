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
import {
  Break,
  cameraFar,
  cameraNear,
  cameraPosition,
  cameraViewMatrix,
  clamp,
  Discard,
  exp2,
  Fn,
  float,
  If,
  Loop,
  max,
  min,
  mix,
  normalize,
  oneMinus,
  perspectiveDepthToViewZ,
  positionWorld,
  step,
  struct,
  texture3D,
  uniform,
  vec3,
  vec4,
  viewportDepthTexture,
  viewZToPerspectiveDepth,
} from "three/tsl";
import { MeshBasicNodeMaterial } from "three/webgpu";
import { type Affine3, type ChunkId, chunkCounts, chunkKey, type Multiscale, type Selection } from "../../../src/datasource";
import type { BrickSource } from "./brickSource";
import { LoadScheduler, type LoadState } from "./loadScheduler";

const STEPS = 128;
// One raymarch, two outputs: the compositated colour AND the solid-surface clip depth.
// colorNode + depthNode build into the same fragment function, so sharing one struct-
// returning node emits the loop ONCE (vs. a full march per output).
const RayResult = struct({ color: "vec4", depth: "float" });

interface Resident {
  slot: [number, number, number];
  level: number;
  brick: THREE.Data3DTexture;
  use: number;
}

export class VolumeRenderer {
  readonly group = new THREE.Group();
  private renderer: WebGPURendererLike;
  private ms: Multiscale;
  private source: BrickSource;
  private B: number; // brick edge (chunk voxels)
  private slotsPerAxis: number;
  private pageDims: [number, number, number];
  private atlas: THREE.Data3DTexture;
  private pageTable: THREE.Data3DTexture;
  private pageData: Uint8Array;
  private free: number[] = [];
  private resident = new Map<string, Resident>();
  // Bounded, nearest-first load queue (replaces fire-every-desired-immediately). Set in the
  // constructor once `resident`/`source` exist.
  private scheduler: LoadScheduler;
  private readonly MAX_CONCURRENT_LOADS = 6;
  private useClock = 0;
  // desired = the chunks whose in-flight loads we still want to commit (selected + floor).
  private desired = new Set<string>();
  // selectedKeys = just the view-selected chunks — the regions the page table actually draws
  // (each resolved to itself or a resident ancestor). The floor is NEVER drawn directly; it is
  // reached only via the walk-up, so it can't overwrite finer resident chunks.
  private selectedKeys: string[] = [];
  // Resident bricks currently referenced by the page table (exact desired OR fallback ancestor).
  // Recomputed each rebuild; eviction protects these (retain-on-refine + keep fallbacks live).
  private inUse = new Set<string>();
  // The coarsest level — proactively kept resident as a no-holes floor (walk-up always finds it).
  private floorIds: ChunkId[] = [];
  private floorKeys = new Set<string>();
  private mesh: THREE.Mesh;
  private uCmin = uniform(0.15);
  private uCmax = uniform(1.0);
  private uGamma = uniform(1.0);
  private uSolid = uniform(0.55);
  private uDepthAware = uniform(1); // 1 = respect scene depth (cull), 0 = draw as a flat overlay
  // world → normalised array coords [0,1]³. Recomputed from the mesh's matrix whenever the
  // volume is moved (gizmo), so the raymarch tracks a live transform.
  private uWorldToNorm = uniform(new THREE.Matrix4());
  private mat: MeshBasicNodeMaterial | null = null;

  constructor(ms: Multiscale, source: BrickSource, renderer: WebGPURendererLike) {
    this.ms = ms;
    this.source = source;
    this.renderer = renderer;
    this.B = ms.chunkShape[0];
    this.pageDims = [
      Math.ceil(ms.voxelDims0[0] / ms.chunkShape[0]),
      Math.ceil(ms.voxelDims0[1] / ms.chunkShape[1]),
      Math.ceil(ms.voxelDims0[2] / ms.chunkShape[2]),
    ];
    // Atlas big enough to hold a generous working set of bricks. 12³ = 1728 slots (a 384³ R8
    // texture, ~54 MB) — headroom for finer base resolutions than 256³ (which alone is 512
    // chunks) and for the LOD-fallback ancestors the page table will lean on (step 3).
    this.slotsPerAxis = 12; // 12³ = 1728 slots
    const atlasEdge = this.slotsPerAxis * this.B;
    this.atlas = makeR8Texture(new Uint8Array(atlasEdge * atlasEdge * atlasEdge), atlasEdge, atlasEdge, atlasEdge, THREE.LinearFilter);
    for (let i = this.slotsPerAxis ** 3 - 1; i >= 0; i--) this.free.push(i);

    const [pw, ph, pd] = this.pageDims;
    this.pageData = new Uint8Array(pw * ph * pd * 4);
    this.pageTable = makeRGBA8Texture(this.pageData, pw, ph, pd);

    this.mesh = this.makeMesh();
    this.group.add(this.mesh);
    this.syncTransform(); // initialise uWorldToNorm from the base placement

    this.scheduler = new LoadScheduler(
      this.MAX_CONCURRENT_LOADS,
      (id) => this.load(id),
      (key) => this.resident.has(key),
    );

    // The fallback floor: the coarsest level that is still a *full* chunk grid, always requested
    // and never evicted, so the page-table walk-up can always find SOME resident ancestor and
    // there is never a transparent hole — worst case is blur that sharpens. It must not be finer
    // OR coarser than this: coarser levels are sub-chunk partials whose finest-cell span is
    // < 2^level, which the shader's `cells = 2^level` mapping (page table → brick-local coord)
    // gets wrong. The valid coarsest level is where 2^level == pageDims: log2(pageDims).
    const maxValidLevel = Math.floor(Math.log2(Math.max(1, Math.min(...this.pageDims))));
    const coarse = Math.max(0, Math.min(ms.levelCount - 1, maxValidLevel));
    const [ncx, ncy, ncz] = chunkCounts(ms, coarse);
    for (let z = 0; z < ncz; z++)
      for (let y = 0; y < ncy; y++) for (let x = 0; x < ncx; x++) this.floorIds.push({ level: coarse, x, y, z });
    this.floorKeys = new Set(this.floorIds.map((id) => chunkKey(id)));
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
    // Desired = the selected chunks PLUS the coarse floor (so a floor brick's in-flight result
    // is committed even when select() never picks the coarsest level).
    this.selectedKeys = sel.chunks.map((c) => chunkKey(c.id));
    this.desired = new Set(this.selectedKeys);
    for (const k of this.floorKeys) this.desired.add(k);
    for (const c of sel.chunks) {
      const r = this.resident.get(chunkKey(c.id));
      if (r) r.use = ++this.useClock; // keep resident desired chunks fresh for the LRU
    }
    // Queue the not-yet-resident chunks, nearest-first; the scheduler bounds concurrency and
    // drops queued requests the camera has since left (it reads `resident` to skip the rest).
    // The floor loads first (priority below any nearestDepth), so a fallback exists immediately.
    const reqs = sel.chunks.map((c) => ({ key: chunkKey(c.id), id: c.id, priority: c.nearestDepth }));
    for (const id of this.floorIds) reqs.push({ key: chunkKey(id), id, priority: -1 });
    this.scheduler.reconcile(reqs);
    this.rebuildPageTable();
  }

  /** The chunk's state for the decision view: resident, or the scheduler's pending/loading, or
   *  missing (wanted but neither queued nor in-flight — transiently, between reconciles). */
  chunkState(key: string): "resident" | LoadState | "missing" {
    if (this.resident.has(key)) return "resident";
    return this.scheduler.state(key) ?? "missing";
  }

  /** Queued (pending) + in-flight (loading) counts, for the HUD. */
  loadCounts(): { pending: number; loading: number } {
    return { pending: this.scheduler.pendingCount, loading: this.scheduler.loadingCount };
  }

  private async load(id: ChunkId): Promise<void> {
    const k = chunkKey(id);
    const data = await this.source.brick(id);
    // Commit only if still desired: a slow in-flight load for a chunk the camera has left
    // shouldn't consume an atlas slot (the scheduler already drops the *queued* stale ones).
    if (!this.desired.has(k)) return;
    const slot = this.allocSlot();
    if (slot === null) return; // atlas full — the LOD fallback (step 3) will cover the gap
    const brick = makeR8Texture(data, this.B, this.B, this.B, THREE.LinearFilter);
    brick.needsUpdate = true;
    this.renderer.copyTextureToTexture(
      brick,
      this.atlas,
      new THREE.Box3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(this.B, this.B, this.B)),
      new THREE.Vector3(slot[0] * this.B, slot[1] * this.B, slot[2] * this.B),
    );
    this.resident.set(k, { slot, level: id.level, brick, use: ++this.useClock });
    this.rebuildPageTable();
  }

  private allocSlot(): [number, number, number] | null {
    let idx = this.free.pop();
    if (idx === undefined) {
      // Evict the LRU resident that is neither referenced by the page table (an exact desired
      // brick OR a fallback ancestor still standing in for a not-yet-resident finer chunk — this
      // is what gives retain-on-refine) nor part of the coarse floor.
      let lruKey: string | null = null,
        lruUse = Infinity;
      for (const [k, r] of this.resident) {
        if (this.inUse.has(k) || this.floorKeys.has(k)) continue;
        if (r.use < lruUse) {
          lruUse = r.use;
          lruKey = k;
        }
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

  /** The finest resident brick covering a desired chunk: the chunk itself if resident, else the
   *  coarsest resident ancestor (walking up the pyramid). `undefined` only before the floor loads. */
  private residentOrAncestor(level: number, x: number, y: number, z: number): { key: string; r: Resident } | undefined {
    let lv = level,
      cx = x,
      cy = y,
      cz = z;
    while (lv < this.ms.levelCount) {
      const key = chunkKey({ level: lv, x: cx, y: cy, z: cz });
      const r = this.resident.get(key);
      if (r) return { key, r };
      lv++;
      cx >>= 1;
      cy >>= 1;
      cz >>= 1;
    }
    return undefined;
  }

  private rebuildPageTable(): void {
    this.pageData.fill(0);
    const [pw, ph, pd] = this.pageDims;
    this.inUse.clear();
    for (const k of this.selectedKeys) {
      const [kl, kx, ky, kz] = k.split(":").map(Number);
      const kLevel = kl ?? 0;
      // LOD fallback: show the desired chunk if resident, else its coarsest resident ancestor —
      // blur that sharpens, never a hole. The shader derives the brick-local coord from the
      // stored level, so an ancestor's (slot, level) sampled through a finer cell is exact.
      const hit = this.residentOrAncestor(kLevel, kx ?? 0, ky ?? 0, kz ?? 0);
      if (!hit) continue;
      this.inUse.add(hit.key); // referenced ⇒ protected from eviction
      const r = hit.r;
      // Fill the DESIRED chunk's finest-cell region (span from its level), with the RESOLVED
      // brick's (slot, level). Loop bounds clamped to the page so coarse spans don't over-iterate.
      const span = 2 ** kLevel;
      const fx = (kx ?? 0) * span,
        fy = (ky ?? 0) * span,
        fz = (kz ?? 0) * span;
      const ez = Math.min(span, pd - fz),
        ey = Math.min(span, ph - fy),
        ex = Math.min(span, pw - fx);
      for (let dz = 0; dz < ez; dz++) {
        for (let dy = 0; dy < ey; dy++) {
          for (let dx = 0; dx < ex; dx++) {
            const o = (((fz + dz) * ph + (fy + dy)) * pw + (fx + dx)) * 4;
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
    const dims = this.ms.voxelDims0;
    const pageDimsU = uniform(new THREE.Vector3(this.pageDims[0], this.pageDims[1], this.pageDims[2]));
    const slots = uniform(this.slotsPerAxis);
    const invB = uniform(0.5 / this.B);
    const atlasTex = this.atlas,
      pageTex = this.pageTable;

    const mat = new MeshBasicNodeMaterial();
    mat.transparent = true;
    mat.depthWrite = false; // toggled by setDepthWrite
    mat.depthTest = false; // occlusion is handled inside the ray (depth-aware), not by z-test
    mat.side = THREE.BackSide;

    // ONE march feeding both outputs (colour + solid-surface clip depth). The ray is set up
    // in world space (so t is world distance and the depth math stays simple), but intersected
    // and sampled in the volume's normalised array space, where the box is axis-aligned even
    // when M rotates/shears it in world.
    const march = Fn(() => {
      const camW = cameraPosition;
      const dirW = normalize(positionWorld.sub(camW)); // unit world ray ⇒ t is world distance
      // Camera + ray direction in normalised array space (direction ⇒ w = 0, no translation).
      const camN = this.uWorldToNorm.mul(vec4(camW, 1.0)).xyz;
      const dirN = this.uWorldToNorm.mul(vec4(dirW, 0.0)).xyz;
      // Ray ∩ the unit box [0,1]³ (slab test). worldToNorm is affine, so this shares t with
      // the world ray — tEnter/tExit come out as world distances, matching the depth math.
      const invD = vec3(1.0).div(dirN);
      const tA = vec3(0.0).sub(camN).mul(invD);
      const tB = vec3(1.0).sub(camN).mul(invD);
      const tmin = min(tA, tB);
      const tmax = max(tA, tB);
      const tEnter = max(max(tmin.x, tmin.y), tmin.z).max(0.0);
      const tExit = min(min(tmax.x, tmax.y), tmax.z);
      // Occlusion bail (gated by uDepthAware): clamp the exit to the opaque scene surface along
      // this ray. view-z is affine in t (rigid view matrix), so solve viewZ(t) = sceneViewZ —
      // this stops the ray at the image plane AND packs all STEPS into the *visible* span.
      const existingDepth = viewportDepthTexture(); // perspective depth [0,1] of prior opaque geometry
      const sceneViewZ = perspectiveDepthToViewZ(existingDepth, cameraNear, cameraFar);
      const a0 = cameraViewMatrix.mul(vec4(camW, 1.0)).z;
      const ad = cameraViewMatrix.mul(vec4(dirW, 0.0)).z;
      const tSurface = sceneViewZ.sub(a0).div(ad);
      const tExitEff = mix(tExit, min(tExit, max(tEnter, tSurface)), this.uDepthAware);
      const stepLen = tExitEff.sub(tEnter).div(float(STEPS));

      const col = vec3(0.0).toVar();
      const alpha = float(0.0).toVar();
      const found = float(0.0).toVar();
      const outDepth = float(1.0).toVar();
      const dmPrev = float(0.0).toVar(); // density at the previous step, for iso-interpolation
      const tPrev = tEnter.toVar();
      // Skip the whole march when there's no visible span: the ray misses the oriented box, or
      // (depth-aware) an opaque surface sits in front of where it would enter — fully occluded.
      // Those fragments fall through to the Discard below untouched, so we don't burn STEPS
      // samples on hidden rays. `discard` alone wouldn't help — in WGSL it doesn't halt the
      // shader — but an If around the loop genuinely branches past it.
      If(tExitEff.greaterThan(tEnter), () => {
        Loop(STEPS, ({ i }) => {
          const t = tEnter.add(stepLen.mul(float(i).add(0.5)));
          const p = camN.add(dirN.mul(t)); // normalised array coords [0,1]³
          // Sample density at p via page table → atlas brick (0 where empty).
          const pt = texture3D(pageTex, p);
          const mask = pt.a.mul(255.0).min(1.0); // 1 if occupied, 0 if empty
          const level = pt.a.mul(255.0).sub(1.0).max(0.0);
          const cells = exp2(level);
          const fo = p.mul(pageDimsU).floor().div(cells).floor().mul(cells);
          const localP = clamp(p.mul(pageDimsU).sub(fo).div(cells), invB, float(1.0).sub(invB));
          const slot = pt.rgb.mul(255.0).add(0.5).floor();
          const density = texture3D(atlasTex, slot.add(localP).div(slots)).r.mul(mask);
          // Transfer function: contrast window + gamma → normalised intensity.
          const dm = clamp(density.sub(this.uCmin).div(this.uCmax.sub(this.uCmin)), 0.0, 1.0).pow(this.uGamma);
          // Emission-absorption compositing.
          const a = dm.mul(0.1).mul(oneMinus(alpha));
          const sc = mix(vec3(0.1, 0.03, 0.18), vec3(1.0, 0.86, 0.55), dm);
          col.addAssign(sc.mul(a));
          alpha.addAssign(a);
          // First crossing of uSolid → the surface depth. Interpolate the exact iso-position
          // between the last two samples instead of snapping to the step centre, so the written
          // depth is smooth rather than banded to stepLen (which makes anything depth-tested
          // against it — the probe rod — stair-step at the occlusion edge). pHit is world-space.
          const crossing = step(this.uSolid, dm).mul(oneMinus(found));
          const frac = clamp(this.uSolid.sub(dmPrev).div(max(dm.sub(dmPrev), float(1e-4))), 0.0, 1.0);
          const pHit = camW.add(dirW.mul(mix(tPrev, t, frac)));
          const clipD = viewZToPerspectiveDepth(cameraViewMatrix.mul(vec4(pHit, 1.0)).z, cameraNear, cameraFar);
          outDepth.assign(mix(outDepth, clipD, crossing));
          found.assign(max(found, crossing));
          dmPrev.assign(dm);
          tPrev.assign(t);
          // Early-out once the ray is effectively opaque (the solid depth is already fixed).
          If(alpha.greaterThan(0.995), () => {
            Break();
          });
        });
      });
      // Empty ray (no visible density AND no solid): discard so we don't tint the pixel.
      // (An optimisation — the depth min below is what actually prevents a far-depth clobber.)
      Discard(alpha.lessThan(0.003).and(found.equal(0.0)));
      // Never write a depth FARTHER than the buffer already holds: emulate a LESS depth test.
      // We can't switch the hardware one on — depthTest is off because this is a BackSide box
      // whose fragment depth is the far wall, not the volume content. So the volume only wins
      // the z-buffer where its solid surface is genuinely nearer than the opaque geometry there.
      return RayResult(vec4(col, alpha), min(outDepth, existingDepth));
    })();

    mat.colorNode = march.get("color") as ReturnType<typeof vec4>;
    mat.depthNode = march.get("depth") as ReturnType<typeof float>;

    this.mat = mat;

    // Proxy geometry: the array box as a CENTRED box in local space (spans ±dims/2). The mesh's
    // matrix maps it to world with the mesh origin at the volume CENTRE, so a gizmo rotates about
    // the centre. The local→array half-voxel shift is baked into worldToNorm by syncTransform().
    const geo = new THREE.BoxGeometry(dims[0], dims[1], dims[2]);
    const mesh = new THREE.Mesh(geo, mat);
    const half = new THREE.Matrix4().makeTranslation(dims[0] / 2, dims[1] / 2, dims[2] / 2);
    this.baseMatrix().multiply(half).decompose(mesh.position, mesh.quaternion, mesh.scale);
    return mesh;
  }

  /** The base array→world affine as a Matrix4 (columns = array axes, translation = origin). */
  private baseMatrix(): THREE.Matrix4 {
    const { axes, origin } = this.ms.worldFromArray;
    const [ax0, ax1, ax2] = axes;
    return new THREE.Matrix4()
      .makeBasis(
        new THREE.Vector3(ax0[0], ax0[1], ax0[2]),
        new THREE.Vector3(ax1[0], ax1[1], ax1[2]),
        new THREE.Vector3(ax2[0], ax2[1], ax2[2]),
      )
      .setPosition(origin[0], origin[1], origin[2]);
  }

  /** The gizmo target — attach TransformControls here to move/rotate the volume. */
  get transformTarget(): THREE.Object3D {
    return this.mesh;
  }

  /** Recompute world→normalised-array from the mesh's current transform. Call after the gizmo
   *  moves the mesh so the raymarch tracks it. */
  syncTransform(): void {
    this.mesh.updateMatrix();
    const d = this.ms.voxelDims0;
    // world → normalised array coords = T(0.5) · diag(1/dims) · meshMatrix⁻¹ (local is centred).
    this.uWorldToNorm.value = new THREE.Matrix4()
      .makeTranslation(0.5, 0.5, 0.5)
      .multiply(new THREE.Matrix4().makeScale(1 / d[0], 1 / d[1], 1 / d[2]))
      .multiply(this.mesh.matrix.clone().invert());
  }

  /** The volume's live array→world placement (base ∘ gizmo) — for select()/overlays. */
  effectiveWorldFromArray(): Affine3 {
    this.mesh.updateMatrix();
    const d = this.ms.voxelDims0;
    // M_eff (array→world) = meshMatrix · T(−dims/2), undoing the centring shift.
    const m = this.mesh.matrix.clone().multiply(new THREE.Matrix4().makeTranslation(-d[0] / 2, -d[1] / 2, -d[2] / 2));
    const e = m.elements; // column-major
    return {
      origin: [e[12] ?? 0, e[13] ?? 0, e[14] ?? 0],
      axes: [
        [e[0] ?? 0, e[1] ?? 0, e[2] ?? 0],
        [e[4] ?? 0, e[5] ?? 0, e[6] ?? 0],
        [e[8] ?? 0, e[9] ?? 0, e[10] ?? 0],
      ],
    };
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
