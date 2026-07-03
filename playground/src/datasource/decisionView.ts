// The Decision view (ADR-0008 slice 1b): the true linked pair, in one third-person
// scene. A *view* camera (orbit) is what you look through; a separate *select* camera
// (first-person) is what drives the pure `select()`. We draw the select camera's
// frustum slicing through the data, its enclosed chunks tinted by chosen level — so
// the receding-resolution gradient (near = fine/warm, far = coarse/cool) is legible
// at a glance, without hunting for a camera angle. No tile textures yet (slice 2).
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { WebGPURenderer } from "three/webgpu";
import { add, cross, dot, frustumCorners, normalize, scale, select, sub, worldAabbOfArrayBox, type Aabb, type Camera, type Multiscale, type Selection, type Vec3 } from "../../../src/datasource";
import { boundsOverlay, chunkOverlays, disposeGroup, frustumOverlay } from "./overlays";

export interface DecisionStats {
  q: number;
  chunkCount: number;
  totalBytes: number;
  countByLevel: readonly number[];
}

export class DecisionView {
  private renderer: WebGPURenderer;
  private scene = new THREE.Scene();
  private viewCamera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private overlays = new THREE.Group();
  private ms: Multiscale;
  private center: THREE.Vector3;
  private size: number;
  private bounds: Aabb;
  private isPlane: boolean;

  private q = 1;
  private selDistMult = 0.6;
  private selPitchDeg = 14;
  private selYawDeg = 32;
  private animate = false;

  private dirty = true;
  private disposed = false;
  private onStats?: (s: DecisionStats) => void;

  constructor(canvas: HTMLCanvasElement, ms: Multiscale, onStats?: (s: DecisionStats) => void) {
    this.ms = ms;
    this.onStats = onStats;
    this.renderer = new WebGPURenderer({ canvas, antialias: true, alpha: true });

    const b = worldAabbOfArrayBox(ms.worldFromArray, [0, 0, 0], [ms.voxelDims0[0], ms.voxelDims0[1], ms.voxelDims0[2]]);
    this.bounds = b;
    this.isPlane = ms.voxelDims0[2] === 1;
    this.center = new THREE.Vector3((b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2);
    this.size = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]) || 1;

    // Third-person view camera, pulled back to see the data AND the select frustum.
    this.viewCamera = new THREE.PerspectiveCamera(50, 1, this.size * 0.004, this.size * 40);
    this.viewCamera.position.set(this.center.x + this.size * 1.4, this.center.y + this.size * 1.1, this.center.z - this.size * 2.2);
    this.controls = new OrbitControls(this.viewCamera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.copy(this.center);
    this.controls.addEventListener("change", () => { this.dirty = true; });

    this.scene.add(boundsOverlay(ms));
    this.scene.add(this.overlays);
    this.resize(canvas.clientWidth, canvas.clientHeight);
  }

  setQ(q: number): void { this.q = q; this.dirty = true; }
  setSelectDistance(mult: number): void { this.selDistMult = mult; this.dirty = true; }
  setSelectPitch(deg: number): void { this.selPitchDeg = deg; this.dirty = true; }
  setSelectYaw(deg: number): void { this.selYawDeg = deg; this.dirty = true; }
  setAnimate(on: boolean): void { this.animate = on; this.dirty = true; }

  resize(width: number, height: number): void {
    this.viewCamera.aspect = width / Math.max(1, height);
    this.viewCamera.updateProjectionMatrix();
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setSize(width, height, false);
    this.dirty = true;
  }

  private dataCorners(): Vec3[] {
    const b = this.bounds;
    const out: Vec3[] = [];
    for (let i = 0; i < 8; i++) out.push([i & 1 ? b.max[0] : b.min[0], i & 2 ? b.max[1] : b.min[1], i & 4 ? b.max[2] : b.min[2]]);
    return out;
  }

  /** The datasource `Camera` for the current select-camera pose, plus its frustum corners
   *  sized to the data slab. A plane is viewed *grazing* from just beyond one edge (the
   *  money shot: near edge fine, far edge coarse); a volume is orbited from outside. */
  private selectCamera(): { ds: Camera; corners: Vec3[] } {
    const size = this.size;
    const c: Vec3 = [this.center.x, this.center.y, this.center.z];
    const pitch = (this.selPitchDeg * Math.PI) / 180;
    const yaw = (this.selYawDeg * Math.PI) / 180;

    let eye: Vec3, forward: Vec3, up: Vec3;
    if (this.isPlane) {
      const ax = this.ms.worldFromArray.axes;
      const inU = normalize([ax[0][0], ax[0][1], ax[0][2]]);
      const inV = normalize([ax[1][0], ax[1][1], ax[1][2]]);
      const normal = normalize(cross(inU, inV));
      const across = normalize(add(scale(inV, Math.cos(yaw)), scale(inU, Math.sin(yaw))));
      const half = size * 0.5;
      const pull = this.selDistMult * size * 0.35; // beyond the near edge
      const height = Math.max(size * 0.03, Math.tan(pitch) * size * 0.9);
      eye = add(add(c, scale(across, -(half + pull))), scale(normal, height));
      forward = normalize(sub(add(c, scale(across, half * 0.25)), eye)); // aim across, toward the far side
      up = normal;
    } else {
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const dist = this.selDistMult * size;
      eye = [c[0] + cp * Math.sin(yaw) * dist, c[1] + sp * dist, c[2] + cp * Math.cos(yaw) * dist];
      forward = normalize(sub(c, eye));
      up = [0, 1, 0];
    }

    const ds: Camera = {
      eye, forward, up,
      fovY: (50 * Math.PI) / 180, aspect: 1.4,
      near: size * 0.005, far: size * 6,
      viewportHeightPx: 150, // a "virtual sensor" resolution; exaggerates the LOD spread for legibility
    };

    // Size the drawn frustum to the data: project its corners onto the view axis.
    let dmin = Infinity, dmax = -Infinity;
    for (const p of this.dataCorners()) {
      const d = dot(sub(p, eye), forward);
      if (d < dmin) dmin = d;
      if (d > dmax) dmax = d;
    }
    const nearViz = Math.max(size * 0.02, dmin * 0.6);
    const farViz = Math.max(nearViz + size * 0.1, dmax);
    return { ds, corners: frustumCorners(ds, nearViz, farViz) };
  }

  private rebuild(): Selection {
    const { ds, corners } = this.selectCamera();
    const sel = select(this.ms, ds, { q: this.q });
    disposeGroup(this.overlays);
    this.overlays.clear();
    this.overlays.add(chunkOverlays(this.ms, sel));
    this.overlays.add(frustumOverlay(corners, ds.eye, ds.forward));
    return sel;
  }

  async start(): Promise<void> {
    await this.renderer.init();
    this.renderer.setAnimationLoop(() => {
      if (this.disposed) return;
      this.controls.update();
      if (this.animate) { this.selYawDeg = (this.selYawDeg + 0.3) % 360; this.dirty = true; }
      if (this.dirty) {
        this.dirty = false;
        const sel = this.rebuild();
        this.onStats?.({ q: this.q, chunkCount: sel.chunks.length, totalBytes: sel.totalApproxBytes, countByLevel: sel.countByLevel });
      }
      this.renderer.render(this.scene, this.viewCamera);
    });
  }

  dispose(): void {
    this.disposed = true;
    this.renderer.setAnimationLoop(null);
    this.controls.dispose();
    disposeGroup(this.overlays);
    this.renderer.dispose();
  }
}
