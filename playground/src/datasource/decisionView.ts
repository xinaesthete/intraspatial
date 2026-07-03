// The Decision view (ADR-0008 slice 1): a plain, imperative three.js WebGPU scene —
// no React — that turns the pure `select()` into pixels. An orbit camera doubles as
// the *select* camera; each time the view or `q` changes we re-select and redraw the
// level-tinted chunk Overlays. Orbit/dolly → watch the levels recede and the
// fetch-budget fall. No tile textures yet (that is slice 2).
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { WebGPURenderer } from "three/webgpu";
import { select, worldAabbOfArrayBox, type Camera, type Multiscale, type Selection } from "../../../src/datasource";
import { boundsOverlay, chunkOverlays, disposeGroup } from "./overlays";

export interface DecisionStats {
  q: number;
  chunkCount: number;
  totalBytes: number;
  countByLevel: readonly number[];
}

/** Build the datasource `Camera` (ADR-0008) from the live three.js camera + viewport. */
function cameraFromThree(cam: THREE.PerspectiveCamera, viewportHeightPx: number): Camera {
  const fwd = new THREE.Vector3();
  cam.getWorldDirection(fwd);
  return {
    eye: [cam.position.x, cam.position.y, cam.position.z],
    forward: [fwd.x, fwd.y, fwd.z],
    up: [cam.up.x, cam.up.y, cam.up.z],
    fovY: (cam.fov * Math.PI) / 180,
    aspect: cam.aspect,
    near: cam.near,
    far: cam.far,
    viewportHeightPx,
  };
}

export class DecisionView {
  private renderer: WebGPURenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private overlays = new THREE.Group();
  private ms: Multiscale;
  private q = 1;
  private dirty = true;
  private disposed = false;
  private onStats?: (s: DecisionStats) => void;

  constructor(canvas: HTMLCanvasElement, ms: Multiscale, onStats?: (s: DecisionStats) => void) {
    this.ms = ms;
    this.onStats = onStats;
    this.renderer = new WebGPURenderer({ canvas, antialias: true, alpha: true });

    // Frame the dataset: centre + extent from its world bounds.
    const b = worldAabbOfArrayBox(ms.worldFromArray, [0, 0, 0], [ms.voxelDims0[0], ms.voxelDims0[1], ms.voxelDims0[2]]);
    const center = new THREE.Vector3((b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2);
    const size = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]) || 1;

    this.camera = new THREE.PerspectiveCamera(50, 1, size * 0.002, size * 20);
    this.camera.position.set(center.x + size * 0.9, center.y + size * 0.7, center.z - size * 1.4);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.copy(center);
    this.controls.addEventListener("change", () => { this.dirty = true; });

    this.scene.add(boundsOverlay(ms));
    this.scene.add(this.overlays);
    this.resize(canvas.clientWidth, canvas.clientHeight);
  }

  setQ(q: number): void {
    this.q = q;
    this.dirty = true;
  }

  resize(width: number, height: number): void {
    this.camera.aspect = width / Math.max(1, height);
    this.camera.updateProjectionMatrix();
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setSize(width, height, false);
    this.dirty = true;
  }

  private rebuild(): Selection {
    const dsCam = cameraFromThree(this.camera, this.renderer.domElement.clientHeight || 600);
    const sel = select(this.ms, dsCam, { q: this.q });
    disposeGroup(this.overlays);
    this.overlays.clear();
    this.overlays.add(chunkOverlays(this.ms, sel));
    return sel;
  }

  async start(): Promise<void> {
    await this.renderer.init();
    this.renderer.setAnimationLoop(() => {
      if (this.disposed) return;
      this.controls.update();
      if (this.dirty) {
        this.dirty = false;
        const sel = this.rebuild();
        this.onStats?.({ q: this.q, chunkCount: sel.chunks.length, totalBytes: sel.totalApproxBytes, countByLevel: sel.countByLevel });
      }
      this.renderer.render(this.scene, this.camera);
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
