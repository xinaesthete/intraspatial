// Dual view (ADR-0008): the real app on the left, the decision illustration inset on
// the right — both live, both independently orbitable. The MAIN camera is the app/
// select camera: it renders the real data (textured plane / raymarched volume) exactly
// as an app would, and *its* view drives the pure select(). The INSET is a third-person
// camera watching that select camera's frustum slice through the level-tinted chunk grid.
// One WebGPURenderer, two scissor viewports, three.js layers to split the content
// (data = layer 0, seen by both; overlays = layer 1, inset only).
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import type { WebGPURenderer } from "three/webgpu";
import {
  type Aabb,
  type Camera,
  cross,
  dot,
  frustumCorners,
  type Loader,
  type Multiscale,
  mandelbrotField,
  normalize,
  type Selection,
  select,
  sub,
  type Vec3,
  worldAabbOfArrayBox,
} from "../../../src/datasource";
import type { GpuBackend } from "../../../src/gpu/graph/backend";
import { type LatencyModel, mandelbulbBrickSource, slowBrickSource } from "./brickSource";
import { NaiveVolumeRenderer } from "./naiveVolumeRenderer";
import { boundsOverlay, chunkOverlays, disposeGroup, frustumOverlay } from "./overlays";
import { TileRenderer } from "./tileRenderer";
import { VolumeRenderer } from "./volumeRenderer";

export interface DecisionStats {
  q: number;
  chunkCount: number;
  totalBytes: number;
  countByLevel: readonly number[];
}

/** Optional per-frame hooks (e.g. mrdoob Stats begin/end). */
export interface FrameMonitor {
  begin(): void;
  end(): void;
}

const OVERLAY_LAYER = 1;
const INSET_W = 340,
  INSET_H = 250,
  INSET_M = 16;

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

export class DualView {
  private renderer: WebGPURenderer;
  private scene = new THREE.Scene();
  private appCamera: THREE.PerspectiveCamera; // main = select/app camera
  private decisionCamera: THREE.PerspectiveCamera; // inset overview
  private appControls: OrbitControls;
  private decisionControls: OrbitControls;
  private overlays = new THREE.Group();
  private ms: Multiscale;
  private loader: Loader;
  private backend?: GpuBackend; // GPU device seam for GPU-generated volume bricks (adopted from three)
  // Mutable latency model shared with the volume's brick source — the "slow network" sim knob.
  private netModel: LatencyModel = { base: 0, jitter: 0 };
  private bounds: Aabb;
  private isPlane: boolean;
  private center: THREE.Vector3;
  private size: number;

  private tiles: TileRenderer | null = null;
  private volume: VolumeRenderer | NaiveVolumeRenderer | null = null;
  private image: THREE.Mesh | null = null;
  private probe: THREE.Mesh | null = null;
  private transform: TransformControls | null = null;
  private volumeTransform: TransformControls | null = null;
  private onKey: ((e: KeyboardEvent) => void) | null = null;
  private q = 1;
  private showTextures = true;
  private showWireframe = true;
  private dirty = true;
  private lastOverlayRefresh = 0; // throttle decision-view recolour while chunks stream in
  private wasLoading = false; // to force one final recolour when the queue drains
  private decisionVisible = true; // inset shown? when collapsed, skip overlay build + inset pass
  private disposed = false;
  private width = 1;
  private height = 1;
  private onStats?: (s: DecisionStats) => void;
  private frameMonitor?: FrameMonitor;

  constructor(
    canvas: HTMLCanvasElement,
    insetEl: HTMLElement,
    renderer: WebGPURenderer,
    source: { ms: Multiscale; loader: Loader; makeMaterial?: (tex: THREE.Texture) => THREE.Material },
    onStats?: (s: DecisionStats) => void,
    combined = false,
    naive = false,
    frameMonitor?: FrameMonitor,
    backend?: GpuBackend,
  ) {
    const ms = source.ms;
    this.ms = ms;
    this.loader = source.loader;
    this.backend = backend;
    this.onStats = onStats;
    this.frameMonitor = frameMonitor;
    // Renderer (and its GPU device) is created once by the host and reused across source
    // switches — re-creating it per switch leaks GPU devices until WebGPU stops working.
    this.renderer = renderer;
    this.renderer.autoClear = false;

    const b = worldAabbOfArrayBox(ms.worldFromArray, [0, 0, 0], [ms.voxelDims0[0], ms.voxelDims0[1], ms.voxelDims0[2]]);
    this.bounds = b;
    this.isPlane = ms.voxelDims0[2] === 1;
    this.center = new THREE.Vector3((b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2);
    this.size = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]) || 1;
    const s = this.size,
      c = this.center;

    // App camera: front-on for a plane, outside-diagonal for a volume.
    this.appCamera = new THREE.PerspectiveCamera(50, 1, s * 0.002, s * 30);
    if (this.isPlane) {
      const ax = ms.worldFromArray.axes;
      const normal = normalize(cross([ax[0][0], ax[0][1], ax[0][2]], [ax[1][0], ax[1][1], ax[1][2]]));
      this.appCamera.position.set(c.x - normal[0] * s * 1.25 + s * 0.12, c.y + s * 0.18, c.z - normal[2] * s * 1.25 - s * 0.6);
    } else {
      this.appCamera.position.set(c.x + s * 0.9, c.y + s * 0.7, c.z - s * 1.6);
    }
    this.appControls = new OrbitControls(this.appCamera, canvas);
    this.appControls.enableDamping = true;
    this.appControls.dampingFactor = 0.08;
    this.appControls.target.copy(c);
    this.appControls.addEventListener("change", () => {
      this.dirty = true;
    });

    // Decision (inset) camera: pulled back to see the data AND the app frustum.
    this.decisionCamera = new THREE.PerspectiveCamera(45, INSET_W / INSET_H, s * 0.01, s * 60);
    this.decisionCamera.position.set(c.x + s * 1.9, c.y + s * 1.5, c.z - s * 2.8);
    this.decisionCamera.layers.enable(OVERLAY_LAYER); // sees data (0) + overlays (1)
    this.decisionControls = new OrbitControls(this.decisionCamera, insetEl);
    this.decisionControls.enableDamping = true;
    this.decisionControls.dampingFactor = 0.08;
    this.decisionControls.target.copy(c);

    const bounds = boundsOverlay(ms);
    bounds.layers.set(OVERLAY_LAYER);
    this.scene.add(bounds);
    if (this.isPlane) {
      this.tiles = new TileRenderer(ms, this.loader, { makeMaterial: source.makeMaterial });
      this.scene.add(this.tiles.group);
    } else if (naive) {
      // The pass-per-chunk baseline (no gizmo / depth-culling — a rendering-technique A/B).
      this.volume = new NaiveVolumeRenderer(ms, this.loader);
      this.volume.group.renderOrder = 0;
      this.scene.add(this.volume.group);
    } else {
      if (!this.backend) throw new Error("GPU volume renderer requires a GpuBackend (adopted from the three.js device)");
      // Bricks are GPU-generated (Mandelbulb compute) on the renderer's own device, then delayed
      // by the latency model (the network sim) before the scheduler commits them to the atlas.
      const source = slowBrickSource(mandelbulbBrickSource(ms, this.backend, ms.chunkShape[0]), this.netModel);
      const v = new VolumeRenderer(ms, source, this.renderer);
      this.volume = v;
      v.group.renderOrder = 0;
      this.scene.add(v.group);
      if (combined) {
        this.addImagePlane(canvas);
        this.addProbe();
      } else {
        this.addVolumeGizmo(canvas, v); // move/rotate the volume; select() + overlays + raymarch track it
      }
    }
    this.scene.add(this.overlays);
    this.resize(canvas.clientWidth, canvas.clientHeight);
  }

  setQ(q: number): void {
    this.q = q;
    this.dirty = true;
  }
  setTextures(on: boolean): void {
    this.showTextures = on;
    if (this.tiles) this.tiles.group.visible = on;
    if (this.volume) this.volume.group.visible = on;
    this.dirty = true;
  }
  setWireframe(on: boolean): void {
    this.showWireframe = on;
    this.dirty = true;
  }
  setTransfer(cmin: number, cmax: number, gamma: number): void {
    this.volume?.setTransfer(cmin, cmax, gamma);
  }
  setDepthRead(on: boolean): void {
    this.volume?.setDepthRead(on);
  }
  setDepthWrite(on: boolean): void {
    this.volume?.setDepthWrite(on);
  }
  setSolid(threshold: number): void {
    this.volume?.setSolid(threshold);
  }
  /** Show/hide the decision inset. When hidden, rebuild() skips the (potentially huge) overlay
   *  build, the frame loop skips the inset render pass, and the inset's orbit is disabled — so
   *  collapsing it is also the escape hatch from slow rebuilds at large finest resolutions. */
  setDecisionVisible(on: boolean): void {
    this.decisionVisible = on;
    this.decisionControls.enabled = on;
    this.dirty = true; // rebuild to add/clear the overlays
  }
  /** Set the simulated network latency (ms) for the volume's brick source: each brick waits
   *  `base + rand·jitter` before generating. Live — mutates the shared model in place. */
  setNetwork(base: number, jitter: number): void {
    this.netModel.base = base;
    this.netModel.jitter = jitter;
  }

  /** A movable opaque image quad (mandelbrot) + a translate/rotate gizmo — slide it
   *  through the volume to see depth-culling. Opaque ⇒ it writes depth before the
   *  transparent volume, whose depth-aware ray then clips against it. */
  private addImagePlane(canvas: HTMLElement): void {
    const N = 256;
    const data = new Uint8Array(N * N * 4);
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const g = Math.round(mandelbrotField([i / N, j / N, 0]) * 255);
        const o = (j * N + i) * 4;
        data[o] = g;
        data[o + 1] = g * 0.7;
        data[o + 2] = 40 + g * 0.4;
        data[o + 3] = 255;
      }
    }
    const tex = new THREE.DataTexture(data, N, N);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    const s = this.size;
    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(s * 1.15, s * 1.15),
      new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide }),
    );
    mesh.position.copy(this.center);
    this.image = mesh;
    this.scene.add(mesh);

    const tc = new TransformControls(this.appCamera, canvas);
    tc.attach(mesh);
    tc.addEventListener("dragging-changed", (e) => {
      this.appControls.enabled = !(e as unknown as { value: boolean }).value;
    });
    this.scene.add(tc.getHelper());
    this.transform = tc;
  }

  /** A bright rod skewering the volume — the depth-WRITE demonstrator. It's drawn AFTER
   *  the (transparent) volume and depth-tests against it (renderOrder 1 + depthTest),
   *  so when the volume writes its solid-surface depth the rod is swallowed where it
   *  passes behind that surface. Toggle "volume writes depth" off ⇒ the whole rod shows
   *  through. (An opaque object couldn't show this — the opaque pass draws before the
   *  transparent volume, so the volume's depth write would come too late to occlude it.) */
  private addProbe(): void {
    const s = this.size;
    const geo = new THREE.CylinderGeometry(s * 0.03, s * 0.03, s * 2.4, 20);
    geo.rotateZ(Math.PI / 2); // lie along the X axis, through the volume
    const mat = new THREE.MeshBasicMaterial({ color: 0x34f5c8 });
    mat.transparent = true; // join the transparent pass so it sorts AFTER the volume
    mat.depthTest = true; // read the volume's written depth → get occluded by it
    mat.depthWrite = true;
    // Where the rod is exactly coplanar with the volume's solid surface it would z-fight
    // (dashed seam). Bias it very slightly away so the surface wins the tie cleanly.
    mat.polygonOffset = true;
    mat.polygonOffsetFactor = 1;
    mat.polygonOffsetUnits = 1;
    const rod = new THREE.Mesh(geo, mat);
    rod.position.copy(this.center);
    rod.renderOrder = 1; // after the volume (renderOrder 0)
    this.probe = rod;
    this.scene.add(rod);
  }

  /** A translate/rotate gizmo on the volume itself. Moving it reorients the whole placement —
   *  the raymarch (via syncTransform), the chunk overlays AND select() all track it live, which
   *  exercises the de-axialised pipeline end to end. Press R for rotate, T for translate. */
  private addVolumeGizmo(canvas: HTMLElement, vol: VolumeRenderer): void {
    const tc = new TransformControls(this.appCamera, canvas);
    tc.attach(vol.transformTarget);
    tc.addEventListener("dragging-changed", (e) => {
      this.appControls.enabled = !(e as unknown as { value: boolean }).value;
    });
    tc.addEventListener("objectChange", () => {
      vol.syncTransform();
      this.dirty = true; // re-select + rebuild overlays against the new placement
    });
    this.scene.add(tc.getHelper());
    this.volumeTransform = tc;

    this.onKey = (ev: KeyboardEvent): void => {
      if (ev.key === "r" || ev.key === "R") tc.setMode("rotate");
      else if (ev.key === "t" || ev.key === "T") tc.setMode("translate");
    };
    window.addEventListener("keydown", this.onKey);
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.appCamera.aspect = width / Math.max(1, height);
    this.appCamera.updateProjectionMatrix();
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

  private rebuild(): Selection {
    const ds = cameraFromThree(this.appCamera, this.height || 600);
    // Use the volume's live placement (base ∘ gizmo) so select + overlays follow the gizmo.
    // (Only the brick-page VolumeRenderer tracks a transform; the naive baseline stays at base.)
    const ms = this.volume instanceof VolumeRenderer ? { ...this.ms, worldFromArray: this.volume.effectiveWorldFromArray() } : this.ms;
    const sel = select(ms, ds, { q: this.q });

    disposeGroup(this.overlays);
    this.overlays.clear();
    // Building the level-tinted chunk boxes is O(selected chunks) of geometry — the dominant cost
    // of rebuild() at large finest resolutions (tens of thousands of boxes). Since overlays are
    // decision-inset-only (layer 1), skip the whole build when the inset is collapsed.
    if (this.decisionVisible) {
      // Colour chunk boxes by load state (brick-page volume only): hue = level, opacity = state.
      const vol = this.volume instanceof VolumeRenderer ? this.volume : null;
      const stateOf = vol ? (k: string) => vol.chunkState(k) : undefined;
      if (this.showWireframe) this.overlays.add(chunkOverlays(ms, sel, stateOf));
      // Draw the app camera's frustum, sized to the data slab.
      let dmin = Infinity,
        dmax = -Infinity;
      for (const p of this.dataCorners()) {
        const d = dot(sub(p, ds.eye), ds.forward);
        if (d < dmin) dmin = d;
        if (d > dmax) dmax = d;
      }
      const nearViz = Math.max(this.size * 0.02, dmin * 0.6);
      this.overlays.add(frustumOverlay(frustumCorners(ds, nearViz, Math.max(nearViz + this.size * 0.1, dmax)), ds.eye, ds.forward));
      this.overlays.traverse((o) => o.layers.set(OVERLAY_LAYER));
    }

    if (this.tiles && this.showTextures) this.tiles.update(sel);
    if (this.volume && this.showTextures) this.volume.update(sel);
    return sel;
  }

  /** While the brick-page volume streams chunks in, keep the decision view's colours current
   *  (throttled full recolour, since arrivals don't set `dirty`) and pulse the loading boxes each
   *  frame (cheap — opacity only). One final recolour when the queue drains clears the pulse. */
  private animateLoadingOverlays(): void {
    if (!this.decisionVisible) return; // no overlays to recolour/pulse when the inset is collapsed
    const vol = this.volume instanceof VolumeRenderer ? this.volume : null;
    if (!vol) return;
    const { pending, loading } = vol.loadCounts();
    const busy = pending + loading > 0;
    if (busy) {
      const now = performance.now();
      if (now - this.lastOverlayRefresh > 120) {
        this.lastOverlayRefresh = now;
        this.dirty = true; // rebuild overlays next frame with fresh per-chunk state
      }
      const pulse = 0.3 + 0.35 * (0.5 + 0.5 * Math.sin(now * 0.006));
      this.overlays.traverse((o) => {
        if (o instanceof THREE.LineSegments && o.userData.pulse) (o.material as THREE.LineBasicMaterial).opacity = pulse;
      });
    } else if (this.wasLoading) {
      this.dirty = true; // queue drained — recolour once so everything reads as resident
    }
    this.wasLoading = busy;
  }

  private renderInsetRect(): { x: number; y: number; w: number; h: number } {
    // WebGPURenderer's viewport/scissor origin is top-left here, so y measures from the top —
    // matches the #inset div (CSS bottom-right).
    return { x: this.width - INSET_W - INSET_M, y: this.height - INSET_H - INSET_M, w: INSET_W, h: INSET_H };
  }

  async start(): Promise<void> {
    this.renderer.setAnimationLoop(() => {
      if (this.disposed) return;
      this.frameMonitor?.begin();
      this.appControls.update();
      this.decisionControls.update();
      if (this.dirty) {
        this.dirty = false;
        const sel = this.rebuild();
        this.onStats?.({ q: this.q, chunkCount: sel.chunks.length, totalBytes: sel.totalApproxBytes, countByLevel: sel.countByLevel });
      }
      this.animateLoadingOverlays();
      // Main pass: app camera, full canvas, data only (layer 0).
      this.renderer.setScissorTest(false);
      this.renderer.setViewport(0, 0, this.width, this.height);
      this.renderer.setClearColor(0x0b1020, 1);
      this.renderer.clear();
      this.renderer.render(this.scene, this.appCamera);
      // Inset pass: decision camera, its own scissor region, data + overlays (layer 1). Skipped
      // entirely when the inset is collapsed (nothing to draw there).
      if (this.decisionVisible) {
        const r = this.renderInsetRect();
        this.renderer.setScissorTest(true);
        this.renderer.setScissor(r.x, r.y, r.w, r.h);
        this.renderer.setViewport(r.x, r.y, r.w, r.h);
        this.renderer.setClearColor(0x0a0f1c, 1);
        this.renderer.clear();
        this.renderer.render(this.scene, this.decisionCamera);
      }
      this.frameMonitor?.end();
    });
  }

  /** Tear down this scene's resources + controls, but NOT the shared renderer/device. */
  dispose(): void {
    this.disposed = true;
    this.renderer.setAnimationLoop(null);
    this.appControls.dispose();
    this.decisionControls.dispose();
    disposeGroup(this.overlays);
    this.tiles?.dispose();
    this.volume?.dispose();
    this.transform?.dispose();
    this.volumeTransform?.dispose();
    if (this.onKey) window.removeEventListener("keydown", this.onKey);
    if (this.image) {
      this.image.geometry.dispose();
      (this.image.material as THREE.Material).dispose();
    }
    if (this.probe) {
      this.probe.geometry.dispose();
      (this.probe.material as THREE.Material).dispose();
    }
    // Deliberately NOT this.renderer.dispose() — the host owns and reuses it.
  }
}
