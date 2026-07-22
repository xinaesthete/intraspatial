// MultiImageScene — several SpatialData images in ONE WebGPU view, each with an editable 3D
// placement (translate / rotate / scale) driven by a three.js gizmo, plus click-to-select picking.
// This is the multi-image counterpart to the single-image `DualView`: it reuses the SAME streaming
// building blocks per image (TileRenderer + sd.js Loader + ChannelComposite) but drops the
// dual-view "decision inset" — one shared app camera, N images.
//
// Placement (1b co-registration): when the store carries a real sd.js `global` transform per image
// (`SpatialDataImage.globalFromArray`), we place every image through ONE shared global→world
// similarity (normalise+centre the first image's global footprint to ~worldSpan units). Images that
// share the `global` coordinate system then land co-registered with an IDENTITY editable transform;
// the user nudges the gizmo to fine-tune. Stores without a stored transform fall back to a stagger.
//
// Cheap gizmo: TileRenderer bakes the *base* array→world affine into tile geometry (unchanged). Each
// image's `tiles.group` is parented under an editable `xform` Object3D the gizmo moves — the scene
// graph re-transforms baked geometry for free (no rebuild on drag). Per frame the effective placement
// `xform.matrixWorld ∘ base` is fed to the pure `select()` so LOD tracks the pose.
//
// Blend/depth (the "layers" model, viv/MDV-shaped): co-registered images are near-coplanar, so a
// single shared depth buffer would make them z-fight/occlude. One shared depth buffer can't isolate
// per-image LOD AND blend across images (mid-frame depth clears aren't reliable on the WebGPU
// backend; true isolation needs render-to-texture). So the FIRST image is the opaque base — it keeps
// depth on, so intra-image LOD (finer tiles win via LEVEL_Z_BIAS) works — and later images are
// overlays: transparent, drawn depth-test-off in add order (renderOrder), each with its own opacity
// and Normal(over)/Additive blend. Overlays therefore composite over the base by layer order rather
// than true 3-D depth (correct for co-registration; the seam for a later layers panel + RTT).
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import type { WebGPURenderer } from "three/webgpu";
import { type Affine3, type Camera, type Multiscale, select, worldAabbOfArrayBox } from "../../../src/datasource";
import type { SpatialDataImage } from "./spatialDataLoader";
import { type BlendMode, ChannelComposite, type ChannelSettings } from "./tileChannelMaterial";
import { TileRenderer } from "./tileRenderer";

export type GizmoMode = "translate" | "rotate" | "scale";
/** How an image composites over the ones beneath it: `normal` = alpha-over, `additive` = sum. */
export type LayerBlend = "normal" | "additive";

const WORLD_SPAN = 256; // the first co-registered image's largest global extent maps to ~this many world units
const OVERLAY_OPACITY = 0.65; // default opacity for images added on top of the base layer

/** A rigid-ish 3D placement as position + quaternion + scale — the serialisable form of an image's
 *  editable transform (maps 1:1 onto `THREE.Object3D`). */
export interface ImageTransform {
  translation: [number, number, number];
  rotation: [number, number, number, number]; // quaternion (x, y, z, w)
  scale: [number, number, number];
}

/** Per-image metadata surfaced to the React panel (no three.js objects leak out). */
export interface SceneImageInfo {
  readonly id: string;
  readonly name: string;
  readonly label: string;
  readonly visible: boolean;
  readonly opacity: number;
  readonly blend: LayerBlend;
  readonly coregistered: boolean; // placed from a real sd.js global transform (vs staggered fallback)
  readonly channels: readonly ChannelSettings[];
}

/** One image's serialisable state — everything needed to rebuild it (runtime-only today). */
export interface SceneImageState {
  readonly id: string;
  readonly store: string;
  readonly element: string;
  readonly transform: ImageTransform;
  readonly opacity: number;
  readonly blend: LayerBlend;
  readonly channels: ChannelSettings[];
}

export interface SceneState {
  readonly version: 1;
  readonly images: SceneImageState[];
}

interface SceneImage {
  id: string;
  name: string;
  store: string;
  label: string;
  ms: Multiscale; // with worldFromArray = the baked base placement
  baseMatrix: THREE.Matrix4; // array→local placement baked into the tiles' geometry
  xform: THREE.Object3D; // editable local→world placement (what the gizmo moves)
  tiles: TileRenderer;
  composite: ChannelComposite;
  channels: ChannelSettings[];
  pickPlane: THREE.Mesh; // invisible, full-extent, for robust picking before tiles stream in
  opacity: number;
  blend: LayerBlend;
  coregistered: boolean;
}

const IDENTITY_QUAT: [number, number, number, number] = [0, 0, 0, 1];
const blendOf = (b: LayerBlend): THREE.Blending => (b === "additive" ? THREE.AdditiveBlending : THREE.NormalBlending);

/** Build a `Camera` (the pure select input) from a three.js perspective camera. */
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

/** An engine `Affine3` (origin + column axes) as a column-major `Matrix4`. */
function affineToMatrix(a: Affine3): THREE.Matrix4 {
  const [x, y, z] = a.axes;
  return new THREE.Matrix4()
    .makeBasis(new THREE.Vector3(x[0], x[1], x[2]), new THREE.Vector3(y[0], y[1], y[2]), new THREE.Vector3(z[0], z[1], z[2]))
    .setPosition(a.origin[0], a.origin[1], a.origin[2]);
}

/** A column-major `Matrix4` as an engine `Affine3` (columns → axes, translation → origin). */
function matrixToAffine(m: THREE.Matrix4): Affine3 {
  const e = m.elements;
  return {
    origin: [e[12] ?? 0, e[13] ?? 0, e[14] ?? 0],
    axes: [
      [e[0] ?? 0, e[1] ?? 0, e[2] ?? 0],
      [e[4] ?? 0, e[5] ?? 0, e[6] ?? 0],
      [e[8] ?? 0, e[9] ?? 0, e[10] ?? 0],
    ],
  };
}

const readTransform = (o: THREE.Object3D): ImageTransform => ({
  translation: [o.position.x, o.position.y, o.position.z],
  rotation: [o.quaternion.x, o.quaternion.y, o.quaternion.z, o.quaternion.w],
  scale: [o.scale.x, o.scale.y, o.scale.z],
});

export class MultiImageScene {
  private renderer: WebGPURenderer;
  private canvas: HTMLCanvasElement;
  private scene = new THREE.Scene(); // all image xforms + the gizmo helper, drawn in one pass
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private gizmo: TransformControls;
  private raycaster = new THREE.Raycaster();
  private images: SceneImage[] = [];
  private selectedId: string | null = null;
  private q = 1;
  private channelBlend: BlendMode = "additive";
  private dirty = true; // recompute per-image LOD select on the next frame
  private width = 1;
  private height = 1;
  private idSeq = 0;
  private disposed = false;

  // The shared global→world similarity, seeded from the first co-registered image (null until then).
  private gCenter: THREE.Vector3 | null = null;
  private gScale = 1;

  // Pointer bookkeeping so an orbit-drag isn't mistaken for a select-click.
  private downPos: { x: number; y: number } | null = null;

  private onImagesChange?: (images: SceneImageInfo[]) => void;
  private onSelectionChange?: (id: string | null) => void;
  private onTransformChange?: (id: string, t: ImageTransform) => void;

  constructor(canvas: HTMLCanvasElement, renderer: WebGPURenderer) {
    this.canvas = canvas;
    this.renderer = renderer;
    this.renderer.autoClear = true;
    this.renderer.setClearColor(0x000000, 0); // transparent clear so the page background shows through

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100000);
    this.camera.position.set(0, 0, 600);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.addEventListener("change", () => {
      this.dirty = true;
    });

    // One gizmo, re-attached to the selected image's xform. Same wiring as dualView's volume gizmo:
    // disable orbit while dragging, re-select LOD on change, R/T/S switch mode.
    this.gizmo = new TransformControls(this.camera, canvas);
    this.gizmo.addEventListener("dragging-changed", (e) => {
      this.controls.enabled = !(e as unknown as { value: boolean }).value;
    });
    this.gizmo.addEventListener("objectChange", () => {
      this.dirty = true;
      const img = this.images.find((i) => i.id === this.selectedId);
      if (img) this.onTransformChange?.(img.id, readTransform(img.xform));
    });
    this.scene.add(this.gizmo.getHelper());

    canvas.addEventListener("pointerdown", this.handlePointerDown);
    canvas.addEventListener("pointerup", this.handlePointerUp);
    window.addEventListener("keydown", this.handleKey);

    this.renderer.setAnimationLoop(this.frame);
  }

  // ---- React-facing subscriptions -----------------------------------------------------------

  setCallbacks(cbs: {
    onImagesChange?: (images: SceneImageInfo[]) => void;
    onSelectionChange?: (id: string | null) => void;
    onTransformChange?: (id: string, t: ImageTransform) => void;
  }): void {
    this.onImagesChange = cbs.onImagesChange;
    this.onSelectionChange = cbs.onSelectionChange;
    this.onTransformChange = cbs.onTransformChange;
  }

  private emitImages(): void {
    this.onImagesChange?.(
      this.images.map((i) => ({
        id: i.id,
        name: i.name,
        label: i.label,
        visible: i.xform.visible,
        opacity: i.opacity,
        blend: i.blend,
        coregistered: i.coregistered,
        channels: i.channels,
      })),
    );
  }

  // ---- Placement -----------------------------------------------------------------------------

  /** Apply the shared global→world similarity to a global-space affine (uniform scale about the
   *  seeded global centre). Preserves relative geometry, so co-registered images stay aligned. */
  private worldFromGlobal(gfa: Affine3): Affine3 {
    const c = this.gCenter ?? new THREE.Vector3();
    const s = this.gScale;
    return {
      origin: [(gfa.origin[0] - c.x) * s, (gfa.origin[1] - c.y) * s, (gfa.origin[2] - c.z) * s],
      axes: [
        [gfa.axes[0][0] * s, gfa.axes[0][1] * s, gfa.axes[0][2] * s],
        [gfa.axes[1][0] * s, gfa.axes[1][1] * s, gfa.axes[1][2] * s],
        [gfa.axes[2][0] * s, gfa.axes[2][1] * s, gfa.axes[2][2] * s],
      ],
    };
  }

  // ---- Scene mutation ------------------------------------------------------------------------

  /** Add a built SpatialData image. If it carries a real sd.js `global` transform it lands
   *  co-registered (identity editable transform); otherwise it's staggered beside the others.
   *  Returns the new image id. */
  addImage(src: SpatialDataImage, opts: { store: string; element: string }): string {
    const id = `img${++this.idSeq}`;
    const dims = src.ms.voxelDims0;
    const isBase = this.images.length === 0;

    // Decide the baked base placement + initial editable transform.
    let baseAffine: Affine3;
    const xform = new THREE.Object3D();
    const coregistered = !!src.globalFromArray;
    if (src.globalFromArray) {
      // Seed the shared global→world similarity from the FIRST co-registered image.
      if (!this.gCenter) {
        const gb = worldAabbOfArrayBox(src.globalFromArray, [0, 0, 0], [dims[0], dims[1], 1]);
        this.gCenter = new THREE.Vector3((gb.min[0] + gb.max[0]) / 2, (gb.min[1] + gb.max[1]) / 2, (gb.min[2] + gb.max[2]) / 2);
        const ext = Math.max(gb.max[0] - gb.min[0], gb.max[1] - gb.min[1], gb.max[2] - gb.min[2]) || 1;
        this.gScale = WORLD_SPAN / ext;
      }
      baseAffine = this.worldFromGlobal(src.globalFromArray);
      // xform starts at identity — the stored transform already co-registers the image.
    } else {
      // Fallback: the loader's demo-normalised, axis-aligned placement, staggered along +X so
      // images don't overlap and are individually pickable.
      baseAffine = src.ms.worldFromArray;
      const wb = worldAabbOfArrayBox(baseAffine, [0, 0, 0], [dims[0], dims[1], 1]);
      const spanX = (wb.max[0] - wb.min[0]) * 1.15 || 300;
      xform.position.set(this.images.length * spanX, 0, 0);
    }

    const ms: Multiscale = { ...src.ms, worldFromArray: baseAffine };
    // The base layer is opaque (keeps depth on → intra-image LOD); overlays are transparent and
    // composited over it depth-test-off, in add order.
    const opacity = isBase ? 1 : OVERLAY_OPACITY;
    const composite = new ChannelComposite(src.channels, this.channelBlend, { transparent: !isBase, opacity });
    const tiles = new TileRenderer(ms, src.loader, { makeMaterial: composite.makeMaterial });
    tiles.setBlending(blendOf("normal"));
    if (!isBase) tiles.setDepth(false, false);
    tiles.setRenderOrder(this.images.length); // add order = layer order

    xform.add(tiles.group);
    const pickPlane = this.makePickPlane(ms, id);
    xform.add(pickPlane);
    this.scene.add(xform);
    this.scene.updateMatrixWorld(true);

    const img: SceneImage = {
      id,
      name: opts.element,
      store: opts.store,
      label: src.label,
      ms,
      baseMatrix: affineToMatrix(baseAffine),
      xform,
      tiles,
      composite,
      channels: src.channels,
      pickPlane,
      opacity,
      blend: "normal",
      coregistered,
    };
    this.images.push(img);

    if (this.images.length === 1) this.frameCamera();
    this.dirty = true;
    this.emitImages();
    this.selectImage(id);
    return id;
  }

  /** A full-extent, render-invisible-but-raycastable quad in the image's local (array→local) space,
   *  so clicks land on the image even before any tiles have streamed in. */
  private makePickPlane(ms: Multiscale, name: string): THREE.Mesh {
    const box = worldAabbOfArrayBox(ms.worldFromArray, [0, 0, 0], [ms.voxelDims0[0], ms.voxelDims0[1], 1]);
    const w = box.max[0] - box.min[0];
    const h = box.max[1] - box.min[1];
    const geo = new THREE.PlaneGeometry(w || 1, h || 1);
    const mat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0 });
    mat.colorWrite = false; // never draws colour…
    mat.depthWrite = false; // …nor disturbs depth — it exists only to be raycast.
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set((box.min[0] + box.max[0]) / 2, (box.min[1] + box.max[1]) / 2, 0);
    mesh.renderOrder = -1;
    mesh.name = `pick:${name}`;
    return mesh;
  }

  removeImage(id: string): void {
    const idx = this.images.findIndex((i) => i.id === id);
    if (idx === -1) return;
    const img = this.images[idx];
    if (!img) return;
    if (this.selectedId === id) this.selectImage(null);
    this.scene.remove(img.xform);
    img.tiles.dispose();
    img.pickPlane.geometry.dispose();
    (img.pickPlane.material as THREE.Material).dispose();
    this.images.splice(idx, 1);
    this.dirty = true;
    this.emitImages();
  }

  /** Select an image (attach the gizmo to it) or clear the selection. */
  selectImage(id: string | null): void {
    this.selectedId = id;
    const img = id ? this.images.find((i) => i.id === id) : undefined;
    if (img) this.gizmo.attach(img.xform);
    else this.gizmo.detach();
    this.onSelectionChange?.(id);
  }

  setGizmoMode(mode: GizmoMode): void {
    this.gizmo.setMode(mode);
  }

  /** Set an image's editable transform from the numeric panel (React is the source of truth here,
   *  so no `onTransformChange` echo). */
  setTransform(id: string, t: ImageTransform): void {
    const img = this.images.find((i) => i.id === id);
    if (!img) return;
    img.xform.position.set(...t.translation);
    img.xform.quaternion.set(...t.rotation);
    img.xform.scale.set(...t.scale);
    img.xform.updateMatrixWorld(true);
    this.dirty = true;
  }

  /** Reset an image to identity (back to its stored co-registered pose, or origin for staggered). */
  resetTransform(id: string): void {
    this.setTransform(id, { translation: [0, 0, 0], rotation: IDENTITY_QUAT, scale: [1, 1, 1] });
    const img = this.images.find((i) => i.id === id);
    if (img) this.onTransformChange?.(id, readTransform(img.xform));
  }

  setVisible(id: string, visible: boolean): void {
    const img = this.images.find((i) => i.id === id);
    if (!img) return;
    img.xform.visible = visible;
    if (!visible && this.selectedId === id) this.selectImage(null);
    this.dirty = true;
    this.emitImages();
  }

  /** Set an image's layer opacity (live; how strongly it shows over the layers beneath it). */
  setOpacity(id: string, opacity: number): void {
    const img = this.images.find((i) => i.id === id);
    if (!img) return;
    img.opacity = opacity;
    img.composite.setOpacity(opacity);
    this.emitImages();
  }

  /** Set an image's layer blend mode (normal = over, additive = sum). */
  setBlend(id: string, blend: LayerBlend): void {
    const img = this.images.find((i) => i.id === id);
    if (!img) return;
    img.blend = blend;
    img.tiles.setBlending(blendOf(blend));
    this.emitImages();
  }

  setQ(q: number): void {
    this.q = q;
    this.dirty = true;
  }

  /** The across-CHANNEL composite mode (additive/max), applied to every image's material. */
  setChannelBlend(blend: BlendMode): void {
    this.channelBlend = blend;
    for (const i of this.images) i.composite.update(i.channels, blend);
  }

  /** Re-push a single image's channel settings into its live material (colour/window/visibility). */
  updateChannels(id: string): void {
    const img = this.images.find((i) => i.id === id);
    img?.composite.update(img.channels, this.channelBlend);
  }

  getTransform(id: string): ImageTransform | null {
    const img = this.images.find((i) => i.id === id);
    return img ? readTransform(img.xform) : null;
  }

  /** The serialisable scene — runtime-only today; the seam for localStorage / sd.js write. */
  getState(): SceneState {
    return {
      version: 1,
      images: this.images.map((i) => ({
        id: i.id,
        store: i.store,
        element: i.name,
        transform: readTransform(i.xform),
        opacity: i.opacity,
        blend: i.blend,
        channels: i.channels.map((c) => ({ ...c })),
      })),
    };
  }

  // ---- Camera --------------------------------------------------------------------------------

  private frameCamera(): void {
    const box = new THREE.Box3();
    for (const img of this.images) {
      const b = worldAabbOfArrayBox(matrixToAffine(this.effectiveMatrix(img)), [0, 0, 0], [img.ms.voxelDims0[0], img.ms.voxelDims0[1], 1]);
      box.expandByPoint(new THREE.Vector3(...b.min));
      box.expandByPoint(new THREE.Vector3(...b.max));
    }
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    this.camera.near = size * 0.001;
    this.camera.far = size * 50;
    this.camera.position.set(center.x, center.y, center.z + size * 0.9);
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.update();
  }

  resize(width: number, height: number): void {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setSize(this.width, this.height, false);
    this.dirty = true;
  }

  // ---- Frame loop ----------------------------------------------------------------------------

  /** Effective array→world matrix for an image (`xform.matrixWorld ∘ base`). */
  private effectiveMatrix(img: SceneImage): THREE.Matrix4 {
    return new THREE.Matrix4().multiplyMatrices(img.xform.matrixWorld, img.baseMatrix);
  }

  private frame = (): void => {
    if (this.disposed) return;
    this.controls.update();
    this.scene.updateMatrixWorld(true); // xform world matrices current for select, render, and picking

    if (this.dirty) {
      const cam = cameraFromThree(this.camera, this.height);
      for (const img of this.images) {
        if (!img.xform.visible) continue;
        const eff = matrixToAffine(this.effectiveMatrix(img));
        img.tiles.update(select({ ...img.ms, worldFromArray: eff }, cam, { q: this.q }));
      }
      this.dirty = false;
    }

    // One pass: the base draws opaque with depth; overlays draw depth-test-off in renderOrder, each
    // blending over what's beneath (the layers model — see the file header).
    this.renderer.render(this.scene, this.camera);
  };

  // ---- Input ---------------------------------------------------------------------------------

  private handlePointerDown = (e: PointerEvent): void => {
    this.downPos = { x: e.clientX, y: e.clientY };
  };

  private handlePointerUp = (e: PointerEvent): void => {
    const down = this.downPos;
    this.downPos = null;
    if (!down) return;
    // Ignore drags (orbit) and any interaction consumed by the gizmo.
    if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
    if ((this.gizmo as unknown as { dragging?: boolean }).dragging) return;
    this.pickAt(e.clientX, e.clientY);
  };

  private pickAt(clientX: number, clientY: number): void {
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const planes = this.images.filter((i) => i.xform.visible).map((i) => i.pickPlane);
    const hits = this.raycaster.intersectObjects(planes, false);
    const first = hits[0]?.object;
    const picked = this.images.find((i) => i.pickPlane === first);
    this.selectImage(picked ? picked.id : null);
  }

  private handleKey = (e: KeyboardEvent): void => {
    if (e.key === "r" || e.key === "R") this.setGizmoMode("rotate");
    else if (e.key === "t" || e.key === "T") this.setGizmoMode("translate");
    else if (e.key === "s" || e.key === "S") this.setGizmoMode("scale");
    else if (e.key === "Escape") this.selectImage(null);
  };

  dispose(): void {
    this.disposed = true;
    this.renderer.setAnimationLoop(null);
    this.canvas.removeEventListener("pointerdown", this.handlePointerDown);
    this.canvas.removeEventListener("pointerup", this.handlePointerUp);
    window.removeEventListener("keydown", this.handleKey);
    this.gizmo.detach();
    this.gizmo.dispose();
    this.controls.dispose();
    for (const img of this.images) {
      this.scene.remove(img.xform);
      img.tiles.dispose();
      img.pickPlane.geometry.dispose();
      (img.pickPlane.material as THREE.Material).dispose();
    }
    this.images = [];
  }
}
