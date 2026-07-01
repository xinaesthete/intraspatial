// The 3D stage — three.js on its WebGPU backend (WebGPURenderer). The swarm is one
// InstancedMesh of small oriented cones (so facing/spin reads), plus motion trails drawn
// as additive line segments fed by the TrailBuffer ring-buffer primitive. Supports hover
// picking (→ which dancer) and highlighting a set of dancers + a connecting line for a
// hovered pair (the distance-matrix cross-link). Orbit camera, soft lighting, glow.
import * as THREE from "three";
import { StorageInstancedBufferAttribute, WebGPURenderer } from "three/webgpu";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TrailBuffer } from "../../../../src/gpu/sim/trails";

const TRAIL_LEN = 56;
const TRAIL_HEAD: [number, number, number] = [0.5, 0.72, 1.0];

export interface Highlight {
  /** Dancers to emphasise (enlarge + brighten). */
  agents: number[];
  /** A pair to connect with a line (the hovered matrix cell), or null. */
  pair: [number, number] | null;
}

export interface DancerRenderer {
  readonly renderer: WebGPURenderer;
  update(positions: Float32Array, orientations: Float32Array, speeds: Float32Array, highlight?: Highlight): void;
  render(): void;
  resize(width: number, height: number): void;
  /** Pick the dancer under normalised device coords (x,y ∈ [-1,1]); null if none. */
  pick(ndcX: number, ndcY: number): number | null;
  /** Turn on GPU-driven instance matrices: the render reads pose from the storage
   *  `instanceMatrix` buffer (written by our compute) — `update()` stops touching matrices. */
  setGpuMatrix(on: boolean): void;
  /** The raw GPUBuffer backing `instanceMatrix` (materialising it with a render if needed), so
   *  our TypeGPU compute can write model matrices into it. Null if not yet available. */
  gpuInstanceMatrixBuffer(): Promise<GPUBuffer | null>;
  /** True while the camera is being manipulated (plus a short damping cooldown). */
  isInteracting(): boolean;
  dispose(): void;
}

export async function createDancerRenderer(canvas: HTMLCanvasElement, n: number): Promise<DancerRenderer> {
  const renderer = new WebGPURenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  await renderer.init();

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
  camera.position.set(0, 6, 16);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);

  // Track active camera interaction so the caller can suspend the (periodic, GPU-syncing)
  // snapshot readback while dragging — the readback's mapAsync contends with the heavier render
  // during a drag and shows up as regular pauses. Add a short cooldown so inertial damping
  // after release also stays readback-free.
  let interactUntil = 0;
  const now = (): number => (typeof performance !== "undefined" ? performance.now() : 0);
  controls.addEventListener("start", () => {
    interactUntil = Number.POSITIVE_INFINITY;
  });
  controls.addEventListener("end", () => {
    interactUntil = now() + 400;
  });

  scene.add(new THREE.AmbientLight(0x6070a0, 1.2));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(6, 10, 8);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x4060ff, 1.0);
  rim.position.set(-8, -4, -6);
  scene.add(rim);

  // dancers — oriented cones (tip → +z)
  const geometry = new THREE.ConeGeometry(0.1, 0.2, 10);
  geometry.rotateX(Math.PI / 2);
  const material = new THREE.MeshStandardMaterial({
    color: 0x9fc4ff,
    emissive: 0x2a4a8a,
    emissiveIntensity: 0.6,
    roughness: 0.4,
    metalness: 0.1,
  });
  const mesh = new THREE.InstancedMesh(geometry, material, n);
  // A storage instanceMatrix so the render can read pose straight from a GPUBuffer our compute
  // writes (zero readback). In CPU-fallback mode we still setMatrixAt + needsUpdate (three
  // uploads the array); in GPU mode we never touch it and never set needsUpdate.
  mesh.instanceMatrix = new StorageInstancedBufferAttribute(new Float32Array(n * 16), 16);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
  scene.add(mesh);
  let gpuMatrix = false;
  let lastPositions: Float32Array | null = null;

  // trails — additive line segments from the ring buffer
  const trail = new TrailBuffer(n, TRAIL_LEN);
  const maxVerts = n * (TRAIL_LEN - 1) * 2;
  const trailGeo = new THREE.BufferGeometry();
  const trailPos = new THREE.BufferAttribute(new Float32Array(maxVerts * 3), 3);
  const trailCol = new THREE.BufferAttribute(new Float32Array(maxVerts * 3), 3);
  trailPos.setUsage(THREE.DynamicDrawUsage);
  trailCol.setUsage(THREE.DynamicDrawUsage);
  trailGeo.setAttribute("position", trailPos);
  trailGeo.setAttribute("color", trailCol);
  trailGeo.setDrawRange(0, 0);
  const trailMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
  const trails = new THREE.LineSegments(trailGeo, trailMat);
  trails.frustumCulled = false;
  scene.add(trails);

  // pair-connection line (hovered matrix cell)
  const pairGeo = new THREE.BufferGeometry();
  const pairPos = new THREE.BufferAttribute(new Float32Array(2 * 3), 3);
  pairPos.setUsage(THREE.DynamicDrawUsage);
  pairGeo.setAttribute("position", pairPos);
  const pairLine = new THREE.Line(pairGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 }));
  pairLine.frustumCulled = false;
  pairLine.visible = false;
  scene.add(pairLine);

  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();

  // scratch objects
  const m = new THREE.Matrix4();
  const pos = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const axis = new THREE.Vector3();
  const scl = new THREE.Vector3(1, 1, 1);
  const col = new THREE.Color();

  const update = (positions: Float32Array, orientations: Float32Array, speeds: Float32Array, highlight?: Highlight): void => {
    const count = Math.min(n, (positions.length / 3) | 0);
    lastPositions = positions;
    const hi = highlight ? new Set(highlight.agents) : null;
    for (let i = 0; i < count; i++) {
      // In GPU mode the compute owns instanceMatrix (rendered straight from its buffer); we
      // only maintain per-instance colour here. In CPU mode we compose the matrix too.
      if (!gpuMatrix) {
        const px = positions[i * 3] ?? 0, py = positions[i * 3 + 1] ?? 0, pz = positions[i * 3 + 2] ?? 0;
        pos.set(px, py, pz);
        const ax = orientations[i * 3] ?? 0, ay = orientations[i * 3 + 1] ?? 0, az = orientations[i * 3 + 2] ?? 0;
        const angle = Math.hypot(ax, ay, az);
        if (angle > 1e-6) {
          axis.set(ax / angle, ay / angle, az / angle);
          quat.setFromAxisAngle(axis, angle);
        } else {
          quat.identity();
        }
        const on = hi?.has(i) ?? false;
        scl.setScalar(on ? 2.4 : 1);
        m.compose(pos, quat, scl);
        mesh.setMatrixAt(i, m);
      }
      const on = hi?.has(i) ?? false;
      if (on) {
        col.setRGB(1, 1, 1);
      } else {
        const t = Math.min(1, (speeds[i] ?? 0) / 1.2);
        col.setRGB(0.35 + 0.5 * t, 0.55 + 0.35 * t, 0.9 + 0.1 * t);
      }
      mesh.setColorAt(i, col);
    }
    mesh.count = count;
    if (!gpuMatrix) mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

    // trails
    trail.push(positions);
    const verts = trail.fillSegments(trailPos.array as Float32Array, trailCol.array as Float32Array, TRAIL_HEAD);
    trailGeo.setDrawRange(0, verts);
    trailPos.needsUpdate = true;
    trailCol.needsUpdate = true;

    // pair line
    const pair = highlight?.pair ?? null;
    if (pair && pair[0] < count && pair[1] < count) {
      const a = pair[0], b = pair[1];
      const arr = pairPos.array as Float32Array;
      arr[0] = positions[a * 3] ?? 0; arr[1] = positions[a * 3 + 1] ?? 0; arr[2] = positions[a * 3 + 2] ?? 0;
      arr[3] = positions[b * 3] ?? 0; arr[4] = positions[b * 3 + 1] ?? 0; arr[5] = positions[b * 3 + 2] ?? 0;
      pairPos.needsUpdate = true;
      pairLine.visible = true;
    } else {
      pairLine.visible = false;
    }
  };

  const render = (): void => {
    controls.update();
    void renderer.renderAsync(scene, camera);
  };

  const resize = (width: number, height: number): void => {
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(1, height);
    camera.updateProjectionMatrix();
  };

  const pickVec = new THREE.Vector3();
  const pick = (ndcX: number, ndcY: number): number | null => {
    // GPU mode: instanceMatrix.array is stale (compute writes the GPU buffer), so raycasting
    // won't work — pick the nearest projected dancer from the latest CPU positions instead.
    if (gpuMatrix && lastPositions) {
      const count = Math.min(n, (lastPositions.length / 3) | 0);
      let best: number | null = null;
      let bestD = 0.03; // NDC radius
      for (let i = 0; i < count; i++) {
        pickVec.set(lastPositions[i * 3] ?? 0, lastPositions[i * 3 + 1] ?? 0, lastPositions[i * 3 + 2] ?? 0).project(camera);
        if (pickVec.z < -1 || pickVec.z > 1) continue;
        const dx = pickVec.x - ndcX, dy = pickVec.y - ndcY;
        const dist = Math.hypot(dx, dy);
        if (dist < bestD) {
          bestD = dist;
          best = i;
        }
      }
      return best;
    }
    ndc.set(ndcX, ndcY);
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObject(mesh);
    const hit = hits[0];
    return hit && hit.instanceId !== undefined ? hit.instanceId : null;
  };

  const setGpuMatrix = (on: boolean): void => {
    gpuMatrix = on;
  };

  const isInteracting = (): boolean => now() < interactUntil;

  const gpuInstanceMatrixBuffer = async (): Promise<GPUBuffer | null> => {
    // The attribute's GPUBuffer is created on first render; ensure one has happened.
    const backend = renderer.backend as unknown as { get(o: object): { buffer?: GPUBuffer } | undefined };
    let data = backend.get(mesh.instanceMatrix);
    if (!data?.buffer) {
      await renderer.renderAsync(scene, camera);
      data = backend.get(mesh.instanceMatrix);
    }
    return data?.buffer ?? null;
  };

  const dispose = (): void => {
    controls.dispose();
    geometry.dispose();
    material.dispose();
    mesh.dispose();
    trailGeo.dispose();
    trailMat.dispose();
    pairGeo.dispose();
    renderer.dispose();
  };

  return { renderer, update, render, resize, pick, setGpuMatrix, gpuInstanceMatrixBuffer, isInteracting, dispose };
}
