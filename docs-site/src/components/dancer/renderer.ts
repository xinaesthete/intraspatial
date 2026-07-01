// The 3D stage — three.js on its WebGPU backend (WebGPURenderer). The swarm is one
// InstancedMesh of small oriented shapes (cones) so each dancer's facing/spin reads, not
// just a cloud of dots. In this phase the CPU sim (sim.ts) writes the per-instance
// matrices each frame; Phase 5 moves the sim onto the GPU (TSL compute over storage
// buffers) and the mesh reads those buffers directly. Orbit camera, soft lighting, a
// little emissive glow.
import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export interface DancerRenderer {
  readonly renderer: WebGPURenderer;
  update(positions: Float32Array, orientations: Float32Array, speeds: Float32Array): void;
  render(): void;
  resize(width: number, height: number): void;
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

  scene.add(new THREE.AmbientLight(0x6070a0, 1.2));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(6, 10, 8);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x4060ff, 1.0);
  rim.position.set(-8, -4, -6);
  scene.add(rim);

  // A cone pointing +z (its local forward), so orientation is visible.
  const geometry = new THREE.ConeGeometry(0.14, 0.5, 10);
  geometry.rotateX(Math.PI / 2); // tip → +z
  const material = new THREE.MeshStandardMaterial({
    color: 0x9fc4ff,
    emissive: 0x2a4a8a,
    emissiveIntensity: 0.6,
    roughness: 0.4,
    metalness: 0.1,
  });

  const mesh = new THREE.InstancedMesh(geometry, material, n);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
  scene.add(mesh);

  // scratch objects to avoid per-frame allocation
  const m = new THREE.Matrix4();
  const pos = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const axis = new THREE.Vector3();
  const scl = new THREE.Vector3(1, 1, 1);
  const col = new THREE.Color();

  const update = (positions: Float32Array, orientations: Float32Array, speeds: Float32Array): void => {
    const count = Math.min(n, (positions.length / 3) | 0);
    for (let i = 0; i < count; i++) {
      pos.set(positions[i * 3] ?? 0, positions[i * 3 + 1] ?? 0, positions[i * 3 + 2] ?? 0);
      // angle-axis rotation vector → quaternion
      const ax = orientations[i * 3] ?? 0, ay = orientations[i * 3 + 1] ?? 0, az = orientations[i * 3 + 2] ?? 0;
      const angle = Math.hypot(ax, ay, az);
      if (angle > 1e-6) {
        axis.set(ax / angle, ay / angle, az / angle);
        quat.setFromAxisAngle(axis, angle);
      } else {
        quat.identity();
      }
      m.compose(pos, quat, scl);
      mesh.setMatrixAt(i, m);
      // colour by speed: slow = deep blue, fast = bright cyan/white
      const t = Math.min(1, (speeds[i] ?? 0) / 1.2);
      col.setRGB(0.35 + 0.5 * t, 0.55 + 0.35 * t, 0.9 + 0.1 * t);
      mesh.setColorAt(i, col);
    }
    mesh.count = count;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
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

  const dispose = (): void => {
    controls.dispose();
    geometry.dispose();
    material.dispose();
    mesh.dispose();
    renderer.dispose();
  };

  return { renderer, update, render, resize, dispose };
}
