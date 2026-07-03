// The 3D stage — three.js on its WebGPU backend (WebGPURenderer). The swarm is one InstancedMesh
// of small oriented cones whose per-instance model transform AND visual traits are built entirely
// in the shader from the sim's own state buffers — no instanceMatrix, no bridge/channel kernels:
//   • positionNode reads pos + angPos per instance and composes the transform explicitly (scale the
//     local cone vertex → Rodrigues-rotate by angPos → translate by pos). We own the transform, so
//     it's the level the superegg deformation will also work at.
//   • colour is an okLCH ramp on |vel| (speed); per-instance scale grows with |angVel| (spin).
// The sim computes its state INTO these three-owned storage buffers (dancerGpu.init render buffers),
// so the render reads them with zero readback. Trails are the same idea: read the GPU history ring.
// Hover-highlight rides a few int uniforms. Picking projects the latest CPU position snapshot.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { clamp, float, instanceIndex, int, length, max, mix, normalize, oneMinus, positionGeometry, smoothstep, storage, uniform, varying, vec3, vertexIndex } from "three/tsl";
import { LineBasicNodeMaterial, MeshStandardNodeMaterial, StorageBufferAttribute, WebGPURenderer } from "three/webgpu";
import type { RenderStateBuffers } from "../../../../src/gpu/sim/dancerGpu";
import { srgbToOklab } from "../../../../src/color/oklab";
import { oklabToLinear, oklchToLinear } from "../../lib/oklabTsl";
import { orientToForward } from "../../lib/tslTransform";

// Trail history depth (frames of position per agent). The GPU sim appends the current position into
// a per-agent ring in a storage buffer three renders straight from — full per-step time resolution.
const TRAIL_CAP = 64;
const TRAIL_HEAD_COL: [number, number, number] = [0.5, 0.72, 1.0]; // newest end
const TRAIL_TAIL_COL: [number, number, number] = [0.12, 0.16, 0.42]; // oldest end

/** Breedable appearance traits the renderer maps from motion (subset of DancerParams). */
export interface RenderTraits {
  hueSlow: number;
  hueFast: number;
  chroma: number;
  lightSlow: number;
  lightFast: number;
  speedRef: number;
  sizeBase: number;
  sizeSpin: number;
}

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
  /** The three-owned per-instance state buffers (materialising them with a render if needed), for
   *  the sim to compute its pos/angPos/vel/angVel INTO. Null if not yet available. */
  gpuStateBuffers(): Promise<RenderStateBuffers | null>;
  /** CPU-fallback: when on, `update()` uploads pos+orientation into the state buffers so the same
   *  shader renders the CPU sim (GPU mode leaves them to the compute). */
  setCpuMode(on: boolean): void;
  /** Drive the colour/size mapping from the dancer's breedable render traits (call when they change
   *  — e.g. adopting a bred specimen). */
  setRenderTraits(t: RenderTraits): void;
  /** True while the camera is being manipulated (plus a short damping cooldown). */
  isInteracting(): boolean;
  /** The raw GPUBuffer backing the trail history ring (materialising it with a render if needed),
   *  so our compute can append the current position each step. Null if not yet available. */
  gpuTrailBuffer(): Promise<GPUBuffer | null>;
  /** Ring capacity (frames of history per agent). */
  trailCapacity(): number;
  /** Point the trail shader at the newest-written slot; walks back from here. Call each frame. */
  setTrailHead(head: number): void;
  dispose(): void;
}

/** Options for a dancer view. A cell in a grid shares one central GPUDevice (many canvases, one
 *  device — the webgpu multipleCanvases pattern, via three's WebGPUBackendParameters.device) and is
 *  non-interactive (auto-rotates instead of OrbitControls). */
export interface DancerRendererOptions {
  device?: GPUDevice;
  interactive?: boolean; // default true
}

export async function createDancerRenderer(canvas: HTMLCanvasElement, n: number, opts: DancerRendererOptions = {}): Promise<DancerRenderer> {
  const interactive = opts.interactive ?? true;
  const renderer = new WebGPURenderer({ canvas, antialias: true, alpha: true, ...(opts.device ? { device: opts.device } : {}) });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  await renderer.init();

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
  camera.position.set(0, 6, 16);

  // Interactive stage: OrbitControls. Non-interactive cell: no controls, a gentle auto-rotate.
  const controls = interactive ? new OrbitControls(camera, canvas) : null;
  if (controls) {
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);
  }

  // Track active camera interaction so the caller can suspend the (periodic, GPU-syncing) snapshot
  // readback while dragging — the readback's mapAsync contends with the drag render (regular pauses).
  let interactUntil = 0;
  const now = (): number => (typeof performance !== "undefined" ? performance.now() : 0);
  if (controls) {
    controls.addEventListener("start", () => {
      interactUntil = Number.POSITIVE_INFINITY;
    });
    controls.addEventListener("end", () => {
      interactUntil = now() + 400;
    });
  }

  scene.add(new THREE.AmbientLight(0x6070a0, 1.2));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(6, 10, 8);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x4060ff, 1.0);
  rim.position.set(-8, -4, -6);
  scene.add(rim);

  // dancers — oriented cones (tip → +z), transformed + coloured in-shader from the sim state buffers
  const geometry = new THREE.ConeGeometry(0.1, 0.2, 10);
  geometry.rotateX(Math.PI / 2);

  // per-instance sim state, three-owned so the sim (writes) and the render (reads) share the buffers
  const posAttr = new StorageBufferAttribute(n, 4);
  const velAttr = new StorageBufferAttribute(n, 4);
  const angVelAttr = new StorageBufferAttribute(n, 4);
  const posNode = storage(posAttr, "vec4", n).toReadOnly();
  const velNode = storage(velAttr, "vec4", n).toReadOnly();
  const angVelNode = storage(angVelAttr, "vec4", n).toReadOnly();

  // mapping uniforms — driven by the dancer's breedable RENDER TRAITS (setRenderTraits), so a bred
  // specimen changes how it looks. okLCH endpoints are (lightness, chroma, hue-radians).
  const lchSlow = uniform(new THREE.Vector3(0.52, 0.14, 4.6)); // colour at rest
  const lchFast = uniform(new THREE.Vector3(0.86, 0.14, 3.5)); // colour when moving
  const speedRef = uniform(1.2); // speed reaching the top of the colour ramp
  const scaleBase = uniform(0.8); // size at rest
  const scaleGain = uniform(0.9); // spin → extra size
  const hlA = uniform(-1, "int"); // up to three highlighted instances (hovered dancer + matrix pair)
  const hlB = uniform(-1, "int");
  const hlC = uniform(-1, "int");

  const iPos = posNode.element(instanceIndex).xyz; // vertex stage
  const iVel = velNode.element(instanceIndex).xyz;
  const spd = length(iVel);
  const spin = length(angVelNode.element(instanceIndex).xyz);
  const ii = int(instanceIndex);
  const isHi = ii.equal(hlA).or(ii.equal(hlB)).or(ii.equal(hlC));
  const instScale = scaleBase.add(spin.mul(scaleGain)).mul(float(isHi).mul(0.7).add(1)); // highlight ×1.7

  // facing: point the cone (+z tip) ALONG velocity; at near-zero speed blend to radial (outward) so a
  // settled swarm keeps varied headings instead of all snapping to one axis. Roll about the axis is
  // free for now (angular-momentum 'up' is future work). /max keeps the directions NaN-free at 0.
  const velDir = iVel.div(max(spd, 1e-4));
  const radialDir = iPos.div(max(length(iPos), 1e-4));
  const forward = normalize(mix(radialDir, velDir, smoothstep(0.008, 0.05, spd)).add(vec3(0, 0, 1e-4)));

  const material = new MeshStandardNodeMaterial({ roughness: 0.4, metalness: 0.1 });
  // explicit model transform: scale local vertex → orient +z along `forward` → translate by pos
  material.positionNode = orientToForward(forward, positionGeometry.mul(instScale)).add(iPos);
  // NOTE: per-instance normal rotation is deferred — feeding the vertex-stage `forward` into the
  // fragment-stage normalNode tripped a bad render-target allocation in three. Lighting uses the
  // unrotated geometry normal for now (fine for small cones); revisit for the superegg surfaces.

  const speedV = varying(spd); // → fragment
  const hiV = varying(float(isHi));
  const t = clamp(speedV.div(speedRef), 0, 1);
  const baseCol = oklchToLinear(mix(lchSlow, lchFast, t));
  material.colorNode = mix(baseCol, vec3(1, 1, 1), hiV); // highlighted → white
  material.emissiveNode = baseCol.mul(0.3); // gentle self-glow tracking the mapped colour

  const mesh = new THREE.InstancedMesh(geometry, material, n); // instanceMatrix stays identity — unused
  scene.add(mesh);
  let cpuMode = false;
  let lastPositions: Float32Array | null = null;

  // trails — GPU-resident. The compute appends each agent's position into a per-agent ring in
  // `trailHistory`; the render draws n·(cap-1) line segments whose vertex positions are read straight
  // from that buffer by a TSL positionNode (walking the ring from `trailHead`), colour + opacity by
  // trail age. No CPU snapshot, no vertex attributes — setDrawRange sets the count (zero vertex memory).
  const SEGS = TRAIL_CAP - 1;
  const PER_AGENT = SEGS * 2;
  const trailVerts = n * PER_AGENT;
  const trailHistory = new StorageBufferAttribute(n * TRAIL_CAP, 4); // vec3f, 16-byte stride
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setDrawRange(0, trailVerts);

  const trailHead = uniform(0, "int"); // newest slot; updated each frame
  const trailCapU = uniform(TRAIL_CAP, "int");
  const histNode = storage(trailHistory, "vec4", n * TRAIL_CAP).toReadOnly();
  const vi = int(vertexIndex);
  const perAgent = int(PER_AGENT);
  const agent = vi.div(perAgent);
  const local = vi.mod(perAgent);
  const age = local.div(int(2)).add(local.mod(int(2))); // segment endpoints → 0,1,1,2,2,… (0 = newest)
  const slot = trailHead.sub(age).add(trailCapU).mod(trailCapU); // walk back through the ring
  const idx = agent.mul(trailCapU).add(slot);
  const ageFrac = varying(float(age).div(float(TRAIL_CAP - 1))); // 0 newest → 1 oldest, interpolated

  const headLab = srgbToOklab(TRAIL_HEAD_COL); // endpoints pre-converted to okLab on the CPU
  const tailLab = srgbToOklab(TRAIL_TAIL_COL);
  const trailMat = new LineBasicNodeMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  trailMat.positionNode = histNode.element(idx).xyz;
  trailMat.colorNode = oklabToLinear(mix(vec3(...headLab), vec3(...tailLab), ageFrac)); // mix in okLab → linear
  trailMat.opacityNode = oneMinus(ageFrac).pow(1.5); // fade toward the tail
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

  const update = (positions: Float32Array, _orientations: Float32Array, _speeds: Float32Array, highlight?: Highlight): void => {
    const count = Math.min(n, (positions.length / 3) | 0);
    lastPositions = positions;

    // Highlight rides three int uniforms (hovered dancer + a hovered matrix pair = ≤3), read by the
    // colour + scale nodes — no per-instance CPU colour upload.
    const ag = highlight?.agents ?? [];
    hlA.value = ag[0] ?? -1;
    hlB.value = ag[1] ?? -1;
    hlC.value = ag[2] ?? -1;

    // GPU mode: the compute owns the state buffers (pose + traits read straight from them). CPU
    // fallback: upload positions so the same shader renders the CPU sim (vel/angVel stay seeded →
    // base colour/scale and radial facing — a degraded but functional safety net).
    if (cpuMode) {
      const parr = posAttr.array as Float32Array;
      for (let i = 0; i < count; i++) {
        parr[i * 4] = positions[i * 3] ?? 0;
        parr[i * 4 + 1] = positions[i * 3 + 1] ?? 0;
        parr[i * 4 + 2] = positions[i * 3 + 2] ?? 0;
      }
      posAttr.needsUpdate = true;
    }
    mesh.count = count;

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
    if (controls) {
      controls.update();
    } else {
      // gentle auto-rotate around the swarm (non-interactive cell view)
      const a = 0.0035;
      const x = camera.position.x, z = camera.position.z;
      camera.position.x = x * Math.cos(a) - z * Math.sin(a);
      camera.position.z = x * Math.sin(a) + z * Math.cos(a);
      camera.lookAt(0, 0, 0);
    }
    void renderer.renderAsync(scene, camera);
  };

  const resize = (width: number, height: number): void => {
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(1, height);
    camera.updateProjectionMatrix();
  };

  // Picking: the pose lives in GPU buffers (instanceMatrix is unused, geometry is shader-deformed),
  // so raycasting can't see it — pick the nearest projected dancer from the latest CPU position snapshot.
  const pickVec = new THREE.Vector3();
  const pick = (ndcX: number, ndcY: number): number | null => {
    if (!lastPositions) return null;
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
  };

  const setCpuMode = (on: boolean): void => {
    cpuMode = on;
  };

  const setRenderTraits = (tr: RenderTraits): void => {
    lchSlow.value.set(tr.lightSlow, tr.chroma, tr.hueSlow);
    lchFast.value.set(tr.lightFast, tr.chroma, tr.hueFast);
    speedRef.value = tr.speedRef;
    scaleBase.value = tr.sizeBase;
    scaleGain.value = tr.sizeSpin;
  };

  const isInteracting = (): boolean => now() < interactUntil;

  const rawBuffer = (attr: object): GPUBuffer | undefined => {
    const backend = renderer.backend as unknown as { get(o: object): { buffer?: GPUBuffer } | undefined };
    return backend.get(attr)?.buffer;
  };

  const gpuStateBuffers = async (): Promise<RenderStateBuffers | null> => {
    // The storage buffers are created when the cone material (whose nodes reference them) first renders.
    const attrs = [posAttr, velAttr, angVelAttr];
    if (attrs.some((a) => !rawBuffer(a))) await renderer.renderAsync(scene, camera);
    const pos = rawBuffer(posAttr), vel = rawBuffer(velAttr), angVel = rawBuffer(angVelAttr);
    if (!pos || !vel || !angVel) return null;
    return { pos, vel, angVel };
  };

  const gpuTrailBuffer = async (): Promise<GPUBuffer | null> => {
    if (!rawBuffer(trailHistory)) await renderer.renderAsync(scene, camera);
    return rawBuffer(trailHistory) ?? null;
  };

  const trailCapacity = (): number => TRAIL_CAP;
  const setTrailHead = (head: number): void => {
    trailHead.value = head;
  };

  const dispose = (): void => {
    controls?.dispose();
    geometry.dispose();
    material.dispose();
    mesh.dispose();
    trailGeo.dispose();
    trailMat.dispose();
    pairGeo.dispose();
    renderer.dispose();
  };

  return {
    renderer,
    update,
    render,
    resize,
    pick,
    gpuStateBuffers,
    setCpuMode,
    setRenderTraits,
    isInteracting,
    gpuTrailBuffer,
    trailCapacity,
    setTrailHead,
    dispose,
  };
}
