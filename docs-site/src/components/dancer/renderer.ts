// The 3D stage — three.js on its WebGPU backend (WebGPURenderer). The swarm is one InstancedMesh
// of swept CREATURES — a superegg nose that tapers into a tube tail — built entirely in the shader
// from the sim's own state buffers, no instanceMatrix, no bridge/channel kernels:
//   • positionNode synthesises each vertex (ring, θ) attribute-less: the nose extends forward along
//     the heading from the current position; the body sweeps a tapering tube over the GPU trail ring
//     (the tube IS the trail now — the old line-segment trails are retired). We own the transform.
//   • colour is an okLCH ramp on |vel| (speed), fading head→tail like a comet; girth grows with
//     |angVel| (spin) and thins with speed (fast → thin+long, slow → fat+short).
// The sim computes its state INTO these three-owned storage buffers (dancerGpu.init render buffers),
// so the render reads them with zero readback, including the trail history ring the body follows.
// Hover-highlight rides a few int uniforms. Picking projects the latest CPU position snapshot.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  clamp,
  cos,
  float,
  floor,
  instanceIndex,
  int,
  length,
  max,
  mix,
  normalize,
  oneMinus,
  sin,
  smoothstep,
  storage,
  transformNormalToView,
  uniform,
  varying,
  vec3,
} from "three/tsl";
import { MeshStandardNodeMaterial, StorageBufferAttribute, WebGPURenderer } from "three/webgpu";
import type { RenderStateBuffers } from "../../../../src/gpu/sim/dancerGpu";
import {
  bodyTaper,
  CREATURE_HEAD_SEGMENTS,
  CREATURE_SEGMENTS,
  CREATURE_VERTEX_COUNT,
  creatureCell,
  noseAxial,
  noseRadial,
} from "../../lib/creatureTsl";
import { oklchToLinear } from "../../lib/oklabTsl";
import { orientToForward } from "../../lib/tslTransform";

// the creature assembly builds a shader graph, so its intermediate nodes are loosely typed.
// biome-ignore lint/suspicious/noExplicitAny: three TSL node types are inconsistent (see oklabTsl);
type Tsl = any;

// Trail history depth (frames of position per agent). The GPU sim appends the current position into
// a per-agent ring in a storage buffer three renders straight from — full per-step time resolution.
const TRAIL_CAP = 64;

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
  noseRound: number;
  tubeRadius: number;
  tubeTaper: number;
  thinSpeed: number;
  noseAspect: number;
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

export async function createDancerRenderer(
  canvas: HTMLCanvasElement,
  n: number,
  opts: DancerRendererOptions = {},
): Promise<DancerRenderer> {
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

  // dancers — swept CREATURES synthesised ENTIRELY in the shader from vertexIndex: a superegg NOSE
  // that tapers into a TUBE TAIL swept over the motion trail. The geometry carries no real vertex
  // data, only a zero position attribute for the vertex count; every vertex's (ring, θ) → a point
  // on a circular cross-section whose centreline is read live from the GPU trail-history ring (no
  // readback). The nose sticks forward along the heading; the body follows where the dancer went.
  const geometry = new THREE.BufferGeometry();
  // A zero-filled `position` attribute carries NO geometry — the shader synthesises every vertex —
  // but it gives three the vertex count and satisfies the InstancedMesh setup, which reads
  // positionLocal to fold in instanceMatrix (identity here) *before* our positionNode overrides it.
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(CREATURE_VERTEX_COUNT * 3), 3));
  const SEG = CREATURE_SEGMENTS; // cross-sections along the body (0 = nose tip … SEG = tail)
  const HEAD = CREATURE_HEAD_SEGMENTS; // front segments forming the nose; the rest are the tail
  const BODY = SEG - HEAD; // body segments (follow the trail)
  const AGE_SPAN = TRAIL_CAP - 1; // frames of trail history the body sweeps over

  // trail-history ring — GPU-resident, shared with the compute (which appends each agent's position
  // per step at `trailHead`). The creature's body reads its centreline straight from here; the old
  // line-segment trails are retired (the tube IS the trail now). Buffer/head plumbing stays exposed
  // so the sim can target it and the loop can advance the head.
  const trailHistory = new StorageBufferAttribute(n * TRAIL_CAP, 4); // vec3 in vec4 lanes, 16-byte stride
  const trailHead = uniform(0, "int"); // newest slot; updated each frame
  const trailCapU = uniform(TRAIL_CAP, "int");
  const histNode = storage(trailHistory, "vec4", n * TRAIL_CAP).toReadOnly();

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
  const scaleBase = uniform(0.8); // overall size at rest
  const scaleGain = uniform(0.9); // spin → extra size
  const noseRound = uniform(0.78); // nose profile exponent (superellipse front lobe; 1 = hemispherical)
  const tubeRadius = uniform(0.14); // base body girth at unit size
  const tubeTaper = uniform(1.4); // how sharply the tail thins (higher = pointier)
  const thinSpeed = uniform(0.5); // speed → thinner body (the thin-fast/fat-slow energy counter-balance)
  const noseAspect = uniform(1.6); // nose forward reach as a multiple of girth
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

  // creature dimensions. rMax (girth) SHRINKS with speed — and because a fast dancer also covers
  // more ground per trail frame, its tube grows LONGER: the thin-fast / fat-slow energy balance.
  const rMax = tubeRadius.mul(instScale).div(max(float(1).add(spd.mul(thinSpeed)), 1e-3));
  const noseLen = rMax.mul(noseAspect); // nose forward reach ∝ girth

  const cell = creatureCell(); // (ring, θ) from vertexIndex — attribute-less
  const riN = cell.x; // 0 = nose tip … SEG = tail
  const thetaN = cell.y;
  const agentIdx = int(instanceIndex);

  // Sample the trail ring at a fractional age (0 = newest ≈ current position), linear interpolation.
  // Slots walk back from `trailHead`; ages are clamped so k+1 can't wrap past the oldest frame.
  const sampleTrail = (age: Tsl): Tsl => {
    const a = age.clamp(0, AGE_SPAN);
    const k0: Tsl = int(floor(a));
    const f = a.sub(float(k0));
    const k1 = k0.add(int(1)).min(int(AGE_SPAN));
    const slot0 = trailHead.sub(k0).add(trailCapU).mod(trailCapU);
    const slot1 = trailHead.sub(k1).add(trailCapU).mod(trailCapU);
    const p0 = histNode.element(agentIdx.mul(trailCapU).add(slot0)).xyz;
    const p1 = histNode.element(agentIdx.mul(trailCapU).add(slot1)).xyz;
    return mix(p0, p1, f);
  };

  // Centreline at ring `ri`: the nose (a superegg front lobe extending forward of the current
  // position) for the head segments, the trail path for the body. They meet exactly at the shoulder.
  const centreline = (ri: Tsl): Tsl => {
    const isHead = ri.lessThanEqual(float(HEAD));
    const s = ri.div(float(HEAD)).clamp(0, 1); // 0 tip → 1 shoulder
    const headC = iPos.add(forward.mul(noseAxial(s, noseRound).mul(noseLen)));
    const age = ri.sub(float(HEAD)).clamp(0, BODY).div(float(BODY)).mul(AGE_SPAN);
    return isHead.select(headC, sampleTrail(age));
  };
  // Radius at ring `ri`: the nose lobe grows 0 → rMax, then the tail tapers rMax → 0.
  const radiusAt = (ri: Tsl): Tsl => {
    const isHead = ri.lessThanEqual(float(HEAD));
    const s = ri.div(float(HEAD)).clamp(0, 1);
    const x = ri.sub(float(HEAD)).div(float(BODY)).clamp(0, 1);
    return rMax.mul(isHead.select(noseRadial(s, noseRound), bodyTaper(x, tubeTaper)));
  };

  // Per-ring frame from the local centreline tangent. The cross-section is a circle, so roll about
  // the axis is invisible — a world-up reference frame (orientToForward) is enough, no parallel
  // transport needed. `radial` is the outward direction of this vertex's point on the circle.
  const cHead = centreline(riN.sub(1)); // toward the tip
  const cHere = centreline(riN);
  const cTail = centreline(riN.add(1)); // toward the tail
  const axisVec = cTail.sub(cHead); // tailward tangent (∝ dC/dring)
  const axisLen = max(length(axisVec), 1e-4);
  const tHead = cHead.sub(cTail).div(axisLen); // headward unit tangent → +z of the frame
  const radial = orientToForward(tHead, vec3(cos(thetaN), sin(thetaN), 0));
  const R = radiusAt(riN);

  const material = new MeshStandardNodeMaterial({ roughness: 0.45, metalness: 0.1 });
  material.positionNode = cHere.add(radial.mul(R)); // mesh is untransformed, so this is world space
  // Twist-immune analytic normal of a surface of revolution with a bent axis:
  //   N ∝ radial·|C'| − R'·T_u   (pure radial where the radius is constant, tilting to round the
  // nose and close the tail). Uses only local terms, so the frame's free roll can't corrupt it.
  const dR = radiusAt(riN.add(1)).sub(radiusAt(riN.sub(1)));
  const tU = axisVec.div(axisLen);
  const worldN = normalize(radial.sub(tU.mul(dR.div(axisLen))));
  material.normalNode = transformNormalToView(varying(worldN)).normalize();

  const speedV = varying(spd); // → fragment
  const uV: Tsl = varying(riN.div(float(SEG))); // 0 head → 1 tail
  const hiV = varying(float(isHi));
  const t = clamp(speedV.div(speedRef), 0, 1);
  const baseCol = oklchToLinear(mix(lchSlow, lchFast, t));
  material.colorNode = mix(baseCol.mul(oneMinus(uV.mul(0.7))), vec3(1, 1, 1), hiV); // comet: bright head → dark tail, highlight → white
  material.emissiveNode = baseCol.mul(oneMinus(uV).mul(0.3)); // self-glow strongest at the head

  const mesh = new THREE.InstancedMesh(geometry, material, n); // instanceMatrix stays identity — unused
  mesh.frustumCulled = false; // attribute-less geometry has no bounding sphere; the shader owns positions
  scene.add(mesh);
  let cpuMode = false;
  let lastPositions: Float32Array | null = null;

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
      const a = pair[0],
        b = pair[1];
      const arr = pairPos.array as Float32Array;
      arr[0] = positions[a * 3] ?? 0;
      arr[1] = positions[a * 3 + 1] ?? 0;
      arr[2] = positions[a * 3 + 2] ?? 0;
      arr[3] = positions[b * 3] ?? 0;
      arr[4] = positions[b * 3 + 1] ?? 0;
      arr[5] = positions[b * 3 + 2] ?? 0;
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
      const x = camera.position.x,
        z = camera.position.z;
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
      const dx = pickVec.x - ndcX,
        dy = pickVec.y - ndcY;
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
    noseRound.value = tr.noseRound;
    tubeRadius.value = tr.tubeRadius;
    tubeTaper.value = tr.tubeTaper;
    thinSpeed.value = tr.thinSpeed;
    noseAspect.value = tr.noseAspect;
  };

  const isInteracting = (): boolean => now() < interactUntil;

  const rawBuffer = (attr: object): GPUBuffer | undefined => {
    const backend = renderer.backend as unknown as { get(o: object): { buffer?: GPUBuffer } | undefined };
    return backend.get(attr)?.buffer;
  };

  const gpuStateBuffers = async (): Promise<RenderStateBuffers | null> => {
    // The storage buffers are created when the creature material (whose nodes reference them) first renders.
    const attrs = [posAttr, velAttr, angVelAttr];
    if (attrs.some((a) => !rawBuffer(a))) await renderer.renderAsync(scene, camera);
    const pos = rawBuffer(posAttr),
      vel = rawBuffer(velAttr),
      angVel = rawBuffer(angVelAttr);
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
