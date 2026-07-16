// Hybrid decomposition, rendered (ADR-0013). One implicit model — a plane-based house smooth-unioned
// with a noise-displaced growth — is split by `nonPlanarRegions`: the planar skeleton (the house) is
// meshed exactly by the plane BSP and lit as normal three.js PBR; the non-planar region (the growth
// and the blend seam around it) is raymarched, confined to its reported box, and composited against
// the mesh through the shared depth buffer. Neither renderer sees the other's geometry — they meet at
// the box edge, where the field is still the bare house, so the seam is invisible.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  Break,
  cameraFar,
  cameraNear,
  cameraPosition,
  cameraViewMatrix,
  clamp,
  Discard,
  dot,
  Fn,
  float,
  If,
  Loop,
  max,
  min,
  mix,
  normalize,
  normalWorld,
  positionWorld,
  struct,
  uniform,
  vec3,
  vec4,
  viewZToPerspectiveDepth,
  wgslFn,
} from "three/tsl";
import { MeshBasicNodeMaterial, WebGPURenderer } from "three/webgpu";
import {
  type AABB,
  aabbFinite,
  brepToMesh,
  evaluateBrep,
  type Implicit,
  mergeCoplanar,
  nonPlanarRegions,
  planarSkeleton,
} from "../../src/geometry";
import { hybridGrowth, hybridHouse } from "./geometryShapes";

const RayResult = struct({ color: "vec4", depth: "float" });
const BOUNDS = 1.8;
// The raymarch drives the noise animation; `uNoiseTime` in the codegen'd field reads this each frame.
const uTime = uniform(0);

// One shading node, used for BOTH the meshed house and the raymarched growth, so they render
// identically (same albedo, same key-light + hemisphere, same node-material output path — no
// PBR-vs-approximation or tone-mapping mismatch). Our lights are directional, so shading needs only the
// surface normal; the mesh feeds its geometry normal, the raymarch feeds its gradient normal.
const HOUSE_COLOR = vec3(0.78, 0.7, 0.6); // ≈ 0xc7b299
function shade(n: ReturnType<typeof vec3>): ReturnType<typeof vec3> {
  const lig = normalize(vec3(3, 5, 2)); // matches the scene key-light direction
  const dif = clamp(dot(n, lig), 0, 1).mul(1.05);
  const hemi = clamp(n.y.mul(0.5).add(0.5), 0, 1);
  const amb = mix(vec3(0.14, 0.11, 0.08), vec3(0.5, 0.56, 0.72), hemi).mul(0.55);
  return HOUSE_COLOR.mul(dif.add(amb));
}

/** A raymarch material for one non-planar region: sphere-traces the FULL model, but only within `box`
 *  (a ray-box slab bounds the march), and shades to match the meshed house so the two read as one
 *  surface across the smooth-union seam. */
function regionMaterial(model: Implicit, box: AABB): MeshBasicNodeMaterial {
  const sdField = wgslFn(`
fn sdField(p: vec3<f32>, t: f32) -> f32 { uNoiseTime = t; return sdScene(p); }
${model.toWgsl({ bakeConstants: true })}
`);
  const bmin = vec3(box.min[0], box.min[1], box.min[2]);
  const bmax = vec3(box.max[0], box.max[1], box.max[2]);
  const mat = new MeshBasicNodeMaterial();

  const march = Fn(() => {
    const ro = cameraPosition;
    const rd = normalize(positionWorld.sub(cameraPosition));

    // Ray vs the region box (slab test) → the [tNear, tFar] the march is allowed to live in. Outside
    // it the field is the bare (meshed) house, so there is nothing here to draw.
    const t1 = bmin.sub(ro).div(rd);
    const t2 = bmax.sub(ro).div(rd);
    const tmn = min(t1, t2);
    const tmx = max(t1, t2);
    const tNear = max(max(tmn.x, tmn.y), tmn.z);
    const tFar = min(min(tmx.x, tmx.y), tmx.z);
    Discard(tNear.greaterThan(tFar)); // ray misses the box

    // Under-relaxed sphere-trace (the noise/blend field over-estimates distance), confined to the box.
    const relax = float(0.85);
    const hitEps = float(0.0012);
    const t = max(tNear, float(0)).toVar();
    const tPrev = t.toVar();
    const dPrev = float(1e9).toVar();
    const hit = float(0).toVar();
    const tHit = float(0).toVar();
    Loop(160, () => {
      const p = ro.add(rd.mul(t));
      const d = sdField({ p, t: uTime });
      If(d.lessThan(hitEps), () => {
        hit.assign(1);
        const f = dPrev.div(max(dPrev.sub(d), float(1e-5)));
        tHit.assign(tPrev.add(t.sub(tPrev).mul(f)));
        Break();
      });
      tPrev.assign(t);
      dPrev.assign(d);
      t.addAssign(max(d.mul(relax), float(0.002)));
      If(t.greaterThan(tFar), () => {
        Break();
      });
    });
    Discard(hit.equal(0)); // ray crossed the box without hitting → let the mesh / background show
    const pHit = ro.add(rd.mul(tHit));

    // Tetrahedral gradient normal.
    const e = float(0.0015);
    const k0 = vec3(1, -1, -1);
    const k1 = vec3(-1, -1, 1);
    const k2 = vec3(-1, 1, -1);
    const k3 = vec3(1, 1, 1);
    const n = normalize(
      k0
        .mul(sdField({ p: pHit.add(k0.mul(e)), t: uTime }))
        .add(k1.mul(sdField({ p: pHit.add(k1.mul(e)), t: uTime })))
        .add(k2.mul(sdField({ p: pHit.add(k2.mul(e)), t: uTime })))
        .add(k3.mul(sdField({ p: pHit.add(k3.mul(e)), t: uTime }))),
    );

    // The SAME shade node the meshed house uses — so the growth is the same material, seamlessly.
    const col = shade(n);

    const hitViewZ = cameraViewMatrix.mul(vec4(pHit, 1)).z;
    const depth = viewZToPerspectiveDepth(hitViewZ, cameraNear, cameraFar);
    return RayResult(vec4(col, 1), depth);
  })();

  mat.colorNode = march.get("color");
  mat.depthNode = march.get("depth");
  return mat;
}

/** A BufferGeometry from a plane-BSP triangle mesh. */
function toBufferGeometry(iso: ReturnType<typeof brepToMesh>): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(iso.positions, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(iso.normals, 3));
  g.setIndex(new THREE.BufferAttribute(iso.indices, 1));
  return g;
}

/** Clamp an (possibly unbounded) region to a proxy-sizeable box. */
function proxyBox(r: AABB): AABB {
  if (aabbFinite(r)) return r;
  const s = BOUNDS * 1.5;
  return { min: [-s, -s, -s], max: [s, s, s] };
}

function fail(msg: string): void {
  const err = document.getElementById("err");
  if (err) {
    err.style.display = "grid";
    err.innerHTML = msg;
  }
}

async function main(): Promise<void> {
  const canvas = document.getElementById("stage") as HTMLCanvasElement | null;
  if (!canvas) return;
  if (!navigator.gpu) {
    fail("This view renders with <b>WebGPU</b>. Try a recent Chrome, Edge, or Safari.");
    return;
  }

  const renderer = new WebGPURenderer({ canvas, antialias: true });
  await renderer.init().catch((e) => {
    fail(`Couldn’t initialise WebGPU: ${e}`);
    throw e;
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1020);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.02, 100);
  camera.position.set(3.4, 2.2, 4.1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.target.set(0, 0.5, 0);

  scene.add(new THREE.HemisphereLight(0x9fb4ff, 0x20160c, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(3, 5, 2);
  scene.add(key);

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), new THREE.MeshStandardMaterial({ color: 0x2a3350, roughness: 0.95 }));
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  // The one model: a plane-based house smooth-unioned with a noise growth. `k` blends the growth into
  // the roof rather than letting it merely rest on it.
  const model = hybridHouse().smoothUnion(hybridGrowth(), 0.16);

  // Mesh the planar skeleton (the house) with the BSP; a slight polygon offset lets the region raymarch
  // win the depth test where the two overlap inside the box.
  const skel = planarSkeleton(model.node);
  if (!skel) {
    fail("Model has no planar skeleton to mesh.");
    return;
  }
  const brep = mergeCoplanar(evaluateBrep(skel, { bounds: BOUNDS }));

  // Same node shading as the raymarch (via the geometry normal), so mesh and raymarch are one material.
  const houseMat = new MeshBasicNodeMaterial();
  houseMat.colorNode = shade(normalWorld);
  houseMat.polygonOffset = true; // let the region raymarch (and the wire) win the depth test where they overlap
  houseMat.polygonOffsetFactor = 1;
  houseMat.polygonOffsetUnits = 1;
  const houseGeom = toBufferGeometry(brepToMesh(brep));
  const house = new THREE.Mesh(houseGeom, houseMat);
  scene.add(house);

  // Raw-triangulation wireframe overlay (every triangle, fan diagonals and all — the technical-artist
  // view of the actual mesh, not just clean feature edges). Depth-tested, so the raymarched growth
  // occludes the triangles behind it — you can read where the mesh hands off to the raymarch; the
  // house's polygon offset keeps the lines above its own faces.
  const wire = new THREE.LineSegments(new THREE.WireframeGeometry(houseGeom), new THREE.LineBasicMaterial({ color: 0x18324e }));
  wire.visible = false;
  scene.add(wire);

  // One raymarch proxy per non-planar region — sized to the region box; its fragments march the model.
  const regions = nonPlanarRegions(model.node);
  const proxies: THREE.Mesh[] = [];
  const helpers: THREE.Box3Helper[] = [];
  for (const r of regions) {
    const b = proxyBox(r);
    const size: [number, number, number] = [b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]];
    const proxy = new THREE.Mesh(new THREE.BoxGeometry(size[0], size[1], size[2]), regionMaterial(model, b));
    proxy.position.set((b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2);
    scene.add(proxy);
    proxies.push(proxy);
    const helper = new THREE.Box3Helper(
      new THREE.Box3(new THREE.Vector3(...b.min), new THREE.Vector3(...b.max)),
      new THREE.Color(0xff8a3c),
    );
    helper.visible = false;
    scene.add(helper);
    helpers.push(helper);
  }

  // ── UI ────────────────────────────────────────────────────────────────────────────────
  const houseBox = document.getElementById("house") as HTMLInputElement;
  const wireBox = document.getElementById("wire") as HTMLInputElement;
  const regionBox = document.getElementById("region") as HTMLInputElement;
  houseBox.addEventListener("change", () => {
    house.visible = houseBox.checked;
  });
  wireBox.addEventListener("change", () => {
    wire.visible = wireBox.checked;
  });
  regionBox.addEventListener("change", () => {
    for (const h of helpers) h.visible = regionBox.checked;
  });

  function resize(): void {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h || 1;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  const start = performance.now();
  renderer.setAnimationLoop(() => {
    uTime.value = (performance.now() - start) / 1000; // seconds → drifts the noise domain
    controls.update();
    renderer.render(scene, camera);
  });
}

main().catch((e) => fail(`Error: ${e}`));
