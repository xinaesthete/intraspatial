// Interactive MESH view of the implicit (SDF/CSG) geometry-kind (ADR-0010). The GEOMETRY is ours:
// `Implicit.toMesh` (CPU surface nets / dual contouring) produces the mesh; this file only hands the
// vertex data to three.js and shades it with a STANDARD PBR material — the deliberate boundary
// (WebGPU-first / TSL-boundary principle): three.js is presentation only, no geometry logic in TSL.
//
// The raymarch view (`raymarch.html` / `raymarchMain.ts`) renders the same shapes exactly (no mesh).
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { WebGPURenderer } from "three/webgpu";
import { brepEdges, brepToMesh, evaluateBrep, type IsoMesh, mergeCoplanar } from "../../src/geometry";
import { SHAPES } from "./geometryShapes";

const SPHERE_ORANGE = 0xff8a3c;

function toBufferGeometry(m: IsoMesh): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(m.positions, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(m.normals, 3));
  g.setIndex(new THREE.BufferAttribute(m.indices, 1));
  return g;
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

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  camera.position.set(2.7, 2.7, 2.9);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.target.set(0, 0.4, 0.25);

  // Lighting — a key/fill/hemisphere rig so the PBR material reads well.
  scene.add(new THREE.HemisphereLight(0x9fb4ff, 0x201509, 1.0));
  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(3, 5, 2);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x88aaff, 0.7);
  fill.position.set(-3, 1, -2);
  scene.add(fill);

  const material = new THREE.MeshStandardMaterial({ color: SPHERE_ORANGE, roughness: 0.5, metalness: 0.06, flatShading: false });
  const wireMaterial = new THREE.MeshBasicMaterial({ color: 0xbfe0ff, wireframe: true });
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
  scene.add(mesh);
  // Feature edges (BSP only): the merged faces' outlines as clean line-work — no fan diagonals, no
  // interior split-edges. Shown in place of triangle-wireframe when the exact mesher is active.
  const edges = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xbfe0ff }));
  edges.visible = false;
  scene.add(edges);

  // ── UI ────────────────────────────────────────────────────────────────────────────────
  const shapeSel = document.getElementById("shape") as HTMLSelectElement;
  const mesherSel = document.getElementById("mesher") as HTMLSelectElement;
  const mergeBox = document.getElementById("merge") as HTMLInputElement;
  const resInput = document.getElementById("res") as HTMLInputElement;
  const resVal = document.getElementById("resv") as HTMLSpanElement;
  const sharpenBox = document.getElementById("sharpen") as HTMLInputElement;
  const wireBox = document.getElementById("wire") as HTMLInputElement;
  const stat = document.getElementById("stat") as HTMLDivElement;
  SHAPES.forEach((s, i) => {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = s.name;
    shapeSel.appendChild(o);
  });

  // Feature edges are only meaningful for the BSP; the grid falls back to triangle-wireframe.
  let hasEdges = false;
  function applyWire(): void {
    const wire = wireBox.checked;
    if (wire && hasEdges) {
      mesh.visible = false;
      edges.visible = true;
    } else {
      mesh.visible = true;
      edges.visible = false;
      mesh.material = wire ? wireMaterial : material;
    }
  }

  function rebuild(): void {
    const shape = SHAPES[Number(shapeSel.value)] ?? SHAPES[0];
    if (!shape) return;
    const res = Number(resInput.value);
    resVal.textContent = String(res);
    const sharpen = sharpenBox.checked;
    // The merge toggle only applies to the BSP; grid-DC options only to the grid. Grey out the inapplicable.
    const bsp = mesherSel.value === "bsp";
    resInput.disabled = bsp;
    sharpenBox.disabled = bsp;
    mergeBox.disabled = !bsp;

    let iso: IsoMesh;
    let method: string;
    const t0 = performance.now();
    if (bsp) {
      try {
        const raw = evaluateBrep(shape.make().node, { bounds: shape.bounds });
        const brep = mergeBox.checked ? mergeCoplanar(raw) : raw;
        iso = brepToMesh(brep);
        edges.geometry.dispose();
        const eg = new THREE.BufferGeometry();
        eg.setAttribute("position", new THREE.BufferAttribute(brepEdges(brep), 3));
        edges.geometry = eg;
        hasEdges = true;
        method = `plane BSP (exact) · ${brep.faces.length} faces${mergeBox.checked ? "" : " (raw)"}`;
      } catch (e) {
        // Non-polyhedral (curved/smooth) shapes: fall back to the grid so the view isn't empty, and
        // say why the exact mesher declined.
        iso = shape.make().toMesh({ bounds: shape.bounds, res, sharpen });
        hasEdges = false;
        mesh.geometry.dispose();
        mesh.geometry = toBufferGeometry(iso);
        applyWire();
        stat.textContent = `BSP declined (${String(e).replace(/^Error:\s*bsp:\s*/, "")}) — showing surface nets · ${(iso.indices.length / 3).toLocaleString()} tris`;
        return;
      }
    } else {
      iso = shape.make().toMesh({ bounds: shape.bounds, res, sharpen });
      hasEdges = false;
      method = sharpen ? "dual contouring" : "surface nets";
    }
    const ms = performance.now() - t0;
    mesh.geometry.dispose();
    mesh.geometry = toBufferGeometry(iso);
    applyWire();
    stat.textContent = `${iso.vertexCount.toLocaleString()} verts · ${(iso.indices.length / 3).toLocaleString()} tris · extracted in ${ms.toFixed(0)} ms (CPU ${method})`;
  }
  shapeSel.addEventListener("change", rebuild);
  mesherSel.addEventListener("change", rebuild);
  mergeBox.addEventListener("change", rebuild);
  resInput.addEventListener("input", rebuild);
  sharpenBox.addEventListener("change", rebuild);
  wireBox.addEventListener("change", applyWire);
  rebuild();

  function resize(): void {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h || 1;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
}

main().catch((e) => fail(`Error: ${e}`));
