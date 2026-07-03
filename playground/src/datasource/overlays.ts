// Overlays (ADR-0008: framework-free three.js Object3D factories, reusable by the
// production renderer's debug mode and by the Decision view). Slice 1 draws the
// selected chunks as level-tinted wireframe boxes plus a faint dataset bound — the
// receding-resolution gradient made visible, driven entirely by the pure `select()`.
import * as THREE from "three";
import { chunkWorldAabb, worldAabbOfArrayBox, type Multiscale, type Selection } from "../../../src/datasource";

/** okLCH-ish ramp by level: fine = warm & bright, coarse = cool & dim. */
export function levelColor(level: number, maxLevel: number): THREE.Color {
  const t = maxLevel > 0 ? level / maxLevel : 0;
  const hue = 0.08 + t * 0.55; // orange → blue
  const light = 0.62 - t * 0.3;
  return new THREE.Color().setHSL(hue, 0.75, Math.max(0.15, light));
}

const BOX_EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 0], // bottom
  [4, 5], [5, 6], [6, 7], [7, 4], // top
  [0, 4], [1, 5], [2, 6], [3, 7], // verticals
];

function pushBoxEdges(out: number[], min: THREE.Vector3Like, max: THREE.Vector3Like): void {
  const corners: number[][] = [
    [min.x, min.y, min.z], [max.x, min.y, min.z], [max.x, max.y, min.z], [min.x, max.y, min.z],
    [min.x, min.y, max.z], [max.x, min.y, max.z], [max.x, max.y, max.z], [min.x, max.y, max.z],
  ];
  for (const [a, b] of BOX_EDGES) {
    const ca = corners[a], cb = corners[b];
    if (!ca || !cb) continue;
    out.push(ca[0] ?? 0, ca[1] ?? 0, ca[2] ?? 0, cb[0] ?? 0, cb[1] ?? 0, cb[2] ?? 0);
  }
}

/** A group of level-tinted wireframe boxes for a Selection (one LineSegments per level). */
export function chunkOverlays(ms: Multiscale, sel: Selection): THREE.Group {
  const group = new THREE.Group();
  const maxLevel = ms.levelCount - 1;
  const byLevel = new Map<number, number[]>();
  for (const c of sel.chunks) {
    const box = chunkWorldAabb(ms, c.id);
    const arr = byLevel.get(c.id.level) ?? [];
    pushBoxEdges(arr, { x: box.min[0], y: box.min[1], z: box.min[2] }, { x: box.max[0], y: box.max[1], z: box.max[2] });
    byLevel.set(c.id.level, arr);
  }
  for (const [level, positions] of byLevel) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const mat = new THREE.LineBasicMaterial({ color: levelColor(level, maxLevel), transparent: true, opacity: 0.9 });
    group.add(new THREE.LineSegments(geo, mat));
  }
  return group;
}

/** A faint outline of the whole dataset extent, for context. */
export function boundsOverlay(ms: Multiscale): THREE.LineSegments {
  const box = worldAabbOfArrayBox(ms.worldFromArray, [0, 0, 0], [ms.voxelDims0[0], ms.voxelDims0[1], ms.voxelDims0[2]]);
  const positions: number[] = [];
  pushBoxEdges(positions, { x: box.min[0], y: box.min[1], z: box.min[2] }, { x: box.max[0], y: box.max[1], z: box.max[2] });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.5 }));
}

/** Free the geometries/materials of a previously-built overlay group. */
export function disposeGroup(group: THREE.Group): void {
  group.traverse((o) => {
    const any = o as THREE.LineSegments;
    if (any.geometry) any.geometry.dispose();
    const m = any.material;
    if (Array.isArray(m)) m.forEach((mm) => mm.dispose());
    else if (m) m.dispose();
  });
}
