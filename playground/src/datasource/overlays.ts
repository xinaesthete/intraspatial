// Overlays (ADR-0008: framework-free three.js Object3D factories, reusable by the
// production renderer's debug mode and by the Decision view). Slice 1 draws the
// selected chunks as level-tinted wireframe boxes plus a faint dataset bound — the
// receding-resolution gradient made visible, driven entirely by the pure `select()`.
import * as THREE from "three";
import {
  type Affine3,
  applyAffine,
  chunkArrayBox,
  chunkKey,
  type Multiscale,
  type Selection,
  type Vec3,
  worldFromArrayOf,
} from "../../../src/datasource";

/** okLCH-ish ramp by level: fine = warm & bright, coarse = cool & dim. */
export function levelColor(level: number, maxLevel: number): THREE.Color {
  const t = maxLevel > 0 ? level / maxLevel : 0;
  const hue = 0.08 + t * 0.55; // orange → blue
  const light = 0.62 - t * 0.3;
  return new THREE.Color().setHSL(hue, 0.75, Math.max(0.15, light));
}

const BOX_EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 0], // bottom
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 4], // top
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7], // verticals
];

// Which (lo|hi) each of the 8 corners takes per axis. Order matches BOX_EDGES:
// bottom face 0-1-2-3, top face 4-5-6-7.
const CORNER_SIGN: ReadonlyArray<readonly [0 | 1, 0 | 1, 0 | 1]> = [
  [0, 0, 0],
  [1, 0, 0],
  [1, 1, 0],
  [0, 1, 0],
  [0, 0, 1],
  [1, 0, 1],
  [1, 1, 1],
  [0, 1, 1],
];

/** The 8 world-space corners of an array-space box [lo,hi] under the affine. An oblique
 *  `worldFromArray` yields a genuinely oriented (rotated/sheared) box, not an AABB — the
 *  same per-corner `applyAffine` the tile renderer uses to place data. */
function orientedCorners(a: Affine3, lo: Vec3, hi: Vec3): Vec3[] {
  const x: [number, number] = [lo[0], hi[0]];
  const y: [number, number] = [lo[1], hi[1]];
  const z: [number, number] = [lo[2], hi[2]];
  return CORNER_SIGN.map(([ix, iy, iz]) => applyAffine(a, [x[ix], y[iy], z[iz]]));
}

function pushBoxEdges(out: number[], corners: readonly Vec3[]): void {
  for (const [a, b] of BOX_EDGES) {
    const ca = corners[a],
      cb = corners[b];
    if (!ca || !cb) continue;
    out.push(ca[0], ca[1], ca[2], cb[0], cb[1], cb[2]);
  }
}

/** Load state of a chunk, for the decision view's colouring (see `chunkOverlays`). */
export type ChunkVizState = "resident" | "loading" | "pending" | "missing";

// State is orthogonal to level: level stays the HUE (the receding-resolution gradient the
// decision view exists to show), state is the OPACITY, plus a gentle pulse on loading. So you
// read both at once — which LOD a chunk is, and where it is in the pending→loading→resident wave.
const STATE_OPACITY: Record<ChunkVizState, number> = {
  resident: 0.95,
  loading: 0.6,
  pending: 0.22,
  missing: 0.22,
};

/** Level-tinted wireframe boxes for a Selection. With `stateOf`, each box's opacity encodes its
 *  load state and loading boxes pulse (tagged `userData.pulse`); without it, all are solid
 *  (the pre-loading-UX behaviour). One LineSegments per (level, state) group. */
export function chunkOverlays(ms: Multiscale, sel: Selection, stateOf?: (key: string) => ChunkVizState): THREE.Group {
  const group = new THREE.Group();
  const maxLevel = ms.levelCount - 1;
  const groups = new Map<string, { level: number; state: ChunkVizState; positions: number[] }>();
  for (const c of sel.chunks) {
    const state = stateOf ? stateOf(chunkKey(c.id)) : "resident";
    const gkey = `${c.id.level}:${state}`;
    let g = groups.get(gkey);
    if (!g) {
      g = { level: c.id.level, state, positions: [] };
      groups.set(gkey, g);
    }
    const [lo, hi] = chunkArrayBox(ms, c.id);
    pushBoxEdges(g.positions, orientedCorners(worldFromArrayOf(ms), lo, hi));
  }
  for (const { level, state, positions } of groups.values()) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const opacity = STATE_OPACITY[state];
    const mat = new THREE.LineBasicMaterial({ color: levelColor(level, maxLevel), transparent: true, opacity });
    const ls = new THREE.LineSegments(geo, mat);
    if (state === "loading") {
      ls.userData.pulse = true; // dualView animates opacity around this base each frame
      ls.userData.baseOpacity = opacity;
    }
    group.add(ls);
  }
  return group;
}

/** A faint outline of the whole dataset extent, for context. */
export function boundsOverlay(ms: Multiscale): THREE.LineSegments {
  const positions: number[] = [];
  pushBoxEdges(positions, orientedCorners(worldFromArrayOf(ms), [0, 0, 0], [ms.voxelDims0[0], ms.voxelDims0[1], ms.voxelDims0[2]]));
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.5 }));
}

/** The select camera's frustum as a wireframe (near quad, far quad, connectors),
 *  plus a short line down its optical axis so its pose reads at a glance. */
export function frustumOverlay(corners: readonly Vec3[], eye: Vec3, forward: Vec3): THREE.Group {
  const group = new THREE.Group();
  const seg: number[] = [];
  const line = (a: Vec3, b: Vec3): void => {
    seg.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  };
  const c = (i: number): Vec3 => corners[i] ?? [0, 0, 0];
  // near quad 0-1-2-3, far quad 4-5-6-7, connectors i↔i+4
  for (let i = 0; i < 4; i++) {
    line(c(i), c((i + 1) % 4));
    line(c(4 + i), c(4 + ((i + 1) % 4)));
    line(c(i), c(4 + i));
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(seg, 3));
  group.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.85 })));

  // A dot at the eye so the apex is obvious.
  const dotGeo = new THREE.BufferGeometry();
  dotGeo.setAttribute("position", new THREE.Float32BufferAttribute([eye[0], eye[1], eye[2]], 3));
  group.add(new THREE.Points(dotGeo, new THREE.PointsMaterial({ color: 0x7dd3fc, size: 8, sizeAttenuation: false })));
  void forward;
  return group;
}

/** Free the geometries/materials of a previously-built overlay group. */
export function disposeGroup(group: THREE.Group): void {
  group.traverse((o) => {
    const any = o as THREE.LineSegments;
    if (any.geometry) any.geometry.dispose();
    const m = any.material;
    if (Array.isArray(m)) for (const mm of m) mm.dispose();
    else if (m) m.dispose();
  });
}
