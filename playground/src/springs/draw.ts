// Canvas renderings of a spring-model scenario. Two families:
//   analytic  — what is where and how loaded it is: species, bonds coloured by strain, pins;
//   aesthetic — the same state as light: additive glows and filaments that leave trails.
// Both take a `View` that maps model coordinates to canvas pixels.
import {
  CELL,
  DEAD,
  KIND_LIGAND,
  MATRIX,
  NO_BOND,
  type Scenario,
  SLOT_WORDS,
  type SpringEvent,
  slotKind,
  slotPartner,
} from "../../../src/gpu/sim/springs/index";

export interface View {
  readonly width: number;
  readonly height: number;
  /** Model units across half the shorter side. */
  readonly extent: number;
  /** Rotation about the vertical axis, for 3D scenarios. */
  readonly yaw: number;
}

interface Projected {
  readonly x: number;
  readonly y: number;
  /** Depth after rotation; larger is nearer. */
  readonly z: number;
}

function projector(view: View, cx = 0, cy = 0, cz = 0): (x: number, y: number, z: number) => Projected {
  const scale = Math.min(view.width, view.height) / 2 / view.extent;
  const c = Math.cos(view.yaw);
  const s = Math.sin(view.yaw);
  return (x, y, z) => {
    const dx = x - cx;
    const dz = z - cz;
    return { x: view.width / 2 + (dx * c + dz * s) * scale, y: view.height / 2 - (y - cy) * scale, z: -dx * s + dz * c };
  };
}

/** Centre of the live particles: 3D scenarios are drawn about it. */
function centre(sc: Scenario): [number, number, number] {
  const { state } = sc;
  let x = 0;
  let y = 0;
  let z = 0;
  let c = 0;
  for (let i = 0; i < state.n; i++) {
    if (state.species[i] === DEAD) continue;
    x += state.pos[i * 4]!;
    y += state.pos[i * 4 + 1]!;
    z += state.pos[i * 4 + 2]!;
    c++;
  }
  return c > 0 ? [x / c, y / c, z / c] : [0, 0, 0];
}

/** Blue (compressed) through grey to red (stretched), for a strain in about ±0.3. */
function strainColour(strain: number, alpha: number): string {
  const t = Math.max(-1, Math.min(1, strain / 0.3));
  const r = Math.round(120 + 135 * Math.max(0, t));
  const g = Math.round(130 - 70 * Math.abs(t));
  const b = Math.round(140 + 115 * Math.max(0, -t));
  return `rgba(${r},${g},${b},${alpha})`;
}

function eachBond(sc: Scenario, visit: (i: number, j: number, kind: number, strain: number) => void): void {
  const { state, topology: t, params: p } = sc;
  for (let i = 0; i < t.n; i++) {
    const base = i * t.K * SLOT_WORDS;
    for (let k = 0; k < t.K; k++) {
      const o = base + k * SLOT_WORDS;
      const bits = t.slotBits[o]!;
      if (bits === NO_BOND) continue;
      const j = slotPartner(bits);
      if (j < i) continue;
      const len = Math.hypot(
        state.pos[j * 4]! - state.pos[i * 4]!,
        state.pos[j * 4 + 1]! - state.pos[i * 4 + 1]!,
        state.pos[j * 4 + 2]! - state.pos[i * 4 + 2]!,
      );
      const rest = t.slots[o + 1]! * p.springlen + p.springlenfix;
      visit(i, j, slotKind(bits), len / rest - 1);
    }
  }
}

/** Analytic view of a 2D scenario: discs by species, bonds by strain, pinned particles ringed. */
export function drawSheetAnalytic(ctx: CanvasRenderingContext2D, sc: Scenario, view: View): void {
  const { state, topology } = sc;
  const proj = projector({ ...view, yaw: 0 });
  const scale = Math.min(view.width, view.height) / 2 / view.extent;
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = "#0b1020";
  ctx.fillRect(0, 0, view.width, view.height);

  ctx.lineWidth = Math.max(1, scale * 0.08);
  eachBond(sc, (i, j, kind, strain) => {
    const a = proj(state.pos[i * 4]!, state.pos[i * 4 + 1]!, 0);
    const b = proj(state.pos[j * 4]!, state.pos[j * 4 + 1]!, 0);
    ctx.strokeStyle = kind === KIND_LIGAND ? "rgba(250,204,21,0.8)" : strainColour(strain, 0.75);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });

  for (let i = 0; i < state.n; i++) {
    const sp = state.species[i]!;
    if (sp === DEAD) continue;
    const q = proj(state.pos[i * 4]!, state.pos[i * 4 + 1]!, 0);
    const r = state.radius[i]! * scale;
    ctx.beginPath();
    if (sp === CELL) {
      // Cells shade from teal to white as they get crowded: the pressure that stops division.
      const crowd = Math.max(0, Math.min(1, (state.pos[i * 4 + 3]! - 6) / 8));
      ctx.fillStyle = `rgb(${Math.round(45 + 190 * crowd)},${Math.round(212 - 10 * crowd)},${Math.round(191 + 40 * crowd)})`;
      ctx.arc(q.x, q.y, r * 0.92, 0, 2 * Math.PI);
      ctx.fill();
    } else {
      ctx.fillStyle = sp === MATRIX ? "#64748b" : "#a78bfa";
      ctx.arc(q.x, q.y, Math.max(1, r * 0.3), 0, 2 * Math.PI);
      ctx.fill();
    }
    if (topology.fix[i * 4] === 1) {
      ctx.strokeStyle = "#f8fafc";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(q.x, q.y, Math.max(2, r * 0.5), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.lineWidth = Math.max(1, scale * 0.08);
    }
  }
}

/** Analytic view of a chain: the polyline in 3D, coloured by position along the chain. */
export function drawChainAnalytic(ctx: CanvasRenderingContext2D, sc: Scenario, view: View): void {
  const { state } = sc;
  const proj = projector(view, ...centre(sc));
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = "#0b1020";
  ctx.fillRect(0, 0, view.width, view.height);
  const n = state.n;
  // Springs other than the backbone: the ties a contact line or a drawn line adds.
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(226,232,240,0.35)";
  eachBond(sc, (i, j) => {
    if (j - i <= 1) return;
    const a = proj(state.pos[i * 4]!, state.pos[i * 4 + 1]!, state.pos[i * 4 + 2]!);
    const b = proj(state.pos[j * 4]!, state.pos[j * 4 + 1]!, state.pos[j * 4 + 2]!);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });
  ctx.lineWidth = 1.5;
  let prev = proj(state.pos[0]!, state.pos[1]!, state.pos[2]!);
  for (let i = 1; i < n; i++) {
    const q = proj(state.pos[i * 4]!, state.pos[i * 4 + 1]!, state.pos[i * 4 + 2]!);
    ctx.strokeStyle = `hsl(${(i / n) * 300}, 80%, 60%)`;
    ctx.beginPath();
    ctx.moveTo(prev.x, prev.y);
    ctx.lineTo(q.x, q.y);
    ctx.stroke();
    prev = q;
  }
  for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 400))) {
    const q = proj(state.pos[i * 4]!, state.pos[i * 4 + 1]!, state.pos[i * 4 + 2]!);
    ctx.fillStyle = `hsl(${(i / n) * 300}, 80%, 70%)`;
    ctx.fillRect(q.x - 1.5, q.y - 1.5, 3, 3);
  }
}

/** Fade the previous frame, so moving things leave trails. */
function fade(ctx: CanvasRenderingContext2D, view: View, amount: number): void {
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = `rgba(4,5,14,${amount})`;
  ctx.fillRect(0, 0, view.width, view.height);
  ctx.globalCompositeOperation = "lighter";
}

/**
 * Aesthetic view of a 2D scenario. Light is additive and each frame fades by a fixed fraction,
 * so a mark of alpha `a` drawn every frame settles at about `a / fade`; the alphas are small. The matrix is drawn only where it is loaded: each bond glows
 * in proportion to its strain, so stress shows as light and relaxed matrix is dark. Cells are
 * warm glows that brighten as they move.
 */
export function drawSheetAesthetic(ctx: CanvasRenderingContext2D, sc: Scenario, view: View): void {
  const { state } = sc;
  const proj = projector({ ...view, yaw: 0 });
  const scale = Math.min(view.width, view.height) / 2 / view.extent;
  fade(ctx, view, 0.12);

  ctx.lineWidth = Math.max(1, scale * 0.12);
  eachBond(sc, (i, j, kind, strain) => {
    const load = Math.min(1, Math.abs(strain) / 0.25);
    if (load < 0.04 && kind !== KIND_LIGAND) return;
    const a = proj(state.pos[i * 4]!, state.pos[i * 4 + 1]!, 0);
    const b = proj(state.pos[j * 4]!, state.pos[j * 4 + 1]!, 0);
    ctx.strokeStyle =
      kind === KIND_LIGAND
        ? "rgba(255,220,120,0.08)"
        : strain > 0
          ? `rgba(255,${Math.round(90 + 90 * load)},160,${0.02 + 0.1 * load})`
          : `rgba(80,${Math.round(150 + 80 * load)},255,${0.02 + 0.1 * load})`;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });

  for (let i = 0; i < state.n; i++) {
    if (state.species[i] !== CELL) continue;
    const q = proj(state.pos[i * 4]!, state.pos[i * 4 + 1]!, 0);
    const speed = Math.hypot(state.pos[i * 4]! - state.prev[i * 4]!, state.pos[i * 4 + 1]! - state.prev[i * 4 + 1]!);
    const heat = Math.min(1, speed / 0.08);
    const r = state.radius[i]! * scale * 1.3;
    const glow = ctx.createRadialGradient(q.x, q.y, 0, q.x, q.y, r);
    glow.addColorStop(0, `rgba(255,${Math.round(170 + 70 * heat)},${Math.round(90 + 140 * heat)},0.07)`);
    glow.addColorStop(1, "rgba(255,120,60,0)");
    ctx.fillStyle = glow;
    ctx.fillRect(q.x - r, q.y - r, 2 * r, 2 * r);
  }
  ctx.globalCompositeOperation = "source-over";
}

/** Aesthetic view of a chain: a ribbon of light whose width follows depth, with trails. */
export function drawChainAesthetic(ctx: CanvasRenderingContext2D, sc: Scenario, view: View): void {
  const { state } = sc;
  const proj = projector(view, ...centre(sc));
  fade(ctx, view, 0.07);
  const n = state.n;
  ctx.lineCap = "round";
  let prev = proj(state.pos[0]!, state.pos[1]!, state.pos[2]!);
  for (let i = 1; i < n; i++) {
    const q = proj(state.pos[i * 4]!, state.pos[i * 4 + 1]!, state.pos[i * 4 + 2]!);
    const near = Math.max(0, Math.min(1, 0.5 + q.z / (2 * view.extent)));
    ctx.strokeStyle = `hsla(${190 + (i / n) * 150}, 90%, ${45 + 25 * near}%, ${0.12 + 0.35 * near})`;
    ctx.lineWidth = 1 + 5 * near;
    ctx.beginPath();
    ctx.moveTo(prev.x, prev.y);
    ctx.lineTo(q.x, q.y);
    ctx.stroke();
    prev = q;
  }
  ctx.globalCompositeOperation = "source-over";
}

/** A square matrix as a heat map: dark is near, bright is far. */
export function drawMatrix(ctx: CanvasRenderingContext2D, values: Float32Array, size: number, max: number): void {
  const image = ctx.createImageData(size, size);
  for (let k = 0; k < size * size; k++) {
    const t = Math.max(0, Math.min(1, values[k]! / max));
    image.data[k * 4] = Math.round(20 + 235 * t);
    image.data[k * 4 + 1] = Math.round(30 + 190 * t * t);
    image.data[k * 4 + 2] = Math.round(80 + 100 * (1 - t));
    image.data[k * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

/** Mark this frame's events on a 2D view: a spark where a bond broke, a ring where a cell
 *  divided. In the aesthetic view the trails make them linger. */
export function drawEvents(ctx: CanvasRenderingContext2D, events: readonly SpringEvent[], view: View): void {
  const proj = projector({ ...view, yaw: 0 });
  const scale = Math.min(view.width, view.height) / 2 / view.extent;
  ctx.globalCompositeOperation = "lighter";
  for (const e of events) {
    const q = proj(e.x, e.y, 0);
    if (e.type === "break") {
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.beginPath();
      ctx.arc(q.x, q.y, Math.max(1.5, scale * 0.25), 0, 2 * Math.PI);
      ctx.fill();
    } else {
      ctx.strokeStyle = "rgba(255,210,120,0.9)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(q.x, q.y, scale * 0.9, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }
  ctx.globalCompositeOperation = "source-over";
}
