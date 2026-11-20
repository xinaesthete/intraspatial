// Measurements of a spring-model state: the numbers the analytic views plot and the phase
// diagram is drawn from.
import type { SpringParams } from "./params";
import { DEAD, type SpringMatrices, type SpringState } from "./state";
import { NO_BOND, SLOT_WORDS, type SpringTopology, slotKind, slotPartner } from "./topology";

export interface ShapeReadout {
  /** Area of the projected outline, holes filled. */
  readonly area: number;
  readonly perimeter: number;
  /** `4π · area / perimeter²`: 1 for a disc, falling as the outline grows fingers. */
  readonly circularity: number;
  /** Centroid of the particles measured. */
  readonly cx: number;
  readonly cy: number;
  /** Largest distance of a measured particle's edge from the centroid. */
  readonly maxRadius: number;
  /** Root-mean-square distance from the centroid. */
  readonly gyration: number;
  /** Particles measured. */
  readonly count: number;
}

export interface ShapeOptions {
  /** Pixel side of the working raster; default a quarter of the mean radius. */
  readonly pixel?: number;
  /** Width of the smoothing kernel, in particle diameters. Default 0.8. Larger values ignore
   *  more of the bumpiness of individual particles. */
  readonly smoothing?: number;
}

/**
 * Outline of one species projected onto the xy plane: the measurement the paper makes on its
 * micrographs. Each particle adds a Gaussian bump to a raster, and the outline is the contour
 * at half the value found inside a close-packed sheet, which puts it on the edge of the outer
 * row of particles without following each one. Holes are filled and only the largest connected
 * piece is kept. Area is a pixel count; perimeter is the marching-squares contour length.
 */
export function projectedShape(s: SpringState, species: number, opts: ShapeOptions = {}): ShapeReadout {
  const ids: number[] = [];
  let cx = 0;
  let cy = 0;
  let rsum = 0;
  for (let i = 0; i < s.n; i++) {
    if (s.species[i] !== species) continue;
    ids.push(i);
    cx += s.pos[i * 4]!;
    cy += s.pos[i * 4 + 1]!;
    rsum += s.radius[i]!;
  }
  const count = ids.length;
  if (count === 0) return { area: 0, perimeter: 0, circularity: 0, cx: 0, cy: 0, maxRadius: 0, gyration: 0, count: 0 };
  cx /= count;
  cy /= count;
  const diameter = (2 * rsum) / count;
  const sigma = (opts.smoothing ?? 0.8) * diameter;
  const h = opts.pixel ?? diameter / 8;
  // Field value deep inside a triangular packing is `2πσ² / (√3/2 · d²)`; the edge is at half.
  const threshold = (Math.PI * sigma * sigma) / ((Math.sqrt(3) / 2) * diameter * diameter);
  const reach = 3 * sigma;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxRadius = 0;
  let gyr = 0;
  for (const i of ids) {
    const x = s.pos[i * 4]!;
    const y = s.pos[i * 4 + 1]!;
    minX = Math.min(minX, x - reach);
    minY = Math.min(minY, y - reach);
    maxX = Math.max(maxX, x + reach);
    maxY = Math.max(maxY, y + reach);
    const d = Math.hypot(x - cx, y - cy);
    maxRadius = Math.max(maxRadius, d + s.radius[i]!);
    gyr += d * d;
  }
  // Two pixels of margin, so the outside is one connected region around the shape.
  const W = Math.ceil((maxX - minX) / h) + 5;
  const H = Math.ceil((maxY - minY) / h) + 5;
  const x0 = minX - 2 * h;
  const y0 = minY - 2 * h;

  // `sd` is threshold minus field: negative inside, like a signed distance.
  const sd = new Float32Array(W * H).fill(threshold);
  const inv = 1 / (2 * sigma * sigma);
  for (const i of ids) {
    const x = s.pos[i * 4]!;
    const y = s.pos[i * 4 + 1]!;
    const ax = Math.max(0, Math.floor((x - reach - x0) / h));
    const bx = Math.min(W - 1, Math.ceil((x + reach - x0) / h));
    const ay = Math.max(0, Math.floor((y - reach - y0) / h));
    const by = Math.min(H - 1, Math.ceil((y + reach - y0) / h));
    for (let py = ay; py <= by; py++) {
      for (let px = ax; px <= bx; px++) {
        const ex = x0 + px * h - x;
        const ey = y0 + py * h - y;
        sd[py * W + px]! -= Math.exp(-(ex * ex + ey * ey) * inv);
      }
    }
  }

  // Label: 0 unvisited, 1 outside (reached from the border), 2.. inside components.
  const label = new Int32Array(W * H);
  const stack: number[] = [];
  const flood = (seed: number, want: boolean, id: number): number => {
    let size = 0;
    stack.push(seed);
    label[seed] = id;
    while (stack.length > 0) {
      const k = stack.pop()!;
      size++;
      const px = k % W;
      const py = (k - px) / W;
      for (let d = 0; d < 4; d++) {
        const qx = px + (d === 0 ? 1 : d === 1 ? -1 : 0);
        const qy = py + (d === 2 ? 1 : d === 3 ? -1 : 0);
        if (qx < 0 || qy < 0 || qx >= W || qy >= H) continue;
        const q = qy * W + qx;
        if (label[q] === 0 && sd[q]! < 0 === want) {
          label[q] = id;
          stack.push(q);
        }
      }
    }
    return size;
  };
  flood(0, false, 1);
  let bestId = 0;
  let bestSize = 0;
  let nextId = 2;
  for (let k = 0; k < W * H; k++) {
    if (label[k] === 0 && sd[k]! < 0) {
      const size = flood(k, true, nextId);
      if (size > bestSize) {
        bestSize = size;
        bestId = nextId;
      }
      nextId++;
    }
  }
  // Keep the largest piece; fill its holes (unvisited positive pixels are enclosed).
  let area = 0;
  for (let k = 0; k < W * H; k++) {
    const l = label[k]!;
    if (l === 0) sd[k] = -threshold;
    else if (l !== 1 && l !== bestId) sd[k] = threshold;
    if (sd[k]! < 0) area++;
  }
  area *= h * h;
  let perimeter = 0;
  // Marching squares: each cell contributes the segment(s) between its edge crossings.
  const cross = (a: number, b: number) => a / (a - b);
  for (let py = 0; py < H - 1; py++) {
    for (let px = 0; px < W - 1; px++) {
      const a = sd[py * W + px]!;
      const b = sd[py * W + px + 1]!;
      const c = sd[(py + 1) * W + px + 1]!;
      const d = sd[(py + 1) * W + px]!;
      const pts: number[] = [];
      if (a < 0 !== b < 0) pts.push(cross(a, b), 0);
      if (b < 0 !== c < 0) pts.push(1, cross(b, c));
      if (d < 0 !== c < 0) pts.push(cross(d, c), 1);
      if (a < 0 !== d < 0) pts.push(0, cross(a, d));
      for (let q = 0; q + 3 < pts.length; q += 4) {
        perimeter += Math.hypot(pts[q + 2]! - pts[q]!, pts[q + 3]! - pts[q + 1]!) * h;
      }
    }
  }
  const circularity = perimeter > 0 ? Math.min(1, (4 * Math.PI * area) / (perimeter * perimeter)) : 0;
  return { area, perimeter, circularity, cx, cy, maxRadius, gyration: Math.sqrt(gyr / count), count };
}

export interface BondReadout {
  /** Bonds, each counted once. */
  readonly count: number;
  /** Mean of `len / rest - 1`. */
  readonly meanStrain: number;
  /**
   * Virial of the bond forces, `Σ f ⊗ r` over bonds, as `[xx, yy, zz, xy]`. Divided by the
   * volume it is the stress the network carries; tracking it at fixed shape is a stress
   * relaxation test.
   */
  readonly virial: readonly [number, number, number, number];
}

/** Bond statistics, for all kinds or those in a bitmask. */
export function bondReadout(s: SpringState, t: SpringTopology, p: SpringParams, kinds = 0xff): BondReadout {
  let count = 0;
  let strain = 0;
  let xx = 0;
  let yy = 0;
  let zz = 0;
  let xy = 0;
  for (let i = 0; i < t.n; i++) {
    const base = i * t.K * SLOT_WORDS;
    for (let k = 0; k < t.K; k++) {
      const o = base + k * SLOT_WORDS;
      const bits = t.slotBits[o]!;
      if (bits === NO_BOND) continue;
      const j = slotPartner(bits);
      const kind = slotKind(bits);
      if (j < i || ((kinds >> kind) & 1) === 0) continue;
      const dx = s.pos[j * 4]! - s.pos[i * 4]!;
      const dy = s.pos[j * 4 + 1]! - s.pos[i * 4 + 1]!;
      const dz = s.pos[j * 4 + 2]! - s.pos[i * 4 + 2]!;
      const len = Math.hypot(dx, dy, dz);
      const rest = t.slots[o + 1]! * p.springlen + p.springlenfix;
      const f = (len - rest) * t.slots[o + 2]! * (p.springforce * p.roleforces[kind]! + p.roleforcesFix[kind]!);
      count++;
      strain += len / rest - 1;
      if (len > 0) {
        xx += (f * dx * dx) / len;
        yy += (f * dy * dy) / len;
        zz += (f * dz * dz) / len;
        xy += (f * dx * dy) / len;
      }
    }
  }
  return { count, meanStrain: count > 0 ? strain / count : 0, virial: [xx, yy, zz, xy] };
}

/** Pairwise distances between the first `size` particles, row-major. CSynth's distance map. */
export function distanceMatrix(s: SpringState, size: number): Float32Array {
  const out = new Float32Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = i + 1; j < size; j++) {
      const d = Math.hypot(s.pos[j * 4]! - s.pos[i * 4]!, s.pos[j * 4 + 1]! - s.pos[i * 4 + 1]!, s.pos[j * 4 + 2]! - s.pos[i * 4 + 2]!);
      out[i * size + j] = d;
      out[j * size + i] = d;
    }
  }
  return out;
}

/**
 * How well the model reproduces its target: Pearson correlation between model and reference
 * distances over all pairs more than `minSeparation` apart along the chain.
 */
export function distanceCorrelation(model: Float32Array, reference: Float32Array, size: number, minSeparation = 2): number {
  let nn = 0;
  let sa = 0;
  let sb = 0;
  let saa = 0;
  let sbb = 0;
  let sab = 0;
  for (let i = 0; i < size; i++) {
    for (let j = i + minSeparation; j < size; j++) {
      const a = model[i * size + j]!;
      const b = reference[i * size + j]!;
      nn++;
      sa += a;
      sb += b;
      saa += a * a;
      sbb += b * b;
      sab += a * b;
    }
  }
  const cov = sab - (sa * sb) / nn;
  const va = saa - (sa * sa) / nn;
  const vb = sbb - (sb * sb) / nn;
  return va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : 0;
}

/** Contact matrix from a distance matrix, `contact = (scale / dist)^power`, the inverse of the
 *  distance-to-contact relation CSynth assumes when it turns Hi-C counts into spring lengths. */
export function contactsFromDistances(dist: Float32Array, size: number, scale = 1, power = 1): SpringMatrices {
  const contact = new Float32Array(size * size);
  for (let k = 0; k < size * size; k++) contact[k] = dist[k]! > 0 ? (scale / dist[k]!) ** power : 0;
  return { size, contact, dist };
}

/** Live particles and their mean density. */
export function populationReadout(s: SpringState, species: number): { count: number; meanDensity: number } {
  let count = 0;
  let density = 0;
  for (let i = 0; i < s.n; i++) {
    if (s.species[i] !== species || s.species[i] === DEAD) continue;
    count++;
    density += s.pos[i * 4 + 3]!;
  }
  return { count, meanDensity: count > 0 ? density / count : 0 };
}
