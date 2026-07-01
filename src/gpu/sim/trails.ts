// A ring buffer of recent positions per agent — the primitive behind motion trails. It
// stores the last `capacity` frames of an N-agent point cloud in a circular buffer and
// can emit them as line-segment vertices (oldest→newest) with a tail→head colour fade, so
// a renderer just uploads the result. Pure, allocation-light (one backing buffer), and
// reusable beyond the dancer (any point cloud with history). Checked access, no `!`.
import { readVec3, scale, writeVec3, type Vec3 } from "./vec3";

export class TrailBuffer {
  readonly n: number;
  readonly capacity: number;
  /** frame-major ring: frame slot `f` holds N vec3s at `f*n*3`. */
  private readonly buf: Float32Array;
  private cursor = 0; // next slot to write
  private filled = 0; // frames stored so far (≤ capacity)

  constructor(n: number, capacity: number) {
    this.n = n;
    this.capacity = Math.max(2, capacity);
    this.buf = new Float32Array(this.capacity * n * 3);
  }

  /** Record the current positions ([x,y,z]×N). */
  push(positions: Float32Array): void {
    const need = this.n * 3;
    const base = this.cursor * need;
    this.buf.set(positions.subarray(0, need), base);
    this.cursor = (this.cursor + 1) % this.capacity;
    if (this.filled < this.capacity) this.filled++;
  }

  reset(): void {
    this.cursor = 0;
    this.filled = 0;
    this.buf.fill(0);
  }

  get frames(): number {
    return this.filled;
  }

  /** Vertices produced by `fillSegments`: n · (frames−1) · 2 (a GL_LINES pair per step). */
  segmentVertexCount(): number {
    return this.n * Math.max(0, this.filled - 1) * 2;
  }

  /** Write line-segment vertices (positions + per-vertex colours) connecting each agent's
   *  consecutive recorded frames, oldest→newest. Colour fades from ~0 at the tail to
   *  `head` at the newest sample (quadratic), so on an additive/dark background the trail
   *  dissolves behind each dancer. Returns the number of vertices written. */
  fillSegments(outPositions: Float32Array, outColors: Float32Array, head: Vec3): number {
    const { n, capacity, filled, cursor } = this;
    if (filled < 2) return 0;
    const start = (cursor - filled + capacity) % capacity; // oldest slot
    let v = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < filled - 1; j++) {
        const s0 = (start + j) % capacity;
        const s1 = (start + j + 1) % capacity;
        const p0 = readVec3(this.buf, s0 * n + i);
        const p1 = readVec3(this.buf, s1 * n + i);
        const t0 = j / (filled - 1);
        const t1 = (j + 1) / (filled - 1);
        writeVec3(outPositions, v, p0);
        writeVec3(outColors, v, scale(head, t0 * t0));
        v++;
        writeVec3(outPositions, v, p1);
        writeVec3(outColors, v, scale(head, t1 * t1));
        v++;
      }
    }
    return v;
  }
}
