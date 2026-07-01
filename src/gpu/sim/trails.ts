// Motion trails — now a thin *view* over the general field-history ring buffer
// (./ringBuffer). The ring stores the last `capacity` frames of the N×vec3 position field;
// `TrailBuffer` just reads them back as fading line-segment vertices. The exact same
// RingBuffer records grids, matrices, or any field (see FieldRing) — trails are simply the
// points case. Checked access, no `!`.
import { RingBuffer } from "../graph/ringBuffer";
import { readVec3, scale, writeVec3, type Vec3 } from "./vec3";

export class TrailBuffer {
  readonly n: number;
  readonly capacity: number;
  private readonly ring: RingBuffer;

  constructor(n: number, capacity: number) {
    this.n = n;
    this.capacity = Math.max(2, capacity);
    this.ring = new RingBuffer(n * 3, this.capacity);
  }

  /** Record the current positions ([x,y,z]×N). */
  push(positions: Float32Array): void {
    this.ring.push(positions);
  }

  reset(): void {
    this.ring.reset();
  }

  get frames(): number {
    return this.ring.frames;
  }

  /** Vertices produced by `fillSegments`: n · (frames−1) · 2. */
  segmentVertexCount(): number {
    return this.n * Math.max(0, this.ring.frames - 1) * 2;
  }

  /** Write line-segment vertices (positions + fading colours) connecting each agent's
   *  consecutive recorded frames, oldest→newest. Colour fades quadratically from ~0 at the
   *  tail to `head` at the newest sample. Returns the number of vertices written. */
  fillSegments(outPositions: Float32Array, outColors: Float32Array, head: Vec3): number {
    const { n } = this;
    const filled = this.ring.frames;
    if (filled < 2) return 0;
    let v = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < filled - 1; j++) {
        // oldest → newest: frame(filled-1) is the oldest, frame(0) the newest
        const older = this.ring.frame(filled - 1 - j);
        const newer = this.ring.frame(filled - 1 - (j + 1));
        const p0 = readVec3(older, i);
        const p1 = readVec3(newer, i);
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
