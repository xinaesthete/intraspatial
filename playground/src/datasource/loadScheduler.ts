// LoadScheduler — turns "load every desired chunk immediately" into a bounded, prioritised
// queue, so a slow source (see the latency sim) streams chunks in nearest-first instead of
// firing hundreds of concurrent requests. It's the piece that makes the loading process
// legible: at any moment a chunk is `pending` (queued), `loading` (in-flight), or neither
// (resident / not wanted). The decision view colours by exactly these states (step 4).
//
// Reconciled against the current Selection every update: queued-but-unstarted requests that
// are no longer wanted are dropped (a fast camera move doesn't waste the source on chunks it
// has already left); in-flight requests are left to finish but the renderer commits their
// result only if still desired.
import type { ChunkId } from "../../../src/datasource";

export type LoadState = "pending" | "loading";

export interface LoadRequest {
  readonly key: string;
  readonly id: ChunkId;
  /** Lower = higher priority. The chunk's nearest-point depth (front-to-back refinement). */
  readonly priority: number;
}

export class LoadScheduler {
  private pending: LoadRequest[] = [];
  private inFlight = new Set<string>();

  constructor(
    private readonly maxConcurrent: number,
    /** Runs one load to completion (fetch/generate + commit). Rejections are swallowed — a
     *  failed chunk simply never becomes resident. */
    private readonly run: (id: ChunkId) => Promise<void>,
    /** Already-resident predicate, so reconcile never re-queues a resident chunk. */
    private readonly isResident: (key: string) => boolean,
  ) {}

  /** Rebuild the queue to match `desired`: drop queued items no longer wanted, enqueue newly
   *  wanted ones (skipping resident/in-flight/already-queued), sort nearest-first, and pump. */
  reconcile(desired: readonly LoadRequest[]): void {
    const want = new Set(desired.map((d) => d.key));
    this.pending = this.pending.filter((p) => want.has(p.key));
    const queued = new Set(this.pending.map((p) => p.key));
    for (const d of desired) {
      if (this.isResident(d.key) || this.inFlight.has(d.key) || queued.has(d.key)) continue;
      this.pending.push(d);
      queued.add(d.key);
    }
    this.pending.sort((a, b) => a.priority - b.priority);
    this.pump();
  }

  /** The chunk's current load state, or `undefined` if it's neither queued nor in-flight. */
  state(key: string): LoadState | undefined {
    if (this.inFlight.has(key)) return "loading";
    return this.pending.some((p) => p.key === key) ? "pending" : undefined;
  }

  /** Number of chunks queued (not yet started) — for the HUD. */
  get pendingCount(): number {
    return this.pending.length;
  }
  /** Number of chunks in-flight — for the HUD. */
  get loadingCount(): number {
    return this.inFlight.size;
  }

  private pump(): void {
    while (this.inFlight.size < this.maxConcurrent && this.pending.length > 0) {
      const next = this.pending.shift();
      if (!next) break;
      this.inFlight.add(next.key);
      void this.run(next.id)
        .catch(() => {})
        .finally(() => {
          this.inFlight.delete(next.key);
          this.pump();
        });
    }
  }
}
