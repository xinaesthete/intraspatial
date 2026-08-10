// A small pool of HTJ2K decode workers, and the zarrita codec that routes through it.
//
// Why a pool and not one worker: the viewer asks for 19–27 bricks at once after a camera move, and a
// single worker serialises them — the main thread stops blocking but the bricks still arrive one
// 45 ms decode at a time. Several workers turn that into wall-clock parallelism on cores that are
// otherwise idle while the main thread renders.

import type { DecodeRequest, DecodeResponse } from "./htj2kDecodeWorker";

export interface DecodedChunk {
  data: Uint8Array | Uint16Array | Uint32Array;
  width: number;
  height: number;
  components: number;
}

interface Job {
  bytes: Uint8Array;
  resolve: (c: DecodedChunk) => void;
  reject: (e: Error) => void;
}

/** Decode is CPU-bound, so more workers than cores only adds contention; leave one core for the
 *  main thread, which is busy rendering. Capped because each worker holds its own wasm instance. */
function workerCount(): number {
  const cores = navigator.hardwareConcurrency || 4;
  return Math.max(1, Math.min(4, cores - 1));
}

class DecodePool {
  private workers: Worker[] = [];
  private idle: Worker[] = [];
  private queue: Job[] = [];
  private pending = new Map<number, Job>();
  private busyOf = new Map<Worker, number>(); // worker → in-flight job id
  private seq = 0;

  constructor(n: number) {
    for (let i = 0; i < n; i++) {
      const w = new Worker(new URL("./htj2kDecodeWorker.ts", import.meta.url), { type: "module" });
      w.onmessage = (e: MessageEvent<DecodeResponse>) => this.onDone(w, e.data);
      w.onerror = (e) => this.onFatal(w, e.message || "worker error");
      this.workers.push(w);
      this.idle.push(w);
    }
  }

  get size(): number {
    return this.workers.length;
  }

  decode(bytes: Uint8Array): Promise<DecodedChunk> {
    return new Promise<DecodedChunk>((resolve, reject) => {
      const job: Job = { bytes, resolve, reject };
      const w = this.idle.pop();
      if (w) this.dispatch(w, job);
      else this.queue.push(job);
    });
  }

  private dispatch(w: Worker, job: Job): void {
    const id = ++this.seq;
    this.pending.set(id, job);
    this.busyOf.set(w, id);
    // The compressed bytes are NOT transferred: they belong to zarrita's decode pipeline, and
    // detaching a buffer it may still hold would be a very confusing bug for a 2–5 MB copy.
    const req: DecodeRequest = { id, bytes: job.bytes };
    w.postMessage(req);
  }

  private onDone(w: Worker, res: DecodeResponse): void {
    const job = this.pending.get(res.id);
    this.pending.delete(res.id);
    this.busyOf.delete(w);
    if (job) {
      if (res.ok) {
        const Ctor = res.bytesPerSample === 1 ? Uint8Array : res.bytesPerSample === 2 ? Uint16Array : Uint32Array;
        job.resolve({ data: new Ctor(res.data), width: res.width, height: res.height, components: res.components });
      } else {
        job.reject(new Error(res.error));
      }
    }
    const next = this.queue.shift();
    if (next) this.dispatch(w, next);
    else this.idle.push(w);
  }

  /** A worker that dies takes its in-flight job with it. Fail that job loudly rather than leaving
   *  the caller's promise pending forever — a brick that never resolves is indistinguishable from a
   *  slow network, which is the worst way to learn the decoder is broken. */
  private onFatal(w: Worker, message: string): void {
    const id = this.busyOf.get(w);
    if (id !== undefined) {
      this.pending.get(id)?.reject(new Error(`htj2k decode worker failed: ${message}`));
      this.pending.delete(id);
      this.busyOf.delete(w);
    }
    const next = this.queue.shift();
    if (next) this.dispatch(w, next);
    else this.idle.push(w);
  }
}

let pool: DecodePool | undefined;
export function decodePool(): { decode: (bytes: Uint8Array) => Promise<DecodedChunk>; size: number } {
  pool ??= new DecodePool(workerCount());
  return { decode: (b) => (pool as DecodePool).decode(b), size: pool.size };
}
