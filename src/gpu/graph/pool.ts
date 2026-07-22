// The Tier-2 resident buffer pool (ADR-0017, resource-sync invariant 3 — "pool reuse
// respects liveness").
//
// Backend-agnostic on purpose: it needs nothing but a `GPUDevice`, so the Node (Dawn) and
// browser backends share one implementation and one set of semantics. Before this, every
// module grew its own `makePool`/`ensurePool` singleton (convolveSeparable, emptySpace,
// threshold, the fdwt/idwt family, a dozen spatial kernels) — all grow-only, single-tenant,
// and freed by nothing. This is the multi-tenant counterpart: buffers are handed out by
// *lease* and come back by *liveness*, so two fields whose lifetimes don't overlap can share
// one allocation.
//
// TWO RULES, both load-bearing:
//
//  1. NEVER `destroy()`. `release` returns a buffer to the free list and nothing else.
//     Destroying a buffer mid-process segfaults Dawn-on-Node (ADR-0002/0003), which is why
//     the ad-hoc pools that grow (convolveSeparable.ts, emptySpace.ts) drop the old buffer on
//     the floor rather than freeing it. Since buffers are never destroyed and the pool
//     outlives any single pull, the steady-state footprint converges to peak liveness.
//
//  2. NOTHING MAPPABLE IN THE POOL. WebGPU forbids combining mappable and non-mappable usage
//     (`MAP_READ` may combine only with `COPY_DST`), so `STORAGE | MAP_READ` is not a legal
//     buffer and there is no universal allocation to over-provision toward. Mappable buffers
//     also live in host-visible memory, so pooling them would defeat the point. Readback
//     therefore creates its short-lived staging buffer at the download boundary and never
//     pools it — which is what TypeGPU's `.read()` already does internally, and what
//     splatDensity.ts:111-114 records crashing the vitest worker when done by hand.

import type { LeaseToken, ResidentBuffer } from "./handle";

/** Smallest lease. Buckets are powers of two from here up, so a pool holds at most ~log2(max)
 *  distinct sizes per usage class and fragmentation stays bounded. */
const MIN_BYTES = 256;

/** WebGPU's `GPUBufferUsage` is installed as a global by device acquisition (see
 *  src/gpu/device.ts), so it cannot be read at module scope. Spec values are fixed, and are
 *  used as the fallback when the global is not yet present. */
function bufferUsage(): { STORAGE: number; COPY_SRC: number; COPY_DST: number } {
  const U = (globalThis as unknown as { GPUBufferUsage?: Record<string, number> }).GPUBufferUsage;
  return {
    STORAGE: U?.STORAGE ?? 0x80,
    COPY_SRC: U?.COPY_SRC ?? 0x04,
    COPY_DST: U?.COPY_DST ?? 0x08,
  };
}

/** The *resident class*: readable and writable by compute, and copyable in both directions so a
 *  value can be uploaded at a source and downloaded at a sink. Over-provisioning here is
 *  essentially free — these flags are mostly a placement hint and such buffers live in
 *  device-local memory regardless.
 *
 *  `lease` takes an arbitrary `usage`, so a class the pool doesn't name (e.g. `| VERTEX` for
 *  geometry a render pass binds) works today by passing it explicitly; the pool keys its free
 *  lists on the flags, so classes never alias. Whether VERTEX should simply be folded in here is
 *  open — see ADR-0017's consequences. There is deliberately no named vertex class until
 *  something in the graph actually needs one. */
export function residentUsage(): number {
  const U = bufferUsage();
  return U.STORAGE | U.COPY_SRC | U.COPY_DST;
}

/** Round up to the next power of two, floored at `MIN_BYTES`. */
function bucket(byteLength: number): number {
  let cap = MIN_BYTES;
  while (cap < byteLength) cap *= 2;
  return cap;
}

export interface PoolStats {
  /** Buffers currently leased out (not yet released). */
  live: number;
  /** Buffers sitting in the free lists, available for reuse. */
  free: number;
  /** Physical `createBuffer` calls — the number that should stop growing once a graph
   *  reaches steady state. */
  created: number;
  /** Total bytes physically allocated (capacity, not logical size). */
  bytes: number;
}

/** A liveness-based buffer pool over one device. Buffers are bucketed by (usage, capacity);
 *  a lease is satisfied from the matching free list when one exists and allocated otherwise. */
export class BufferPool {
  private readonly free = new Map<string, GPUBuffer[]>();
  private readonly live = new Set<number>();
  private seq = 0;
  private created = 0;
  private bytes = 0;

  constructor(private readonly device: GPUDevice) {}

  private static slot(usage: number, capacity: number): string {
    return `${usage}:${capacity}`;
  }

  /** Lease a buffer of at least `byteLength` bytes. The returned `ResidentBuffer` records the
   *  *logical* length; the physical buffer is the bucketed capacity and may be larger. */
  lease(byteLength: number, usage: number = residentUsage()): ResidentBuffer {
    const capacity = bucket(Math.max(1, byteLength));
    const slot = BufferPool.slot(usage, capacity);
    const list = this.free.get(slot);
    let buffer = list?.pop();
    if (!buffer) {
      buffer = this.device.createBuffer({ size: capacity, usage });
      this.created++;
      this.bytes += capacity;
    }
    const lease: LeaseToken = { id: this.seq++, usage, capacity };
    this.live.add(lease.id);
    return { buffer, byteLength, lease };
  }

  /** Return a leased buffer to its free list. Never destroys. Releasing a token twice is a
   *  bug in the caller's liveness accounting — it would hand the same buffer to two live
   *  values — so it throws rather than silently corrupting the pool. */
  release(b: ResidentBuffer): void {
    if (!this.live.delete(b.lease.id)) {
      throw new Error(`BufferPool.release: lease ${b.lease.id} is not live (double release?)`);
    }
    const slot = BufferPool.slot(b.lease.usage, b.lease.capacity);
    const list = this.free.get(slot);
    if (list) list.push(b.buffer);
    else this.free.set(slot, [b.buffer]);
  }

  stats(): PoolStats {
    let free = 0;
    for (const list of this.free.values()) free += list.length;
    return { live: this.live.size, free, created: this.created, bytes: this.bytes };
  }
}
