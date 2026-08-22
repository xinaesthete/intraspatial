// Tier-2 graph node wrapping `encodeGridIndex` (points -> offset list) — ADR-0022's index as a
// node, design note `docs/gpu-spatial-index-and-scan.md` §3.3.
//
// ONE output: a `gridIndex` **bundle** (ADR-0023) whose parts are `start`, `items` and `lattice`.
// Three sibling ports was the first shape (ADR-0022), and it let a graph wire `start` from one
// index and `items` from another — both are `points`/u32, so the mistake typechecks and returns a
// plausible wrong neighbourhood. A bundle makes that unrepresentable. `gridIndex.start` and its
// siblings (from `extractOp`) take a part out, borrowing the buffer rather than copying it.
//
// ## What a consumer must know
//
// **`start` and `items` are u32, and the host bridge is f32-only** (ADR-0017 stage 1), so a
// `pull()` of the bundle throws naming the offending part. They feed *resident* consumers
// (`cellCounts` is the first); `pullResident` hands the bundle over whole, and `mode: "cpu"` runs
// `cpuGolden`, which produces host arrays instead.
//
// **The lattice is a param, not a measurement.** `inferShapes` must produce `points{M+1}` at
// graph-build time, when no values exist, so the extent cannot be read off the points — it comes
// from the `minX`/`minY`/`maxX`/`maxY` params, exactly as `buildBucketGrid`'s optional `bounds`
// does. Points outside are clamped into the edge cells (the CPU build's rule, and safe for a 3×3
// query: a clamped point is at least as far away as its cell implies).
import type { Vec3 } from "../../../coords";
import { type BucketGrid, buildBucketGrid, type GridLattice, latticeFor, numCells } from "../../../spatial/bucketGrid";
import { encodeGridIndex, getGridIndexCtx } from "../../spatial/gridIndex";
import type { FieldValue, ResolvedPlacement, Shape } from "../handle";
import type { ExecCtx, OpType, Params } from "../op";
import { type BundleSpec, bundleValue, combineOp, extractOp } from "./bundleOps";

/** The bundle this op produces; `extractOp`/`combineOp` generate its accessors from it. */
export const GRID_INDEX_BUNDLE: BundleSpec = { name: "gridIndex", label: "Grid index", parts: ["start", "items", "lattice"] };

/** What the `lattice` port carries. Shaped so a consumer can rebuild the cell arithmetic without
 *  reading either buffer, and so a readback can be compared against `buildBucketGrid`. */
export interface GridLatticePayload extends GridLattice {
  /** Discriminator — the part's *shape* is a grid (that is what makes a consumer's extent
   *  inferable), so the payload is what says "this grid is an index lattice". */
  readonly kind: typeof GRID_LATTICE;
  /** Cell count (`cols * rows`), i.e. `start.length - 1`. */
  readonly cells: number;
  /** Points indexed. */
  readonly n: number;
  /** Where the lattice sits in world space, when the points were placed. */
  readonly placement?: ResolvedPlacement;
}

/** Tag on the lattice payload, so a reader can recognise it. */
export const GRID_LATTICE = "gridLattice";

function pointsN(s: Shape): number {
  if (s.kind !== "points") throw new Error("gridIndex: input must be a points cloud");
  return s.n;
}

function latticeOf(params: Params): GridLattice {
  const cell = params.cell as number;
  if (!(cell > 0)) throw new Error(`gridIndex: cell must be > 0, got ${cell}`);
  const bounds = [params.minX, params.minY, params.maxX, params.maxY] as [number, number, number, number];
  if (bounds.some((b) => !Number.isFinite(b))) throw new Error(`gridIndex: bounds must be finite, got [${bounds.join(", ")}]`);
  if (bounds[2] < bounds[0] || bounds[3] < bounds[1]) throw new Error(`gridIndex: max bound below min in [${bounds.join(", ")}]`);
  // `latticeFor` is the CPU build's own arithmetic, so the op, `buildBucketGrid` and the kernel
  // all agree on which cell a boundary point lands in.
  return latticeFor([], [], cell, bounds);
}

/** The lattice's own placement (note §3.2). Array space for a points cloud is the point
 *  coordinates themselves, so cell (cx, cy) sits at point `(minX + cx·cell, minY + cy·cell)`:
 *
 *      world = origin + (minX + cx·cell)·axes[0] + (minY + cy·cell)·axes[1]
 *            = [origin + minX·axes[0] + minY·axes[1]] + cx·(cell·axes[0]) + cy·(cell·axes[1])
 *
 *  i.e. `worldFromArray · translate(origin) · scale(cell)`, corner-indexed like
 *  `decimatedPlacement` — cell (cx, cy) covers `[cx, cx+1) × [cy, cy+1)` of the lattice, and its
 *  centre is half an axis step in. A per-cell consumer is therefore already placed, and
 *  `systemsAgree` rejects a foreign query cloud.
 */
function latticePlacement(pts: ResolvedPlacement | undefined, lattice: GridLattice): ResolvedPlacement | undefined {
  if (!pts) return undefined;
  const { origin, axes } = pts.worldFromArray;
  const [ax, ay, az] = axes;
  const scale = (v: Vec3, k: number): Vec3 => [v[0] * k, v[1] * k, v[2] * k];
  const shifted: Vec3 = [
    origin[0] + ax[0] * lattice.minX + ay[0] * lattice.minY,
    origin[1] + ax[1] * lattice.minX + ay[1] * lattice.minY,
    origin[2] + ax[2] * lattice.minX + ay[2] * lattice.minY,
  ];
  return { system: pts.system, worldFromArray: { origin: shifted, axes: [scale(ax, lattice.cell), scale(ay, lattice.cell), az] } };
}

function payload(lattice: GridLattice, n: number, placement?: ResolvedPlacement): GridLatticePayload {
  return { kind: GRID_LATTICE, ...lattice, cells: numCells(lattice), n, placement };
}

export const gridIndexOp: OpType = {
  name: "gridIndex",
  label: "Uniform grid index",
  category: "Spatial",
  describe: "Bucket a points cloud into a uniform grid: per-cell start offsets and point indices, built on the device.",
  help: {
    detail:
      "The offset list every neighbourhood query walks: with the cell size set to the query radius, " +
      "each point's 3×3 cell neighbourhood holds every point within that radius. `start[b]..start[b+1]` " +
      "is cell b's slice of `items`. Both outputs are u32 and stay on the device.",
  },
  inputs: [{ name: "points", kind: "points", dtype: "f32" }],
  outputs: [{ name: "index", kind: "bundle" }],
  params: [
    { name: "cell", type: "number", default: 1, min: 1e-6, describe: "Cell side in world units — set it to the query radius." },
    { name: "minX", type: "number", default: 0, describe: "Lattice origin x." },
    { name: "minY", type: "number", default: 0, describe: "Lattice origin y." },
    { name: "maxX", type: "number", default: 1, describe: "Lattice extent x (the far corner, not a width)." },
    { name: "maxY", type: "number", default: 1, describe: "Lattice extent y." },
  ],
  inferShapes(inputs, params) {
    const n = pointsN(inputs[0]!);
    const lattice = latticeOf(params);
    return [
      {
        kind: "bundle",
        name: GRID_INDEX_BUNDLE.name,
        parts: {
          start: { kind: "points", n: numCells(lattice) + 1 },
          items: { kind: "points", n: Math.max(n, 1) },
          // The lattice part IS the cell grid, so its shape says so: `cols × rows`. That makes
          // a consumer's output extent (`cellCounts`) inferable at build time, and it is more
          // honest than `opaque` — the payload beside it carries cell size, origin and placement.
          lattice: { kind: "grid", width: lattice.cols, height: lattice.rows },
        },
      },
    ];
  },
  inferPlacement() {
    // The bundle is not itself placed: `start`/`items` are index lists with no position, and the
    // `lattice` part carries the cell grid's placement (stamped in `execute` / `cpuGolden`).
    return [undefined];
  },
  resident: true,
  async execute(ctx: ExecCtx, inputs: FieldValue[], params: Params) {
    const pts = inputs[0]!;
    const n = pointsN(pts.shape);
    const lattice = latticeOf(params);
    const cells = numCells(lattice);
    const gridCtx = await getGridIndexCtx();
    const { device } = gridCtx.scan;

    // A host points cloud arriving at a resident op is uploaded by the executor's bridge, but an
    // op may still be pulled directly with host data (`pullResident` on a source edge), so accept
    // both. The upload is a lease this call owns and returns.
    const uploaded = pts.buffer ? undefined : await ctx.backend.upload(pts.data as Float32Array);
    const src = pts.buffer ?? uploaded!;

    // One lease per output port (ADR-0017: the executor releases per `(node, port)`, so two ports
    // may not share a buffer). The index's own buffers are POOLED under a global key, so they are
    // copied into the leases in the same submit: a second `gridIndex` node in the same tick reuses
    // that pool, and only the copies are private to this node.
    const startOut = await ctx.backend.lease((cells + 1) * 4);
    const itemsOut = await ctx.backend.lease(Math.max(n, 1) * 4);

    const enc = device.createCommandEncoder({ label: "gridIndexOp" });
    const idx = encodeGridIndex(gridCtx, src.buffer, n, lattice, enc);
    enc.copyBufferToBuffer(idx.start, 0, startOut.buffer, 0, (cells + 1) * 4);
    if (n > 0) enc.copyBufferToBuffer(idx.items, 0, itemsOut.buffer, 0, n * 4);
    device.queue.submit([enc.finish()]);

    if (uploaded) ctx.backend.release(uploaded);

    const place = latticePlacement(pts.placement, lattice);
    return [
      bundleValue(GRID_INDEX_BUNDLE, [
        { shape: { kind: "points", n: cells + 1 }, dtype: "u32", buffer: startOut },
        { shape: { kind: "points", n: Math.max(n, 1) }, dtype: "u32", buffer: itemsOut },
        {
          shape: { kind: "grid", width: lattice.cols, height: lattice.rows },
          dtype: "f32",
          payload: payload(lattice, n, place),
          placement: place,
        },
      ]),
    ];
  },
  cpuGolden(inputs, params) {
    const pts = inputs[0]!;
    const n = pointsN(pts.shape);
    const lattice = latticeOf(params);
    const flat = pts.data as Float32Array;
    const xs = new Float32Array(n);
    const ys = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      xs[i] = flat[2 * i]!;
      ys[i] = flat[2 * i + 1]!;
    }
    const grid: BucketGrid = buildBucketGrid(xs, ys, lattice.cell, [
      lattice.minX,
      lattice.minY,
      params.maxX as number,
      params.maxY as number,
    ]);
    const items = new Uint32Array(Math.max(n, 1));
    items.set(grid.items);
    const place = latticePlacement(pts.placement, lattice);
    return [
      bundleValue(GRID_INDEX_BUNDLE, [
        { shape: { kind: "points", n: grid.start.length }, dtype: "u32", data: new Uint32Array(grid.start) },
        { shape: { kind: "points", n: Math.max(n, 1) }, dtype: "u32", data: items },
        {
          shape: { kind: "grid", width: lattice.cols, height: lattice.rows },
          dtype: "f32",
          payload: payload(lattice, n, place),
          placement: place,
        },
      ]),
    ];
  },
};

/** `gridIndex.start` / `.items` / `.lattice` — take one part out, borrowing the buffer rather
 *  than copying it (ADR-0023). */
export const gridIndexPartOps: OpType[] = [
  extractOp(GRID_INDEX_BUNDLE, "start", "Per-cell start offsets into `items` (u32, device-only)."),
  extractOp(GRID_INDEX_BUNDLE, "items", "Point indices grouped by cell (u32, device-only)."),
  extractOp(GRID_INDEX_BUNDLE, "lattice", "The lattice the index was built over — cols, rows, cell, origin."),
];

/** `gridIndex.bundle` — reassemble an index from parts, for a producer that builds them itself. */
export const gridIndexCombineOp: OpType = combineOp(GRID_INDEX_BUNDLE, { start: "points", items: "points", lattice: "opaque" });
