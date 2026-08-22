// `gridIndex` bundle -> grid of per-cell point counts (ADR-0023's first consumer).
//
// The count of cell b is `start[b+1] - start[b]` — the offset list already holds it, so this is a
// one-line kernel over `cols*rows` threads and no second pass over the points. Statistically it
// is the quadrat count the spatial-stats front keeps reaching for (`quadratCorrelationGpu`,
// `getisOrd`'s input, a null model's expected density); visually it is what makes an index
// legible, which is why the composer example ends here rather than at a dangling port.
//
// It reads the bundle whole rather than taking a `start` port, which is the point of ADR-0023: a
// consumer cannot be handed one index's offsets and another's lattice.
//
// The output grid inherits the LATTICE's placement, not the points': one cell of this grid is one
// cell of the index, so it is already in world space via `worldFromArray · translate(origin) ·
// scale(cell)`.
import { checkBindingSize, compileShader, dispatchGrid, MAX_WORKGROUPS_PER_DIM, sized } from "../../device";
import { ensureBuf, getScanCtx } from "../../scan/prefixSum";
import type { FieldValue, Shape } from "../handle";
import type { ExecCtx, OpType, Params } from "../op";
import type { GridLatticePayload } from "./gridIndex";
import { GRID_INDEX_BUNDLE, GRID_LATTICE } from "./gridIndex";

const WG = 64;

const SHADER = /* wgsl */ `
struct Uni { cells: u32, gridX: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> start: array<u32>;
@group(0) @binding(2) var<storage, read_write> counts: array<f32>;

@compute @workgroup_size(${WG})
fn cellCounts(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let b = (wid.x + wid.y * U.gridX) * ${WG}u + lid.x;
  if (b >= U.cells) { return; }
  // start is M+1 long, so start[b + 1] is always in range for b < M.
  counts[b] = f32(start[b + 1u] - start[b]);
}
`;

interface Ctx {
  device: GPUDevice;
  layout: GPUBindGroupLayout;
  pipeline: GPUComputePipeline;
}
let ctxCache: Promise<Ctx> | undefined;

function getCtx(): Promise<Ctx> {
  ctxCache ??= (async () => {
    // The scan context owns the device this index was built on; sharing it keeps the pool keys
    // and the device in one place.
    const { device } = await getScanCtx();
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const module = await compileShader(device, SHADER, "cellCounts");
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: "cellCounts" },
    });
    return { device, layout, pipeline };
  })();
  return ctxCache;
}

function latticeOf(v: FieldValue, who: string): GridLatticePayload {
  const lat = v.parts?.lattice?.payload as GridLatticePayload | undefined;
  if (lat?.kind !== GRID_LATTICE)
    throw new Error(`${who}: input is not a grid-index bundle (its lattice part carries no ${GRID_LATTICE} payload)`);
  return lat;
}

/** The output extent is the index's own cell grid, which the bundle's `lattice` part declares —
 *  so the shape is exact at graph-build time, no run-time surprise. */
function latticeShape(s: Shape, who: string): { width: number; height: number } {
  if (s.kind !== "bundle" || s.name !== GRID_INDEX_BUNDLE.name) {
    throw new Error(`${who}: expected a "${GRID_INDEX_BUNDLE.name}" bundle, got ${s.kind}`);
  }
  const lat = s.parts.lattice;
  if (!lat || lat.kind !== "grid") throw new Error(`${who}: the bundle's lattice part is not a grid`);
  return { width: lat.width, height: lat.height };
}

export const cellCountsOp: OpType = {
  name: "cellCounts",
  label: "Points per cell",
  category: "Spatial & TDA",
  describe: "How many points fell in each cell of a bucket grid — a coarse density raster at the cell size.",
  help: {
    detail:
      "Differences of the bucket grid's cell offsets: cell b holds `start[b+1] - start[b]` points. " +
      "Statisticians call it a quadrat count; here it is also the cheapest way to SEE what the " +
      "bucketing did, since the counts are f32 and can come back to the host while the index itself " +
      "cannot.",
  },
  inputs: [{ name: "buckets", kind: "bundle" }],
  outputs: [{ name: "counts", kind: "grid", dtype: "f32" }],
  params: [],
  inferShapes(inputs) {
    return [{ kind: "grid", ...latticeShape(inputs[0]!, "cellCounts") }];
  },
  inferPlacement() {
    // The counts sit on the index's cell grid, so the placement comes from the lattice part at
    // run time (`execute` stamps it) rather than passing the bundle's through.
    return [undefined];
  },
  resident: true,
  async execute(_ctx: ExecCtx, inputs: FieldValue[], _params: Params) {
    const bundle = inputs[0]!;
    const lat = latticeOf(bundle, "cellCounts");
    const startPart = bundle.parts!.start!;
    const src = startPart.buffer;
    if (!src) throw new Error("cellCounts: the bundle's start part is not resident");

    const { device, layout, pipeline } = await getCtx();
    const cells = lat.cells;
    checkBindingSize(device, `cellCounts: ${cells} cells`, (cells + 1) * 4);

    // Its own lease, not a borrow: this op creates a value rather than passing one through.
    const out = await _ctx.backend.lease(cells * 4);
    const uni = ensureBuf(device, "cellCounts:uni", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const { x: gridX, y: gridY } = dispatchGrid(Math.ceil(cells / WG), MAX_WORKGROUPS_PER_DIM);
    device.queue.writeBuffer(uni, 0, new Uint32Array([cells, gridX, 0, 0]));

    const bind = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: sized(uni, 16) },
        { binding: 1, resource: sized(src.buffer, (cells + 1) * 4) },
        { binding: 2, resource: sized(out.buffer, cells * 4) },
      ],
    });
    const enc = device.createCommandEncoder({ label: "cellCounts" });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(gridX, gridY);
    pass.end();
    device.queue.submit([enc.finish()]);

    return [
      {
        shape: { kind: "grid", width: lat.cols, height: lat.rows },
        dtype: "f32",
        buffer: out,
        placement: lat.placement,
      },
    ];
  },
  cpuGolden(inputs) {
    const bundle = inputs[0]!;
    const lat = latticeOf(bundle, "cellCounts");
    const start = bundle.parts!.start!.data as Uint32Array | Int32Array | undefined;
    if (!start) throw new Error("cellCounts: the bundle's start part has no host data");
    const out = new Float32Array(lat.cells);
    for (let b = 0; b < lat.cells; b++) out[b] = start[b + 1]! - start[b]!;
    return [{ shape: { kind: "grid", width: lat.cols, height: lat.rows }, dtype: "f32", data: out, placement: lat.placement }];
  },
};
