// TcmLayer on a real device (Dawn): the Γ it leaves in its field buffer, for the grid it planned, is
// the TCM of the two subsets — computed independently from separate A and B clouds over a mark grid
// covering the whole region. This pins the layer's wiring: the shared point buffer and its two
// weightings, ρ_B over `region`, the mark grid it plans (wide enough for every A cell that reaches
// the view), and the padded rows it is drawn from. luma's Model is stubbed; the draw shader is
// covered by `layerShaders.gpu.test.ts`.
import { OrthographicViewport } from "@deck.gl/core";
import { describe, expect, it, vi } from "vite-plus/test";
import { getDevice, readBackBytes } from "../../../src/gpu/device";
import { TcmRenderer, tcmRowFloats } from "../../../src/gpu/spatial/tcmRender";
import { QUARTIC } from "../../../src/spatial/kernels";

vi.mock("intraspatial/gpu/spatial/tcmRender", () => import("../../../src/gpu/spatial/tcmRender"));
vi.mock("intraspatial/spatial/kernels", () => import("../../../src/spatial/kernels"));
vi.mock("@luma.gl/engine", () => ({
  Model: class {
    shaderInputs = { setProps() {} };
    setBindings() {}
    draw() {}
    destroy() {}
  },
}));

const { TcmLayer } = await import("./TcmLayer");

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const REGION = [0, 0, 100, 100] as [number, number, number, number];
const RADIUS = 6;
const SIGMA = 3;

/** Three types on one point set: A clustered with B on the left, A away from B on the right, and a
 *  third type neither weighting includes. Plus a NaN point, which both must leave out. */
function scene() {
  const rnd = mulberry32(0xc0ffee);
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() || 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const x: number[] = [];
  const y: number[] = [];
  const type: number[] = [];
  const add = (px: number, py: number, t: number) => (x.push(px), y.push(py), type.push(t));
  for (let i = 0; i < 1500; i++) add(rnd() * 100, rnd() * 100, rnd() < 0.5 ? 1 : 2);
  for (let i = 0; i < 600; i++) add(30 + gauss() * 5, 50 + gauss() * 5, 1);
  for (let i = 0; i < 300; i++) add(30 + gauss() * 4, 50 + gauss() * 4, 0);
  for (let i = 0; i < 300; i++) add(75 + gauss() * 4, 50 + gauss() * 4, 0);
  add(Number.NaN, 10, 0);
  const keep = x.map((px, i) => Number.isFinite(px) && px >= 0 && px <= 100 && y[i]! >= 0 && y[i]! <= 100);
  return {
    x: x.filter((_, i) => keep[i] || Number.isNaN(x[i]!)),
    y: y.filter((_, i) => keep[i] || Number.isNaN(x[i]!)),
    type: type.filter((_, i) => keep[i] || Number.isNaN(x[i]!)),
  };
}

function storage(device: GPUDevice, data: Float32Array, label: string): GPUBuffer {
  const buf = device.createBuffer({ label, size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
  return buf;
}

describe("TcmLayer on a device", () => {
  it("leaves the TCM of the two subsets in its field buffer, for the grid it planned", async () => {
    const device = await getDevice();
    const { x, y, type } = scene();
    const context = {
      device: { type: "webgpu", handle: device, createBuffer: (p: { id?: string }) => ({ id: p.id, destroy() {} }) },
      viewport: new OrthographicViewport({ width: 300, height: 300, target: [50, 50, 0], zoom: 1.5 }),
      shaderAssembler: {},
      defaultShaderModules: [],
      resourceManager: { unsubscribe() {} },
      onError: (e: Error) => {
        throw e;
      },
    };
    const layer = new TcmLayer({
      id: "tcm-gpu",
      x: Float32Array.from(x),
      y: Float32Array.from(y),
      weightsA: Float32Array.from(type, (t) => (t === 0 ? 1 : 0)),
      weightsB: Float32Array.from(type, (t) => (t === 1 ? 1 : 0)),
      radius: RADIUS,
      sigma: SIGMA,
      region: REGION,
      // Smooth, so the comparison can be tight: the top-hat's step makes a fragment on the disk's edge
      // flip with the last bit of the world-to-raster mapping, which differs between the two mark grids.
      kernel: QUARTIC,
    });
    Object.assign(layer, { context, internalState: { needsRedraw: false, uniformTransitions: { clear() {} }, finalize() {} } });
    layer.initializeState();
    layer.updateState({ props: layer.props, oldProps: {}, context, changeFlags: { somethingChanged: true } } as never);
    const plan = layer.state.shown!;
    expect(plan).not.toBeNull();
    // The view is a window: the layer must have planned less than the whole region.
    expect(plan.bbox[0] > 0 || plan.bbox[2] < 100).toBe(true);

    const rowFloats = tcmRowFloats(plan.width);
    const bytes = rowFloats * plan.height * 4;
    const field = new Float32Array(await readBackBytes(device, "test:tcmLayer:field", layer.state.field!, 0, bytes));
    const maxBits = new Float32Array(await readBackBytes(device, "test:tcmLayer:max", layer.state.max!, 0, 4))[0]!;

    // The reference: A and B as separate clouds (no shared buffer, no weights but 1), ρ_B over the
    // region, and marks over the whole region on the layer's mark lattice (min(cell, r/8)).
    const pick = (t: number) => {
      const xy: number[] = [];
      for (let i = 0; i < x.length; i++) if (type[i] === t && Number.isFinite(x[i]!)) xy.push(x[i]!, y[i]!);
      return Float32Array.from(xy);
    };
    const a = pick(0);
    const b = pick(1);
    const nA = a.length / 2;
    const nB = b.length / 2;
    const markCell = Math.min(plan.cell, RADIUS / 8);
    const lo = (v: number) => Math.floor(v / markCell) * markCell;
    const hi = (v: number) => Math.ceil(v / markCell) * markCell;
    const markBox = [lo(-RADIUS), lo(-RADIUS), hi(100 + RADIUS), hi(100 + RADIUS)] as const;
    const ref = new TcmRenderer(device, "test:tcmLayer:ref");
    const out = device.createBuffer({
      label: "test:tcmLayer:refField",
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    ref.renderResident(
      {
        a: { points: storage(device, a, "test:a"), weights: storage(device, new Float32Array(nA).fill(1), "test:aw"), count: nA },
        b: { points: storage(device, b, "test:b"), weights: storage(device, new Float32Array(nB).fill(1), "test:bw"), count: nB },
        rhoB: nB / (100 * 100),
      },
      {
        radius: RADIUS,
        sigma: SIGMA,
        alpha: 5,
        kernel: QUARTIC,
        radiusSigma: 4,
        grid: plan,
        markGrid: {
          bbox: markBox,
          width: Math.round((markBox[2] - markBox[0]) / markCell),
          height: Math.round((markBox[3] - markBox[1]) / markCell),
        },
        out: { buffer: out },
      },
    );
    const expected = new Float32Array(await readBackBytes(device, "test:tcmLayer:refField", out, 0, bytes));

    let peak = 0;
    let err = 0;
    let pos = 0;
    let neg = 0;
    for (let row = 0; row < plan.height; row++) {
      for (let col = 0; col < plan.width; col++) {
        const e = expected[row * rowFloats + col]!;
        peak = Math.max(peak, Math.abs(e));
        err = Math.max(err, Math.abs(field[row * rowFloats + col]! - e));
        if (e > 0) pos++;
        if (e < 0) neg++;
      }
    }
    expect(peak).toBeGreaterThan(0);
    // The scene has both: A clustered with B (Γ > 0) and A excluded from B (Γ < 0).
    expect(pos).toBeGreaterThan(0);
    expect(neg).toBeGreaterThan(0);
    expect(err / peak).toBeLessThan(1e-4);
    expect(maxBits).toBeCloseTo(peak, 3);

    layer.finalizeState(context as never);
    ref.destroy();
  });
});
