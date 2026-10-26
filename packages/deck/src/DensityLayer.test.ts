// Teardown while a splat is in flight. deck finalizes a layer whenever a host replaces it, and the
// splat keeps running after that: it awaits (pipeline compiles, per-device state) before each submit.
// A buffer destroyed in that window fails the submit with "used in submit while destroyed". These
// tests drive the layer against a fake device that records any use of a destroyed buffer, with the
// two kernels stubbed to wait on the test before they touch their buffers.

import { OrthographicViewport } from "@deck.gl/core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vite-plus/test";

type FakeBuffer = { label?: string; size: number; destroyed: boolean; destroy(): void };

const gpu = vi.hoisted(() => {
  const errors: string[] = [];
  /** What a submit or writeBuffer does to a buffer: fail if it was destroyed. */
  const use = (b: FakeBuffer, how: string) => {
    if (b.destroyed) errors.push(`[Buffer "${b.label ?? "unlabeled"}"] used in ${how} while destroyed`);
  };
  /** Kernel calls parked at their internal await, released by the test. */
  const parked: { kernel: string; release: () => void }[] = [];
  const park = (kernel: string) => new Promise<void>((release) => parked.push({ kernel, release }));
  return { errors, use, parked, park };
});

vi.mock("@luma.gl/engine", () => ({
  Model: class {
    shaderInputs = { setProps() {} };
    setBindings() {}
    draw() {}
    destroy() {}
  },
}));

vi.mock("intraspatial/gpu/spatial/splatDensity", () => ({
  async splatDensityResident(points: FakeBuffer, _n: number, dst: FakeBuffer) {
    await gpu.park("splat"); // the real one awaits its per-device state before encoding
    gpu.use(points, "submit");
    gpu.use(dst, "submit");
  },
}));

vi.mock("intraspatial/gpu/spatial/fieldMax", () => ({
  async fieldMaxResident(src: FakeBuffer, _n: number, dst: FakeBuffer) {
    await gpu.park("fieldMax"); // getCtx
    gpu.use(dst, "writeBuffer");
    gpu.use(src, "submit");
    gpu.use(dst, "submit");
  },
}));

const { DensityLayer } = await import("./DensityLayer");

class FakeGPUDevice {
  buffers: FakeBuffer[] = [];
  features = new Set<string>();
  queue = { writeBuffer: (b: FakeBuffer) => gpu.use(b, "writeBuffer") };
  createBuffer({ size, label }: { size: number; label?: string }): FakeBuffer {
    const b: FakeBuffer = {
      label,
      size,
      destroyed: false,
      destroy() {
        b.destroyed = true;
      },
    };
    this.buffers.push(b);
    return b;
  }
}

beforeEach(() => {
  vi.stubGlobal("GPUDevice", FakeGPUDevice);
  vi.stubGlobal("GPUBufferUsage", { STORAGE: 0x80, COPY_DST: 0x8, COPY_SRC: 0x4, UNIFORM: 0x40 });
  gpu.errors.length = 0;
  gpu.parked.length = 0;
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** A layer as deck's LayerManager would have it after its first update: a splat in flight. */
function mountLayer() {
  const handle = new FakeGPUDevice();
  const layerErrors: Error[] = [];
  const context = {
    device: {
      type: "webgpu",
      handle,
      createBuffer: (props: { id?: string }) => ({ id: props.id, destroy() {} }),
    },
    viewport: new OrthographicViewport({ width: 200, height: 200, target: [0.5, 0.5, 0], zoom: 8 }),
    shaderAssembler: {},
    defaultShaderModules: [],
    resourceManager: { unsubscribe() {} },
    onError: (e: Error) => layerErrors.push(e),
  };
  const n = 100;
  const layer = new DensityLayer({
    id: "density-7",
    x: Float32Array.from({ length: n }, (_, i) => (i % 10) / 10),
    y: Float32Array.from({ length: n }, (_, i) => Math.floor(i / 10) / 10),
    bandwidth: 0.05,
  });
  const internalState = { needsRedraw: false, uniformTransitions: { clear() {} }, finalize() {} };
  Object.assign(layer, { context, internalState });
  layer.initializeState();
  layer.updateState({
    props: layer.props,
    oldProps: {},
    context,
    changeFlags: { somethingChanged: true },
  } as never);
  return { layer, context, handle, layerErrors };
}

/** Let every settled promise run its continuations. */
const flush = () => new Promise<void>((r) => setTimeout(r, 0));

async function release(kernel: string) {
  const i = gpu.parked.findIndex((p) => p.kernel === kernel);
  expect(i, `${kernel} is parked`).toBeGreaterThanOrEqual(0);
  gpu.parked.splice(i, 1)[0]!.release();
  await flush();
}

describe("DensityLayer teardown", () => {
  it("finalized before the splat submits: no destroyed buffer is used, and all are freed after", async () => {
    const { layer, context, handle, layerErrors } = mountLayer();
    expect(layer.state.busy).toBe(true);
    layer.finalizeState(context as never);
    await release("splat");
    await flush();
    expect(gpu.errors).toEqual([]);
    expect(layerErrors).toEqual([]);
    // A finalized layer submits no further work…
    expect(gpu.parked).toEqual([]);
    // …and once its splat has settled, nothing it created is left alive.
    expect(handle.buffers.filter((b) => !b.destroyed).map((b) => b.label)).toEqual([]);
  });

  it("finalized between the splat and the max: no destroyed buffer is used, and all are freed after", async () => {
    const { layer, context, handle } = mountLayer();
    await release("splat");
    expect(layer.state.busy).toBe(true);
    layer.finalizeState(context as never);
    await release("fieldMax");
    expect(gpu.errors).toEqual([]);
    expect(layer.state.shown).toBeNull();
    expect(handle.buffers.filter((b) => !b.destroyed).map((b) => b.label)).toEqual([]);
  });

  it("finalized while idle frees everything at once", async () => {
    const { layer, context, handle } = mountLayer();
    await release("splat");
    await release("fieldMax");
    expect(layer.state.busy).toBe(false);
    expect(layer.state.shown).not.toBeNull();
    layer.finalizeState(context as never);
    expect(handle.buffers.filter((b) => !b.destroyed).map((b) => b.label)).toEqual([]);
    expect(gpu.errors).toEqual([]);
  });

  it("labels every buffer with the layer id, so a device error names its layer", async () => {
    const { handle } = mountLayer();
    expect(handle.buffers.map((b) => b.label ?? "").sort()).toEqual(
      ["density-7:field", "density-7:max", "density-7:points", "density-7:ramp"].sort(),
    );
  });
});
