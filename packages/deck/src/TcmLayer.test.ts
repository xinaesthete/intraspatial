// TcmLayer's lifecycle against a fake device: what it hands the renderer, when it renders, and that
// teardown and data replacement never leave a destroyed buffer for a submit to use. The renderer is
// stubbed: it "submits" synchronously, as the real one does, recording any use of a destroyed buffer.
// (The render itself is pinned on a real device in intraspatial's `tcmRenderer.gpu.test.ts`.)

import { OrthographicViewport } from "@deck.gl/core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vite-plus/test";

type FakeBuffer = { label?: string; size: number; destroyed: boolean; destroy(): void };

const gpu = vi.hoisted(() => {
  const errors: string[] = [];
  const use = (b: FakeBuffer, how: string) => {
    if (b.destroyed) errors.push(`[Buffer "${b.label ?? "unlabeled"}"] used in ${how} while destroyed`);
  };
  type Call = { input: { a: Record<string, unknown>; b: Record<string, unknown>; rhoB: number }; o: Record<string, unknown> };
  const renders: Call[] = [];
  const renderers: { label: string; destroyed: boolean }[] = [];
  return { errors, use, renders, renderers };
});

vi.mock("@luma.gl/engine", () => ({
  Model: class {
    shaderInputs = { setProps() {} };
    setBindings() {}
    draw() {}
    destroy() {}
  },
}));

vi.mock("intraspatial/gpu/spatial/tcmRender", () => ({
  tcmRowFloats: (w: number) => Math.ceil((w * 4) / 256) * 64,
  TcmRenderer: class {
    destroyed = false;
    constructor(
      device: { features: Set<string> },
      readonly label: string,
    ) {
      if (!device.features.has("float32-blendable")) throw new Error(`${label}: this device lacks 'float32-blendable'`);
      gpu.renderers.push(this);
    }
    renderResident(input: never, o: never) {
      const { a, b } = input as { a: { points: FakeBuffer; weights: FakeBuffer }; b: { points: FakeBuffer; weights: FakeBuffer } };
      const { out } = o as { out: { buffer: FakeBuffer; max: FakeBuffer } };
      if (this.destroyed) throw new Error("used after destroy()");
      for (const buf of [a.points, a.weights, b.points, b.weights, out.buffer, out.max]) gpu.use(buf, "submit");
      gpu.renders.push({ input, o });
      return {};
    }
    destroy() {
      this.destroyed = true;
    }
  },
}));

// Mocked like every intraspatial import in this package's CPU tests: the CPU suite resolves
// `intraspatial` to its build output, which a test run does not have.
vi.mock("intraspatial/spatial/kernels", () => ({ TOPHAT: { kind: "poly", order: 0 } }));

const { TcmLayer } = await import("./TcmLayer");

class FakeGPUDevice {
  buffers: FakeBuffer[] = [];
  features = new Set<string>(["float32-blendable"]);
  limits = { maxTextureDimension2D: 8192 };
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
  gpu.renders.length = 0;
  gpu.renderers.length = 0;
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const N = 100;
/** A 10×10 lattice on [0, 0.9]²: even indices are type A, multiples of 3 type B. */
const X = Float32Array.from({ length: N }, (_, i) => (i % 10) / 10);
const Y = Float32Array.from({ length: N }, (_, i) => Math.floor(i / 10) / 10);
const WA = Float32Array.from({ length: N }, (_, i) => (i % 2 === 0 ? 1 : 0));
const WB = Float32Array.from({ length: N }, (_, i) => (i % 3 === 0 ? 1 : 0));

type Props = ConstructorParameters<typeof TcmLayer>[0];

function mountLayer(props: Partial<Props> = {}, handle = new FakeGPUDevice(), type = "webgpu") {
  const layerErrors: Error[] = [];
  const context = {
    device: { type, handle, createBuffer: (p: { id?: string }) => ({ id: p.id, destroy() {} }) },
    viewport: new OrthographicViewport({ width: 200, height: 200, target: [0.5, 0.5, 0], zoom: 8 }),
    shaderAssembler: {},
    defaultShaderModules: [],
    resourceManager: { unsubscribe() {} },
    onError: (e: Error) => layerErrors.push(e),
  };
  const layer = new TcmLayer({ id: "tcm-3", x: X, y: Y, weightsA: WA, weightsB: WB, radius: 0.15, sigma: 0.05, ...props });
  const internalState = { needsRedraw: false, uniformTransitions: { clear() {} }, finalize() {} };
  Object.assign(layer, { context, internalState });
  layer.initializeState();
  const update = (next: InstanceType<typeof TcmLayer>, oldProps: object) =>
    next.updateState({ props: next.props, oldProps, context, changeFlags: { somethingChanged: true } } as never);
  update(layer, {});
  /** deck's prop update: a new layer instance with the same state. */
  const setProps = (more: Partial<Props>) => {
    const next = new TcmLayer({ ...layer.props, ...more } as Props);
    Object.assign(next, { context, internalState, state: layer.state });
    update(next, layer.props);
    return next;
  };
  return { layer, context, handle, layerErrors, setProps };
}

const alive = (handle: FakeGPUDevice) => handle.buffers.filter((b) => !b.destroyed).map((b) => b.label);

describe("TcmLayer", () => {
  it("renders synchronously on mount, A and B sharing one point buffer", () => {
    const { layer, layerErrors } = mountLayer();
    expect(layerErrors).toEqual([]);
    expect(gpu.renders).toHaveLength(1);
    expect(layer.state.shown).not.toBeNull();
    const { input, o } = gpu.renders[0]!;
    expect(input.a.points).toBe(input.b.points);
    expect(input.a.weights).not.toBe(input.b.weights);
    expect(input.a.count).toBe(N);
    // ρ_B = 34 B cells over the bounds of every point, [0, 0.9]² (0.9 as an f32).
    expect(input.rhoB).toBeCloseTo(34 / 0.81, 4);
    expect(o).toMatchObject({ radius: 0.15, sigma: 0.05, alpha: 5, radiusSigma: 4 });
  });

  it("takes ρ_B over `region` when given", () => {
    mountLayer({ region: [0, 0, 2, 2] });
    expect(gpu.renders[0]!.input.rhoB).toBeCloseTo(34 / 4, 6);
  });

  it("does not render again while the field still serves, and does when a parameter of Γ changes", () => {
    const { setProps } = mountLayer();
    setProps({ fillOpacity: 0.5 });
    expect(gpu.renders).toHaveLength(1);
    setProps({ sigma: 0.08 });
    expect(gpu.renders).toHaveLength(2);
  });

  it("draws nothing, and renders nothing, when no point is of type A", () => {
    const { layer } = mountLayer({ weightsA: new Float32Array(N) });
    expect(gpu.renders).toHaveLength(0);
    expect(layer.state.shown).toBeNull();
  });

  it("finalize frees every buffer and the renderer at once, with no use after destroy", () => {
    const { layer, context, handle } = mountLayer();
    layer.finalizeState(context as never);
    expect(alive(handle)).toEqual([]);
    expect(gpu.renderers.every((r) => r.destroyed)).toBe(true);
    expect(gpu.errors).toEqual([]);
  });

  it("replacing the data with more points frees the old buffers and never submits them again", () => {
    const { handle, setProps } = mountLayer();
    const big = 1000;
    const next = setProps({
      x: Float32Array.from({ length: big }, (_, i) => (i % 40) / 40),
      y: Float32Array.from({ length: big }, (_, i) => Math.floor(i / 40) / 40),
      weightsA: new Float32Array(big).fill(1),
      weightsB: new Float32Array(big).fill(1),
    });
    expect(gpu.renders).toHaveLength(2);
    expect(gpu.errors).toEqual([]);
    expect(next.state.pointCount).toBe(big);
    // The first points/weights buffers were too small: they are gone, the new ones live.
    expect(handle.buffers.filter((b) => b.destroyed).map((b) => b.label)).toEqual(["tcm-3:xy", "tcm-3:weightsA", "tcm-3:weightsB"]);
  });

  it("labels every buffer with the layer id, so a device error names its layer", () => {
    const { handle } = mountLayer();
    expect(handle.buffers.map((b) => b.label ?? "").sort()).toEqual(
      ["tcm-3:field", "tcm-3:max", "tcm-3:ramp", "tcm-3:weightsA", "tcm-3:weightsB", "tcm-3:xy"].sort(),
    );
    expect(gpu.renderers.map((r) => r.label)).toEqual(["tcm-3:tcm"]);
  });

  it("reports a device without float32-blendable once and stays inert", () => {
    const handle = new FakeGPUDevice();
    handle.features.clear();
    const { layer, layerErrors } = mountLayer({}, handle);
    expect(layerErrors.map((e) => e.message)).toEqual([expect.stringMatching(/float32-blendable/)]);
    expect(layer.state.ready).toBe(false);
    expect(gpu.renders).toHaveLength(0);
  });

  it("reports a WebGL deck and stays inert", () => {
    const { layer, layerErrors } = mountLayer({}, new FakeGPUDevice(), "webgl");
    expect(layerErrors.map((e) => e.message)).toEqual([expect.stringMatching(/needs deck on WebGPU/)]);
    expect(layer.state.ready).toBe(false);
  });
});
