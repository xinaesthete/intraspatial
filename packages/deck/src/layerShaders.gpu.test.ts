// The layers' draw shaders as luma assembles them — deck's WGSL assembler, deck's modules (`project32`,
// `color`) and each layer's uniform module — compiled and made into a render pipeline on Dawn. A WGSL
// error or a binding the pipeline rejects otherwise shows up only in a browser, as a blank layer.
// What the shaders draw is not checked here.
import { getShaderAssembler, type Layer, LayerManager } from "@deck.gl/core";
import { describe, expect, it, vi } from "vite-plus/test";
import { compileShader, getDevice } from "../../../src/gpu/device";

// This suite resolves `intraspatial` to its build output, which a test run does not have: point the
// layers' imports at the source instead.
vi.mock("intraspatial/gpu/spatial/tcmRender", () => import("../../../src/gpu/spatial/tcmRender"));
vi.mock("intraspatial/gpu/spatial/splatDensity", () => import("../../../src/gpu/spatial/splatDensity"));
vi.mock("intraspatial/gpu/spatial/fieldMax", () => import("../../../src/gpu/spatial/fieldMax"));
vi.mock("intraspatial/spatial/kernels", () => import("../../../src/spatial/kernels"));

const { DensityLayer } = await import("./DensityLayer");
const { TcmLayer } = await import("./TcmLayer");

/** The shaders deck would hand luma for `layer`: with the default modules a LayerManager gives every
 *  layer's context (`layerUniforms`). */
function shadersOf(layer: Layer): { source: string; modules: never[] } {
  const manager = new LayerManager(null as never, {}) as unknown as { context: { defaultShaderModules: unknown[] } };
  Object.assign(layer, { context: { defaultShaderModules: manager.context.defaultShaderModules } });
  // Both layers override getShaders() without arguments.
  return (layer as unknown as { getShaders(): { source: string; modules: never[] } }).getShaders();
}

/** deck 9.4 types `getShaderAssembler` as the GLSL assembler; for "wgsl" it is the WGSL one. */
type WgslAssembler = { assembleWGSLShader(props: unknown): { source: string } };

const PREMULTIPLIED = {
  color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
} as const satisfies GPUBlendState;

async function compileLayer(name: string, shaders: { source: string; modules: never[] }): Promise<string | null> {
  const device = await getDevice();
  const { source } = (getShaderAssembler("wgsl") as unknown as WgslAssembler).assembleWGSLShader({
    platformInfo: { type: "webgpu", shaderLanguage: "wgsl", shaderLanguageVersion: 100, gpu: "test", features: new Set(), limits: {} },
    ...shaders,
  } as never);
  const module = await compileShader(device, source, name); // throws with the WGSL diagnostics
  device.pushErrorScope("validation");
  device.createRenderPipeline({
    label: `${name}:test`,
    layout: "auto",
    vertex: { module, entryPoint: "vertexMain" },
    fragment: { module, entryPoint: "fragmentMain", targets: [{ format: "rgba8unorm", blend: PREMULTIPLIED }] },
    primitive: { topology: "triangle-list" },
  });
  const error = await device.popErrorScope();
  return error ? error.message : null;
}

describe("layer draw shaders, as luma assembles them", () => {
  it("DensityLayer compiles and makes a valid pipeline", async () => {
    const layer = new DensityLayer({ id: "d", x: [0], y: [0], bandwidth: 1 });
    expect(await compileLayer("DensityLayer", shadersOf(layer))).toBeNull();
  });

  it("TcmLayer compiles and makes a valid pipeline", async () => {
    const layer = new TcmLayer({ id: "t", x: [0], y: [0], radius: 1, sigma: 1 });
    expect(await compileLayer("TcmLayer", shadersOf(layer))).toBeNull();
  });
});
