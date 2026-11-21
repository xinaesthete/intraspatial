// The part every spring-model op repeats, written once: a resident op with one TGSL kernel, some
// read-only inputs and one output, plus its CPU golden. An op file declares its bind-group
// layout and kernel (TGSL needs those at module scope) and hands them here with the two things
// that differ: how to fill the uniform block, and the same maths in TypeScript.
import tgpu from "typegpu";
import { submitDispatch } from "#src/gpu/device";
import { typegpuComputePipeline } from "#src/gpu/typegpuPipeline";
import type { Root } from "../../backend";
import type { ElementType, FieldValue, Shape } from "../../handle";
import type { OpHelp, OpType, ParamSpec, Params } from "../../op";
import { type BindEntry, rawBindGroup } from "../../residentBind";
import { VEC4 } from "./bondTable";
import { VEC2 } from "./common";

const WG = 64;
/** The workgroup size every kernel made for `kernelOp` must declare. */
export const KERNEL_WORKGROUP = WG;

type Lanes = 1 | 2 | 4;

export interface KernelPort {
  readonly name: string;
  readonly kind: "points" | "matrix" | "scalar";
  readonly dtype: "f32" | "u32";
  /** Floats per sample: 1 (the default) for a scalar field, 2 or 4 for a vec. */
  readonly lanes?: Lanes;
}

const elementOf = (lanes: Lanes | undefined): ElementType => (lanes === 4 ? VEC4 : lanes === 2 ? VEC2 : { kind: "scalar" });

function count(shape: Shape): number {
  if (shape.kind === "points") return shape.n;
  if (shape.kind === "matrix") return shape.rows * shape.cols;
  return 1;
}

/** A uniform buffer that takes a value of `U`. */
export type UniformBlock<U> = BindEntry & { write(value: U): void };

export interface KernelOpConfig<U> {
  readonly name: string;
  readonly label: string;
  readonly describe: string;
  readonly help?: OpHelp;
  /** In the order the layout declares them, after the uniform block and before the output. */
  readonly inputs: readonly KernelPort[];
  /** The output: one sample per sample of the first input. */
  readonly output: Omit<KernelPort, "kind">;
  readonly params: ParamSpec[];
  readonly layout: Parameters<typeof rawBindGroup>[2];
  readonly kernel: Parameters<typeof tgpu.resolveWithContext>[0][number];
  /** The kernel's `$name`, when several ops share one kernel and so it is not the op's name. */
  readonly entryPoint?: string;
  /** The uniform block, made once per device. */
  uniform(root: Root): UniformBlock<U>;
  /** Values for the uniform block. `n` is the number of output samples. */
  uniforms(n: number, params: Params, shapes: readonly Shape[]): U;
  /** The same maths on the host: one array per input, in order. */
  cpu(inputs: readonly ArrayLike<number>[], n: number, params: Params, shapes: readonly Shape[]): Float32Array | Uint32Array;
  /** Extra build-time checks on the input shapes. */
  check?(shapes: readonly Shape[]): void;
}

/** A resident graph op around one kernel. */
/**
 * A resident graph op around one kernel. Give `U` explicitly, as `d.Infer<typeof TheStruct>`: a
 * TypeGPU buffer's `write` is overloaded, so it cannot be inferred from the buffer.
 */
export function kernelOp<U>(cfg: KernelOpConfig<U>): OpType {
  const pipes = new WeakMap<object, { pipeline: GPUComputePipeline; uniform: UniformBlock<U> }>();
  const getPipe = (device: GPUDevice, root: Root) => {
    let p = pipes.get(root as object);
    if (!p) {
      const resolved = tgpu.resolveWithContext([cfg.kernel], { names: "strict" });
      p = { pipeline: typegpuComputePipeline(device, root, resolved, cfg.entryPoint ?? cfg.name, cfg.name), uniform: cfg.uniform(root) };
      pipes.set(root as object, p);
    }
    return p;
  };
  const outLanes = cfg.output.lanes ?? 1;
  const outShape = (shapes: readonly Shape[]): Shape => shapes[0]!;
  const checked = (shapes: readonly Shape[]): number => {
    const first = shapes[0]!;
    cfg.inputs.forEach((port, k) => {
      const s = shapes[k]!;
      if (s.kind !== port.kind) throw new Error(`${cfg.name}.${port.name}: expected a ${port.kind} field, got ${s.kind}`);
      if (port.kind === "points" && first.kind === "points" && count(s) !== count(first)) {
        throw new Error(`${cfg.name}.${port.name}: for ${count(s)} particles, but ${cfg.inputs[0]!.name} is for ${count(first)}`);
      }
    });
    cfg.check?.(shapes);
    return count(first);
  };
  const value = (shapes: readonly Shape[], payload: Pick<FieldValue, "data" | "buffer">): FieldValue => ({
    shape: outShape(shapes),
    dtype: cfg.output.dtype,
    element: elementOf(cfg.output.lanes),
    ...payload,
  });

  return {
    name: cfg.name,
    label: cfg.label,
    describe: cfg.describe,
    category: "Springs",
    help: cfg.help,
    inputs: cfg.inputs.map((p) => ({ name: p.name, kind: p.kind, dtype: p.dtype })),
    outputs: [{ name: cfg.output.name, kind: cfg.inputs[0]!.kind, dtype: cfg.output.dtype }],
    params: cfg.params,
    inferShapes(shapes) {
      checked(shapes);
      return [outShape(shapes)];
    },
    inferElements(elements) {
      cfg.inputs.forEach((port, k) => {
        const el = elements[k];
        const lanes = port.lanes ?? 1;
        const got = !el || el.kind === "scalar" ? 1 : el.kind === "vec" ? el.n : 0;
        if (got !== lanes) throw new Error(`${cfg.name}.${port.name}: expected ${lanes === 1 ? "a scalar" : `a vec${lanes}`} field`);
      });
      return [elementOf(cfg.output.lanes)];
    },
    resident: true,
    async execute(ctx, inputs, params) {
      const shapes = inputs.map((v) => v.shape);
      const n = checked(shapes);
      const device = await ctx.backend.getDevice();
      const root = await ctx.backend.getRoot();
      const pipe = getPipe(device, root);
      const out = await ctx.backend.lease(Math.max(n, 1) * outLanes * 4);
      pipe.uniform.write(cfg.uniforms(n, params, shapes));
      const buffers = inputs.map((v, k) => {
        if (!v.buffer) throw new Error(`${cfg.name}.${cfg.inputs[k]!.name}: resident op received a non-resident input`);
        return v.buffer.buffer;
      });
      const bind = rawBindGroup(device, root, cfg.layout, [pipe.uniform, ...buffers, out.buffer], cfg.name);
      submitDispatch(device, cfg.name, pipe.pipeline, bind, Math.max(1, Math.ceil(n / WG)));
      return [value(shapes, { buffer: out })];
    },
    cpuGolden(inputs, params) {
      const shapes = inputs.map((v) => v.shape);
      const n = checked(shapes);
      const data = inputs.map((v, k) => {
        if (!v.data) throw new Error(`${cfg.name}.${cfg.inputs[k]!.name}: cpu golden needs host data`);
        return v.data;
      });
      return [value(shapes, { data: cfg.cpu(data, n, params, shapes) })];
    },
  };
}
