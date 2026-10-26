// A labelled compute pipeline from a TypeGPU resolution.
//
// Every TypeGPU kernel here goes resolve → shader module → pipeline layout (built from the bind-group
// layouts the generated code actually uses) → compute pipeline. All three objects carry the same
// label, so whichever one a validation error cites, it names the kernel.
//
// One entry point per module: a caller that runs several entry points off one module (dancerGpu)
// builds them by hand rather than compiling the same WGSL once per pipeline.

import type tgpu from "typegpu";
import type { TgpuRoot } from "typegpu";

export function typegpuComputePipeline(
  device: GPUDevice,
  root: TgpuRoot,
  resolved: Pick<ReturnType<typeof tgpu.resolveWithContext>, "code" | "usedBindGroupLayouts">,
  entryPoint: string,
  label: string,
): GPUComputePipeline {
  const module = device.createShaderModule({ label, code: resolved.code });
  const layout = device.createPipelineLayout({ label, bindGroupLayouts: resolved.usedBindGroupLayouts.map((l) => root.unwrap(l)) });
  return device.createComputePipeline({ label, layout, compute: { module, entryPoint } });
}
